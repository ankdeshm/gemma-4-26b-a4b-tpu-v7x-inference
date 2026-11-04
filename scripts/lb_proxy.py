#!/usr/bin/env python3
"""Minimal least-outstanding-requests HTTP load balancer for OpenAI-compatible servers.

Stands in for a production NGINX/Envoy `least_conn` front end so that a single `vllm bench serve`
client can drive several independent replicas (e.g. 2x or 4x single-host servers = 8 or 16 chips).
Streaming responses are forwarded chunk-by-chunk (no buffering).

  python3 lb_proxy.py --listen 0.0.0.0:8000 --backends http://10.0.0.1:30000,http://10.0.0.2:30000
"""
import argparse
import asyncio
import itertools
import json
import time

import aiohttp
from aiohttp import web

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "content-length", "host", "content-encoding"}


class Backend:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.inflight = 0
        self.total = 0
        self.errors = 0


def make_app(backends, stats_path):
    rr = itertools.count()
    session_holder = {}

    async def on_startup(app):
        session_holder["s"] = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, ttl_dns_cache=300),
            timeout=aiohttp.ClientTimeout(total=None, sock_read=7200))

    async def on_cleanup(app):
        await session_holder["s"].close()

    def pick():
        start = next(rr) % len(backends)
        order = backends[start:] + backends[:start]
        return min(order, key=lambda b: b.inflight)

    async def lb_stats(request):
        return web.json_response({b.url: {"inflight": b.inflight, "total": b.total, "errors": b.errors}
                                  for b in backends})

    async def proxy(request: web.Request):
        b = pick()
        b.inflight += 1
        b.total += 1
        body = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        try:
            async with session_holder["s"].request(request.method, b.url + request.rel_url.path_qs,
                                                   data=body, headers=headers) as up:
                resp = web.StreamResponse(status=up.status,
                                          headers={k: v for k, v in up.headers.items() if k.lower() not in HOP})
                await resp.prepare(request)
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        except Exception:
            b.errors += 1
            raise
        finally:
            b.inflight -= 1

    app = web.Application(client_max_size=1024 ** 3)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/lb_stats", lb_stats)
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="0.0.0.0:8000")
    ap.add_argument("--backends", required=True)
    ap.add_argument("--stats-path", default="")
    a = ap.parse_args()
    host, port = a.listen.rsplit(":", 1)
    backends = [Backend(u) for u in a.backends.split(",") if u.strip()]
    print(json.dumps({"lb_listen": a.listen, "backends": [b.url for b in backends], "t": time.time()}), flush=True)
    web.run_app(make_app(backends, a.stats_path), host=host, port=int(port), access_log=None)


if __name__ == "__main__":
    main()
