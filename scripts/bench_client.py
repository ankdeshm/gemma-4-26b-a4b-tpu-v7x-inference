#!/usr/bin/env python3
"""Async OpenAI-compatible serving benchmark client (one methodology for SGLang and vLLM).

Why a custom client: the same code measures both frameworks, so SGLang-vs-vLLM numbers are
methodologically matched. Definitions follow sglang.bench_serving / vllm bench serve:

  * prompts: exactly --isl random token ids (special/added tokens excluded), sent as a token-id
    list to /v1/completions, so no chat template or re-tokenization changes the input length;
    every request uses a different random sequence (no shared prefixes)
  * output: max_tokens = --osl with ignore_eos=true, temperature 0; every request's
    completion_tokens is checked against --osl from the server's usage block
  * TTFT = first streamed chunk carrying choices - request send time
  * ITL  = gaps between consecutive streamed chunks (a chunk may carry >1 token)
  * TPOT = (E2E - TTFT) / (completion_tokens - 1), per request
  * throughput = tokens / benchmark duration (first measured send -> last measured completion)
  * warm-up requests (same ISL, short OSL, same concurrency) run first and are excluded
  * concurrency = max in-flight requests (closed loop, request rate = inf), like --max-concurrency
  * several --base-urls = client-side least-outstanding-requests load balancer over replicas

Outputs: <out>.summary.json (all metrics + config) and <out>.requests.jsonl (one row per request).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import resource
import socket
import statistics
import sys
import time

import aiohttp

try:
    import orjson

    def jloads(b):
        return orjson.loads(b)
except Exception:  # noqa: BLE001
    def jloads(b):
        return json.loads(b)

try:
    import uvloop  # type: ignore

    HAVE_UVLOOP = True
except Exception:  # noqa: BLE001
    HAVE_UVLOOP = False

PCTS = (50, 90, 95, 99)


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stats_ms(values_s):
    v = [x * 1000.0 for x in values_s if x is not None]
    if not v:
        return {}
    out = {"mean": statistics.fmean(v), "std": statistics.pstdev(v) if len(v) > 1 else 0.0,
           "min": min(v), "max": max(v), "n": len(v)}
    for p in PCTS:
        out[f"p{p}"] = pct(v, p)
    return out


def build_vocab(tokenizer_path: str | None, vocab_size: int | None, seed: int):
    """Returns the list of token ids prompts are sampled from (special/added tokens removed)."""
    excluded = set()
    if tokenizer_path:
        from transformers import AutoTokenizer  # local files only

        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
        n = len(tok)
        excluded.update(int(i) for i in (tok.all_special_ids or []))
        try:
            excluded.update(int(i) for i in tok.get_added_vocab().values())
        except Exception:  # noqa: BLE001
            pass
        base_vocab = getattr(tok, "vocab_size", n) or n
        upper = min(n, base_vocab)
    else:
        upper = int(vocab_size)
    ids = [i for i in range(upper) if i not in excluded]
    return ids, {"tokenizer": tokenizer_path, "sample_upper_bound": upper, "excluded_ids": len(excluded),
                 "candidate_ids": len(ids)}


class Endpoint:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.inflight = 0
        self.sent = 0


async def one_request(session, ep: Endpoint, model, prompt_ids, osl, extra, timeout_s, idx, phase):
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": osl,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    payload.update(extra)
    rec = {"idx": idx, "phase": phase, "endpoint": ep.url, "prompt_len": len(prompt_ids), "osl_req": osl}
    ep.inflight += 1
    ep.sent += 1
    t0 = time.perf_counter()
    rec["t_send"] = t0
    chunk_ts = []
    usage = None
    err = None
    first_text = None
    try:
        async with session.post(f"{ep.url}/v1/completions", json=payload,
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {body[:300]}")
            buf = b""
            async for raw in resp.content:
                buf += raw
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line or not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    now = time.perf_counter()
                    obj = jloads(data)
                    ch = obj.get("choices")
                    if ch:
                        chunk_ts.append(now)
                        if first_text is None:
                            first_text = ch[0].get("text")
                    u = obj.get("usage")
                    if u:
                        usage = u
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {str(exc)[:300]}"
    t1 = time.perf_counter()
    ep.inflight -= 1
    rec["t_done"] = t1
    rec["e2e_s"] = t1 - t0
    rec["n_chunks"] = len(chunk_ts)
    if chunk_ts:
        rec["ttft_s"] = chunk_ts[0] - t0
        itl = [b - a for a, b in zip(chunk_ts, chunk_ts[1:])]
        rec["itl_s"] = itl
    else:
        rec["ttft_s"] = None
        rec["itl_s"] = []
    rec["completion_tokens"] = (usage or {}).get("completion_tokens")
    rec["prompt_tokens"] = (usage or {}).get("prompt_tokens")
    if rec["completion_tokens"] is None and chunk_ts:
        rec["completion_tokens_estimated_from_chunks"] = True
        rec["completion_tokens"] = len(chunk_ts)
    ct = rec["completion_tokens"]
    if ct and ct > 1 and rec["ttft_s"] is not None:
        rec["tpot_s"] = (rec["e2e_s"] - rec["ttft_s"]) / (ct - 1)
    else:
        rec["tpot_s"] = None
    if err is None and not chunk_ts:
        err = "no streamed chunks"
    rec["error"] = err
    rec["first_text"] = (first_text or "")[:40]
    return rec


async def run_phase(endpoints, model, prompts, osl, concurrency, extra, timeout_s, phase, start_idx=0):
    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, force_close=False)
    results = []
    rr = [0]

    def pick():
        # least outstanding requests, round-robin tie-break
        n = len(endpoints)
        best = None
        for k in range(n):
            ep = endpoints[(rr[0] + k) % n]
            if best is None or ep.inflight < best.inflight:
                best = ep
        rr[0] = (rr[0] + 1) % n
        return best

    async with aiohttp.ClientSession(connector=conn, read_bufsize=2 ** 20) as session:
        async def worker(i, p):
            async with sem:
                ep = pick()
                r = await one_request(session, ep, model, p, osl, extra, timeout_s, start_idx + i, phase)
                results.append(r)

        t_start = time.perf_counter()
        await asyncio.gather(*(worker(i, p) for i, p in enumerate(prompts)))
        t_end = time.perf_counter()
    return results, t_start, t_end


async def loop_lag_probe(stop: asyncio.Event, out: list):
    while not stop.is_set():
        t = time.perf_counter()
        await asyncio.sleep(0.05)
        out.append(time.perf_counter() - t - 0.05)


async def main_async(args):
    rng = random.Random(args.seed)
    ids, vocab_info = build_vocab(args.tokenizer, args.vocab_size, args.seed)

    def mk_prompt():
        return [ids[rng.randrange(len(ids))] for _ in range(args.isl)]

    endpoints = [Endpoint(u) for u in args.base_urls.split(",") if u.strip()]
    extra = json.loads(args.extra_body) if args.extra_body else {}

    warm_n = args.warmup if args.warmup >= 0 else args.concurrency
    warm_osl = max(1, min(args.osl, args.warmup_osl))
    warm_res = []
    if warm_n > 0:
        warm_prompts = [mk_prompt() for _ in range(warm_n)]
        warm_res, _, _ = await run_phase(endpoints, args.model, warm_prompts, warm_osl, args.concurrency, extra,
                                         args.request_timeout, "warmup", start_idx=-warm_n)
        await asyncio.sleep(args.warmup_sleep)

    prompts = [mk_prompt() for _ in range(args.num_prompts)]
    stop = asyncio.Event()
    lags: list[float] = []
    lag_task = asyncio.create_task(loop_lag_probe(stop, lags))
    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    res, t_start, t_end = await run_phase(endpoints, args.model, prompts, args.osl, args.concurrency, extra,
                                          args.request_timeout, "measure")
    cpu1 = resource.getrusage(resource.RUSAGE_SELF)
    stop.set()
    await lag_task

    ok = [r for r in res if r["error"] is None]
    failed = [r for r in res if r["error"] is not None]
    duration = max((r["t_done"] for r in res), default=t_end) - min((r["t_send"] for r in res), default=t_start)
    duration = max(duration, 1e-9)
    out_tok = sum((r["completion_tokens"] or 0) for r in ok)
    in_tok = sum((r["prompt_tokens"] if r["prompt_tokens"] is not None else r["prompt_len"]) for r in ok)
    osl_mismatch = [r["idx"] for r in ok if r["completion_tokens"] != args.osl]
    isl_mismatch = [r["idx"] for r in ok if r["prompt_tokens"] is not None and r["prompt_tokens"] != args.isl]
    all_itl = [x for r in ok for x in r["itl_s"]]
    tpots = [r["tpot_s"] for r in ok if r["tpot_s"] is not None]
    chips = args.chips
    summary = {
        "schema": "tpu-v7x-bench/v1",
        "label": args.label,
        "model": args.model,
        "framework": args.framework,
        "base_urls": [e.url for e in endpoints],
        "requests_per_endpoint": {e.url: e.sent for e in endpoints},
        "isl": args.isl,
        "osl": args.osl,
        "concurrency": args.concurrency,
        "num_prompts": args.num_prompts,
        "warmup_requests": warm_n,
        "warmup_osl": warm_osl,
        "warmup_failed": sum(1 for r in warm_res if r["error"] is not None),
        "chips": chips,
        "tensorcores": chips * 2 if chips else None,
        "completed": len(ok),
        "failed": len(failed),
        "failed_examples": [r["error"] for r in failed[:5]],
        "osl_mismatch_count": len(osl_mismatch),
        "isl_mismatch_count": len(isl_mismatch),
        "duration_s": duration,
        "request_throughput_rps": len(ok) / duration,
        "input_throughput_tok_s": in_tok / duration,
        "output_throughput_tok_s": out_tok / duration,
        "total_throughput_tok_s": (in_tok + out_tok) / duration,
        "output_throughput_tok_s_per_chip": (out_tok / duration / chips) if chips else None,
        "total_throughput_tok_s_per_chip": ((in_tok + out_tok) / duration / chips) if chips else None,
        "total_input_tokens": in_tok,
        "total_output_tokens": out_tok,
        "ttft_ms": stats_ms([r["ttft_s"] for r in ok]),
        "tpot_ms": stats_ms(tpots),
        "itl_ms": stats_ms(all_itl),
        "e2e_latency_ms": stats_ms([r["e2e_s"] for r in ok]),
        "per_user_output_tok_s_median": (1000.0 / pct([t * 1000 for t in tpots], 50)) if tpots else None,
        "client": {
            "host": socket.gethostname(), "python": platform.python_version(), "uvloop": HAVE_UVLOOP,
            "cpu_user_s": cpu1.ru_utime - cpu0.ru_utime, "cpu_sys_s": cpu1.ru_stime - cpu0.ru_stime,
            "cpu_util_of_one_core": (cpu1.ru_utime - cpu0.ru_utime + cpu1.ru_stime - cpu0.ru_stime) / duration,
            "event_loop_lag_ms_p99": (pct(lags, 99) or 0) * 1000, "event_loop_lag_ms_max": (max(lags) if lags else 0) * 1000,
        },
        "vocab": vocab_info,
        "seed": args.seed,
        "started_unix": time.time() - (time.perf_counter() - t_start),
        "sampling": {"temperature": 0.0, "ignore_eos": True, "max_tokens": args.osl, "extra_body": extra},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out + ".summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    with open(args.out + ".requests.jsonl", "w") as f:
        for r in sorted(res, key=lambda x: x["idx"]):
            itl = r.pop("itl_s")
            r["itl_ms_p50"] = (pct(itl, 50) or 0) * 1000 if itl else None
            r["itl_ms_p99"] = (pct(itl, 99) or 0) * 1000 if itl else None
            r["itl_ms_max"] = max(itl) * 1000 if itl else None
            f.write(json.dumps(r) + "\n")
    brief = {k: summary[k] for k in ("label", "isl", "osl", "concurrency", "completed", "failed", "osl_mismatch_count",
                                     "duration_s", "output_throughput_tok_s", "total_throughput_tok_s",
                                     "output_throughput_tok_s_per_chip")}
    brief["ttft_p50_ms"] = summary["ttft_ms"].get("p50")
    brief["tpot_p50_ms"] = summary["tpot_ms"].get("p50")
    brief["e2e_p50_ms"] = summary["e2e_latency_ms"].get("p50")
    print("BENCH_RESULT " + json.dumps(brief), flush=True)
    return 0 if (ok and not failed) else 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-urls", required=True, help="comma-separated server base URLs (replicas)")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--framework", default="")
    ap.add_argument("--tokenizer", default=None, help="local tokenizer dir (for the sampling vocabulary)")
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--isl", type=int, required=True)
    ap.add_argument("--osl", type=int, required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--num-prompts", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=-1, help="warm-up requests (-1 = concurrency)")
    ap.add_argument("--warmup-osl", type=int, default=16)
    ap.add_argument("--warmup-sleep", type=float, default=2.0)
    ap.add_argument("--chips", type=int, default=0, help="physical TPU chips serving (per-chip normalization)")
    ap.add_argument("--request-timeout", type=float, default=3600)
    ap.add_argument("--extra-body", default="", help="JSON merged into every request body")
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True, help="output prefix")
    args = ap.parse_args()
    if HAVE_UVLOOP:
        uvloop.install()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
