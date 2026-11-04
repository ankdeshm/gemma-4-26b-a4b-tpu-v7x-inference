#!/usr/bin/env python3
"""Benchmark agent: runs `vllm bench serve` on request (sidecar container, no TPU attached).

Runs in the CPU client sidecar built from the vllm-tpu image (`vllm bench serve` CLI),
whose benchmark harness executes cleanly without TPU devices attached. The orchestrator in the server
container POSTs a JSON list of CLI arguments; results land on the shared /dev/shm volume.

  POST /run   {"args": [...], "timeout_s": 7200, "log": "/dev/shm/.../x.client.log"} -> {"rc":..,"wall_s":..}
  GET  /health
  GET  /help  -> `vllm bench serve --help` text (recorded once per run for provenance)
"""
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "JAX_PLATFORMS": "cpu",
       "TPU_SKIP_MDS_QUERY": "true", "TPU_VISIBLE_CHIPS": "", "PYTHONUNBUFFERED": "1"}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        return

    def _send(self, code, obj):
        body = json.dumps(obj).encode() if not isinstance(obj, bytes) else obj
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True})
        if self.path == "/help":
            p = subprocess.run(["vllm", "bench", "serve", "--help"], capture_output=True, text=True, env=ENV, timeout=300)
            return self._send(200, {"rc": p.returncode, "out": p.stdout[-60000:], "err": p.stderr[-5000:]})
        if self.path == "/versions":
            p = subprocess.run([sys.executable, "-c", "import vllm,transformers;print(vllm.__version__, transformers.__version__)"],
                               capture_output=True, text=True, env=ENV, timeout=300)
            return self._send(200, {"out": p.stdout.strip(), "err": p.stderr[-2000:]})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path != "/run":
            return self._send(404, {"error": "not found"})
        args = req["args"]
        cmd = ["vllm", "bench", "serve", *args]
        t0 = time.time()
        log_path = req.get("log")
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, env=ENV, timeout=float(req.get("timeout_s", 7200)),
                               cwd="/tmp")
            rc, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as exc:
            rc = -9
            out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err = "TIMEOUT"
        if log_path:
            with open(log_path, "w") as f:
                f.write(" ".join(cmd) + "\n\n" + out[-40000:] + "\n--- stderr ---\n" + err[-40000:])
        return self._send(200, {"rc": rc, "wall_s": round(time.time() - t0, 1), "tail": out[-3000:],
                                "err_tail": err[-3000:]})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9100
    print(f"bench agent on 127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
