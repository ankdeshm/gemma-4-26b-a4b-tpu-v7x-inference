#!/usr/bin/env python3
"""Unified Ironwood TPU (v7x) server launcher & `vllm bench serve` suite runner.

Supports:
  - vllm-tpu (JAX / tpu-inference / Flax NNX)
  - vllm-torchtpu (PyTorch/TPU + vLLM V1)
  - sglang-torchtpu (SGLang TPU runtime, single-engine TP8 or multi-replica TP2/TP4)
  - 1-host (4 chips), 2-host (8 chips), and 4-host (16 chips) load-balanced sweeps
    using `lb_proxy.py` (least-outstanding-requests HTTP reverse proxy)
  - Closed-loop (`--request-rate=inf --max-concurrency=C`) and Open-loop Poisson
    (`--request-rate=R`) benchmarks via `vllm bench serve` with SLA goodput tracking.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
GOODPUT_SLA = ["ttft:500", "tpot:10", "e2el:60000"]

UNSET_SLICE_ENVS = [
    "TPU_WORKER_ID",
    "TPU_WORKER_HOSTNAMES",
    "TPU_PROCESS_ADDRESSES",
    "TPU_PROCESS_PORT",
    "TPU_TOPOLOGY",
    "TPU_TOPOLOGY_WRAP",
    "TPU_TOPOLOGY_ALT",
    "TPU_HOST_BOUNDS",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_ACCELERATOR_TYPE",
    "CLOUD_TPU_TASK_ID",
    "JOB_COMPLETION_INDEX",
    "JOB_COMPLETION_MODE",
]


def log(msg: str) -> None:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}][v7x-bench] {msg}", flush=True)


def stage_to_shm(src_dir: Path, dst_dir: Path, workers: int = 16) -> float:
    marker = dst_dir / ".staged"
    if marker.exists() and (dst_dir / "config.json").exists():
        log(f"Checkpoint already in RAM: {dst_dir}")
        return 0.0
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(src_dir.rglob("*")) if p.is_file()]
    t0 = time.time()

    def _copy_one(fp: Path) -> int:
        rel = fp.relative_to(src_dir)
        out = dst_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.stat().st_size == fp.stat().st_size:
            return 0
        buf = bytearray(16 * 1024 * 1024)
        mv = memoryview(buf)
        total = 0
        with fp.open("rb", buffering=0) as fi, out.open("wb") as fo:
            while True:
                n = fi.readinto(mv)
                if not n:
                    break
                fo.write(mv[:n])
                total += n
        return total

    copied = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for b in ex.map(_copy_one, files):
            copied += b
    marker.write_text("ok\n")
    dt_s = time.time() - t0
    log(f"Staged {src_dir} -> {dst_dir} ({copied / 1e9:.2f} GB copied in {dt_s:.1f}s)")
    return dt_s


def setup_single_host_isolation() -> dict[str, str]:
    """Ensure mock_metadata.py is running and compile metadata_hook.so if compiler exists."""
    port = 18080
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/computeMetadata/v1/instance/attributes/TPU_WORKER_ID",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                pass
    except Exception:
        mock_py = HERE / "mock_metadata.py"
        if mock_py.exists():
            clean_env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
            clean_env["MOCK_METADATA_PORT"] = str(port)
            subprocess.Popen(
                [sys.executable, str(mock_py)],
                env=clean_env,
                stdout=open("/tmp/mock_metadata.log", "ab"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            time.sleep(1.0)

    hook_so = Path("/tmp/metadata_hook.so")
    hook_c = HERE / "metadata_hook.c"
    if not hook_so.exists() and hook_c.exists():
        cc = shutil.which("gcc") or shutil.which("clang")
        if cc:
            subprocess.run(
                [cc, "-O2", "-fPIC", "-shared", str(hook_c), "-o", str(hook_so), "-ldl"],
                check=False,
            )

    overrides = {
        "MOCK_METADATA_PORT": str(port),
        "GCE_METADATA_HOST": f"127.0.0.1:{port}",
        "GCE_METADATA_IP": "127.0.0.1",
    }
    if hook_so.exists():
        overrides["LD_PRELOAD"] = str(hook_so)
    return overrides


def build_clean_server_env(extra_env: dict[str, str], tpu_env_mode: str = "gke_single_host") -> dict[str, str]:
    env = os.environ.copy()
    if tpu_env_mode in ("gke_single_host", "sglang_single_host"):
        for k in UNSET_SLICE_ENVS:
            env.pop(k, None)
        for k in list(env.keys()):
            if k.startswith("MEGASCALE"):
                env.pop(k, None)
        if tpu_env_mode == "gke_single_host":
            iso = setup_single_host_isolation()
            existing_preload = extra_env.get("LD_PRELOAD", "")
            if "LD_PRELOAD" in iso and existing_preload:
                iso["LD_PRELOAD"] = f"{existing_preload}:{iso['LD_PRELOAD']}"
            env.update(iso)
            env.update(
                {
                    "TPU_SKIP_MDS_QUERY": "true",
                    "TPU_TOPOLOGY": "2x2x1",
                    "TPU_HOST_BOUNDS": "1,1,1",
                    "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
                    "TPU_TOPOLOGY_WRAP": "false,false,false",
                    "TPU_TOPOLOGY_ALT": "false",
                    "TPU_ACCELERATOR_TYPE": "tpu7x-8",
                    "TPU_WORKER_ID": "0",
                    "TPU_WORKER_HOSTNAMES": "localhost",
                    "TPU_PROCESS_ADDRESSES": "localhost:8471",
                }
            )
        elif tpu_env_mode == "sglang_single_host":
            env["TPU_SKIP_MDS_QUERY"] = "true"
            env["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
            env["FORCE_SINGLE_HOST"] = "1"
    env.update({k: str(v) for k, v in extra_env.items()})
    return env


def http_get(url: str, timeout: float = 10.0) -> tuple[int | None, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except Exception as exc:
        return None, str(exc).encode()


def http_post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_server_ready(base_url: str, proc: subprocess.Popen | None, served_model: str, timeout_s: float) -> tuple[bool, str]:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            return False, f"server exited with code {proc.returncode}"
        st, _ = http_get(f"{base_url}/health", timeout=15)
        if st == 200:
            try:
                out = http_post(
                    f"{base_url}/v1/completions",
                    {
                        "model": served_model,
                        "prompt": "The capital of France is",
                        "max_tokens": 8,
                        "temperature": 0.0,
                    },
                    timeout=600,
                )
                if out.get("choices"):
                    txt = out["choices"][0].get("text", "")
                    return True, f"ready in {time.time() - t0:.1f}s (sample={txt!r})"
            except Exception as exc:
                log(f"Completion warmup probe pending: {exc}")
        time.sleep(10)
    return False, f"timeout after {timeout_s}s"


def stop_all_servers(proc: subprocess.Popen | None = None) -> None:
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
    subprocess.run(
        [
            "pkill",
            "-9",
            "-f",
            r"[v]llm\.entrypoints|[v]llm serve|[V]LLM::|[E]ngineCore|[s]glang\.launch_server|[s]glang::",
        ],
        check=False,
    )
    for lf in Path("/tmp").glob("libtpu_lockfile*"):
        lf.unlink(missing_ok=True)
    time.sleep(8)


def run_sanity_prompts(base_url: str, served_model: str, out_file: Path) -> list[dict]:
    prompts = [
        "Explain in two sentences why optical interconnects improve distributed tensor parallelism.",
        "If an investment of $4,000 earns 5% simple interest per year, how much interest is earned in 3 years?",
        "Write a concise Python function `is_prime(n: int) -> bool`.",
    ]
    rows = []
    for p in prompts:
        t0 = time.time()
        try:
            resp = http_post(
                f"{base_url}/v1/completions",
                {"model": served_model, "prompt": p, "max_tokens": 128, "temperature": 0.0},
                timeout=300,
            )
            rows.append(
                {
                    "prompt": p,
                    "output": resp["choices"][0].get("text", ""),
                    "usage": resp.get("usage"),
                    "latency_s": round(time.time() - t0, 2),
                }
            )
        except Exception as exc:
            rows.append({"prompt": p, "error": str(exc)[:300]})
    out_file.write_text(json.dumps(rows, indent=2))
    return rows


def run_vllm_bench_cell(
    base_url: str,
    served_model: str,
    tokenizer_dir: str,
    cell: dict,
    out_dir: Path,
    bench_agent_url: str = "",
) -> dict:
    isl = int(cell["isl"])
    osl = int(cell["osl"])
    num_prompts = int(cell["num_prompts"])
    rps = cell.get("rps", "inf")
    conc = cell.get("concurrency")
    load_tag = f"c{conc}" if conc is not None else f"rps{rps}"
    tag = f"{cell['suite']}_isl{isl}_osl{osl}_{load_tag}_n{num_prompts}"
    res_file = out_dir / f"{tag}.json"
    log_file = out_dir / f"{tag}.client.log"

    bench_args = [
        "--backend", "vllm",
        "--base-url", base_url,
        "--endpoint", "/v1/completions",
        "--model", served_model,
        "--tokenizer", tokenizer_dir,
        "--trust-remote-code",
        "--dataset-name", "random",
        "--random-input-len", str(isl),
        "--random-output-len", str(osl),
        "--num-prompts", str(num_prompts),
        "--request-rate", str(rps),
        "--ignore-eos",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,95,99",
        "--goodput", *GOODPUT_SLA,
        "--seed", str(cell.get("seed", 42)),
        "--save-result",
        "--result-dir", str(out_dir),
        "--result-filename", f"{tag}.json",
    ]
    if conc is not None:
        bench_args += ["--max-concurrency", str(conc)]
    if cell.get("num_warmups") is not None:
        bench_args += ["--num-warmups", str(cell["num_warmups"])]

    timeout_s = float(cell.get("timeout_s", 5400))
    t0 = time.time()
    if bench_agent_url:
        resp = http_post(
            f"{bench_agent_url}/run",
            {"args": bench_args, "timeout_s": timeout_s, "log": str(log_file)},
            timeout=timeout_s + 120,
        )
        rc = resp["rc"]
        wall_s = resp["wall_s"]
    else:
        env = {
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "TPU_SKIP_MDS_QUERY": "true",
            "TPU_VISIBLE_CHIPS": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        for k in ("PJRT_DEVICE", "LD_PRELOAD", *UNSET_SLICE_ENVS):
            env.pop(k, None)
        cmd = ["vllm", "bench", "serve", *bench_args]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
            rc, so, se = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as exc:
            rc = -9
            so = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            se = "TIMEOUT"
        wall_s = round(time.time() - t0, 1)
        log_file.write_text(" ".join(cmd) + "\n\n" + so[-30000:] + "\n--- stderr ---\n" + se[-30000:])

    out = {"tag": tag, **cell, "rc": rc, "wall_s": wall_s}
    if res_file.exists():
        try:
            d = json.loads(res_file.read_text())
            out["metrics"] = {k: v for k, v in d.items() if not isinstance(v, list)}
        except Exception as exc:
            out["parse_error"] = str(exc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to JSON benchmark config")
    ap.add_argument("--serve-only", action="store_true", help="Keep server running after readiness check")
    ap.add_argument("--bench-only-url", default="", help="Skip server launch and benchmark existing endpoint/LB URL")
    ap.add_argument("--bench-agent-url", default="", help="Optional sidecar bench_agent URL (e.g. http://127.0.0.1:9100)")
    ap.add_argument("--out-dir", default="", help="Override output directory")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{cfg['name']}_{ts}"
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.get("out_root", "/models/benchmark-results/v7x-poc")) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage checkpoints to /dev/shm if requested
    for item in cfg.get("stage", []):
        stage_to_shm(Path(item["src"]), Path(item["dst"]))

    meta = {
        "run_id": run_id,
        "config": cfg,
        "node": os.environ.get("NODE_NAME", os.uname().nodename),
        "pod": os.environ.get("HOSTNAME", ""),
        "started_utc": ts,
        "cells": [],
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2))

    proc = None
    served_model = cfg["served_model"]
    tokenizer_dir = cfg["tokenizer_dir"]
    port = int(cfg.get("port", 8000))

    if args.bench_only_url:
        base_url = args.bench_only_url.rstrip("/")
        log(f"Bench-only mode against {base_url}")
    else:
        stop_all_servers()
        srv_env = build_clean_server_env(
            cfg.get("env", {}),
            tpu_env_mode=cfg.get("tpu_env_mode", "gke_single_host"),
        )
        for p in cfg.get("mkdirs", []):
            Path(p).mkdir(parents=True, exist_ok=True)
        slog = out_dir / "server.log"
        argv = [str(x) for x in cfg["argv"]]
        log(f"Launching server: {' '.join(argv)}")
        with slog.open("w") as lf:
            proc = subprocess.Popen(
                argv,
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=srv_env,
                cwd=cfg.get("cwd", "/"),
                start_new_session=True,
            )
        base_url = f"http://127.0.0.1:{port}"
        ok, msg = wait_server_ready(base_url, proc, served_model, float(cfg.get("ready_timeout_s", 3600)))
        meta["server_ready"] = ok
        meta["server_ready_msg"] = msg
        log(f"Server readiness: ok={ok} ({msg})")
        if not ok:
            (out_dir / "summary.json").write_text(json.dumps(meta, indent=2))
            stop_all_servers(proc)
            sys.exit(2)

        meta["sanity"] = run_sanity_prompts(base_url, served_model, out_dir / "sanity.json")
        (out_dir / "summary.json").write_text(json.dumps(meta, indent=2))

        if args.serve_only:
            log(f"Server ready on {base_url} (--serve-only active); waiting until terminated.")
            proc.wait()
            return

    chips = int(cfg.get("chips", 4))
    for cell in cfg.get("cells", []):
        if proc is not None and proc.poll() is not None:
            log(f"ERROR: server exited unexpectedly with rc={proc.returncode}")
            meta["status"] = "SERVER_DIED"
            break
        log(f"Running cell: {cell}")
        res = run_vllm_bench_cell(
            base_url=base_url,
            served_model=served_model,
            tokenizer_dir=tokenizer_dir,
            cell=cell,
            out_dir=out_dir,
            bench_agent_url=args.bench_agent_url,
        )
        m = res.get("metrics", {})
        if m and chips:
            m["output_throughput_per_chip"] = round((m.get("output_throughput") or 0.0) / chips, 3)
            m["total_throughput_per_chip"] = round((m.get("total_token_throughput") or 0.0) / chips, 3)
        meta["cells"].append(res)
        (out_dir / "summary.json").write_text(json.dumps(meta, indent=2))
        log(
            f"Completed {res['tag']}: rc={res['rc']} wall={res['wall_s']}s "
            f"done={m.get('completed')} qps={m.get('request_throughput')} "
            f"out_tok_s={m.get('output_throughput')} ({m.get('output_throughput_per_chip')}/chip) "
            f"goodput={m.get('request_goodput')} "
            f"ttft_p50={m.get('median_ttft_ms')} ttft_p95={m.get('p95_ttft_ms')} "
            f"tpot_p50={m.get('median_tpot_ms')} tpot_p95={m.get('p95_tpot_ms')} "
            f"e2e_p50={m.get('median_e2el_ms')} e2e_p95={m.get('p95_e2el_ms')}"
        )

    meta["finished_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta.setdefault("status", "DONE")
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2))
    if proc is not None and not cfg.get("keep_server_alive", False):
        stop_all_servers(proc)
    log(f"Finished run {run_id} -> {out_dir}")


if __name__ == "__main__":
    main()
