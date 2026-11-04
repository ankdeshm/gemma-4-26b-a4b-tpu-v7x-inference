#!/usr/bin/env python3
"""Summarize raw `vllm bench serve` JSON artifacts into Markdown SLA & throughput tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(v: float | int | None, digits: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:,.{digits}f}"


def check_sla(m: dict) -> str:
    ttft = m.get("p95_ttft_ms") or m.get("median_ttft_ms")
    tpot = m.get("p95_tpot_ms") or m.get("median_tpot_ms")
    e2e = m.get("p95_e2el_ms") or m.get("median_e2el_ms")
    if ttft is None or tpot is None or e2e is None:
        return "—"
    ok_ttft = ttft <= 500.0
    ok_tpot = tpot <= 10.0
    ok_e2e = e2e <= 60000.0
    if ok_ttft and ok_tpot and ok_e2e:
        return "PASS (All)"
    parts = []
    if ok_ttft:
        parts.append("TTFT")
    if ok_tpot:
        parts.append("TPOT")
    if ok_e2e:
        parts.append("E2E")
    return f"Partial ({'+'.join(parts)})" if parts else "Above SLA"


def render_run(summary_path: Path) -> str:
    d = json.loads(summary_path.read_text())
    cfg = d.get("config", {})
    chips = int(cfg.get("chips", 4))
    lines = [
        f"### `{cfg.get('name', summary_path.parent.name)}`",
        f"- **Model**: `{cfg.get('model_family')}` | **Framework**: `{cfg.get('framework')}`",
        f"- **Precision**: `{cfg.get('precision')}` | **Topology**: `{cfg.get('topology')}` (`{chips}` physical `tpu7x` chips)",
        "",
        "| Suite | ISL : OSL | Load | Done | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | SLA Status (P95) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for c in d.get("cells", []):
        m = c.get("metrics", {})
        if not m:
            continue
        load = f"CON={c['concurrency']}" if c.get("concurrency") is not None else f"RPS={c.get('rps')}"
        out_tps = m.get("output_throughput") or 0.0
        tot_tps = m.get("total_token_throughput") or 0.0
        lines.append(
            f"| `{c.get('suite')}` | `{c.get('isl'):,} : {c.get('osl'):,}` | `{load}` | "
            f"`{m.get('completed', 0)}` | `{fmt(m.get('request_throughput'), 3)}` | `{fmt(m.get('request_goodput'), 3)}` | "
            f"**`{fmt(out_tps, 1)}`** | **`{fmt(out_tps / chips, 1)}`** | `{fmt(tot_tps, 1)}` | "
            f"`{fmt(m.get('median_ttft_ms'), 1)} / {fmt(m.get('p95_ttft_ms'), 1)} / {fmt(m.get('p99_ttft_ms'), 1)}` | "
            f"`{fmt(m.get('median_tpot_ms'), 2)} / {fmt(m.get('p95_tpot_ms'), 2)} / {fmt(m.get('p99_tpot_ms'), 2)}` | "
            f"`{fmt(m.get('median_itl_ms'), 2)} / {fmt(m.get('p95_itl_ms'), 2)}` | "
            f"`{fmt((m.get('median_e2el_ms') or 0)/1000.0, 2)} / {fmt((m.get('p95_e2el_ms') or 0)/1000.0, 2)}` | "
            f"{check_sla(m)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="Root directory containing run subdirectories with summary.json")
    args = ap.parse_args()
    root = Path(args.results_dir)
    for sp in sorted(root.glob("*/summary.json")):
        print(render_run(sp))


if __name__ == "__main__":
    main()
