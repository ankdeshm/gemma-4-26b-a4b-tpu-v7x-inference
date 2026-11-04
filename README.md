# Serving `google/gemma-4-26B-A4B-it` on Google Cloud Ironwood TPU (v7x)

Production-ready inference recipes, SparseCore MoE optimization patches, Kubernetes (GKE) and Compute Engine (GCE) deployment manifests, and full-spectrum benchmark telemetry for serving **`google/gemma-4-26B-A4B-it`** (26B total parameters, 4B active parameters per token, Mixture-of-Experts) on **Google Cloud Ironwood TPU (`tpu7x`)**.

---

## Hardware & Architecture Overview

| Attribute | Ironwood TPU (`tpu7x`) Specification |
|---|---|
| **Single Host (`tpu7x-standard-4t`)** | `4` physical Ironwood chips (`2x2x1` ICI mesh) = `8` TensorCores + dedicated SparseCores |
| **HBM Capacity & Bandwidth** | `192 GB` HBM3e per chip (`768 GB` HBM3e per 4-chip host) @ `7.4 TB/s` per chip |
| **2-Host Deployment (`8` Chips)** | `2 x tpu7x-standard-4t` (`16` TensorCores, `1.54 TB` aggregate HBM3e) |
| **4-Host Deployment (`16` Chips)** | `4 x tpu7x-standard-4t` (`32` TensorCores, `3.07 TB` aggregate HBM3e) |
| **BF16 Checkpoint** | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (`~52 GB` BF16) |
| **FP8-Dynamic Checkpoint** | [`RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic) (`~28.7 GB` FP8-Dynamic + FP8 KV) |

---

## Headline Performance Highlights (`1`, `2`, and `4` Hosts)

All benchmarks use `vllm bench serve` with uncached random token IDs (`--ignore-eos`, zero prefix-cache reuse) and enforce strict simultaneous per-request Goodput SLAs (**`TTFT <= 500 ms`**, **`TPOT <= 10 ms`**, **`E2EL <= 60 s`**).

| Scale | Chips (Cores) | Serving Engine & Topology | `22k / 4k` `CON=1` TTFT / TPOT P50 | `22k / 4k` Peak Out tok/s | `1k / 1k` Peak Out tok/s (`tok/s/chip`) | `1k / 1k` Strict Goodput | `8k / 1k` Peak Out tok/s (Total tok/s) |
|---|---:|---|---:|---:|---:|---:|---:|
| **1 Host** (Throughput) | `4` (`8`) | `vllm-torchtpu` FP8-Dynamic `TP=2, DP=4, EP` | `1,023.7 ms` / `5.99 ms` | **`5,327.5`** | **`11,261.7`** (**`2,815.4`**) | `4.198 req/s` (`CON=32`) | `5,165.0` (`46,485.1`) |
| **1 Host** (Low-Latency SLA) | `4` (`8`) | `vllm-torchtpu` FP8-Dynamic `TP=8, DP=1, EP` | **`446.7 ms` / `4.50 ms`** (**100% Goodput**) | `3,319.8` | `8,121.9` (`2,030.5`) | `1.786 req/s` | `3,626.4` (`32,637.9`) |
| **2 Hosts** (Low-Latency SLA) | `8` (`16`) | `2x vllm-torchtpu` FP8-Dynamic `TP=8, EP` + LB | **`446.5 ms` / `4.41 ms`** (**100% Goodput**) | `4,657.2` | **`16,024.7`** (`2,003.1`) | `5.831 req/s` | `7,191.4` (`64,722.8`) |
| **4 Hosts** (Low-Latency SLA) | `16` (`32`) | `4x vllm-torchtpu` FP8-Dynamic `TP=8, EP` + LB | **`447.1 ms` / `4.44 ms`** (**100% Goodput**) | **`10,903.5`** | **`21,411.8`** (`1,338.2`) | **`10.884 req/s`** | **`12,016.8`** (**`108,150.8`**) |

See **[`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md)** for complete comparisons across all 6 single-host engine/precision/topology variants (`vllm-torchtpu` BF16 & FP8-Dynamic `TP=2,DP=4,EP` and `TP=8,EP`, plus `vllm-tpu` JAX Flax NNX BF16 & FP8 KV) and multi-host horizontal scaling sweeps.

---

## Choosing Between `TP=8, EP=true` and `TP=2, DP=4, EP=true`

Both winning profiles run on **`vllm-torchtpu`** with **`USE_MOE_SPARSE_CORE=1`**, **`--enable-expert-parallel`**, **`--attention-backend CUSTOM`**, **`--language-model-only`**, and **`RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`** (`--kv-cache-dtype=fp8`):

1. **Strict Long-Context SLA Goodput Profile (`--tensor-parallel-size=8 --data-parallel-size=1 --enable-expert-parallel`)**:
   - **Why it wins strict SLAs**: Sharding the `22,000`-token prefill across all `8` TensorCores on a 4-chip Ironwood host drives **`22k` TTFT P50 down to `446 ms` (`< 500 ms` SLA!)** while SparseCore expert parallelism delivers **`4.4–4.9 ms` TPOT (`> 200 tok/s` per stream, `< 10 ms` SLA!)** and **`18.2–18.4 s` E2EL (`< 60 s` SLA!)** — achieving **`100%` simultaneous SLA Goodput** on `22,000 / 4,000`!
   - Scaling horizontally across `2` hosts (`8` chips) and `4` hosts (`16` chips) via `least_conn` load balancing scales `22k/4k` peak throughput by **`3.28x`** (`3,319.8` -> **`10,903.5` out tok/s** / `70,872.6` total tok/s), reaches **`21,411.8` out tok/s** (`10.884 req/s` strict goodput, **`6.09x`** vs 1-host) on `1k/1k` and **`12,016.8` out tok/s (`108,150.8` total tok/s, `3.31x`)** on `8k/1k`, and sustains **`100%` strict SLA goodput at Open-Loop `2.0 RPS` (`4,613.3` out tok/s, `TTFT P95 = 455.0 ms`)** and **`86.7%` strict SLA goodput (`7,313.2` out tok/s, `TTFT P50 = 443.5 ms`, `TPOT P50 = 6.25 ms`, `E2E P50 = 25.43 s`) at Open-Loop `6.0 RPS`**!
2. **Maximum Per-Chip Throughput Profile (`--tensor-parallel-size=2 --data-parallel-size=4 --enable-expert-parallel`)**:
   - **Why it wins throughput**: Runs `4` independent `TP=2` engines per 4-chip host, reaching **`11,261.7` output tok/s (`2,815.4` out tok/s/chip)** on `1,024 / 1,024` and **`5,327.5` output tok/s (`1,331.9` out tok/s/chip, `34,628.6` total tok/s)** on `22,000 / 4,000` (`TPOT P50 = 8.97 ms`, `E2EL P50 = 47.94 s`) on a **single 4-chip host**!

---

## SparseCore Pallas Kernel Note (`scripts/patches/core_map_helper.py`)

When running `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` with `USE_MOE_SPARSE_CORE=1` on `vllm-torchtpu`, ensure `core_map_helper.py` delegates single-mesh SparseCore Pallas kernels directly through `@jax.jit(pl.kernel(...))` (`scripts/patches/core_map_helper.py`), avoiding zero-operand `AllocateBuffer` aliasing errors during FP8 dynamic expert routing. Both `gce/launch_gce_tpu_v7x_gemma4_26b_a4b.sh` and `k8s/gemma-4-26b-a4b-vllm-torchtpu-v7x.yaml` include this configuration out of the box.

---

## Repository Structure

```text
.
├── README.md                                     # Architecture, SparseCore MoE setup, GKE & GCE guide
├── BENCHMARK_RESULTS.md                          # Complete 1-host (6 stacks), 2-host & 4-host benchmark tables
├── configs/                                      # Reproducible JSON configs for all Wave 1 & Wave 2 suites
├── gce/
│   └── launch_gce_tpu_v7x_gemma4_26b_a4b.sh      # Turnkey GCE TPU VM script (BF16, FP8 TP2xDP4, FP8 TP8, JAX)
├── k8s/
│   ├── gemma-4-26b-a4b-vllm-torchtpu-v7x.yaml    # GKE Deployment (vllm-torchtpu FP8-Dynamic + GCSFuse PVC)
│   └── gemma-4-26b-a4b-vllm-tpu-jax-v7x.yaml     # GKE Deployment (vllm-tpu JAX Flax NNX TP=2, DP=4, FP8 KV)
├── results/                                      # Sanitized raw & summary JSON artifacts for all benchmark suites
└── scripts/
    ├── patches/core_map_helper.py                # JIT-wrapped pl.kernel SparseCore helper for FP8-Dynamic MoE
    ├── stage_hf_to_gcs.py                        # Multi-threaded HF -> GCS checkpoint streamer
    ├── run_v7x_bench.py                          # Unified server launcher & vllm bench serve suite runner
    ├── lb_proxy.py                               # Async least_conn HTTP load balancer for 2h/4h sweeps
    ├── bench_agent.py                            # Sidecar HTTP agent for CPU-isolated vllm bench serve
    ├── bench_client.py                           # Standalone async OpenAI streaming benchmark client
    ├── mock_metadata.py                          # Local GCE metadata proxy for 4-chip single-host isolation
    ├── metadata_hook.c                           # LD_PRELOAD connect() redirector pairing with mock_metadata.py
    └── summarize_results.py                      # Renders Markdown SLA & throughput tables from results/
```

---

## Quickstart: Stage Weights, Deploy (GKE or GCE), and Benchmark

### 1. Stage Checkpoints to GCS

```bash
export HF_TOKEN="<YOUR_HF_TOKEN>"
export GCS_BUCKET="<YOUR_GCS_BUCKET>"

python3 scripts/stage_hf_to_gcs.py \
  --bucket "${GCS_BUCKET}" \
  --prefix models \
  --repos RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic,google/gemma-4-26B-A4B-it \
  --workers 8
```

### 2. Deploy on GKE (`1`, `2`, or `4` Replicas = `4`, `8`, or `16` Ironwood Chips)

```bash
sed -e "s/<YOUR_GCS_BUCKET>/${GCS_BUCKET}/g" \
    -e "s/<YOUR_PROJECT_ID>/<YOUR_PROJECT_ID>/g" \
    k8s/gemma-4-26b-a4b-vllm-torchtpu-v7x.yaml | kubectl apply -f -
kubectl rollout status deployment/gemma4-26b-a4b-torchtpu-v7x --timeout=30m
```

### 3. Deploy on GCE TPU VM (`tpu7x-standard-4t`)

```bash
GCS_BUCKET="<YOUR_GCS_BUCKET>" bash gce/launch_gce_tpu_v7x_gemma4_26b_a4b.sh mount

# Low-latency SLA Goodput mode (FP8-Dynamic, TP=8, EP=true):
GCS_BUCKET="<YOUR_GCS_BUCKET>" bash gce/launch_gce_tpu_v7x_gemma4_26b_a4b.sh serve-torchtpu-fp8-tp8

# Or Peak Per-Chip Throughput mode (FP8-Dynamic, TP=2, DP=4, EP=true):
GCS_BUCKET="<YOUR_GCS_BUCKET>" bash gce/launch_gce_tpu_v7x_gemma4_26b_a4b.sh serve-torchtpu-fp8
```

### 4. Run Reproducible Benchmarks

```bash
# Single-host suite
python3 scripts/run_v7x_bench.py \
  --config configs/gemma4_1h_4c_torchtpu_fp8dyn_tp8_ep.json \
  --bench-agent-url http://127.0.0.1:9100

# 4-host (16-chip) suite across 4 single-host replicas behind lb_proxy.py
python3 scripts/lb_proxy.py --listen 0.0.0.0:8084 \
  --backends http://10.0.0.11:8000,http://10.0.0.12:8000,http://10.0.0.13:8000,http://10.0.0.14:8000 &
python3 scripts/run_v7x_bench.py \
  --config configs/gemma4_4h_16c_torchtpu_fp8dyn_tp8_ep.json \
  --bench-only-url http://127.0.0.1:8084 \
  --bench-agent-url http://127.0.0.1:9100
```
