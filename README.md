# Serving `google/gemma-4-26B-A4B-it` on Google Cloud Ironwood TPU (v7x)

Production-ready inference recipes, SparseCore MoE optimization patches, **Google Kubernetes Engine (GKE) DWS Calendar Mode Reservation** deployment manifests, and full-spectrum benchmark telemetry for serving **`google/gemma-4-26B-A4B-it`** (26B total parameters, 4B active parameters per token, Mixture-of-Experts) on **Google Cloud Ironwood TPU (`tpu7x`)**.

> **First-Time GCP User or Consuming a DWS Calendar Mode Reservation (`tpu7x-standard-4t`)?**
>
> - **GKE is Recommended for Production & Required for Multi-Host Orchestration (`2` Hosts = `8` Chips `2x2x2`, `4` Hosts = `16` Chips `2x2x4`)**: On Ironwood (`tpu7x`), Google Cloud's AI Hypercomputer stack uses **GKE Standard** with regional **Workload Policies (`--placement-policy`)**, **DWS Calendar Reservation affinity (`--reservation-affinity=specific --reservation=<RESERVATION_NAME>`)**, **GCSFuse CSI driver**, and **Ray on GKE / Kubernetes Services** to handle multi-host slice placement, worker discovery, and load balancing.
> - **Single-Host (`1` Host = `4` Chips) Testing**: A standalone Compute Engine (GCE) VM can be used *only* for basic single-host (`4`-chip) testing, as documented in the appendix of [`INFRA_SETUP.md`](INFRA_SETUP.md).
>
> Follow **[`INFRA_SETUP.md`](INFRA_SETUP.md)** (or run [`k8s/provision_gke_dws_calendar.sh`](k8s/provision_gke_dws_calendar.sh)) for the complete step-by-step setup from a blank GCP project.

---

## Hardware & Architecture Overview

| Attribute | Ironwood TPU (`tpu7x`) Specification |
|---|---|
| **Single Host (`tpu7x-standard-4t`)** | `4` physical Ironwood chips (`2x2x1` ICI mesh) = `8` TensorCores + dedicated SparseCores |
| **HBM Capacity & Bandwidth** | `192 GB` HBM3e per chip (`768 GB` HBM3e per 4-chip host) @ `7.4 TB/s` per chip |
| **2-Host Deployment (`8` Chips)** | `2 x tpu7x-standard-4t` (`16` TensorCores, `1.54 TB` aggregate HBM3e) |
| **4-Host Deployment (`16` Chips)** | `4 x tpu7x-standard-4t` (`32` TensorCores, `3.07 TB` aggregate HBM3e) |
| **BF16 Checkpoint** | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (`~52 GB` Pure BF16 weights + BF16 KV) |
| **FP8-Dynamic Checkpoint** | [`RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic) (`~28.7 GB` FP8-Dynamic + FP8 KV; optional, non-BF16) |

---

## Headline Performance Highlights (`1`, `2`, and `4` Hosts on GKE)

All benchmarks use `vllm bench serve` with uncached random token IDs (`--ignore-eos`, zero prefix-cache reuse) and enforce strict simultaneous per-request Goodput SLAs (**`TTFT <= 500 ms`**, **`TPOT <= 10 ms`**, **`E2EL <= 60 s`**). BF16 weights with a BF16 KV cache (`google/gemma-4-26B-A4B-it`) are the primary configuration; FP8-Dynamic rows (third-party `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` checkpoint with FP8 KV) are optional (non-BF16).

| Scale | Chips (Cores) | Serving Engine · Layout · Precision | `22k / 4k` `CON=1` TTFT / TPOT P50 | `n` | `22k / 4k` Closed-Loop Peak Out tok/s | `1k / 1k` Peak Out tok/s [`tok/s/chip`, derived] | `1k / 1k` Highest SLA Goodput | `8k / 1k` Peak Out tok/s (Total tok/s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| **1 Host** (BF16 SLA profile) | `4` (`8`) | `vllm-torchtpu` **BF16** `TP=8, DP=1, EP` | `443.0 ms` / `4.71 ms` (`100%` SLA goodput) | `25` | `3,199.5` (`CON=64`, `n=64`) | `7,609.9` [`1,902.5`] (`CON=128`, `n=256`) | `1.570 req/s` (`CON=8`, `n=64`) | `3,525.0` (`31,725.1`) (`CON=64`, `n=128`) |
| **1 Host** (BF16 throughput profile) | `4` (`8`) | `vllm-torchtpu` **BF16** `TP=2, DP=4, EP` | `1,068.1 ms` / `6.00 ms` | `25` | `4,994.0` (`CON=64`, `n=64`) | `10,940.1` [`2,735.0`] (`CON=128`, `n=256`) | `1.796 req/s` (`CON=32`, `n=128`) | `4,828.9` (`43,459.9`) (`CON=64`, `n=128`) |
| **1 Host** (FP8 throughput — optional, non-BF16) | `4` (`8`) | `vllm-torchtpu` FP8-Dynamic `TP=2, DP=4, EP` | `1,023.7 ms` / `5.99 ms` | `25` | `5,327.5` (`CON=64`, `n=64`) | `11,261.7` [`2,815.4`] (`CON=128`, `n=256`) | `1.403 req/s` (`CON=32`, `n=128`) | `5,165.0` (`46,485.4`) (`CON=64`, `n=128`) |
| **1 Host** (FP8 SLA profile — optional, non-BF16) | `4` (`8`) | `vllm-torchtpu` FP8-Dynamic `TP=8, DP=1, EP` | `446.7 ms` / `4.50 ms` (`100%` SLA goodput) | `25` | `3,319.8` (`CON=64`, `n=64`) | `8,121.9` [`2,030.5`] (`CON=128`, `n=256`) | `1.786 req/s` (`CON=32`, `n=128`) | `3,626.4` (`32,637.9`) (`CON=64`, `n=128`) |
| **2 Hosts** (FP8 SLA profile — optional, non-BF16) | `8` (`16`) | `2x vllm-torchtpu` FP8-Dynamic `TP=8, EP` | `446.5 ms` / `4.41 ms` (`100%` SLA goodput) | `10` | `4,657.2` (`CON=64`, `n=96`) | `16,024.7` [`2,003.1`] (`CON=256`, `n=512`) | `5.831 req/s` (`CON=32`, `n=128`) | `7,191.4` (`64,722.8`) (`CON=128`, `n=256`) |
| **4 Hosts** (FP8 SLA profile — optional, non-BF16) | `16` (`32`) | `4x vllm-torchtpu` FP8-Dynamic `TP=8, EP` | `447.1 ms` / `4.44 ms` (`100%` SLA goodput) | `10` | `10,903.5` (`CON=128`, `n=128`) | `21,411.8` [`1,338.2`] (`CON=256`, `n=512`) | `10.884 req/s` (`CON=64`, `n=192`) | `12,016.8` (`108,150.8`) (`CON=128`, `n=256`) |

Each peak cell shows the concurrency (`CON`) and number of prompts (`n`) of the run that produced it. Per-chip values in brackets are derived (out tok/s ÷ chips). `100%` SLA goodput means every `CON=1` request met `TTFT <= 500 ms`, `TPOT <= 10 ms` and `E2EL <= 60 s`.

**Sample size (`n`)**: the customer workload specifies `n=50`; the `22k / 4k` `CON=1` cells used `n=25` (1 host) and `n=10` (2 and 4 hosts), and the remaining `22k / 4k` cells used `n=48–128`.

See **[`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md)** for complete comparisons across all 6 single-host engine/precision/topology variants (`vllm-torchtpu` BF16 & FP8-Dynamic `TP=2,DP=4,EP` and `TP=8,EP`, plus `vllm-tpu` JAX Flax NNX BF16 & FP8 KV) and multi-host GKE scaling sweeps.

---

## Choosing Between `TP=8, EP=true` and `TP=2, DP=4, EP=true` on `vllm-torchtpu`

Both winning profiles run on **`vllm-torchtpu`** with **`USE_MOE_SPARSE_CORE=1`**, **`--enable-expert-parallel`**, **`--attention-backend CUSTOM`**, and **`--language-model-only`**:

1. **Strict Long-Context SLA Goodput Profile (`--tensor-parallel-size=8 --data-parallel-size=1 --enable-expert-parallel`)**:
   - **Why it wins strict SLAs**: Sharding the `22,000`-token prefill across all `8` TensorCores on a 4-chip Ironwood host keeps **`22k` TTFT P50 under the `500 ms` SLA**, while SparseCore expert parallelism keeps TPOT under `10 ms` and E2EL under `60 s`. At `CON=1` (`n=25`), BF16 weights measure **`443.0 ms` TTFT / `4.71 ms` TPOT / `19.27 s` E2E P50 with `100%` SLA goodput**; FP8-Dynamic (optional, non-BF16) measures `446.7 ms` / `4.50 ms` / `18.44 s`, also `100%` SLA goodput.
   - These multi-host figures are FP8-Dynamic `TP=8` (optional, non-BF16). Scaling across `2` hosts (`8` chips) and `4` hosts (`16` chips) on GKE scales `22k/4k` peak throughput by **`3.28x`** (derived; `3,319.8` at `CON=64` -> **`10,903.5` out tok/s** at `CON=128`, `70,872.6` total tok/s), reaches **`21,411.8` out tok/s** (`CON=256`) on `1k/1k` with a highest SLA goodput of `10.884 req/s` (`CON=64`; **`6.09x`** the 1-host `1.786 req/s`, derived) and **`12,016.8` out tok/s (`108,150.8` total tok/s, `CON=128`; `3.31x`, derived)** on `8k/1k`, and at 4 hosts sustains **`100%` strict SLA goodput at Open-Loop `2.0 RPS` (`4,613.3` out tok/s, `TTFT P95 = 455.0 ms`, `n=56`)** and **`86.7%` strict SLA goodput (`7,313.2` out tok/s, `TTFT P50 = 443.5 ms`, `TPOT P50 = 6.25 ms`, `E2E P50 = 25.43 s`, `n=60`) at Open-Loop `6.0 RPS`**.
2. **Maximum Per-Chip Throughput Profile (`--tensor-parallel-size=2 --data-parallel-size=4 --enable-expert-parallel`)**:
   - **Why it wins throughput**: Runs `4` independent `TP=2` engines per 4-chip host. With BF16 weights it reaches **`10,940.1` output tok/s** (`2,735.0` out tok/s/chip, derived; `CON=128`, `n=256`) on `1,024 / 1,024` and **`4,994.0` output tok/s** (`1,248.5` out tok/s/chip, derived; `32,460.9` total tok/s; `CON=64`, `n=64`; `TPOT P50 = 9.75 ms`, `E2EL P50 = 51.15 s`) on `22,000 / 4,000` on a **single 4-chip host**. FP8-Dynamic (optional, non-BF16) reaches `11,261.7` (`2,815.4`/chip) and `5,327.5` output tok/s (`1,331.9`/chip, `34,628.6` total tok/s; `TPOT P50 = 8.97 ms`, `E2EL P50 = 47.94 s`) at the same concurrency.

---

## SparseCore Pallas Kernel Note (`scripts/patches/core_map_helper.py`)

When running `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` with `USE_MOE_SPARSE_CORE=1` on `vllm-torchtpu`, ensure `core_map_helper.py` delegates single-mesh SparseCore Pallas kernels directly through `@jax.jit(pl.kernel(...))` (`scripts/patches/core_map_helper.py`), avoiding zero-operand `AllocateBuffer` aliasing errors during FP8 dynamic expert routing. Both `k8s/gemma-4-26b-a4b-vllm-torchtpu-v7x.yaml` and `gce/launch_gce_tpu_v7x_gemma4_26b_a4b.sh` include this configuration out of the box.

---

## Repository Structure

```text
.
├── README.md                                     # Architecture, SparseCore MoE setup & GKE guide
├── INFRA_SETUP.md                                # First-time GCP & GKE DWS Calendar Reservation setup guide
├── BENCHMARK_RESULTS.md                          # Complete 1-host (6 stacks), 2-host & 4-host GKE benchmark tables
├── configs/                                      # Reproducible JSON configs for all Wave 1 & Wave 2 suites
├── k8s/
│   ├── provision_gke_dws_calendar.sh             # Turnkey VPC, GCS, GKE Cluster & DWS Calendar Node Pool provisioner
│   ├── gemma-4-26b-a4b-vllm-torchtpu-v7x.yaml    # GKE Deployment (vllm-torchtpu FP8-Dynamic / BF16 + GCSFuse PVC)
│   └── gemma-4-26b-a4b-vllm-tpu-jax-v7x.yaml     # GKE Deployment (vllm-tpu JAX Flax NNX TP=2, DP=4, FP8 KV)
├── gce/
│   └── launch_gce_tpu_v7x_gemma4_26b_a4b.sh      # Optional single-host (4-chip) GCE VM helper script
├── results/                                      # Sanitized raw & summary JSON artifacts for all benchmark suites
└── scripts/
    ├── patches/core_map_helper.py                # JIT-wrapped pl.kernel SparseCore helper for FP8-Dynamic MoE
    ├── stage_hf_to_gcs.py                        # Multi-threaded HF -> GCS checkpoint streamer
    ├── run_v7x_bench.py                          # Unified server launcher & vllm bench serve suite runner
    ├── lb_proxy.py                               # Async least_conn HTTP load balancer for multi-pod sweeps
    ├── bench_agent.py                            # Sidecar HTTP agent for CPU-isolated vllm bench serve
    ├── bench_client.py                           # Standalone async OpenAI streaming benchmark client
    ├── mock_metadata.py                          # Local GCE metadata proxy for 4-chip single-host isolation
    ├── metadata_hook.c                           # LD_PRELOAD connect() redirector pairing with mock_metadata.py
    └── summarize_results.py                      # Renders Markdown SLA & throughput tables from results/
```

---

## Quickstart: Provision GKE + DWS Calendar Node Pool, Stage Weights, Deploy & Benchmark

> For full step-by-step explanations of IAM roles, VPC/Firewall rules, Workload Policies (`2x2x2` / `2x2x4`), and DWS Calendar reservation binding, see **[`INFRA_SETUP.md`](INFRA_SETUP.md)**.

### 1. Provision GCP Network, GCS Bucket, GKE Cluster & DWS-Reserved `tpu7x-standard-4t` Node Pool

```bash
export PROJECT_ID="<YOUR_GCP_PROJECT_ID>"
export RESERVATION_NAME="<YOUR_DWS_CALENDAR_RESERVATION_NAME>"
export GCS_BUCKET="${PROJECT_ID}-tpu-v7x-models"
export NUM_NODES=4          # 1 (4 chips), 2 (8 chips), or 4 (16 chips)
export TPU_TOPOLOGY="2x2x4" # 2x2x1 (single-host pool), 2x2x2 (2 hosts), or 2x2x4 (4 hosts)

bash k8s/provision_gke_dws_calendar.sh
```

### 2. Stage Checkpoints to GCS

```bash
export HF_TOKEN="<YOUR_HF_TOKEN>"
export GCS_BUCKET="<YOUR_GCS_BUCKET>"

python3 scripts/stage_hf_to_gcs.py \
  --bucket "${GCS_BUCKET}" \
  --prefix models \
  --repos google/gemma-4-26B-A4B-it,RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic \
  --workers 8
```

### 3. Deploy on GKE (`1`, `2`, or `4` Replicas = `4`, `8`, or `16` Ironwood Chips)

```bash
sed -e "s/<YOUR_GCS_BUCKET>/${GCS_BUCKET}/g" \
    -e "s/<YOUR_PROJECT_ID>/${PROJECT_ID}/g" \
    k8s/gemma-4-26b-a4b-vllm-torchtpu-v7x.yaml | kubectl apply -f -
kubectl rollout status deployment/gemma4-26b-a4b-torchtpu-v7x --timeout=30m
```

### 4. Run Reproducible Benchmarks on GKE

```bash
# Single-host suite
python3 scripts/run_v7x_bench.py \
  --config configs/gemma4_1h_4c_torchtpu_fp8dyn_tp8_ep.json \
  --bench-agent-url http://127.0.0.1:9100

# 4-host (16-chip) suite across 4 GKE Pod replicas
python3 scripts/run_v7x_bench.py \
  --config configs/gemma4_4h_16c_torchtpu_fp8dyn_tp8_ep.json \
  --bench-only-url http://gemma4-26b-a4b-v7x-svc.default.svc.cluster.local:8000
```
