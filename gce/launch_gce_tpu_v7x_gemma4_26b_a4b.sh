#!/usr/bin/env bash
# Optional Single-Host (1 host = 4 chips) GCE TPU v7x (tpu7x-standard-4t) Helper
# Model: google/gemma-4-26B-A4B-it (Pure BF16 & FP8-Dynamic)
# NOTE: For DWS Calendar Mode reservations and Multi-Host (2-host / 8-chip or 4-host / 16-chip)
# deployments, use the tested GKE workflow in INFRA_SETUP.md and k8s/provision_gke_dws_calendar.sh.
set -euo pipefail

GCS_BUCKET="${GCS_BUCKET:?Set GCS_BUCKET=<YOUR_GCS_BUCKET>}"
MODEL_SUBPATH="${MODEL_SUBPATH:-models/google/gemma-4-26B-A4B-it}"
FP8_MODEL_SUBPATH="${FP8_MODEL_SUBPATH:-models/RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic}"
XLA_CACHE_SUBPATH="${XLA_CACHE_SUBPATH:-xla-cache/gemma-4-26b-a4b}"
MOUNT_DIR="${MOUNT_DIR:-/model-vol-mount}"
VLLM_TPU_IMAGE="${VLLM_TPU_IMAGE:-vllm/vllm-tpu:latest}"
TORCHTPU_IMAGE="${TORCHTPU_IMAGE:-us-central1-docker.pkg.dev/<YOUR_PROJECT_ID>/vllm/vllm-torchtpu-bench:v0.32.0}"
MODE="${1:-help}"

install_deps() {
  sudo apt-get update && sudo apt-get install -y curl git jq fuse3 ca-certificates
  if ! command -v gcsfuse >/dev/null 2>&1; then
    local codename
    codename="$(lsb_release -c -s)"
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt gcsfuse-${codename} main" \
      | sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc >/dev/null
    sudo apt-get update && sudo apt-get install -y gcsfuse
  fi
  if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | sudo sh
  fi
  sudo usermod -aG docker "${USER}" || true
  gcloud auth configure-docker us-central1-docker.pkg.dev --quiet || true
}

mount_gcsfuse() {
  sudo mkdir -p "${MOUNT_DIR}"
  sudo chown -R "${USER}:${USER}" "${MOUNT_DIR}" || true
  if mountpoint -q "${MOUNT_DIR}"; then
    echo "${MOUNT_DIR} is already mounted."
  else
    gcsfuse --implicit-dirs \
      -o allow_other \
      --stat-cache-max-size-mb=-1 \
      --type-cache-max-size-mb=-1 \
      --kernel-list-cache-ttl-secs=-1 \
      --enable-buffered-read \
      --read-ahead-kb=131072 \
      "${GCS_BUCKET}" "${MOUNT_DIR}"
  fi
  mkdir -p "${MOUNT_DIR}/${XLA_CACHE_SUBPATH}"
}

# Profile 1A: Pure BF16 weights + BF16 KV cache on public vllm/vllm-tpu:latest (JAX Flax NNX, TP=2, DP=4)
serve_vllm_tpu_bf16() {
  sudo docker run --rm --privileged --net=host --shm-size=256g \
    -v /dev/vfio:/dev/vfio \
    -v "${MOUNT_DIR}:/model-vol-mount" \
    -e TPU_BACKEND_TYPE=jax \
    -e MODEL_IMPL_TYPE=flax_nnx \
    -e TPU_MULTIPROCESS_DP=1 \
    -e USE_BATCHED_RPA_KERNEL=1 \
    -e JAX_GC_POST_COMPILE=true \
    -e JAX_THREE_PHASE_DEALLOCATION=true \
    -e VLLM_ENGINE_READY_TIMEOUT_S=2400 \
    -e VLLM_XLA_CACHE_PATH="/model-vol-mount/${XLA_CACHE_SUBPATH}-jax-bf16" \
    "${VLLM_TPU_IMAGE}" \
    python3 -m vllm.entrypoints.openai.api_server \
      --host=0.0.0.0 \
      --port=8000 \
      --seed=42 \
      --model="/model-vol-mount/${MODEL_SUBPATH}" \
      --served-model-name=google/gemma-4-26B-A4B-it \
      --max-model-len=32768 \
      --max-num-seqs=128 \
      --data-parallel-size=4 \
      --tensor-parallel-size=2 \
      --max-num-batched-tokens=16384 \
      --enable-chunked-prefill \
      --no-enable-prefix-caching \
      --trust-remote-code \
      --kv-cache-dtype=auto \
      --gpu-memory-utilization=0.90 \
      --async-scheduling \
      --limit-mm-per-prompt='{"image": 0, "video": 0, "audio": 0}' \
      --block-size=256
}

# Profile 1B: Pure BF16 weights + BF16 KV cache on vllm-torchtpu (TP=2, DP=4, EP=true)
serve_vllm_torchtpu_bf16() {
  sudo docker run --rm --privileged --net=host --shm-size=256g \
    -v /dev/vfio:/dev/vfio \
    -v "${MOUNT_DIR}:/model-vol-mount" \
    -e PJRT_DEVICE=TPU \
    -e USE_MOE_SPARSE_CORE=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=2400 \
    -e VLLM_XLA_CACHE_PATH="/model-vol-mount/${XLA_CACHE_SUBPATH}-bf16-tp2dp4" \
    "${TORCHTPU_IMAGE}" \
    vllm serve "/model-vol-mount/${MODEL_SUBPATH}" \
      --served-model-name google/gemma-4-26B-A4B-it \
      --host 0.0.0.0 \
      --port 8000 \
      --tensor-parallel-size=2 \
      --data-parallel-size=4 \
      --enable-expert-parallel \
      --attention-backend CUSTOM \
      --api-server-count=1 \
      --max-model-len=32768 \
      --max-num-batched-tokens=16384 \
      --max-num-seqs=128 \
      --async-scheduling \
      --no-enable-prefix-caching \
      --gpu-memory-utilization=0.90 \
      --kv-cache-dtype=auto \
      --block-size=256 \
      --language-model-only \
      --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
      --hf-overrides '{"architectures":["Gemma4ForCausalLM"]}'
}

# Profile 1C: Pure BF16 weights + BF16 KV cache on vllm-torchtpu (TP=8, DP=1, EP=true for low-latency 22k TTFT)
serve_vllm_torchtpu_bf16_tp8() {
  sudo docker run --rm --privileged --net=host --shm-size=256g \
    -v /dev/vfio:/dev/vfio \
    -v "${MOUNT_DIR}:/model-vol-mount" \
    -e PJRT_DEVICE=TPU \
    -e USE_MOE_SPARSE_CORE=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=2400 \
    -e VLLM_XLA_CACHE_PATH="/model-vol-mount/${XLA_CACHE_SUBPATH}-bf16-tp8" \
    "${TORCHTPU_IMAGE}" \
    vllm serve "/model-vol-mount/${MODEL_SUBPATH}" \
      --served-model-name google/gemma-4-26B-A4B-it \
      --host 0.0.0.0 \
      --port 8000 \
      --tensor-parallel-size=8 \
      --data-parallel-size=1 \
      --enable-expert-parallel \
      --attention-backend CUSTOM \
      --max-model-len=32768 \
      --max-num-batched-tokens=24576 \
      --max-num-seqs=128 \
      --async-scheduling \
      --no-enable-prefix-caching \
      --gpu-memory-utilization=0.90 \
      --kv-cache-dtype=auto \
      --block-size=256 \
      --language-model-only \
      --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
      --hf-overrides '{"architectures":["Gemma4ForCausalLM"]}'
}

# Profile 2A: FP8-Dynamic weights + FP8 KV cache on vllm-torchtpu (TP=2, DP=4, EP=true — Peak Throughput)
serve_vllm_torchtpu_fp8() {
  local repo_root
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  sudo docker run --rm --privileged --net=host --shm-size=256g \
    -v /dev/vfio:/dev/vfio \
    -v "${MOUNT_DIR}:/model-vol-mount" \
    -v "${repo_root}/scripts/patches/core_map_helper.py:/workspace/vllm-torchtpu/src/vllm_torchtpu/kernels/sparse_core/core_map_helper.py:ro" \
    -e PJRT_DEVICE=TPU \
    -e USE_MOE_SPARSE_CORE=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=2400 \
    -e VLLM_XLA_CACHE_PATH="/model-vol-mount/${XLA_CACHE_SUBPATH}-fp8-tp2dp4" \
    "${TORCHTPU_IMAGE}" \
    vllm serve "/model-vol-mount/${FP8_MODEL_SUBPATH}" \
      --served-model-name google/gemma-4-26B-A4B-it \
      --host 0.0.0.0 \
      --port 8000 \
      --tensor-parallel-size=2 \
      --data-parallel-size=4 \
      --enable-expert-parallel \
      --attention-backend CUSTOM \
      --api-server-count=1 \
      --max-model-len=32768 \
      --max-num-batched-tokens=16384 \
      --max-num-seqs=256 \
      --async-scheduling \
      --no-enable-prefix-caching \
      --gpu-memory-utilization=0.90 \
      --kv-cache-dtype=fp8 \
      --block-size=256 \
      --language-model-only \
      --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
      --hf-overrides '{"architectures":["Gemma4ForCausalLM"]}'
}

# Profile 2B: FP8-Dynamic weights + FP8 KV cache on vllm-torchtpu (TP=8, DP=1, EP=true — Strict <500ms TTFT SLA)
serve_vllm_torchtpu_fp8_tp8() {
  local repo_root
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  sudo docker run --rm --privileged --net=host --shm-size=256g \
    -v /dev/vfio:/dev/vfio \
    -v "${MOUNT_DIR}:/model-vol-mount" \
    -v "${repo_root}/scripts/patches/core_map_helper.py:/workspace/vllm-torchtpu/src/vllm_torchtpu/kernels/sparse_core/core_map_helper.py:ro" \
    -e PJRT_DEVICE=TPU \
    -e USE_MOE_SPARSE_CORE=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=2400 \
    -e VLLM_XLA_CACHE_PATH="/model-vol-mount/${XLA_CACHE_SUBPATH}-fp8-tp8" \
    "${TORCHTPU_IMAGE}" \
    vllm serve "/model-vol-mount/${FP8_MODEL_SUBPATH}" \
      --served-model-name google/gemma-4-26B-A4B-it \
      --host 0.0.0.0 \
      --port 8000 \
      --tensor-parallel-size=8 \
      --data-parallel-size=1 \
      --enable-expert-parallel \
      --attention-backend CUSTOM \
      --max-model-len=32768 \
      --max-num-batched-tokens=24576 \
      --max-num-seqs=128 \
      --async-scheduling \
      --no-enable-prefix-caching \
      --gpu-memory-utilization=0.90 \
      --kv-cache-dtype=fp8 \
      --block-size=256 \
      --language-model-only \
      --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
      --hf-overrides '{"architectures":["Gemma4ForCausalLM"]}'
}

# Profile 2C: Public vllm/vllm-tpu:latest (JAX Flax NNX, TP=2, DP=4, FP8 KV cache)
serve_vllm_tpu_fp8kv() {
  sudo docker run --rm --privileged --net=host --shm-size=256g \
    -v /dev/vfio:/dev/vfio \
    -v "${MOUNT_DIR}:/model-vol-mount" \
    -e TPU_BACKEND_TYPE=jax \
    -e MODEL_IMPL_TYPE=flax_nnx \
    -e TPU_MULTIPROCESS_DP=1 \
    -e USE_BATCHED_RPA_KERNEL=1 \
    -e JAX_GC_POST_COMPILE=true \
    -e JAX_THREE_PHASE_DEALLOCATION=true \
    -e VLLM_ENGINE_READY_TIMEOUT_S=2400 \
    -e VLLM_XLA_CACHE_PATH="/model-vol-mount/${XLA_CACHE_SUBPATH}-jax" \
    "${VLLM_TPU_IMAGE}" \
    python3 -m vllm.entrypoints.openai.api_server \
      --host=0.0.0.0 \
      --port=8000 \
      --seed=42 \
      --model="/model-vol-mount/${MODEL_SUBPATH}" \
      --served-model-name=google/gemma-4-26B-A4B-it \
      --max-model-len=32768 \
      --max-num-seqs=256 \
      --data-parallel-size=4 \
      --tensor-parallel-size=2 \
      --max-num-batched-tokens=16384 \
      --enable-chunked-prefill \
      --no-enable-prefix-caching \
      --trust-remote-code \
      --kv-cache-dtype=fp8 \
      --gpu-memory-utilization=0.90 \
      --async-scheduling \
      --limit-mm-per-prompt='{"image": 0, "video": 0, "audio": 0}' \
      --block-size=256
}

run_client_benchmarks() {
  local lb_host="${LB_HOST:-127.0.0.1}"
  local lb_port="${LB_PORT:-8000}"
  local out_dir="${OUT_DIR:-./results}"
  mkdir -p "${out_dir}"

  # 1. Smoke test (1024 in / 21 out, CON = 1, 8)
  for CON in 1 8; do
    vllm bench serve --backend=vllm --dataset-name=random \
      --random-input-len=1024 --random-output-len=21 --num-prompts=32 --ignore-eos \
      --host="${lb_host}" --port="${lb_port}" --endpoint=/v1/completions \
      --model=google/gemma-4-26B-A4B-it --tokenizer="${MOUNT_DIR}/${MODEL_SUBPATH}" \
      --request-rate=inf --max-concurrency="${CON}" \
      --percentile-metrics=ttft,tpot,itl,e2el --metric-percentiles=50,90,95,99 \
      --goodput ttft:500 tpot:10 e2el:60000 --save-result --result-dir="${out_dir}"
  done

  # 2. Long-context Open-Loop Poisson RPS validation (22,000 in / 4,000 out, RPS = 1, 2, 4, 6)
  for RPS in 1 2 4 6; do
    vllm bench serve --backend=vllm --dataset-name=random \
      --random-input-len=22000 --random-output-len=4000 --num-prompts=50 --ignore-eos \
      --host="${lb_host}" --port="${lb_port}" --endpoint=/v1/completions \
      --model=google/gemma-4-26B-A4B-it --tokenizer="${MOUNT_DIR}/${MODEL_SUBPATH}" \
      --request-rate="${RPS}" \
      --percentile-metrics=ttft,tpot,itl,e2el --metric-percentiles=50,90,95,99 \
      --goodput ttft:500 tpot:10 e2el:60000 --save-result --result-dir="${out_dir}"
  done

  # 3. Long-context Closed-Loop Concurrency Sweep (22,000 in / 4,000 out, CON = 1, 6, 16, 32, 64)
  for CON in 1 6 16 32 64; do
    vllm bench serve --backend=vllm --dataset-name=random \
      --random-input-len=22000 --random-output-len=4000 --num-prompts=50 --ignore-eos \
      --host="${lb_host}" --port="${lb_port}" --endpoint=/v1/completions \
      --model=google/gemma-4-26B-A4B-it --tokenizer="${MOUNT_DIR}/${MODEL_SUBPATH}" \
      --request-rate=inf --max-concurrency="${CON}" \
      --percentile-metrics=ttft,tpot,itl,e2el --metric-percentiles=50,90,95,99 \
      --goodput ttft:500 tpot:10 e2el:60000 --save-result --result-dir="${out_dir}"
  done
}

case "${MODE}" in
  install-deps) install_deps ;;
  mount) mount_gcsfuse ;;
  serve-vllmtpu-bf16) serve_vllm_tpu_bf16 ;;
  serve-torchtpu-bf16) serve_vllm_torchtpu_bf16 ;;
  serve-torchtpu-bf16-tp8) serve_vllm_torchtpu_bf16_tp8 ;;
  serve-torchtpu-fp8) serve_vllm_torchtpu_fp8 ;;
  serve-torchtpu-fp8-tp8) serve_vllm_torchtpu_fp8_tp8 ;;
  serve-vllmtpu-fp8kv) serve_vllm_tpu_fp8kv ;;
  bench) run_client_benchmarks ;;
  *)
    echo "Usage: GCS_BUCKET=<YOUR_GCS_BUCKET> $0 {install-deps|mount|serve-vllmtpu-bf16|serve-torchtpu-bf16|serve-torchtpu-bf16-tp8|serve-torchtpu-fp8|serve-torchtpu-fp8-tp8|serve-vllmtpu-fp8kv|bench}"
    exit 1
    ;;
esac
