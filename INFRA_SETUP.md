# End-to-End GCP Infrastructure & DWS Calendar Reservation Setup Guide for Ironwood TPU (`v7x`)

This guide walks first-time Google Cloud users through setting up the complete cloud infrastructure required to deploy and benchmark **`google/gemma-4-26B-A4B-it`** (both **Pure BF16** and **FP8-Dynamic**) on **Ironwood TPU (`tpu7x-standard-4t`)** using a **Dynamic Workload Scheduler (DWS) Calendar Mode reservation**.

---

## 1. Architecture Decision: Can You Run TPU `v7x` on GCE / MIG Without GKE?

**Yes — Google Kubernetes Engine (GKE) is NOT required to serve or benchmark `google/gemma-4-26B-A4B-it` on `1` host (`4` chips), `2` hosts (`8` chips), or `4` hosts (`16` chips).**

### Why Standalone GCE VMs (or a GCE MIG) Work for Both Single-Host and Multi-Host LLM Serving

| Hardware Unit | Physical Spec | How `gemma-4-26B-A4B-it` Fits & Scales on GCE |
|---|---|---|
| **`1 x tpu7x-standard-4t` VM** (Single Host) | `4` Ironwood TPU chips (`2x2x1` intra-host ICI mesh) = **`8` TensorCores** + SparseCores, **`768 GB` HBM3e** (`192 GB`/chip @ `7.4 TB/s`/chip) | The full `google/gemma-4-26B-A4B-it` checkpoint is **`~52 GB` in BF16** (`~28.7 GB` in FP8-Dynamic). A single 4-chip VM holds the entire model (`TP=8, EP` or `TP=2, DP=4, EP`) using `< 7%` of HBM for weights, leaving **`> 700 GB` of HBM3e per host** for long-context (`32k+`) KV cache. |
| **`2 x tpu7x-standard-4t` VMs** (`8` Chips) | `2` independent 4-chip GCE VMs (`16` TensorCores, `1.54 TB` HBM3e) | Each GCE VM runs its own self-contained OpenAI API server on `:8000`. A lightweight `least_conn` load balancer (`scripts/lb_proxy.py` or NGINX) on the CPU Client VM distributes requests across both hosts with zero cross-host ICI barrier overhead. |
| **`4 x tpu7x-standard-4t` VMs** (`16` Chips) | `4` independent 4-chip GCE VMs (`32` TensorCores, `3.07 TB` HBM3e) | Linearly scales throughput (`3.28x–6.09x` Goodput scaling) by running `4` independent 4-chip servers behind the Client VM load balancer. |

### Comparing the 3 Ways to Consume Your DWS Calendar Reservation

| Provisioning Option | Requires GKE? | Best For | Multi-Host Support |
|---|---|---|---|
| **Option A (Recommended): Standalone GCE VMs Bound to DWS Reservation** (`gcloud compute instances create --provisioning-model=RESERVATION_BOUND`) | **No** | Fastest setup for first-time GCP users; direct SSH + Docker control on `1`, `2`, or `4` hosts (`4`, `8`, or `16` chips) | **Yes** — horizontal scale-out via NGINX or `scripts/lb_proxy.py` on the Client VM |
| **Option B: GCE Managed Instance Group (MIG) + Workload Policy** (`gcloud compute instance-groups managed create`) | **No** | Automated multi-host VM lifecycle on GCE, or co-located `2x2x2` (`8`-chip) / `2x2x4` (`16`-chip) high-throughput ICI slice placement | **Yes** — supports both scale-out load balancing and multi-host ICI (`TPU_MULTIHOST_BACKEND=ray`) |
| **Option C: GKE Standard Cluster + DWS Reservation Node Pool** (`gcloud container node-pools create --reservation=...`) | **Yes** | Teams standardizing on Kubernetes manifests (`k8s/`), GCSFuse CSI volumes, and K8s Services | **Yes** — via `spec.replicas: 1 / 2 / 4` or multi-host LeaderWorkerSet |

```
                       ┌────────────────────────────────────────────────────┐
                       │         Regional GCS Bucket (us-central1)          │
                       │  gs://<YOUR_GCS_BUCKET>/models/gemma-4-26B-A4B-it  │
                       └─────────┬────────────────────────────────┬─────────┘
                                 │ gcsfuse                        │ gcsfuse (128MB read-ahead)
                                 ▼                                ▼
┌──────────────────────────────────────────┐     ┌───────────────────────────────────────────┐
│   CPU Client & Benchmark VM (us-central1-c) │     │  DWS Calendar Reserved TPU VMs (1, 2, 4)  │
│   Machine: c2-standard-16 (250GB SSD)    │     │  Machine: tpu7x-standard-4t (4 chips/VM)  │
│                                          │     │                                           │
│  1. Stages HF weights -> GCS bucket      │     │  • tpu-v7x-node-1 (:8000) [4 chips, 768GB]│
│  2. Runs NGINX / lb_proxy.py (:8080) ────┼────►│  • tpu-v7x-node-2 (:8000) [4 chips, 768GB]│
│  3. Runs vllm bench serve (--goodput)    │     │  • tpu-v7x-node-3 (:8000) [4 chips, 768GB]│
│                                          │     │  • tpu-v7x-node-4 (:8000) [4 chips, 768GB]│
└──────────────────────────────────────────┘     └───────────────────────────────────────────┘
```

---

## 2. Step 1 — GCP Project Prerequisites, APIs & IAM Permissions

Run the following commands from **Google Cloud Shell** or your local terminal (after running `gcloud auth login`).

### 2.1 Set Environment Variables

Edit the variables below to match your GCP project and DWS Calendar reservation name:

```bash
export PROJECT_ID="<YOUR_GCP_PROJECT_ID>"
export REGION="us-central1"
export ZONE="us-central1-c"                     # Zone where your DWS Calendar reservation is active
export RESERVATION_NAME="<YOUR_RESERVATION_NAME>" # From: gcloud compute reservations list
export GCS_BUCKET="${PROJECT_ID}-tpu-v7x-models"
export VPC_NAME="tpu-v7x-vpc"
export SUBNET_NAME="tpu-v7x-subnet"
export SA_NAME="tpu-v7x-bench-sa"
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "${PROJECT_ID}"
gcloud config set compute/region "${REGION}"
gcloud config set compute/zone "${ZONE}"
```

### 2.2 Enable Required Google Cloud APIs

```bash
gcloud services enable \
  compute.googleapis.com \
  tpu.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  iap.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  container.googleapis.com
```

### 2.3 Create the Workload Service Account & Grant IAM Roles

Both the TPU VMs and the CPU Client VM use this Service Account to read/write model checkpoints in GCS and pull container images from Artifact Registry:

```bash
gcloud iam service-accounts create "${SA_NAME}" \
  --display-name="TPU v7x Benchmark Service Account"

for ROLE in \
  roles/compute.instanceAdmin.v1 \
  roles/tpu.admin \
  roles/storage.admin \
  roles/artifactregistry.reader \
  roles/logging.logWriter \
  roles/monitoring.metricWriter \
  roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --quiet
done
```

> **User IAM Note**: Ensure your own user account (`gcloud config get-value account`) has `roles/compute.admin`, `roles/storage.admin`, `roles/iam.serviceAccountUser`, and `roles/iap.tunnelResourceAccessor` in the project.

---

## 3. Step 2 — Network Setup (VPC, Subnet, Cloud NAT & Firewall Rules)

For security and maximum throughput, TPU VMs run without public IPv4 addresses (`--no-address`) inside a custom VPC subnet with **Private Google Access** (for full-bandwidth GCS access) and **Cloud NAT** (for outbound Hugging Face and Docker Hub downloads).

```bash
# 1. Create custom VPC network
gcloud compute networks create "${VPC_NAME}" \
  --subnet-mode=custom \
  --mtu=8896 \
  --bgp-routing-mode=regional

# 2. Create regional subnet with Private Google Access enabled
gcloud compute networks subnets create "${SUBNET_NAME}" \
  --network="${VPC_NAME}" \
  --region="${REGION}" \
  --range="10.128.0.0/20" \
  --enable-private-ip-google-access

# 3. Create Cloud Router & Cloud NAT so internal-only VMs can reach Hugging Face, apt, and Docker Hub
gcloud compute routers create "${VPC_NAME}-router" \
  --network="${VPC_NAME}" \
  --region="${REGION}"

gcloud compute routers nats create "${VPC_NAME}-nat" \
  --router="${VPC_NAME}-router" \
  --region="${REGION}" \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges

# 4. Allow all internal traffic between Client VM and TPU VMs (ports 8000-9200, Ray/ICI, etc.)
gcloud compute firewall-rules create "${VPC_NAME}-allow-internal" \
  --network="${VPC_NAME}" \
  --allow=tcp:1-65535,udp:1-65535,icmp \
  --source-ranges="10.128.0.0/20"

# 5. Allow SSH via Google Cloud Identity-Aware Proxy (IAP)
gcloud compute firewall-rules create "${VPC_NAME}-allow-ssh-iap" \
  --network="${VPC_NAME}" \
  --allow=tcp:22 \
  --source-ranges="35.235.240.0/20"
```

---

## 4. Step 3 — Create the Regional GCS Bucket & CPU Client VM

### 4.1 Create the Regional GCS Bucket (`us-central1`)

> **Important**: Always create the GCS bucket in the **same region (`us-central1`)** as your TPU VMs to avoid cross-region egress latency during model weight loading and XLA compilation cache reads.

```bash
gcloud storage buckets create "gs://${GCS_BUCKET}" \
  --location="${REGION}" \
  --uniform-bucket-level-access
```

### 4.2 Create the CPU Client & Benchmark VM (`c2-standard-16`)

This VM is used to download model weights from Hugging Face to GCS, run the multi-host load balancer (`lb_proxy.py` or NGINX), and execute `vllm bench serve` without stealing CPU cores from the TPU servers:

```bash
gcloud compute instances create tpu-bench-client \
  --zone="${ZONE}" \
  --machine-type=c2-standard-16 \
  --network-interface="network=${VPC_NAME},subnetwork=${SUBNET_NAME},nic-type=GVNIC,no-address" \
  --boot-disk-size=250GB \
  --boot-disk-type=pd-ssd \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --service-account="${SA_EMAIL}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

### 4.3 Install Tools & Stage `gemma-4-26B-A4B-it` Checkpoints to GCS

SSH into `tpu-bench-client` using IAP tunnel:

```bash
gcloud compute ssh tpu-bench-client --zone="${ZONE}" --tunnel-through-iap
```

Inside `tpu-bench-client`, install `gcsfuse`, Python/`uv`, and NGINX, then stage the model weights to your GCS bucket:

```bash
# 1. Install system packages & gcsfuse
sudo apt-get update && sudo apt-get install -y curl git jq nginx fuse3 python3-pip python3-venv
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
sudo apt-get update && sudo apt-get install -y gcsfuse

# 2. Clone this repository
git clone https://github.com/ankdeshm/gemma-4-26b-a4b-tpu-v7x-inference.git
cd gemma-4-26b-a4b-tpu-v7x-inference

# 3. Install uv, vllm (for vllm bench serve), and huggingface_hub
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv ~/.venv --python 3.12
source ~/.venv/bin/activate
uv pip install "vllm>=0.8.5" "huggingface_hub[cli,hf_transfer]" google-cloud-storage aiohttp

# 4. Stage both Pure BF16 and FP8-Dynamic checkpoints directly to GCS
export HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>"   # Must have accepted Google Gemma license on HF
export GCS_BUCKET="<YOUR_GCS_BUCKET>"

python3 scripts/stage_hf_to_gcs.py \
  --bucket "${GCS_BUCKET}" \
  --prefix models \
  --repos google/gemma-4-26B-A4B-it,RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic \
  --workers 8

# 5. Also mount the GCS bucket locally on tpu-bench-client so vllm bench serve can read the tokenizer
sudo mkdir -p /model-vol-mount
sudo chown -R $USER:$USER /model-vol-mount
gcsfuse --implicit-dirs "${GCS_BUCKET}" /model-vol-mount
```

---

## 5. Step 4 — Provision TPU `v7x` (`tpu7x-standard-4t`) Using Your DWS Calendar Reservation

First, verify your active DWS Calendar reservation name and available `tpu7x-standard-4t` count (`1` host = `4` chips, `2` hosts = `8` chips, `4` hosts = `16` chips):

```bash
gcloud compute reservations list --project="${PROJECT_ID}"
gcloud compute reservations describe "${RESERVATION_NAME}" --zone="${ZONE}"
```

> **Critical GCE Image Note**: Standard `ubuntu-2404-lts` images do not include the TPU `v7x` VFIO driver rules out of the box. Always use the official accelerator image family:
> - `--image-family=ubuntu-accel-2404-amd64-tpu-tpu7x`
> - `--image-project=ubuntu-os-accelerator-images`
> - `--network-interface=...,nic-type=GVNIC`

### Option A (Recommended): Create `1`, `2`, or `4` Standalone GCE VMs Bound to Your DWS Reservation

Set `NUM_HOSTS=1` (`4` chips), `NUM_HOSTS=2` (`8` chips), or `NUM_HOSTS=4` (`16` chips):

```bash
export NUM_HOSTS=4   # Set to 1 (4 chips), 2 (8 chips), or 4 (16 chips)

for i in $(seq 1 "${NUM_HOSTS}"); do
  gcloud compute instances create "tpu-v7x-node-${i}" \
    --zone="${ZONE}" \
    --machine-type=tpu7x-standard-4t \
    --image-family=ubuntu-accel-2404-amd64-tpu-tpu7x \
    --image-project=ubuntu-os-accelerator-images \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-balanced \
    --network-interface="network=${VPC_NAME},subnetwork=${SUBNET_NAME},nic-type=GVNIC,no-address" \
    --provisioning-model=RESERVATION_BOUND \
    --reservation-affinity=specific \
    --reservation="${RESERVATION_NAME}" \
    --maintenance-policy=TERMINATE \
    --instance-termination-action=DELETE \
    --service-account="${SA_EMAIL}" \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --Async
done

# Wait until all TPU VMs report RUNNING and note their INTERNAL_IP addresses
gcloud compute instances list --filter="name~'^tpu-v7x-node-'"
```

### Option B: Create a GCE Managed Instance Group (MIG) + Workload Policy (No GKE)

If you prefer managing your hosts as a **GCE Managed Instance Group (MIG)** — or if you want GCE to co-locate `2` hosts (`2x2x2` = `8` chips) or `4` hosts (`2x2x4` = `16` chips) on the optical Inter-Chip Interconnect (ICI) fabric:

```bash
# 1. Create a Workload Policy (use 2x2x1 for 1 host, 2x2x2 for 2 hosts / 8 chips, or 2x2x4 for 4 hosts / 16 chips)
gcloud compute resource-policies create workload-policy tpu-v7x-16c-policy \
  --region="${REGION}" \
  --type=HIGH_THROUGHPUT \
  --accelerator-topology=2x2x4

# 2. Create the Instance Template bound to your DWS Calendar Reservation
gcloud compute instance-templates create tpu-v7x-dws-template \
  --region="${REGION}" \
  --machine-type=tpu7x-standard-4t \
  --image-family=ubuntu-accel-2404-amd64-tpu-tpu7x \
  --image-project=ubuntu-os-accelerator-images \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --network-interface="network=${VPC_NAME},subnetwork=${SUBNET_NAME},nic-type=GVNIC,no-address" \
  --provisioning-model=RESERVATION_BOUND \
  --reservation-affinity=specific \
  --reservation="projects/${PROJECT_ID}/zones/${ZONE}/reservations/${RESERVATION_NAME}" \
  --maintenance-policy=TERMINATE \
  --instance-termination-action=DELETE \
  --service-account="${SA_EMAIL}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform

# 3. Create the Zonal Managed Instance Group (size=4 for 16 chips)
gcloud compute instance-groups managed create tpu-v7x-mig \
  --zone="${ZONE}" \
  --template=tpu-v7x-dws-template \
  --size=4 \
  --resource-policies=tpu-v7x-16c-policy

# List the created VM names and internal IPs
gcloud compute instance-groups managed list-instances tpu-v7x-mig --zone="${ZONE}"
```

### Option C (Optional): GKE Standard Cluster + DWS Calendar Node Pool

Only use Option C if your team prefers Kubernetes manifests (`k8s/`) over GCE VMs:

```bash
gcloud container clusters create tpu-v7x-cluster \
  --region="${REGION}" \
  --network="${VPC_NAME}" \
  --subnetwork="${SUBNET_NAME}" \
  --enable-ip-alias \
  --enable-dataplane-v2 \
  --workload-pool="${PROJECT_ID}.svc.id.goog" \
  --addons=GcsFuseCsiDriver \
  --num-nodes=1 \
  --machine-type=e2-standard-8

gcloud container node-pools create tpu-v7x-dws-pool \
  --cluster=tpu-v7x-cluster \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=tpu7x-standard-4t \
  --num-nodes=4 \
  --reservation-affinity=specific \
  --reservation="${RESERVATION_NAME}" \
  --service-account="${SA_EMAIL}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

---

## 6. Step 5 — Prepare Each GCE TPU VM & Launch the Gemma-4 Serving Recipe

Run this step on **each** `tpu-v7x-node-{1..N}` VM (via `gcloud compute ssh tpu-v7x-node-1 --zone=${ZONE} --tunnel-through-iap`).

### 6.1 One-Time TPU VM Setup (Install `gcsfuse` & Mount Bucket)

```bash
# Verify 4 Ironwood TPU chips (8 TensorCores + SparseCores) are exposed via VFIO
ls -la /dev/vfio

# Clone the repo and mount your GCS bucket with 128MB read-ahead & metadata caching
git clone https://github.com/ankdeshm/gemma-4-26b-a4b-tpu-v7x-inference.git
cd gemma-4-26b-a4b-tpu-v7x-inference

export GCS_BUCKET="<YOUR_GCS_BUCKET>"
bash gce/launch_gce_tpu_v7x_gemma4_26b_a4b.sh install-deps
bash gce/launch_gce_tpu_v7x_gemma4_26b_a4b.sh mount
```

### 6.2 Choose and Launch Your Serving Profile on Each TPU VM (`:8000`)

We provide turnkey modes for **both** **Pure BF16** (matching a strict `bfloat16` weights + `bfloat16` KV cache PoC requirement) and **FP8-Dynamic** (for maximum throughput and `< 450 ms` `22k`-input TTFT).

#### Profile 1: Pure `bfloat16` Weights + `bfloat16` KV Cache (`google/gemma-4-26B-A4B-it`)

* **1A — Pure BF16 on Public `vllm/vllm-tpu:latest` (JAX Flax NNX, `TP=2, DP=4`, BF16 KV):**
  Works immediately using the public Docker Hub image `vllm/vllm-tpu:latest`:
  ```bash
  GCS_BUCKET="${GCS_BUCKET}" bash gce/launch_gce_tpu_v7x_gemma4_26b_a4b.sh serve-vllmtpu-bf16
  ```
* **1B — Pure BF16 on `vllm-torchtpu` (`TP=8, DP=1, EP` for Low-Latency `22k` Prefill, or `TP=2, DP=4, EP` for Throughput):**
  ```bash
  # Low-latency TP=8, EP=true (Pure BF16 weights + BF16 KV cache):
  GCS_BUCKET="${GCS_BUCKET}" bash gce/launch_gce_tpu_v7x_gemma4_26b_a4b.sh serve-torchtpu-bf16-tp8

  # Or High-throughput TP=2, DP=4, EP=true (Pure BF16 weights + BF16 KV cache):
  GCS_BUCKET="${GCS_BUCKET}" bash gce/launch_gce_tpu_v7x_gemma4_26b_a4b.sh serve-torchtpu-bf16
  ```

#### Profile 2: FP8-Dynamic Weights + FP8 KV Cache (`RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` or `vllm-tpu` FP8 KV)

* **2A — Strict SLA Goodput Champion (`vllm-torchtpu` FP8-Dynamic, `TP=8, DP=1, EP=true`):**
  Delivers **`446.7 ms` TTFT (`< 500 ms` SLA)** and **`4.50 ms` TPOT (`< 10 ms` SLA)** on `22,000`-token inputs (`100%` Goodput):
  ```bash
  GCS_BUCKET="${GCS_BUCKET}" bash gce/launch_gce_tpu_v7x_gemma4_26b_a4b.sh serve-torchtpu-fp8-tp8
  ```
* **2B — Peak Per-Chip Throughput Champion (`vllm-torchtpu` FP8-Dynamic, `TP=2, DP=4, EP=true`):**
  Delivers **`5,327.5` output tok/s** on `22k/4k` and **`11,261.7` output tok/s (`2,815.4` tok/s/chip)** on `1k/1k` per 4-chip host:
  ```bash
  GCS_BUCKET="${GCS_BUCKET}" bash gce/launch_gce_tpu_v7x_gemma4_26b_a4b.sh serve-torchtpu-fp8
  ```
* **2C — Public `vllm/vllm-tpu:latest` JAX Flax NNX with FP8 KV Cache (`TP=2, DP=4`):**
  ```bash
  GCS_BUCKET="${GCS_BUCKET}" bash gce/launch_gce_tpu_v7x_gemma4_26b_a4b.sh serve-vllmtpu-fp8kv
  ```

Wait until each TPU VM logs `Application startup complete` on port `8000` (initial XLA compilation takes ~8–15 minutes and is cached in `gs://${GCS_BUCKET}/xla-cache/` so subsequent restarts take ~2 minutes).

---

## 7. Step 6 — Multi-Host Load Balancing (`2` Hosts = `8` Chips, `4` Hosts = `16` Chips) & Running the Benchmarks

Run all commands below from your **CPU Client VM (`tpu-bench-client`)**.

### 7.1 Start the Multi-Host Load Balancer on `tpu-bench-client`

Look up the internal IPs of your TPU VMs (`gcloud compute instances list --filter="name~'^tpu-v7x-node-'"`).

* **For Single-Host (`4` chips)**: Point directly to `NODE_1_IP:8000` (`export LB_HOST=<NODE_1_IP> LB_PORT=8000`).
* **For `2` Hosts (`8` chips) or `4` Hosts (`16` chips)**: Start `scripts/lb_proxy.py` (or NGINX `least_conn`) on port `8080` of `tpu-bench-client`:

```bash
# Option 1: Lightweight async least_conn load balancer included in this repo
python3 scripts/lb_proxy.py \
  --listen 0.0.0.0:8080 \
  --backends http://<NODE_1_IP>:8000,http://<NODE_2_IP>:8000,http://<NODE_3_IP>:8000,http://<NODE_4_IP>:8000 &

export LB_HOST="127.0.0.1"
export LB_PORT="8080"
```

<details>
<summary><b>Alternative: NGINX <code>least_conn</code> Load Balancer Configuration</b></summary>

```bash
sudo tee /etc/nginx/conf.d/tpu_v7x_lb.conf << 'EOF'
upstream tpu_v7x_backends {
    least_conn;
    server <NODE_1_IP>:8000 max_fails=0;
    server <NODE_2_IP>:8000 max_fails=0;
    # Uncomment for 4-host (16-chip) benchmarks:
    # server <NODE_3_IP>:8000 max_fails=0;
    # server <NODE_4_IP>:8000 max_fails=0;
    keepalive 128;
}

server {
    listen 8080;
    location / {
        proxy_pass http://tpu_v7x_backends;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        client_max_body_size 100M;
    }
}
EOF
sudo nginx -t && sudo systemctl restart nginx
```
</details>

### 7.2 Execute the PoC Benchmark Suite (`22,000` Input / `4,000` Output & Smoke Tests)

Run the turnkey benchmark sweep from `tpu-bench-client`:

```bash
GCS_BUCKET="${GCS_BUCKET}" LB_HOST="${LB_HOST}" LB_PORT="${LB_PORT}" \
  bash gce/launch_gce_tpu_v7x_gemma4_26b_a4b.sh bench
```

Or run the individual `vllm bench serve` commands directly:

```bash
export TOKENIZER_PATH="/model-vol-mount/models/google/gemma-4-26B-A4B-it"

# 1. Smoke Test (ISL=1024, OSL=21, Concurrency = 1, 8)
for CON in 1 8; do
  vllm bench serve --backend=vllm --dataset-name=random \
    --random-input-len=1024 --random-output-len=21 --num-prompts=32 --ignore-eos \
    --host="${LB_HOST}" --port="${LB_PORT}" --endpoint=/v1/completions \
    --model=google/gemma-4-26B-A4B-it --tokenizer="${TOKENIZER_PATH}" \
    --request-rate=inf --max-concurrency="${CON}" \
    --percentile-metrics=ttft,tpot,itl,e2el --metric-percentiles=50,90,95,99 \
    --goodput ttft:500 tpot:10 e2el:60000
done

# 2. Long-Context Closed-Loop Concurrency Sweep (ISL=22000, OSL=4000, CON = 1, 6, 16, 32, 64)
for CON in 1 6 16 32 64; do
  vllm bench serve --backend=vllm --dataset-name=random \
    --random-input-len=22000 --random-output-len=4000 --num-prompts=50 --ignore-eos \
    --host="${LB_HOST}" --port="${LB_PORT}" --endpoint=/v1/completions \
    --model=google/gemma-4-26B-A4B-it --tokenizer="${TOKENIZER_PATH}" \
    --request-rate=inf --max-concurrency="${CON}" \
    --percentile-metrics=ttft,tpot,itl,e2el --metric-percentiles=50,90,95,99 \
    --goodput ttft:500 tpot:10 e2el:60000 \
    --save-result --result-dir=./results
done

# 3. Long-Context Open-Loop Poisson Arrival Sweep (ISL=22000, OSL=4000, RPS = 1, 2, 4, 6)
for RPS in 1 2 4 6; do
  vllm bench serve --backend=vllm --dataset-name=random \
    --random-input-len=22000 --random-output-len=4000 --num-prompts=50 --ignore-eos \
    --host="${LB_HOST}" --port="${LB_PORT}" --endpoint=/v1/completions \
    --model=google/gemma-4-26B-A4B-it --tokenizer="${TOKENIZER_PATH}" \
    --request-rate="${RPS}" \
    --percentile-metrics=ttft,tpot,itl,e2el --metric-percentiles=50,90,95,99 \
    --goodput ttft:500 tpot:10 e2el:60000 \
    --save-result --result-dir=./results
done
```

---

## 8. Common Pitfalls & Corrections Checklist

| Area | Common Pitfall in Draft Docs | Validated Fix in This Repository |
|---|---|---|
| **GCE OS Image** | Using generic `--image-family=IMAGE_FAMILY` or standard Ubuntu images that lack TPU `v7x` VFIO permissions | Always specify `--image-family=ubuntu-accel-2404-amd64-tpu-tpu7x --image-project=ubuntu-os-accelerator-images` and `nic-type=GVNIC` |
| **Docker TPU Device Access** | Running `python3 -m vllm...` without exposing VFIO devices or adequate shared memory | Pass `--privileged --net=host --shm-size=256g -v /dev/vfio:/dev/vfio -v /model-vol-mount:/model-vol-mount` to `docker run` |
| **Pure BF16 vs. FP8 Precision** | Mixing up a pure `bfloat16` PoC requirement with `qwix` `float8_e4m3fn` + `--kv-cache-dtype=fp8` flags | Use `serve-vllmtpu-bf16` or `serve-torchtpu-bf16-tp8` (`--kv-cache-dtype=auto` on `google/gemma-4-26B-A4B-it`) for pure BF16, and `serve-torchtpu-fp8-tp8` for FP8-Dynamic |
| **`22k` Input `< 500 ms` TTFT SLA** | Running `TP=2, DP=4` when optimizing for single-request `22,000`-token TTFT (`~1,023 ms` on 2 cores) | Use `TP=8, DP=1, --enable-expert-parallel` (`serve-torchtpu-fp8-tp8` or `serve-torchtpu-bf16-tp8`) to shard prefill across all `8` TensorCores (`446.7 ms` TTFT!) |
| **Automated SLA Goodput Measurement** | Omitting `--goodput` from `vllm bench serve` and manually inspecting latency tables | Pass `--goodput ttft:500 tpot:10 e2el:60000` so `vllm bench serve` directly computes strict per-request Goodput (`req/s`) |
