# End-to-End GKE Infrastructure & DWS Calendar Reservation Setup Guide for Ironwood TPU (`v7x`)

This guide walks first-time Google Cloud users through setting up the production-grade cloud infrastructure required to deploy and benchmark **`google/gemma-4-26B-A4B-it`** (both **Pure `bfloat16`** and **FP8-Dynamic**) on **Ironwood TPU (`tpu7x-standard-4t`)** using a **Dynamic Workload Scheduler (DWS) Calendar Mode reservation**.

All configurations and benchmarks in this repository were validated on **Google Kubernetes Engine (GKE)** with `tpu7x-standard-4t` node pools.

---

## 1. Architecture Decision: Why GKE is Required for Multi-Host & Production Ironwood (`tpu7x`)

A common question from teams new to Google Cloud TPUs is whether a **DWS Calendar Mode reservation** (`e.g., 4 x tpu7x-standard-4t = 16 chips`) should be consumed via standalone **Compute Engine (GCE) VMs** or **Google Kubernetes Engine (GKE)**.

| Deployment Scenario | Recommended Platform | Architectural Rationale |
|---|---|---|
| **Single-Host Quick Test (`1 x tpu7x-standard-4t` = `4` Chips / `8` TensorCores)** | **GKE** (Recommended) or **Single GCE VM** | A single `tpu7x-standard-4t` host (`4` physical chips connected via intra-host `2x2x1` ICI, `768 GB` HBM3e) can run as a standalone GCE VM for basic single-node container testing because all 8 TensorCores reside on one motherboard. |
| **Multi-Host (`2` Hosts = `8` Chips `2x2x2`, or `4` Hosts = `16` Chips `2x2x4`) & Production Serving** | **GKE Standard Cluster** (**Mandatory / Strongly Recommended**) | **Do not use raw GCE VMs for multi-host TPU workloads.** Starting with Ironwood (`tpu7x`), Google Cloud's official AI Hypercomputer stack and TPU recipes are engineered around **GKE** for the reasons below. |

### Why Raw GCE VMs Are Not Suitable for Multi-Host `tpu7x`

1. **Legacy Cloud TPU VM API Deprecation & Ironwood Workload Policies**:
   - Earlier TPU generations used the standalone `gcloud compute tpus tpu-vm` API. On **Ironwood (`tpu7x`)**, multi-host slice placement is governed by regional **Compute Engine Workload Policies (`--type=HIGH_THROUGHPUT --accelerator-topology=2x2x2` or `2x2x4`)** attached to GKE node pools via `--placement-policy` (see [About Ironwood (TPU7x) in GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/tpu-ironwood)).
2. **Multi-Host Coordination & Slice Discovery (`RayOperator` / KubeRay Webhook / LWS)**:
   - On raw GCE VMs, hosts have no built-in orchestration to discover peer worker hostnames, assign slice rank IDs, or initialize cross-host optical Inter-Chip Interconnect (ICI) collectives.
   - On **GKE**, the managed **Ray on GKE add-on (`--addons=RayOperator`)** and its KubeRay TPU webhook (or Kubernetes `LeaderWorkerSet`) automatically inject `TPU_WORKER_ID`, `TPU_WORKER_HOSTNAMES`, `TORCH_TPU_TOPOLOGY`, and `TORCH_TPU_SLICEBUILDER_ADDRESSES` into every Pod across the slice (see [Serve LLMs on multi-host TPUs with Ray and TorchTPU](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-llm-tpu-ray-torchtpu)).
3. **Production Load Balancing, Prefix-Aware Routing & Health Recovery**:
   - GKE provides native Kubernetes `Service` load balancing, **GKE Inference Gateway**, and **Ray Serve LLM** (with prefix-aware KV cache routing, direct token streaming, and atomic slice health checks/auto-repair).
4. **Native Cloud Storage FUSE CSI Driver (`--addons=GcsFuseCsiDriver`) & NUMA Binding**:
   - GKE mounts your GCS model bucket directly into Pods via declarative PersistentVolumeClaims (`k8s/*.yaml`) with kernel metadata caching and automatically binds containers to Ironwood's dual-NUMA-node architecture.

### Official Google Cloud Documentation References
- [Request TPUs with future reservation in calendar mode (GKE)](https://cloud.google.com/kubernetes-engine/docs/how-to/tpu-calendar-mode)
- [About Ironwood (TPU7x) in GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/tpu-ironwood)
- [Deploy TPU workloads in GKE Standard](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus)
- [Serve LLMs on multi-host TPUs with Ray and TorchTPU (`Gemma 4 26B-A4B-it`)](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-llm-tpu-ray-torchtpu)

---

## 2. Step 1 — GCP Project Prerequisites, APIs & IAM Permissions

Run the following commands from **Google Cloud Shell** or a workstation with the `gcloud` CLI and `kubectl` installed.

### 2.1 Set Environment Variables

```bash
export PROJECT_ID="<YOUR_GCP_PROJECT_ID>"
export REGION="us-central1"
export ZONE="us-central1-c"                        # Zone where your DWS Calendar reservation is approved
export RESERVATION_NAME="<YOUR_RESERVATION_NAME>"  # From: gcloud compute reservations list
export CLUSTER_NAME="tpu-v7x-gke-cluster"
export GCS_BUCKET="${PROJECT_ID}-tpu-v7x-models"
export VPC_NAME="tpu-v7x-vpc"
export SUBNET_NAME="tpu-v7x-subnet"
export SA_NAME="tpu-v7x-gke-sa"
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "${PROJECT_ID}"
gcloud config set compute/region "${REGION}"
gcloud config set compute/zone "${ZONE}"
```

### 2.2 Enable Required Google Cloud APIs

```bash
gcloud services enable \
  container.googleapis.com \
  compute.googleapis.com \
  tpu.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com
```

### 2.3 Create the GKE Node Service Account & Grant Required IAM Roles

```bash
gcloud iam service-accounts create "${SA_NAME}" \
  --display-name="GKE Ironwood TPU v7x Service Account"

for ROLE in \
  roles/container.defaultNodeServiceAccount \
  roles/compute.instanceAdmin.v1 \
  roles/storage.objectAdmin \
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

> **Required User Roles**: To create the GKE cluster and consume a DWS Calendar reservation, your user account needs `roles/container.admin`, `roles/compute.instanceAdmin.v1` (and `roles/compute.networkAdmin` for VPC creation), `roles/storage.admin`, and `roles/iam.serviceAccountAdmin`.

---

## 3. Step 2 — Network Setup (VPC, Subnet, Cloud NAT & Firewall Rules)

```bash
# 1. Create custom VPC network with jumbo frames (MTU 8896)
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

# 3. Create Cloud Router & Cloud NAT for outbound Hugging Face & container registry pulls
gcloud compute routers create "${VPC_NAME}-router" \
  --network="${VPC_NAME}" \
  --region="${REGION}"

gcloud compute routers nats create "${VPC_NAME}-nat" \
  --router="${VPC_NAME}-router" \
  --region="${REGION}" \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges

# 4. Allow internal communication across the subnet
gcloud compute firewall-rules create "${VPC_NAME}-allow-internal" \
  --network="${VPC_NAME}" \
  --allow=tcp:1-65535,udp:1-65535,icmp \
  --source-ranges="10.128.0.0/20"
```

---

## 4. Step 3 — Create Regional GCS Bucket & Stage `Gemma-4-26B-A4B-it` Weights

### 4.1 Create the Regional GCS Bucket (`us-central1`)

```bash
gcloud storage buckets create "gs://${GCS_BUCKET}" \
  --location="${REGION}" \
  --uniform-bucket-level-access
```

### 4.2 Stage Both Pure `bfloat16` and `FP8-Dynamic` Checkpoints to GCS

```bash
export HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>"   # Must have accepted Google Gemma license on HF

python3 scripts/stage_hf_to_gcs.py \
  --bucket "${GCS_BUCKET}" \
  --prefix models \
  --repos google/gemma-4-26B-A4B-it,RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic \
  --workers 8
```

---

## 5. Step 4 — Create GKE Standard Cluster & DWS Calendar Reservation Node Pool (`tpu7x`)

First, verify your approved DWS Calendar reservation name and status in `${ZONE}`:

```bash
gcloud compute reservations list --project="${PROJECT_ID}"
gcloud compute reservations describe "${RESERVATION_NAME}" --zone="${ZONE}"
```

### 5.1 Create the GKE Standard Cluster (`1.34+` / `1.36`)

Enable **Dataplane V2**, **Image Streaming**, **GCSFuse CSI Driver**, **Workload Identity**, and **Ray Operator**:

```bash
gcloud container clusters create "${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --node-locations="${ZONE}" \
  --cluster-version=latest \
  --network="${VPC_NAME}" \
  --subnetwork="${SUBNET_NAME}" \
  --enable-ip-alias \
  --enable-dataplane-v2 \
  --enable-image-streaming \
  --workload-pool="${PROJECT_ID}.svc.id.goog" \
  --addons=GcsFuseCsiDriver,RayOperator \
  --machine-type=n2-standard-16 \
  --num-nodes=2 \
  --service-account="${SA_EMAIL}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform

gcloud container clusters get-credentials "${CLUSTER_NAME}" \
  --location="${REGION}" \
  --project="${PROJECT_ID}"
```

### 5.2 Grant Workload Identity Access to the GCS Bucket

Allow Kubernetes Pods running under the `default` Kubernetes ServiceAccount to mount `gs://${GCS_BUCKET}` via the GCSFuse CSI driver:

```bash
gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
  --member="principal://iam.googleapis.com/projects/$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/default/sa/default" \
  --role="roles/storage.objectAdmin"
```

### 5.3 Create the Ironwood (`tpu7x-standard-4t`) Node Pool Bound to Your DWS Calendar Reservation

Per Google Cloud's official [Request TPUs with future reservation in calendar mode](https://cloud.google.com/kubernetes-engine/docs/how-to/tpu-calendar-mode) documentation, choose either a **Single-Host Slice Node Pool** (`2x2x1` = `4` chips per node, independent replicas) or a **Multi-Host Slice Node Pool** (`2x2x2` = `8` chips across 2 hosts, or `2x2x4` = `16` chips across 4 hosts):

#### Option 5.3A — Single-Host TPU Slice Node Pool (`1`, `2`, or `4` Nodes x `4` Chips/Node for Scale-Out Serving)

Because `google/gemma-4-26B-A4B-it` (`52 GB` in BF16, `28.7 GB` in FP8-Dynamic) fits inside a single 4-chip host (`768 GB` HBM3e) with `> 700 GB` of HBM3e for KV cache, a single-host node pool with `1`, `2`, or `4` nodes allows Kubernetes to schedule `1`, `2`, or `4` independent 4-chip Pod replicas (`TP=8, EP` or `TP=2, DP=4, EP`) behind a single Kubernetes `Service`:

```bash
gcloud container node-pools create tpu-v7x-singlehost-pool \
  --project="${PROJECT_ID}" \
  --cluster="${CLUSTER_NAME}" \
  --location="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=tpu7x-standard-4t \
  --num-nodes=4 \
  --reservation-affinity=specific \
  --reservation="${RESERVATION_NAME}" \
  --service-account="${SA_EMAIL}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

#### Option 5.3B — Multi-Host TPU Slice Node Pool (`2x2x2` = `8` Chips or `2x2x4` = `16` Chips with Workload Policy)

On Ironwood (`tpu7x`), creating a multi-host TPU slice node pool requires first creating a regional **Compute Engine Workload Policy (`HIGH_THROUGHPUT`)** and attaching it with `--placement-policy` (which replaces `--tpu-topology` on `tpu7x`):

```bash
# 1. Create the regional Workload Policy (use 2x2x2 for 2 hosts / 8 chips, or 2x2x4 for 4 hosts / 16 chips)
gcloud compute resource-policies create workload-policy tpu-v7x-16c-policy \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --type=HIGH_THROUGHPUT \
  --accelerator-topology=2x2x4

# 2. Create the multi-host TPU7x node pool consuming your DWS Calendar Reservation
gcloud container node-pools create tpu-v7x-multihost-pool \
  --project="${PROJECT_ID}" \
  --cluster="${CLUSTER_NAME}" \
  --location="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=tpu7x-standard-4t \
  --num-nodes=4 \
  --reservation-affinity=specific \
  --reservation="${RESERVATION_NAME}" \
  --placement-policy=tpu-v7x-16c-policy \
  --service-account="${SA_EMAIL}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

Verify that your TPU nodes are `Ready`:

```bash
kubectl get nodes -l cloud.google.com/gke-tpu-accelerator=tpu7x
```

---

## 6. Step 5 — Deploy `Gemma-4-26B-A4B-it` on GKE (`vllm-torchtpu` & `vllm-tpu`)

### 6.1 Deploy `vllm-torchtpu` on GKE (`1`, `2`, or `4` Replicas = `4`, `8`, or `16` Ironwood Chips)

`k8s/gemma-4-26b-a4b-vllm-torchtpu-v7x.yaml` deploys the `vllm-torchtpu` server with the GCSFuse CSI volume mount and exposes a Kubernetes `Service` on port `8000`:

```bash
# Set replicas to 1 (4 chips), 2 (8 chips), or 4 (16 chips)
sed -e "s/<YOUR_GCS_BUCKET>/${GCS_BUCKET}/g" \
    -e "s/<YOUR_PROJECT_ID>/${PROJECT_ID}/g" \
    k8s/gemma-4-26b-a4b-vllm-torchtpu-v7x.yaml | kubectl apply -f -

kubectl scale deployment/gemma4-26b-a4b-torchtpu-v7x --replicas=4
kubectl rollout status deployment/gemma4-26b-a4b-torchtpu-v7x --timeout=30m
```

* **To switch between Pure `bfloat16` and `FP8-Dynamic`**:
  - **Pure `bfloat16` (Exact PoC Scope — BF16 weights + BF16 KV cache)**: Set the model path in `k8s/gemma-4-26b-a4b-vllm-torchtpu-v7x.yaml` to `/model-vol-mount/models/google/gemma-4-26B-A4B-it` and `--kv-cache-dtype=auto` (with `--tensor-parallel-size=8 --data-parallel-size=1 --enable-expert-parallel` for **`511.2 ms` TTFT / `4.95 ms` TPOT**, or `--tensor-parallel-size=2 --data-parallel-size=4 --enable-expert-parallel` for **`4,291.5` out tok/s** on `22k/4k`).
  - **FP8-Dynamic (Strict `< 500 ms` TTFT SLA & Peak Throughput)**: Keep `/model-vol-mount/models/RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` and `--kv-cache-dtype=fp8` (achieves **`446.7 ms` TTFT < `500 ms` SLA** and **`4.50 ms` TPOT < `10 ms` SLA** at `TP=8, EP`, or **`5,327.5` out tok/s** on `22k/4k` at `TP=2, DP=4, EP`).

### 6.2 Alternative: Deploy `vllm-tpu` (JAX Flax NNX) on GKE

```bash
sed -e "s/<YOUR_GCS_BUCKET>/${GCS_BUCKET}/g" \
    k8s/gemma-4-26b-a4b-vllm-tpu-jax-v7x.yaml | kubectl apply -f -
```

---

## 7. Step 6 — Run the PoC Benchmark Suite on GKE (`22,000` Input / `4,000` Output)

Run the benchmark client Pod on the CPU node pool (`n2-standard-16`) targeting the Kubernetes Service (`http://gemma4-26b-a4b-v7x-svc.default.svc.cluster.local:8000`):

```bash
kubectl run vllm-bench-client --rm -it --restart=Never \
  --image=vllm/vllm-openai:latest \
  --overrides='{"spec":{"nodeSelector":{"kubernetes.io/arch":"amd64"}}}' \
  -- bash -c '
    export SVC_HOST="gemma4-26b-a4b-v7x-svc.default.svc.cluster.local"
    export SVC_PORT="8000"

    # 1. Smoke Test (ISL=1024, OSL=21, CON=1, 8)
    for CON in 1 8; do
      vllm bench serve --backend=vllm --dataset-name=random \
        --random-input-len=1024 --random-output-len=21 --num-prompts=32 --ignore-eos \
        --host="${SVC_HOST}" --port="${SVC_PORT}" --endpoint=/v1/completions \
        --model=google/gemma-4-26B-A4B-it \
        --request-rate=inf --max-concurrency="${CON}" \
        --percentile-metrics=ttft,tpot,itl,e2el --metric-percentiles=50,90,95,99 \
        --goodput ttft:500 tpot:10 e2el:60000
    done

    # 2. Long-Context Closed-Loop Concurrency Sweep (ISL=22000, OSL=4000, CON=1, 6, 16, 32, 64)
    for CON in 1 6 16 32 64; do
      vllm bench serve --backend=vllm --dataset-name=random \
        --random-input-len=22000 --random-output-len=4000 --num-prompts=50 --ignore-eos \
        --host="${SVC_HOST}" --port="${SVC_PORT}" --endpoint=/v1/completions \
        --model=google/gemma-4-26B-A4B-it \
        --request-rate=inf --max-concurrency="${CON}" \
        --percentile-metrics=ttft,tpot,itl,e2el --metric-percentiles=50,90,95,99 \
        --goodput ttft:500 tpot:10 e2el:60000
    done

    # 3. Long-Context Open-Loop Poisson Arrival Sweep (ISL=22000, OSL=4000, RPS=1, 2, 4, 6)
    for RPS in 1 2 4 6; do
      vllm bench serve --backend=vllm --dataset-name=random \
        --random-input-len=22000 --random-output-len=4000 --num-prompts=50 --ignore-eos \
        --host="${SVC_HOST}" --port="${SVC_PORT}" --endpoint=/v1/completions \
        --model=google/gemma-4-26B-A4B-it \
        --request-rate="${RPS}" \
        --percentile-metrics=ttft,tpot,itl,e2el --metric-percentiles=50,90,95,99 \
        --goodput ttft:500 tpot:10 e2el:60000
    done
  '
```

---

## 8. Appendix: Optional Single-Host (`1` VM = `4` Chips) GCE Quick Test

If you only want to test a **single host (`1 x tpu7x-standard-4t` = `4` chips)** without setting up a multi-host workload, you can bind a single GCE VM to your reservation using the official `tpu7x` accelerator image family (`ubuntu-accel-2404-amd64-tpu-tpu7x` in `ubuntu-os-accelerator-images`) and run [`gce/launch_gce_tpu_v7x_gemma4_26b_a4b.sh`](gce/launch_gce_tpu_v7x_gemma4_26b_a4b.sh). **For any multi-host (`8`-chip or `16`-chip) or production benchmark, use the GKE workflow in Steps 4–7 above.**
