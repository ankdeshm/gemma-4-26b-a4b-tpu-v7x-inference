#!/usr/bin/env bash
# Automated GCP Infrastructure & DWS Calendar Reservation Provisioner for Ironwood TPU (tpu7x-standard-4t)
# Provisions: APIs, Service Account, Custom VPC/Subnet, Cloud NAT, Firewall Rules, Regional GCS Bucket,
# c2-standard-16 Benchmark Client VM, and 1, 2, or 4 DWS-Reserved tpu7x-standard-4t GCE VMs (no GKE required).
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID=<YOUR_GCP_PROJECT_ID>}"
RESERVATION_NAME="${RESERVATION_NAME:?Set RESERVATION_NAME=<YOUR_DWS_RESERVATION_NAME>}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-c}"
NUM_HOSTS="${NUM_HOSTS:-4}" # 1 (4 chips), 2 (8 chips), or 4 (16 chips)
GCS_BUCKET="${GCS_BUCKET:-${PROJECT_ID}-tpu-v7x-models}"
VPC_NAME="${VPC_NAME:-tpu-v7x-vpc}"
SUBNET_NAME="${SUBNET_NAME:-tpu-v7x-subnet}"
SA_NAME="${SA_NAME:-tpu-v7x-bench-sa}"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=== [1/6] Configuring project ${PROJECT_ID} and enabling APIs ==="
gcloud config set project "${PROJECT_ID}"
gcloud services enable \
  compute.googleapis.com \
  tpu.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  iap.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com

echo "=== [2/6] Creating Service Account ${SA_EMAIL} and binding IAM roles ==="
gcloud iam service-accounts create "${SA_NAME}" \
  --display-name="TPU v7x Benchmark Service Account" 2>/dev/null || true

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
    --quiet >/dev/null
done

echo "=== [3/6] Creating VPC ${VPC_NAME}, Subnet ${SUBNET_NAME}, Cloud NAT & Firewall Rules ==="
gcloud compute networks create "${VPC_NAME}" \
  --subnet-mode=custom --mtu=8896 --bgp-routing-mode=regional 2>/dev/null || true

gcloud compute networks subnets create "${SUBNET_NAME}" \
  --network="${VPC_NAME}" --region="${REGION}" --range="10.128.0.0/20" \
  --enable-private-ip-google-access 2>/dev/null || true

gcloud compute routers create "${VPC_NAME}-router" \
  --network="${VPC_NAME}" --region="${REGION}" 2>/dev/null || true

gcloud compute routers nats create "${VPC_NAME}-nat" \
  --router="${VPC_NAME}-router" --region="${REGION}" \
  --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges 2>/dev/null || true

gcloud compute firewall-rules create "${VPC_NAME}-allow-internal" \
  --network="${VPC_NAME}" --allow=tcp:1-65535,udp:1-65535,icmp \
  --source-ranges="10.128.0.0/20" 2>/dev/null || true

gcloud compute firewall-rules create "${VPC_NAME}-allow-ssh-iap" \
  --network="${VPC_NAME}" --allow=tcp:22 \
  --source-ranges="35.235.240.0/20" 2>/dev/null || true

echo "=== [4/6] Creating Regional GCS Bucket gs://${GCS_BUCKET} (${REGION}) ==="
gcloud storage buckets create "gs://${GCS_BUCKET}" \
  --location="${REGION}" --uniform-bucket-level-access 2>/dev/null || true

echo "=== [5/6] Creating CPU Benchmark Client VM (tpu-bench-client, c2-standard-16) ==="
gcloud compute instances create tpu-bench-client \
  --zone="${ZONE}" \
  --machine-type=c2-standard-16 \
  --network-interface="network=${VPC_NAME},subnetwork=${SUBNET_NAME},nic-type=GVNIC,no-address" \
  --boot-disk-size=250GB \
  --boot-disk-type=pd-ssd \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --service-account="${SA_EMAIL}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform 2>/dev/null || true

echo "=== [6/6] Provisioning ${NUM_HOSTS} x tpu7x-standard-4t VMs from DWS Reservation ${RESERVATION_NAME} ==="
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
    --scopes=https://www.googleapis.com/auth/cloud-platform
done

echo "=== Provisioning Complete! Active VMs: ==="
gcloud compute instances list --filter="name~'^(tpu-bench-client|tpu-v7x-node-)'"
