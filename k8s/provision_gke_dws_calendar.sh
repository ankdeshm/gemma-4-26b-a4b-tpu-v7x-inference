#!/usr/bin/env bash
# Official Google Cloud GKE + DWS Calendar Mode Reservation Provisioner for Ironwood TPU (tpu7x-standard-4t)
# References:
#   - https://cloud.google.com/kubernetes-engine/docs/how-to/tpu-calendar-mode
#   - https://cloud.google.com/kubernetes-engine/docs/concepts/tpu-ironwood
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID=<YOUR_GCP_PROJECT_ID>}"
RESERVATION_NAME="${RESERVATION_NAME:?Set RESERVATION_NAME=<YOUR_DWS_CALENDAR_RESERVATION_NAME>}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-c}"
CLUSTER_NAME="${CLUSTER_NAME:-tpu-v7x-gke-cluster}"
NUM_NODES="${NUM_NODES:-4}"                    # 1 (4 chips), 2 (8 chips), or 4 (16 chips)
TPU_TOPOLOGY="${TPU_TOPOLOGY:-2x2x4}"          # 2x2x1 (1 host), 2x2x2 (2 hosts), or 2x2x4 (4 hosts)
WORKLOAD_POLICY="${WORKLOAD_POLICY:-tpu-v7x-policy-${TPU_TOPOLOGY}}"
GCS_BUCKET="${GCS_BUCKET:-${PROJECT_ID}-tpu-v7x-models}"
VPC_NAME="${VPC_NAME:-tpu-v7x-vpc}"
SUBNET_NAME="${SUBNET_NAME:-tpu-v7x-subnet}"
SA_NAME="${SA_NAME:-tpu-v7x-gke-sa}"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=== [1/6] Configuring project ${PROJECT_ID} and enabling APIs ==="
gcloud config set project "${PROJECT_ID}"
gcloud services enable \
  container.googleapis.com \
  compute.googleapis.com \
  tpu.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com

echo "=== [2/6] Creating GKE Node Service Account ${SA_EMAIL} ==="
gcloud iam service-accounts create "${SA_NAME}" \
  --display-name="GKE Ironwood TPU v7x Service Account" 2>/dev/null || true

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

echo "=== [4/6] Creating Regional GCS Bucket gs://${GCS_BUCKET} (${REGION}) ==="
gcloud storage buckets create "gs://${GCS_BUCKET}" \
  --location="${REGION}" --uniform-bucket-level-access 2>/dev/null || true

echo "=== [5/6] Creating GKE Standard Cluster ${CLUSTER_NAME} (GcsFuseCsiDriver + RayOperator) ==="
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
  --location="${REGION}" --project="${PROJECT_ID}"

echo "=== [6/6] Creating Ironwood (tpu7x-standard-4t) Node Pool Bound to DWS Calendar Reservation ${RESERVATION_NAME} ==="
if [ "${TPU_TOPOLOGY}" != "2x2x1" ]; then
  gcloud compute resource-policies create workload-policy "${WORKLOAD_POLICY}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --type=HIGH_THROUGHPUT \
    --accelerator-topology="${TPU_TOPOLOGY}" 2>/dev/null || true

  gcloud container node-pools create tpu-v7x-dws-pool \
    --project="${PROJECT_ID}" \
    --cluster="${CLUSTER_NAME}" \
    --location="${REGION}" \
    --node-locations="${ZONE}" \
    --machine-type=tpu7x-standard-4t \
    --num-nodes="${NUM_NODES}" \
    --reservation-affinity=specific \
    --reservation="${RESERVATION_NAME}" \
    --placement-policy="${WORKLOAD_POLICY}" \
    --service-account="${SA_EMAIL}" \
    --scopes=https://www.googleapis.com/auth/cloud-platform
else
  gcloud container node-pools create tpu-v7x-dws-pool \
    --project="${PROJECT_ID}" \
    --cluster="${CLUSTER_NAME}" \
    --location="${REGION}" \
    --node-locations="${ZONE}" \
    --machine-type=tpu7x-standard-4t \
    --num-nodes="${NUM_NODES}" \
    --reservation-affinity=specific \
    --reservation="${RESERVATION_NAME}" \
    --service-account="${SA_EMAIL}" \
    --scopes=https://www.googleapis.com/auth/cloud-platform
fi

kubectl get nodes -l cloud.google.com/gke-tpu-accelerator=tpu7x
