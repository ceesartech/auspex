# Deployment Guide

## Prerequisites

- Google Cloud SDK (`gcloud`) installed and authenticated
- `kubectl` installed
- `terraform` >= 1.0 installed
- `kustomize` installed
- Docker installed
- Access to the GCP project

## Initial Setup

### 1. Configure GCP

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable container.googleapis.com sqladmin.googleapis.com redis.googleapis.com
```

### 2. Create Terraform State Bucket

```bash
gsutil mb gs://betting-system-terraform-state
gsutil versioning set on gs://betting-system-terraform-state
```

### 3. Deploy Infrastructure

```bash
# Dev environment
cd infrastructure/terraform/environments/dev
terraform init
terraform plan -var="db_password=YOUR_PASSWORD"
terraform apply -var="db_password=YOUR_PASSWORD"

# Production environment
cd infrastructure/terraform/environments/prod
terraform init
terraform plan -var="db_password=YOUR_PASSWORD"
terraform apply -var="db_password=YOUR_PASSWORD"
```

### 4. Get Cluster Credentials

```bash
gcloud container clusters get-credentials betting-system-cluster \
  --region us-central1 \
  --project YOUR_PROJECT_ID
```

### 5. Install Prerequisites on Cluster

```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.3/cert-manager.yaml
```

### 6. Deploy Application

Using Kustomize:
```bash
# Dev
kustomize build infrastructure/kubernetes/overlays/dev | kubectl apply -f -

# Staging
kustomize build infrastructure/kubernetes/overlays/staging | kubectl apply -f -

# Production
kustomize build infrastructure/kubernetes/overlays/prod | kubectl apply -f -
```

Using Helm (alternative):
```bash
helm install betting-system infrastructure/helm/betting-system/ \
  -n betting-system \
  --create-namespace \
  -f infrastructure/helm/betting-system/values.yaml
```

### 7. Verify Deployment

```bash
kubectl get all -n betting-system
kubectl get ingress -n betting-system
curl https://api.betting-system.com/health
```

## CI/CD Setup

### GitHub Secrets Required

| Secret | Description |
|--------|-------------|
| `GCP_PROJECT_ID` | GCP project ID |
| `GCP_SA_KEY` | Service account JSON key |
| `SLACK_WEBHOOK` | Slack webhook URL (optional) |
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection string |
| `GCS_MODEL_BUCKET` | GCS bucket for ML models |
| `API_ADMIN_TOKEN` | Admin JWT token for API |

### Workflow Triggers

- **CI**: Runs on every push/PR to `main` and `develop`
- **Deploy Dev**: Auto-deploys on push to `develop`
- **Deploy Staging**: Triggered by release candidate tags (`v*-rc*`)
- **Deploy Prod**: Auto-deploys on push to `main` (requires environment approval)
- **Model Retrain**: Weekly on Sundays at 2 AM Denver time

## Backup Setup

```bash
# Manual backup
./infrastructure/scripts/backup-db.sh

# Set up automated daily backups via CronJob
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
  namespace: betting-system
spec:
  schedule: "0 3 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: google/cloud-sdk:alpine
            command: ["/scripts/backup-db.sh"]
          restartPolicy: OnFailure
EOF
```

## Cost Optimization

| Resource | Dev | Staging | Prod |
|----------|-----|---------|------|
| GKE Nodes | 1x e2-medium | 1-3x e2-standard-2 | 1-5x e2-medium + ML pool |
| Cloud SQL | db-f1-micro | db-custom-2-4096 | db-custom-2-4096 (HA) |
| Redis | 1GB BASIC | 2GB STANDARD_HA | 2GB STANDARD_HA |
| Preemptible | Yes | Yes | Yes (80% savings) |
| **Est. Monthly** | **~$30** | **~$75** | **~$115** |
