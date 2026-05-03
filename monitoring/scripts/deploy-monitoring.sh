#!/usr/bin/env bash
# Deploy the full monitoring stack to Kubernetes.
# Usage: ./monitoring/scripts/deploy-monitoring.sh [--namespace monitoring]
set -euo pipefail

NAMESPACE="${NAMESPACE:-monitoring}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MONITORING_DIR="${ROOT_DIR}/monitoring"
K8S_MONITORING="${ROOT_DIR}/infrastructure/kubernetes/base/monitoring"

echo "==> Deploying monitoring stack to namespace: ${NAMESPACE}"

# 1. Create namespace if it doesn't exist
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

# 2. Apply RBAC (shared with betting-system namespace via ClusterRole)
kubectl apply -f "${K8S_MONITORING}/rbac.yaml"

# 3. Deploy Prometheus
kubectl apply -f "${K8S_MONITORING}/prometheus-deployment.yaml"

# Load alert rules and recording rules as a ConfigMap
kubectl create configmap prometheus-alerts \
  --from-file="${MONITORING_DIR}/prometheus/alerts/" \
  --from-file="${MONITORING_DIR}/prometheus/rules/" \
  -n "${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -

# Load prometheus.yml
kubectl create configmap prometheus-config \
  --from-file=prometheus.yml="${MONITORING_DIR}/prometheus/prometheus.yml" \
  -n "${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -

# Trigger config reload
kubectl rollout restart deployment/prometheus -n "${NAMESPACE}"

# 4. Deploy Alertmanager
kubectl apply -f "${K8S_MONITORING}/alertmanager-deployment.yaml"

kubectl create configmap alertmanager-config \
  --from-file=alertmanager.yml="${MONITORING_DIR}/alertmanager/alertmanager.yml" \
  -n "${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl rollout restart deployment/alertmanager -n "${NAMESPACE}"

# 5. Deploy Grafana
kubectl apply -f "${K8S_MONITORING}/grafana-deployment.yaml"

# Load dashboard JSON files
kubectl create configmap grafana-dashboards \
  --from-file="${MONITORING_DIR}/grafana/dashboards/" \
  -n "${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl rollout restart deployment/grafana -n "${NAMESPACE}"

# 6. Deploy Loki + Promtail
kubectl apply -f "${K8S_MONITORING}/loki-deployment.yaml"

kubectl create configmap loki-config \
  --from-file=loki-config.yml="${MONITORING_DIR}/loki/loki-config.yml" \
  -n "${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl rollout restart deployment/loki -n "${NAMESPACE}"

# 7. Deploy model retraining CronJobs (in betting-system namespace)
kubectl apply -f "${ROOT_DIR}/infrastructure/kubernetes/base/scraper/model-retraining-cronjob.yaml"

echo ""
echo "==> Waiting for rollouts..."
kubectl rollout status deployment/prometheus   -n "${NAMESPACE}" --timeout=120s
kubectl rollout status deployment/grafana      -n "${NAMESPACE}" --timeout=120s
kubectl rollout status deployment/alertmanager -n "${NAMESPACE}" --timeout=120s
kubectl rollout status deployment/loki         -n "${NAMESPACE}" --timeout=120s

echo ""
echo "==> Validating deployment..."
python "${SCRIPT_DIR}/validate-deployment.py"

echo ""
echo "==> Monitoring stack deployed successfully."
echo ""
echo "Access:"
echo "  Grafana       : kubectl port-forward -n ${NAMESPACE} svc/grafana 3000:3000"
echo "  Prometheus    : kubectl port-forward -n ${NAMESPACE} svc/prometheus 9090:9090"
echo "  Alertmanager  : kubectl port-forward -n ${NAMESPACE} svc/alertmanager 9093:9093"
