#!/bin/bash
set -e

# Comprehensive health check script
NAMESPACE="betting-system"
API_URL="${API_URL:-https://api.betting-system.com}"

echo "=== Betting System Health Check ==="
echo "Timestamp: $(date)"
echo ""

# Check API health
echo "--- API Health ---"
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}/health" 2>/dev/null || echo "000")
if [ "$HTTP_STATUS" = "200" ]; then
  echo "API: HEALTHY (HTTP ${HTTP_STATUS})"
else
  echo "API: UNHEALTHY (HTTP ${HTTP_STATUS})"
fi
echo ""

# Check pods status
echo "--- Pod Status ---"
kubectl get pods -n ${NAMESPACE} -o wide 2>/dev/null || echo "Cannot reach Kubernetes cluster"
echo ""

# Check for crashlooping pods
echo "--- CrashLoopBackOff Check ---"
CRASH_PODS=$(kubectl get pods -n ${NAMESPACE} --field-selector=status.phase!=Running,status.phase!=Succeeded -o name 2>/dev/null || echo "")
if [ -z "$CRASH_PODS" ]; then
  echo "No unhealthy pods detected"
else
  echo "Unhealthy pods found:"
  echo "$CRASH_PODS"
fi
echo ""

# Check services
echo "--- Services ---"
kubectl get services -n ${NAMESPACE} 2>/dev/null || echo "Cannot reach Kubernetes cluster"
echo ""

# Check ingress
echo "--- Ingress ---"
kubectl get ingress -n ${NAMESPACE} 2>/dev/null || echo "No ingress found"
echo ""

# Check recent events (errors only)
echo "--- Recent Warning Events ---"
kubectl get events -n ${NAMESPACE} --field-selector type=Warning --sort-by='.lastTimestamp' 2>/dev/null | tail -10 || echo "No warning events"
echo ""

# Check resource usage
echo "--- Resource Usage ---"
kubectl top pods -n ${NAMESPACE} 2>/dev/null || echo "Metrics server not available"
echo ""

echo "=== Health Check Complete ==="
