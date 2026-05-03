#!/bin/bash
set -e

# Rollback deployment script
NAMESPACE="betting-system"
DEPLOYMENT="${1:-betting-api}"

echo "Rolling back deployment: ${DEPLOYMENT}"

kubectl rollout undo deployment/${DEPLOYMENT} -n ${NAMESPACE}

echo "Waiting for rollback to complete..."
kubectl rollout status deployment/${DEPLOYMENT} -n ${NAMESPACE} --timeout=5m

echo "Rollback completed successfully"
