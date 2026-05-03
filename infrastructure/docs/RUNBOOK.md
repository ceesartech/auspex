# Operational Runbook

## Incident Response

### API Down / 5xx Errors

1. Check pod status:
   ```bash
   kubectl get pods -n betting-system -l app=betting-api
   kubectl describe pod <pod-name> -n betting-system
   ```

2. Check logs:
   ```bash
   kubectl logs -f deployment/betting-api -n betting-system --tail=100
   ```

3. Check if it's a resource issue:
   ```bash
   kubectl top pods -n betting-system
   ```

4. If OOMKilled, increase memory limits or scale horizontally:
   ```bash
   kubectl scale deployment betting-api --replicas=5 -n betting-system
   ```

5. If persistent, rollback:
   ```bash
   ./infrastructure/scripts/rollback.sh betting-api
   ```

### Database Connection Issues

1. Check PostgreSQL pod:
   ```bash
   kubectl get pods -n betting-system -l app=postgres
   kubectl logs statefulset/postgres -n betting-system
   ```

2. Verify connectivity:
   ```bash
   kubectl exec -it deployment/betting-api -n betting-system -- \
     pg_isready -h betting-system-db -p 5432 -U betting_user
   ```

3. Check connection pool exhaustion:
   ```bash
   kubectl exec -it statefulset/postgres -n betting-system -- \
     psql -U betting_user -d betting_system -c "SELECT count(*) FROM pg_stat_activity;"
   ```

### Redis Issues

1. Check Redis pod:
   ```bash
   kubectl get pods -n betting-system -l app=redis
   kubectl logs deployment/redis -n betting-system
   ```

2. Check memory usage:
   ```bash
   kubectl exec -it deployment/redis -n betting-system -- redis-cli INFO memory
   ```

### High Latency

1. Check HPA status:
   ```bash
   kubectl get hpa -n betting-system
   ```

2. Check node resources:
   ```bash
   kubectl top nodes
   ```

3. Scale up if needed:
   ```bash
   kubectl scale deployment betting-api --replicas=5 -n betting-system
   ```

## Scaling

### Manual Scaling

```bash
# Scale API
kubectl scale deployment betting-api --replicas=5 -n betting-system

# Scale frontend
kubectl scale deployment betting-frontend --replicas=3 -n betting-system
```

### HPA Configuration

```bash
# View current HPA
kubectl get hpa -n betting-system

# Edit HPA
kubectl edit hpa betting-api-hpa -n betting-system
```

## Backup and Restore

### Create Manual Backup

```bash
export GCP_PROJECT_ID=your-project-id
./infrastructure/scripts/backup-db.sh
```

### List Available Backups

```bash
gsutil ls -l gs://${GCP_PROJECT_ID}-backups/database/
```

### Restore from Backup

```bash
./infrastructure/scripts/restore-db.sh gs://PROJECT-backups/database/betting-system-backup-20240101_030000.sql
```

## Certificate Management

### Check Certificate Status

```bash
kubectl get certificates -n betting-system
kubectl describe certificate betting-system-tls -n betting-system
```

### Force Certificate Renewal

```bash
kubectl delete secret betting-system-tls -n betting-system
# cert-manager will automatically request a new certificate
```

## Model Retraining

### Manual Trigger

```bash
# Via GitHub Actions
gh workflow run model-retrain.yml -f model_type=all

# Via Kubernetes Job
kubectl apply -f infrastructure/kubernetes/base/scraper/training-job.yaml
kubectl logs -f job/model-training -n betting-system
```

### Check Model Status

```bash
gsutil ls -l gs://${GCP_PROJECT_ID}-ml-models/models/
```

## Disaster Recovery

### Full Cluster Recovery

1. Recreate infrastructure:
   ```bash
   cd infrastructure/terraform/environments/prod
   terraform apply
   ```

2. Get credentials:
   ```bash
   gcloud container clusters get-credentials betting-system-cluster --region us-central1
   ```

3. Deploy application:
   ```bash
   kustomize build infrastructure/kubernetes/overlays/prod | kubectl apply -f -
   ```

4. Restore database:
   ```bash
   ./infrastructure/scripts/restore-db.sh gs://PROJECT-backups/database/LATEST_BACKUP.sql
   ```

5. Verify:
   ```bash
   ./infrastructure/scripts/health-check.sh
   ```
