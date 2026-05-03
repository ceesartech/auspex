# Runbook: Model Drift Detected

**Alert Names:** `ModelAccuracyDrop`, `DataDriftDetected`
**Severity:** Critical
**Component:** ml-models

## Description

Model performance has degraded significantly or feature distribution drift has been
detected, indicating the model may no longer generalise to current data.

## Impact

- Predictions less accurate
- Negative ROI possible
- User trust degraded

---

## Diagnosis

### 1. Check current performance
```bash
kubectl exec -it -n betting-system deployment/betting-api -- \
  python monitoring/scripts/check-model-performance.py
```

### 2. Check drift detection results
```bash
curl http://betting-api:8000/api/v1/models/drift-status | python -m json.tool
```

### 3. Grafana
- Dashboard: **ML Model Performance**
- Check panels: Accuracy Trend, ROI Trend, Data Drift Score, Calibration Error

### 4. Review recent predictions (Postgres)
```sql
SELECT predicted_outcome, actual_outcome, confidence,
       (predicted_outcome = actual_outcome) AS correct
FROM predictions p
JOIN matches m ON p.match_id = m.match_id
WHERE m.status = 'finished'
  AND m.match_date >= NOW() - INTERVAL '7 days'
ORDER BY m.match_date DESC
LIMIT 100;
```

---

## Resolution

### Immediate
1. Alert ML team via Slack `#ml-alerts`.
2. Check scraper health — stale or corrupt data is a common drift trigger.
3. Inspect feature distributions in Grafana **Scraping Status** dashboard.

### Short-term – trigger retraining
```bash
# Auto mode (only retrains if thresholds breached)
python monitoring/scripts/trigger-retraining.py --auto

# Manual override
python monitoring/scripts/trigger-retraining.py --reason "Manual: accuracy drop detected"

# Or directly via kubectl
kubectl create job model-retraining-manual \
  --from=cronjob/model-retraining \
  -n betting-system
```

Watch job progress:
```bash
kubectl logs -f -n betting-system job/model-retraining-manual
```

### Rollback (if new model is worse)
```bash
kubectl rollout undo deployment/betting-api -n betting-system
kubectl rollout status deployment/betting-api -n betting-system
```

### Long-term
1. Root-cause the drift — seasonal effects, league changes, data-source changes.
2. Update feature engineering if data schema changed.
3. A/B test new model version before full rollout.
4. Lower the retraining cadence (e.g., weekly instead of monthly).

---

## Prevention

- Automated weekly retraining (`model-retraining` CronJob).
- Continuous drift monitoring via `DriftDetector`.
- Alert fires when `drift_score > 0.3`.

## Related Runbooks

- `HIGH_ERROR_RATE.md`
- `DATABASE_ISSUES.md`
