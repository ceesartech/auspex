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
docker compose exec -T api python /app/scripts/monitor_models.py --days 30
```

### 2. Check drift detection results
```bash
curl -s http://127.0.0.1:8000/api/v1/models/drift-status | python -m json.tool
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
1. Check the Telegram alerts channel for the drift context (sport + market).
2. Check scraper health — stale or corrupt data is a common drift trigger.
3. Inspect feature distributions in Grafana **Scraping Status** dashboard.

### Short-term – trigger retraining
```bash
# Retraining is Airflow-driven: trigger the retrain_models DAG.
# Its self-gate only promotes models where ΔBrier ≤ -0.005 vs the incumbent,
# so a no-op trigger is safe.
docker compose exec -T airflow-scheduler airflow dags trigger retrain_models
```

Watch job progress in the Airflow UI (`airflow.$AUSPEX_DOMAIN` → `retrain_models`)
or:
```bash
docker compose logs -f airflow-scheduler
```

### Rollback (if a promoted model is worse)
Models are versioned artifacts, not container images — the retrain gate keeps
the prior model unless the new one beats it. To force-revert, restore the
previous model files from a backup (see `OPERATIONS.md`) and restart the API:
```bash
docker compose restart api
```

### Long-term
1. Root-cause the drift — seasonal effects, league changes, data-source changes.
2. Update feature engineering if data schema changed.
3. A/B test new model version before full rollout.
4. Lower the retraining cadence (e.g., weekly instead of monthly).

---

## Prevention

- Automated weekly retraining via the Airflow `retrain_models` DAG.
- Continuous drift monitoring via `scripts/monitor_models.py` (hourly `monitor_models` DAG), including the constant-prior canary on ungraded predictions.
- Alert fires on rolling ECE/Brier degradation per sport+market.

## Related Runbooks

- `HIGH_ERROR_RATE.md`
- `DATABASE_ISSUES.md`
