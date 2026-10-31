#!/bin/bash
# Promote the staged half-time Dixon-Coles artifact into production.
# The configuration was gated by scripts/ab_halftime_dixon_coles.py
# (2026-10-31: HT 1x2 Brier -0.0160 vs the legacy artifact, fold-clustered
# SE 0.0026, 15 folds; every secondary HT market and ht_ft no worse).
# scripts/train_halftime_dixon_coles.py writes the staged model.bin +
# manifest.json; scripts/promote_dc_model.py --model ht (this wrapper):
#   1. checks it: the manifest's sha256 is the staged file's; the staged
#      file is not already production (else "nothing to promote", exit 0);
#      config name, fit on every row, per-league baselines, finite rho, no
#      team/league regression vs production; it serves through the loader;
#   2. backs up the production file + manifest (sha256 printed);
#   3. promotes atomically (copy beside, then rename);
#   4. verifies: production sha256 == staged, and the serve loader
#      (precompute_predictions) loads it and emits a valid HT 1x2.
# precompute_predictions.py reads the file on every run, so the next 15-min
# pipeline tick serves it — no restart. Rollback is printed at the end.
# The retrain_halftime_dc DAG runs the same promoter monthly, after its train.
# Run on the VM host (extra args pass through, e.g. --check-only):
#   bash /opt/auspex/scripts/ops/promote_ht_model.sh 2>&1 | tee -a ~/promote_ht_model.log
set -euo pipefail
cd /opt/auspex
exec docker compose exec -T api python /app/scripts/promote_dc_model.py --model ht "$@"
