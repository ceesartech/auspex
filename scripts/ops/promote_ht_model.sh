#!/bin/bash
# Promote the staged half-time Dixon-Coles artifact into production.
# The configuration was gated by scripts/ab_halftime_dixon_coles.py
# (2026-10-31: HT 1x2 Brier -0.0160 vs the legacy artifact, fold-clustered
# SE 0.0026, 15 folds; every secondary HT market and ht_ft no worse).
# scripts/train_halftime_dixon_coles.py writes the staged file; this:
#   1. checks it: converged fit written by the gated trainer (per-league
#      baselines present, more teams than the legacy 175, finite rho);
#   2. backs up the production file (sha256 of both);
#   3. promotes atomically (copy beside, then rename);
#   4. verifies: production sha256 == staged, and the serve loader
#      (precompute_predictions) loads it and emits valid HT markets.
# precompute_predictions.py reads the file on every run, so the next 15-min
# pipeline tick serves it — no restart. Rollback is printed at the end.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/promote_ht_model.sh 2>&1 | tee -a ~/promote_ht_model.log
set -euo pipefail
cd /opt/auspex
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
STAGE=/app/models/staging-ht/dixon_coles_ht_soccer/1.0.0/model.bin
PROD=/app/models/production/dixon_coles_ht_soccer/1.0.0/model.bin
BACKUP=/app/models/backup-dixon_coles_ht_soccer-$(date -u +%Y%m%dT%H%M%SZ)

log "1/4 check the staged artifact"
$DC exec -T api python - "$STAGE" <<'PY'
import json, math, sys
d = json.load(open(sys.argv[1]))
cfg = (d.get("config") or {}).get("name")
checks = {
    "config name dixon_coles_ht_soccer": cfg == "dixon_coles_ht_soccer",
    "per-league baselines fitted": bool(d.get("league_baselines")) and d.get("global_home_baseline") is not None,
    "more teams than the legacy 175": len(d.get("team_attack") or {}) > 175,
    "finite rho": isinstance(d.get("rho"), (int, float)) and math.isfinite(d["rho"]),
}
for k, ok in checks.items():
    print(("  ok    " if ok else "  FAIL  ") + k)
print(f"  teams={len(d['team_attack'])} leagues={len(d['league_baselines'])} rho={d['rho']:.4f} max_goals={d['max_goals']}")
sys.exit(0 if all(checks.values()) else 1)
PY

log "2/4 back up production -> $BACKUP"
$DC exec -T api sh -c "mkdir -p $BACKUP && cp -a $PROD $BACKUP/model.bin && sha256sum $PROD $BACKUP/model.bin $STAGE"

log "3/4 promote"
$DC exec -T api sh -c "cp $STAGE $PROD.new && mv -f $PROD.new $PROD"

log "4/4 verify"
$DC exec -T api sh -c "cd /app/services/ml-models && python - $STAGE $PROD" <<'PY'
import hashlib, sys
sys.path.insert(0, "/app/services/ml-models/src")
sys.path.insert(0, "/app/scripts")
stage, prod = sys.argv[1], sys.argv[2]
digest = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
assert digest(stage) == digest(prod), "production file does not match the staged one"
import precompute_predictions as pp
from predictors.market_derivation import MAX_GOALS_DERIVE, build_dc_matrix, derive_markets
model = pp._load_halftime_dixon_coles()
assert model is not None, "serve loader returned no model"
league = next(iter(model.league_baselines))
team_a, team_b = list(model.team_attack)[:2]
h, a = model.lambdas_for_match(team_a, team_b, league=league)
m = derive_markets("soccer_halftime", build_dc_matrix(h, a, model.rho, max_goals=MAX_GOALS_DERIVE))
p = m["match_result_ht"]
assert abs(sum(p.values()) - 1) < 1e-6, p
print(f"  serve loader ok: {team_a} v {team_b} HT 1x2 home={p['home']:.3f} draw={p['draw']:.3f} away={p['away']:.3f}")
PY
log "DONE — rollback: docker compose exec -T api cp $BACKUP/model.bin $PROD"
