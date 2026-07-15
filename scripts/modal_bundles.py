"""Single source of truth for the trainable bundles + their ensemble names.

Shared by the Modal training app (modal_train/train_modal.py), the VM-side
artifact pull + promote-gate (scripts/pull_modal_artifacts.py), and the
incumbent-metric store (scripts/model_metrics_store.py) so function naming,
gating, and artifact routing all agree.

Mirrors the keys of SPORT_BUNDLES + each bundle's ensemble_name in
services/ml-models/src/training/train_all_models.py. Horse racing is NOT here —
it trains via its own precompute path, not train_all_models.
"""

# bundle key (train_all_models --sport value) → the ensemble artifact name
# (the dir under the models tree the serve path loads + the gate keys on).
BUNDLE_TO_ENSEMBLE = {
    "soccer_match_result": "ensemble_soccer_match_result",
    "nhl_moneyline": "ensemble_nhl_ml",
    "nhl_regulation": "ensemble_nhl_reg",
    "nhl_puck_line": "ensemble_nhl_pl",
    "nhl_total": "ensemble_nhl_tot",
    "nba_moneyline": "ensemble_nba_ml",
    "nba_spread": "ensemble_nba_sp",
    "nba_total": "ensemble_nba_tot",
    "nfl_moneyline": "ensemble_nfl_ml",
    "nfl_spread": "ensemble_nfl_sp",
    "nfl_total": "ensemble_nfl_tot",
    "tennis_moneyline": "ensemble_tennis_ml",
    "mma_moneyline": "ensemble_mma_ml",
}

BUNDLES = list(BUNDLE_TO_ENSEMBLE.keys())

# B2 object-key scheme, under the shared BACKUP_S3_BUCKET (auspex-backups):
#   postgres/<db>-<ISO>.dump      ← the nightly db_backup_daily dump (data IN)
#   modal-train/<run_id>/<bundle> ← a Modal run's per-bundle artifact tree (OUT)
DUMP_PREFIX = "postgres/"
ARTIFACT_PREFIX = "modal-train/"

# Promote-gate tolerance: admit a challenger iff its served held-out Brier is
# within a tolerance of the incumbent's (lower Brier = better). NOISE_FLOOR is
# the audit's ~0.009 Brier SE at soccer's held-out size (N_REF); gate_tolerance()
# scales it to each bundle's own test size.
NOISE_FLOOR = 0.009
N_REF = 3522  # soccer's held-out test n, where NOISE_FLOOR is calibrated.


def gate_tolerance(n) -> float:
    """Brier tolerance scaled to the held-out test size. Brier SE shrinks
    ~1/sqrt(n), so a fixed floor tuned for soccer (n≈3522) is far too strict for
    small bundles — NFL (n≈128) varies ~0.02–0.05 run-to-run from the unseeded
    NN alone, so a fixed 0.009 would spuriously reject equivalent retrains. Scale
    by sqrt(N_REF/n), capped at 0.10 so a clearly-worse model is still caught,
    and fall back to NOISE_FLOOR when n is unknown."""
    import math

    if not n or n <= 0:
        return NOISE_FLOOR
    return min(0.10, NOISE_FLOOR * math.sqrt(N_REF / n))


def served_brier(gate: dict):
    """The Brier the model will actually SERVE at, from a bundle's calibration
    gate decision dict ({kept, reason, n, raw:{brier}, calibrated:{brier}}):
    calibrated if the gate kept the calibrator, raw otherwise. Returns
    (brier, kept, n) with brier=None when there was no held-out test to gate on.
    Kept identical here + in the trial so both compute the same number."""
    if not isinstance(gate, dict):
        return None, None, None
    kept = gate.get("kept")
    raw = (gate.get("raw") or {}).get("brier")
    cal = (gate.get("calibrated") or {}).get("brier")
    brier = cal if (kept and cal is not None) else raw
    return brier, kept, gate.get("n")
