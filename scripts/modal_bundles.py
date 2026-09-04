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


# ── Frame fingerprint: are two Brier numbers even comparable? ─────────────────
# train_all_models splits 70/15/15 BY ROW RATIO, so a bundle's held-out test set
# is simply the tail of whatever frame THAT run loaded. When the frame changes
# shape (the 2026-08-06 soccer corpus grew ~7x when the all-league loader
# landed), the incumbent's stored Brier and the challenger's are computed on
# different populations and comparing them is meaningless — it rejected the
# better soccer model four weeks running. So every promoted metric carries a
# fingerprint of the frame it was scored on, and the gate only takes the scalar
# path when the two fingerprints are comparable.
# How far the frame may grow between two runs and still leave their held-out
# tails comparable. The tail is the LAST 15% of the frame, so a frame that
# grows by 1/0.85 = 1.176x has a tail that starts after the previous frame
# ENDED: the two test sets are then completely DISJOINT, and comparing their
# Brier scalars is exactly the defect this fingerprint exists to catch. 1.10
# keeps ~39% of the old tail inside the new one; anything looser is a promise
# the number cannot keep.
FINGERPRINT_ROW_BAND = 1.10
HISTORY_LIMIT = 8  # K: runs of served-Brier history kept per ensemble (~2 months weekly)
MIN_PAIRED_N = 100  # fewest shared rows a paired re-scoring may decide on
PAIRED_Z = 1.96  # two-sided 95% on the paired-delta SE


def _norm_date(value):
    """ISO date/datetime (or None) → 'YYYY-MM-DD'. The fingerprint compares the
    corpus START date as a scope key, so sub-day precision is noise."""
    if value is None:
        return None
    text = str(value)
    return text[:10] if len(text) >= 10 else text


def fingerprint_from_report(report) -> dict | None:
    """Frame fingerprint from a bundle's training_report.json, using the fields
    train_all_models already writes: data_quality {rows, date_min, date_max} and
    holdout_test {n}. Returns None when the report carries no row count (nothing
    to compare on) — the caller must then treat comparability as UNKNOWN, never
    as 'matches'."""
    if not isinstance(report, dict):
        return None
    dq = report.get("data_quality") or {}
    holdout = report.get("holdout_test") or {}
    rows = dq.get("rows")
    if rows is None:
        return None
    try:
        rows = int(rows)
    except (TypeError, ValueError):
        return None
    n = holdout.get("n")
    return {
        "rows": rows,
        "date_min": _norm_date(dq.get("date_min")),
        "date_max": _norm_date(dq.get("date_max")),
        "holdout_n": int(n) if isinstance(n, (int, float)) else None,
        # Population keys: a frame with a different feature set or a different
        # number of target classes is a different modelling problem, and its
        # Brier is not on the same scale — comparing the scalars would be the
        # same class of error as comparing across disjoint test sets. These
        # are the scope fields train_all_models already writes; a fuller key
        # (the loader's league/competition set) would need a change to
        # services/ml-models/src/utils/training_data.py's data_quality.
        "feature_count": dq.get("feature_count"),
        "target_classes": dq.get("target_classes"),
    }


def _within_band(a, b) -> bool:
    """True when a and b are within FINGERPRINT_ROW_BAND of each other."""
    if not a or not b or a <= 0 or b <= 0:
        return False
    ratio = float(b) / float(a)
    return (1.0 / FINGERPRINT_ROW_BAND) <= ratio <= FINGERPRINT_ROW_BAND


def fingerprints_comparable(a, b) -> bool:
    """Can two served-Brier numbers be compared as scalars?

    Only when they were scored on frames of the same SHAPE: the same corpus
    start date (a moved date_min means the loader's scope changed), the same
    feature count and target-class count, and row + held-out counts within
    FINGERPRINT_ROW_BAND. A missing fingerprint (a legacy sidecar written
    before this gate existed) is NOT comparable — unknown must fail closed into
    the paired path, not silently reuse the broken comparison. So is a missing
    held-out count on either side: "how many rows was this Brier averaged
    over" is half of what makes two Briers comparable, and ignoring it when it
    is absent is guessing."""
    if not isinstance(a, dict) or not isinstance(b, dict) or not a or not b:
        return False
    if a.get("date_min") != b.get("date_min"):
        return False
    for key in ("feature_count", "target_classes"):
        if a.get(key) != b.get(key):
            return False
    if not _within_band(a.get("rows"), b.get("rows")):
        return False
    if not _within_band(a.get("holdout_n"), b.get("holdout_n")):
        return False
    return True


def paired_decision(delta: float, se: float, n) -> tuple[bool, str]:
    """Promote verdict from a PAIRED re-scoring of challenger vs incumbent on one
    shared set of rows. `delta` is mean(challenger row Brier - incumbent row
    Brier) — negative is better — and `se` its paired standard error.

    Promote iff the challenger is neither statistically worse (delta within
    PAIRED_Z * SE of zero) NOR worse than the n-scaled tolerance the scalar path
    uses. The SE term is what makes this honest at large n (a real 0.002
    regression is caught); the tolerance term is what stops a tiny-n bundle,
    where PAIRED_Z * SE can be 0.05, from waving a clearly worse model through."""
    ci = PAIRED_Z * float(se)
    tol = gate_tolerance(n)
    margin = min(ci, tol)
    ok = float(delta) <= margin
    verb = "<=" if ok else ">"
    return ok, (
        f"paired ΔBrier {delta:+.5f} ± {se:.5f} (n={n}) {verb} margin {margin:.5f} "
        f"= min(1.96·SE {ci:.5f}, tol {tol:.5f})"
    )
