"""Train a halftime Dixon-Coles model for soccer.

The production FT Dixon-Coles model (dixon_coles_soccer_match_result)
is trained on the full-time scoreline. matches.half_time_home /
matches.half_time_away cover 23,452 of 23,456 finished soccer
matches (99.98%), giving us enough data to train a SEPARATE
Dixon-Coles parameterised on halftime goals. The HT model produces
a halftime scoreline matrix that the market-derivation engine
converts into 1x2_ht, over_under_ht, and btts_ht predictions.

Architecture:
  * Pull a soccer training frame with half_time_home/away renamed
    to home_score/away_score so the existing DixonColesPredictor
    code path works unchanged.
  * Fit on ALL of it with DIXON_COLES_CONFIG — the configuration
    scripts/ab_halftime_dixon_coles.py gated on 2026-10-31 (see the
    comment at the fit) — refuse the artifact when the fit is not sound
    (fit_refusal), and write model.bin + manifest.json to STAGING,
    atomically.
  * scripts/promote_dc_model.py --model ht (scripts/ops/promote_ht_model.sh
    on the host) checks the staged manifest, backs up the production
    artifact and promotes; `scripts/precompute_predictions.py` loads
    /app/models/production/dixon_coles_ht_soccer/1.0.0/model.bin on its
    next run and writes one row per HT market_type per match.
  * The retrain_halftime_dc DAG runs both, monthly. --output-dir refuses
    /app/models/staging (the weekly retrain_models DAG's swap_production
    promotes everything in it, with none of these checks) and
    /app/models/production (only the promoter writes there, after a backup)
    — and anything that resolves under either, symlinks and `..` included.

Until 2026-10-31 this script fitted the OLDEST 80% only (the newest 20% was
a validation slice, so no recent match ever reached the fit), used the
legacy pinned config (a configured 0.25 home bump on a ~0.6-goal base: the
served artifact picked home on 89% of matches and under-weighted the HT
draw), and wrote straight over the production file with no gate.

Run on prod via:
    docker compose exec api python /app/scripts/train_halftime_dixon_coles.py
    bash /opt/auspex/scripts/ops/promote_ht_model.sh      # after reading the log
"""

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

LOGGER = logging.getLogger("train_halftime_dixon_coles")

MODEL_NAME = "dixon_coles_ht_soccer"
DEFAULT_OUTPUT_DIR = Path("/app/models/staging-ht/dixon_coles_ht_soccer/1.0.0")
MODEL_FILE = "model.bin"
MANIFEST_FILE = "manifest.json"

# Trees --output-dir must never resolve into, and why (output_dir_refusal).
FORBIDDEN_OUTPUT_ROOTS: Tuple[Tuple[Path, str], ...] = (
    (
        Path("/app/models/staging"),
        "retrain_models' swap_production promotes everything under it every Sunday, with none of the "
        "half-time checks",
    ),
    (
        Path("/app/models/production"),
        "production is written only by scripts/promote_dc_model.py, after its checks and a backup",
    ),
)

# Post-fit refusal bounds (fit_refusal). The level canary: at the MLE fixed
# point the weighted fitted mean total goals equals the observed one (the
# served 2026-10-31 fit sits at -0.1% on today's frame); outside +/-2% the fit
# is over-constrained or not at its optimum, and every derived totals market
# inherits the bias. rho: the gated fit's is -0.002 — a half-time scoreline
# has little low-score dependence left to correct, so a |rho| past 0.1 is the
# optimiser running off, not a finding.
MAX_LEVEL_BIAS = 0.02
MAX_ABS_RHO = 0.1


HT_TRAINING_QUERY = """
    SELECT
        m.id::text AS match_id,
        m.match_date,
        m.league_id::text AS league_id,
        m.home_team_id::text AS home_team_id,
        m.away_team_id::text AS away_team_id,
        ht.name AS home_team,
        at.name AS away_team,
        -- HT goals routed through the same column names so the
        -- DixonColesPredictor + training_data helpers work unchanged.
        m.half_time_home AS home_score,
        m.half_time_away AS away_score,
        -- match_outcome target so the predictor's train() can record
        -- a comparable validation metric, computed from HT outcome.
        CASE
            WHEN m.half_time_home > m.half_time_away THEN 0
            WHEN m.half_time_home = m.half_time_away THEN 1
            ELSE 2
        END AS match_outcome,
        NULL::numeric AS odds_home,
        NULL::numeric AS odds_draw,
        NULL::numeric AS odds_away,
        NULL::numeric AS odds_over25,
        NULL::numeric AS odds_under25,
        NULL::jsonb AS features
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    WHERE l.sport = 'soccer'
      AND m.status = 'finished'
      AND m.half_time_home IS NOT NULL
      AND m.half_time_away IS NOT NULL
      -- A half-time score is trusted only beside a consistent full-time one
      -- (m.home_score / m.away_score here are the TABLE's full-time columns,
      -- not the aliases above). Matches the A/B gate's frame, which dropped
      -- the 2 HT-without-FT rows of 108,228 on 2026-10-31.
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
      AND m.home_score >= m.half_time_home
      AND m.away_score >= m.half_time_away
    ORDER BY m.match_date ASC
"""


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fit_refusal(
    result: Dict[str, Any],
    model: Any,
    fixtures: Iterable[Tuple[str, str, Any]],
    max_level_bias: float = MAX_LEVEL_BIAS,
    max_abs_rho: float = MAX_ABS_RHO,
) -> Optional[str]:
    """Why this fit must NOT be written, or None when it may be.

    `result` is DixonColesPredictor.train()'s return, `model` the fitted
    predictor, `fixtures` the (home_team, away_team, league_id) triples to
    build served matrices for — every distinct training fixture in main().
    Pure apart from the matrix arithmetic, so each refusal is unit-testable.
    """
    if result.get("iterations") is None:
        return "the fit did not converge"

    fitted = result.get("fitted_mean_total_goals")
    observed = result.get("observed_mean_total_goals")
    if not (_finite(fitted) and _finite(observed) and observed > 0):
        return f"no usable level check (fitted mean total goals {fitted!r}, observed {observed!r})"
    level_bias = fitted / observed - 1.0
    if abs(level_bias) > max_level_bias:
        return (
            f"fitted mean total goals {fitted:.4f} vs observed {observed:.4f} ({100 * level_bias:+.2f}%) "
            f"is outside +/-{100 * max_level_bias:.0f}%"
        )

    rho = getattr(model, "rho", None)
    if not _finite(rho):
        return f"rho is not finite ({rho!r})"
    if abs(rho) > max_abs_rho:
        return f"|rho| = {abs(rho):.4f} exceeds {max_abs_rho}"

    import numpy as np
    from predictors.market_derivation import MAX_GOALS_DERIVE, build_dc_matrix  # type: ignore

    for home, away, league in fixtures:
        h_lam, a_lam = model.lambdas_for_match(home, away, league=league)
        P = build_dc_matrix(float(h_lam), float(a_lam), float(rho), max_goals=MAX_GOALS_DERIVE)
        if not (np.isfinite(P).all() and abs(float(P.sum()) - 1.0) < 1e-6):
            return f"invalid served scoreline matrix for {home} v {away} ({league})"
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    model: Any,
    result: Dict[str, Any],
    train_df: Any,
    *,
    run_id: str,
    git_sha: str,
    model_sha256: str,
    holdout_fraction: float,
    written_at: Optional[str] = None,
) -> Dict[str, Any]:
    """The provenance record written beside model.bin. promote_dc_model.py
    trusts a staged model.bin only when this manifest's model_sha256 is the
    file's, and promotes only a fit on every row (holdout_fraction 0)."""
    import pandas as pd

    newest = train_df["match_date"].max()
    return {
        "model": MODEL_NAME,
        "run_id": run_id,
        "git_sha": git_sha,
        "n_rows": int(len(train_df)),
        "newest_match_date": pd.Timestamp(newest).isoformat() if pd.notna(newest) else None,
        "holdout_fraction": float(holdout_fraction),
        "converged_at": result.get("iterations"),
        "fitted_mean_total_goals": result.get("fitted_mean_total_goals"),
        "observed_mean_total_goals": result.get("observed_mean_total_goals"),
        "rho": float(model.rho),
        "teams": len(model.team_attack),
        "leagues": int(result.get("league_count") or 0),
        "model_sha256": model_sha256,
        "written_at": written_at or datetime.now(timezone.utc).isoformat(),
    }


def _fsync(path: Path) -> None:
    with open(path, "rb") as f:
        os.fsync(f.fileno())


def write_artifact(model: Any, output_dir: Path, manifest_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Save model.bin + manifest.json into output_dir atomically.

    Both are written to *.tmp first and renamed into place only once both
    exist, so a reader never sees a half-written model.bin, and a crash
    between the two renames leaves a manifest whose model_sha256 no longer
    matches model.bin — which promote_dc_model.py refuses. `manifest_fields`
    is build_manifest()'s arguments minus model + model_sha256; the hash is
    taken here, from the bytes actually written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tmp = output_dir / (MODEL_FILE + ".tmp")
    manifest_tmp = output_dir / (MANIFEST_FILE + ".tmp")
    model.save(str(model_tmp))
    _fsync(model_tmp)
    manifest = build_manifest(model, model_sha256=sha256_file(model_tmp), **manifest_fields)
    with open(manifest_tmp, "w") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(model_tmp, output_dir / MODEL_FILE)
    os.replace(manifest_tmp, output_dir / MANIFEST_FILE)
    return manifest


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _is_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def output_dir_refusal(output_dir: Path) -> Optional[str]:
    """Why the trainer must not write into output_dir, or None when it may.

    Refuses /app/models/staging and /app/models/production and anything under
    either (staging-ht is a sibling, not under staging). The directory is
    compared both as written (normalised) and through os.path.realpath — as
    is each root — so neither `..` nor a symlink into those trees gets past.
    """
    written, resolved = os.path.abspath(output_dir), os.path.realpath(output_dir)
    for root, why in FORBIDDEN_OUTPUT_ROOTS:
        roots = (os.path.abspath(root), os.path.realpath(root))
        if any(_is_within(path, r) for path in (written, resolved) for r in roots):
            return f"{output_dir} (resolves to {resolved}) is under {root}: {why}"
    return None


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write model.bin + manifest.json (default: staging-ht; promote with "
        "scripts/ops/promote_ht_model.sh). Refused under /app/models/staging (the weekly retrain auto-promotes "
        "it) or /app/models/production (only the promoter writes there), after resolving symlinks.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Recorded in manifest.json; promote_dc_model.py --expect-run-id refuses any other. Default: a UTC stamp.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help="Diagnostic only: hold out the newest fraction and report its metrics. 0 (default) fits everything — "
        "the gated configuration was measured as a fit on all matches before each eval month. The manifest "
        "records it and the promoter refuses anything but 0.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s - %(message)s",
    )
    if not args.database_url:
        LOGGER.error("DATABASE_URL not set.")
        return 1
    if not 0.0 <= args.holdout_fraction < 1.0:
        LOGGER.error("--holdout-fraction must be in [0, 1)")
        return 1
    refusal = output_dir_refusal(args.output_dir)
    if refusal:
        LOGGER.error("Refusing --output-dir: %s", refusal)
        return 1
    run_id = args.run_id or default_run_id()

    sys.path.insert(0, "/app/services/ml-models/src")
    from dataclasses import replace

    from predictors.model_config import DIXON_COLES_CONFIG
    from predictors.poisson_models import DixonColesPredictor
    from utils.training_data import load_training_frame

    LOGGER.info("Run %s: loading halftime training frame...", run_id)
    frame = load_training_frame(database_url=args.database_url, query=HT_TRAINING_QUERY)
    if frame.empty:
        LOGGER.error("The halftime training frame is empty — refusing to fit")
        return 1
    LOGGER.info(
        "Loaded %d halftime matches (%s..%s).",
        len(frame),
        frame["match_date"].min(),
        frame["match_date"].max(),
    )

    # Fit everything by default. The query's ORDER BY keeps match-date order,
    # so a holdout (diagnostic only) is the temporally latest slice.
    n = len(frame)
    cutoff = n - int(n * args.holdout_fraction)
    train_df = frame.iloc[:cutoff].copy()
    val_df = frame.iloc[cutoff:].copy() if cutoff < n else None
    LOGGER.info("Train=%d Holdout=%d", len(train_df), 0 if val_df is None else len(val_df))

    # DIXON_COLES_CONFIG (per-league home/away baselines, a 20-match shrinkage
    # prior, 730-day decay, max_goals 10), gated for THIS artifact by
    # scripts/ab_halftime_dixon_coles.py (2026-10-31, 15 monthly walk-forward
    # folds 2025-08..2026-10, n=8,145): HT 1x2 Brier -0.0160 vs the served
    # artifact (fold-clustered SE 0.0026) and -0.0130 vs a recency-matched
    # per-league constant; fitted once and frozen for 15 months, -0.0138. The
    # legacy pinned config it replaces DIVERGED on the full frame (13/13 folds
    # unconverged, non-finite probabilities in 2026-08) — never fit it again.
    model = DixonColesPredictor(replace(DIXON_COLES_CONFIG, name=MODEL_NAME))
    result = model.train(train_df, val_df=val_df)
    LOGGER.info("Train result: %s", {k: v for k, v in result.items() if k not in ("team_attack", "team_defense")})

    # Converged, level-true, sane rho, and every training fixture serves a
    # valid matrix — or nothing is written.
    pairs = train_df[["home_team", "away_team", "league_id"]].drop_duplicates()
    refusal = fit_refusal(result, model, pairs.itertuples(index=False))
    if refusal:
        LOGGER.error("Run %s: %s — refusing to write an artifact", run_id, refusal)
        return 1
    LOGGER.info(
        "Fit accepted: converged after %s iterations, fitted/observed mean total goals %.4f/%.4f, rho %.4f; "
        "%d distinct fixtures all serve a finite, normalised matrix",
        result["iterations"],
        result["fitted_mean_total_goals"],
        result["observed_mean_total_goals"],
        model.rho,
        len(pairs),
    )

    manifest = write_artifact(
        model,
        args.output_dir,
        dict(
            result=result,
            train_df=train_df,
            run_id=run_id,
            git_sha=os.environ.get("GIT_SHA") or "unknown",
            holdout_fraction=args.holdout_fraction,
        ),
    )
    LOGGER.info(
        "Saved halftime Dixon-Coles to %s (sha256 %s, run %s, %d teams, newest match %s)",
        args.output_dir / MODEL_FILE,
        manifest["model_sha256"],
        run_id,
        manifest["teams"],
        manifest["newest_match_date"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
