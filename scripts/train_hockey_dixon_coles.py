"""Train a Dixon-Coles model for NHL — into STAGING only.

NOT PROMOTABLE. The served NHL artifact
(/app/models/production/dixon_coles_nhl/1.0.0/model.bin) has never been gated,
and neither has anything this script writes: its gate is
scripts/ab_nhl_dixon_coles.py (pre-registered there, not yet run over its full
window). scripts/promote_dc_model.py has no "nhl" entry, and nothing staged here
may be copied into production by hand. Until the gate passes, the config below
stays the pinned legacy one and production keeps serving its current file.

Hockey scorelines are low-discrete just like soccer (modal ~3-2, median total
~5.5), so the same Dixon-Coles + Poisson machinery works directly. This trainer
pulls the NHL goals (matches.home_score / away_score, FINAL scores including
OT/SO) instead of soccer FT/HT scores.

What it feeds once promoted: precompute_predictions_nhl.py loads the production
file (_load_hockey_dixon_coles), takes lambdas_for_match(home, away) — NO
league argument — builds build_dc_matrix(..., rho, MAX_GOALS_DERIVE) and serves
8 markets from derive_hockey_markets (regulation_1x2, puck_line, over_under,
total_goals, double_chance, clean_sheet, win_to_nil, correct_score; its
moneyline is dropped).

Architecture (the half-time trainer's guards, reused — not copied):
  * Fit on ALL rows of NHL_DC_TRAINING_QUERY (preseason excluded) with
    nhl_config() — the pinned legacy config, which converges on the NHL frame
    (7 iterations). --holdout-fraction (default 0) only holds out the newest
    slice as a diagnostic.
  * force_rho_zero: final scores are never 0-0 or 1-1, so the rho NLL runs away
    (~1e27 observed) instead of fitting; rho is forced to 0 (pure Poisson) after
    the fit, as before. The optimiser's value is logged.
  * train_halftime_dixon_coles.fit_refusal, AFTER forcing rho, refuses an
    unconverged fit, a fitted vs observed mean total goals gap past +/-2%, and
    any training fixture whose served matrix (lambdas_for_match WITHOUT a
    league, as precompute_predictions_nhl calls it) is not finite and
    normalised. Its rho bound is trivially met at 0 — the runaway rho is
    handled by the forcing, not refused. Nothing is written on a refusal.
  * write_artifact writes model.bin + manifest.json atomically (run id, git sha,
    rows, newest match, convergence, level, rho, sha256 of model.bin).
  * output_dir_refusal refuses /app/models/staging (the weekly retrain_models
    swap_production promotes everything in it, with no checks) and
    /app/models/production, symlinks and `..` resolved. The default is
    /app/models/staging-nhl/dixon_coles_nhl/1.0.0, which nothing promotes.

Until 2026-10-31 this script fitted the OLDEST 80% only (so no 2025-26 match
reached the fit: 'Utah Mammoth' is unknown to the served file), never checked
convergence or level, and wrote straight over the production file. A
2026-10-31 dry run of this version (to a /tmp dir) on all 8,161 rows: converged
after 7 iterations, fitted vs observed mean total goals 6.189 / 6.189, the
optimiser's rho 8.2e27 forced to 0, 34 teams.

Run on prod via (stages only; read the log):
    docker compose exec api python /app/scripts/train_hockey_dixon_coles.py
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

# The preseason-exclusion predicate is defined ONCE, in
# services/ml-models/src/utils/training_data.py. Same sys.path dance as
# scripts/compute_features_nfl.py: the api container already has that tree
# on PYTHONPATH, and the repo-relative path keeps local dev + the unit
# tests working. Position 0 also guarantees `utils` resolves to
# ml-models' package rather than the same-named one under
# services/data-ingestion/src.
_ML_MODELS_SRC = str(Path(__file__).resolve().parent.parent / "services" / "ml-models" / "src")
for _p in ("/app/services/ml-models/src", _ML_MODELS_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# The half-time trainer's guards, manifest and atomic write live next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_halftime_dixon_coles as halftime  # noqa: E402
from utils.training_data import preseason_exclusion_sql  # noqa: E402

LOGGER = logging.getLogger("train_hockey_dixon_coles")

MODEL_NAME = "dixon_coles_nhl"
DEFAULT_OUTPUT_DIR = Path("/app/models/staging-nhl/dixon_coles_nhl/1.0.0")


# Preseason is excluded here for the same reason it is excluded from every
# other training frame: preseason lineups and scoring rates are structurally
# different, and this artifact feeds 8 live NHL markets via
# derive_hockey_markets. Unlike the NBA/NFL spread+total frames there is no
# incidental odds join to keep it out — nothing but this predicate. NHL rows
# loaded by load_nhl_historical.py carry the legacy metadata.game_type marker
# instead of season_type, which preseason_exclusion_sql also honours.
NHL_DC_TRAINING_QUERY = f"""
    SELECT
        m.id::text AS match_id,
        m.match_date,
        m.league_id::text AS league_id,
        m.home_team_id::text AS home_team_id,
        m.away_team_id::text AS away_team_id,
        ht.name AS home_team,
        at.name AS away_team,
        -- Use FINAL home/away scores (including OT/SO) so the
        -- Dixon-Coles model captures the actual scoring rate per
        -- team. The NHL_REGULATION market gets its own model
        -- elsewhere; this one is for total goals + final-result
        -- markets that include OT.
        m.home_score AS home_score,
        m.away_score AS away_score,
        CASE
            WHEN m.home_score > m.away_score THEN 0
            ELSE 1
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
    WHERE l.sport = 'nhl'
      AND m.status = 'finished'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
      AND {preseason_exclusion_sql('m')}
    ORDER BY m.match_date ASC
"""


def nhl_config() -> Any:
    """The config this trainer fits — scripts/ab_nhl_dixon_coles.py's pinned arm
    imports it, so the gate measures exactly this.

    NOT DIXON_COLES_CONFIG. That object carries the 2026-09 SOCCER refit knobs
    (max_goals 10, a live time_decay, regularization as a 20-effective-match
    prior, per-league baselines), and not one hockey number was measured for
    any of them; ab_nhl_dixon_coles.py's refit arms measure them now. Pinned to
    the pre-refit values until that gate passes. The NHL defects (as corrected
    2026-10-31 in model_config.py, next to
    dixon_coles_config_pinned_to_legacy_fit), each needing its own measurement:
      * level inflation: the served artifact predates the scale-gauge
        correction in PoissonMatchPredictor.train, so its lambdas carry that
        positive level bias (a fit here does not; fit_refusal checks it);
      * a stale fit: the served file has no 2025-26 data, so 'Utah Mammoth' is
        an unknown team there and serves attack = defense = 1.0 (a fit here
        uses every row);
      * rho forced to 0 (force_rho_zero);
      * a regulation-tie deficit: fitted on FINAL scores (OT/SO included), its
        draw cells — what regulation_1x2 and double_chance price — held 15.5%
        on average vs a 23.4% observed regulation-tie rate over the 286 graded
        matches to 2026-10-31. Nothing here changes that.
    (max_goals=6 is not one: precompute_predictions_nhl builds every served
    matrix at market_derivation.MAX_GOALS_DERIVE=10, and home_advantage=0.25
    roughly matches NHL's observed home edge.)
    """
    from predictors.model_config import dixon_coles_config_pinned_to_legacy_fit  # type: ignore

    return dixon_coles_config_pinned_to_legacy_fit(MODEL_NAME)


def force_rho_zero(model: Any, result: Dict[str, Any]) -> Optional[float]:
    """Force rho to 0 on `model` and in `result`; return the optimiser's value.

    NHL goal scoring has no soccer-style low-score dependence (the
    0-0/1-1/0-1/1-0 cluster the DC tau corrects), and final scores are never
    tied, so the rho NLL runs off (~1e27 observed during initial training).
    rho = 0 is pure Poisson — the right baseline for hockey. The team strengths
    are fitted before rho and do not depend on it. result['rho_fitted'] keeps
    the optimiser's value (None when non-finite) for the log.
    """
    fitted = getattr(model, "rho", None)
    model.rho = 0.0
    result["rho_fitted"] = float(fitted) if halftime._finite(fitted) else None
    result["rho"] = 0.0
    return result["rho_fitted"]


def served_fixtures(frame: Any) -> Iterable[Tuple[str, str, None]]:
    """(home, away, None) per distinct fixture: precompute_predictions_nhl calls
    lambdas_for_match(home, away) with no league, so fit_refusal checks the
    matrices production would actually build."""
    pairs = frame[["home_team", "away_team"]].drop_duplicates()
    return ((home, away, None) for home, away in pairs.itertuples(index=False))


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write model.bin + manifest.json (default: staging-nhl — nothing promotes it). Refused under "
        "/app/models/staging (the weekly retrain auto-promotes it) or /app/models/production, after resolving "
        "symlinks.",
    )
    parser.add_argument("--run-id", default=None, help="Recorded in manifest.json. Default: a UTC stamp.")
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help="Diagnostic only: hold out the newest fraction and report its metrics. 0 (default) fits everything. "
        "The manifest records it.",
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
    refusal = halftime.output_dir_refusal(args.output_dir)
    if refusal:
        LOGGER.error("Refusing --output-dir: %s", refusal)
        return 1
    run_id = args.run_id or halftime.default_run_id()

    from predictors.poisson_models import DixonColesPredictor
    from utils.training_data import load_training_frame

    LOGGER.info("Run %s: loading NHL Dixon-Coles training frame...", run_id)
    frame = load_training_frame(database_url=args.database_url, query=NHL_DC_TRAINING_QUERY)
    if frame.empty:
        LOGGER.error("The NHL training frame is empty — refusing to fit")
        return 1
    LOGGER.info(
        "Loaded %d NHL matches (%s..%s).",
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

    model = DixonColesPredictor(nhl_config())
    result = model.train(train_df, val_df=val_df)
    rho_fitted = force_rho_zero(model, result)
    LOGGER.info(
        "Forced rho=0 for hockey (the optimiser produced %s; final scores carry no soccer-style low-score "
        "dependence).",
        "a non-finite value" if rho_fitted is None else f"{rho_fitted:.6g}",
    )
    LOGGER.info("Train result: %s", {k: v for k, v in result.items() if k not in ("team_attack", "team_defense")})

    # Converged, level-true, and every training fixture serves a valid matrix
    # as precompute_predictions_nhl builds it — or nothing is written.
    fixtures = list(served_fixtures(train_df))
    refusal = halftime.fit_refusal(result, model, fixtures)
    if refusal:
        LOGGER.error("Run %s: %s — refusing to write an artifact", run_id, refusal)
        return 1
    LOGGER.info(
        "Fit accepted: converged after %s iterations, fitted/observed mean total goals %.4f/%.4f, rho 0 (forced); "
        "%d distinct fixtures all serve a finite, normalised matrix",
        result["iterations"],
        result["fitted_mean_total_goals"],
        result["observed_mean_total_goals"],
        len(fixtures),
    )

    manifest = halftime.write_artifact(
        model,
        args.output_dir,
        dict(
            result=result,
            train_df=train_df,
            run_id=run_id,
            git_sha=os.environ.get("GIT_SHA") or "unknown",
            holdout_fraction=args.holdout_fraction,
        ),
        model_name=MODEL_NAME,
    )
    LOGGER.info(
        "Staged NHL Dixon-Coles at %s (sha256 %s, run %s, %d teams, newest match %s). NOT promotable: its gate "
        "(scripts/ab_nhl_dixon_coles.py) has not passed; production keeps the served artifact.",
        args.output_dir / halftime.MODEL_FILE,
        manifest["model_sha256"],
        run_id,
        manifest["teams"],
        manifest["newest_match_date"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
