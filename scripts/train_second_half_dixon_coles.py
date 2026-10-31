"""Train a SECOND HALF (2H) Dixon-Coles model for soccer — into STAGING only.

NOT PROMOTABLE. The served second-half artifact
(/app/models/production/dixon_coles_h2_soccer/1.0.0/model.bin, the 2026-06-04
legacy fit) STAYS. Its gate, scripts/ab_second_half_dixon_coles.py (2026-10-31,
monthly walk-forward 2025-08..2026-10, n=8,144), did NOT pass: the pre-registered
rule needed BOTH refit (ht_ft dBrier -0.0057, passed) AND refit_frozen (-0.0046,
failed the -0.005 gate) to pass ship_strict vs served. So
scripts/promote_dc_model.py has no "h2" entry, and nothing this script writes
may be copied into production by hand either. It exists so the next gate
measures an artifact that a sound, reproducible trainer wrote.

Sibling of `train_halftime_dixon_coles.py`. Where that script trains on
`(half_time_home, half_time_away)`, this one trains on
`(home_score - half_time_home, away_score - half_time_away)` — i.e. goals
scored AFTER halftime. The two models together give the HT/FT joint
distribution: precompute_predictions convolves P_HT (this match's HT scoreline)
with P_2H (its 2H scoreline) and aggregates the joint mass by (HT outcome, FT
outcome) into ht_ft_double_result — the only served market this artifact feeds.

Why a separate 2H model:
  * 1H and 2H goal rates differ in real soccer — fatigue, tactical changes and
    2H "open" play push the 2H goal rate ~15% above the 1H one, so the FT or HT
    model alone miscalibrates the 2H distribution.
  * Team attack/defense strengths can shift between halves; Dixon-Coles fits
    them per half from the same rows.

Architecture (the half-time trainer's, reused — not copied):
  * SECOND_HALF_TRAINING_QUERY routes the 2H goals into home_score/away_score
    so DixonColesPredictor works unchanged (scripts/ab_second_half_dixon_coles.py
    imports this query: its frame IS this trainer's frame).
  * Fit on ALL rows with DIXON_COLES_CONFIG under the name dixon_coles_h2_soccer
    (h2_config). Not the pinned legacy config: that one never converges on the
    2H frame (15/15 walk-forward cutoffs in the gate), so a trainer that refuses
    unconverged fits would refuse every run of it. DIXON_COLES_CONFIG is the
    config the gate's refit arm measured. --holdout-fraction (default 0) only
    holds out the newest slice as a diagnostic.
  * train_halftime_dixon_coles.fit_refusal refuses an unconverged fit, a fitted
    vs observed mean total goals gap past +/-2%, a |rho| past 0.1 and any
    training fixture whose served matrix (lambdas_for_match with its league_id,
    build_dc_matrix at MAX_GOALS_DERIVE — what precompute_predictions does) is
    not finite and normalised. Nothing is written then.
  * write_artifact writes model.bin + manifest.json atomically (run id, git sha,
    rows, newest match, convergence, level, rho, sha256 of model.bin).
  * output_dir_refusal refuses /app/models/staging (the weekly retrain_models
    swap_production promotes everything in it, with no checks) and
    /app/models/production, symlinks and `..` resolved. The default is
    /app/models/staging-h2/dixon_coles_h2_soccer/1.0.0, which nothing promotes.

Until 2026-10-31 this script fitted the OLDEST 80% only, with the pinned legacy
config, never checked convergence, and wrote straight over the production file.
A 2026-10-31 dry run of this version (to a /tmp dir) on all 108,228 rows:
converged after 30 iterations, fitted vs observed mean 2H goals 1.519 / 1.528
(-0.6%), rho +0.006, 939 teams, 31 league baselines — inside the shared bounds.

Run on prod via (stages only; read the log):
    docker compose exec api python /app/scripts/train_second_half_dixon_coles.py
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

# The half-time trainer's guards, manifest and atomic write live next to this
# file; /app/scripts is on sys.path when run as a script, and the explicit
# insert keeps importlib-loaded copies (the unit tests) working too.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_halftime_dixon_coles as halftime  # noqa: E402

LOGGER = logging.getLogger("train_second_half_dixon_coles")

MODEL_NAME = "dixon_coles_h2_soccer"
DEFAULT_OUTPUT_DIR = Path("/app/models/staging-h2/dixon_coles_h2_soccer/1.0.0")


SECOND_HALF_TRAINING_QUERY = """
    SELECT
        m.id::text AS match_id,
        m.match_date,
        m.league_id::text AS league_id,
        m.home_team_id::text AS home_team_id,
        m.away_team_id::text AS away_team_id,
        ht.name AS home_team,
        at.name AS away_team,
        -- Second-half goals routed through the same column names so
        -- the DixonColesPredictor + training_data helpers work
        -- unchanged. NULLIF defensively drops the rare bad-data row
        -- where FT < HT (would imply negative goals).
        GREATEST(m.home_score - m.half_time_home, 0) AS home_score,
        GREATEST(m.away_score - m.half_time_away, 0) AS away_score,
        -- match_outcome derived from the 2H scoreline so the
        -- predictor's train() can log a comparable val metric.
        CASE
            WHEN (m.home_score - m.half_time_home) > (m.away_score - m.half_time_away) THEN 0
            WHEN (m.home_score - m.half_time_home) = (m.away_score - m.half_time_away) THEN 1
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
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
      -- Defensive: drop rows where FT < HT (data corruption).
      AND m.home_score >= m.half_time_home
      AND m.away_score >= m.half_time_away
    ORDER BY m.match_date ASC
"""


def h2_config() -> Any:
    """DIXON_COLES_CONFIG under this artifact's name — what
    scripts/ab_second_half_dixon_coles.py's refit arms measured."""
    from dataclasses import replace

    from predictors.model_config import DIXON_COLES_CONFIG  # type: ignore

    return replace(DIXON_COLES_CONFIG, name=MODEL_NAME)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write model.bin + manifest.json (default: staging-h2 — nothing promotes it). Refused under "
        "/app/models/staging (the weekly retrain auto-promotes it) or /app/models/production, after resolving "
        "symlinks.",
    )
    parser.add_argument("--run-id", default=None, help="Recorded in manifest.json. Default: a UTC stamp.")
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help="Diagnostic only: hold out the newest fraction and report its metrics. 0 (default) fits everything — "
        "the gate measured a fit on all matches before each eval month. The manifest records it.",
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

    sys.path.insert(0, "/app/services/ml-models/src")
    from predictors.poisson_models import DixonColesPredictor
    from utils.training_data import load_training_frame

    LOGGER.info("Run %s: loading second-half training frame...", run_id)
    frame = load_training_frame(database_url=args.database_url, query=SECOND_HALF_TRAINING_QUERY)
    if frame.empty:
        LOGGER.error("The second-half training frame is empty — refusing to fit")
        return 1
    LOGGER.info(
        "Loaded %d second-half matches (%s..%s).",
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

    model = DixonColesPredictor(h2_config())
    result = model.train(train_df, val_df=val_df)
    LOGGER.info("Train result: %s", {k: v for k, v in result.items() if k not in ("team_attack", "team_defense")})

    # Converged, level-true, sane rho, and every training fixture serves a
    # valid matrix (with its league, as precompute_predictions asks) — or
    # nothing is written.
    pairs = train_df[["home_team", "away_team", "league_id"]].drop_duplicates()
    refusal = halftime.fit_refusal(result, model, pairs.itertuples(index=False))
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
        "Staged second-half Dixon-Coles at %s (sha256 %s, run %s, %d teams, newest match %s). NOT promotable: "
        "its gate (scripts/ab_second_half_dixon_coles.py) has not passed; production keeps the served artifact.",
        args.output_dir / halftime.MODEL_FILE,
        manifest["model_sha256"],
        run_id,
        manifest["teams"],
        manifest["newest_match_date"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
