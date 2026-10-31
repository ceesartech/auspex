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
    comment at the fit) — and write the artifact to STAGING.
  * scripts/ops/promote_ht_model.sh backs up the production artifact and
    promotes the staged one; `scripts/precompute_predictions.py` loads
    /app/models/production/dixon_coles_ht_soccer/1.0.0/model.bin on its
    next run and writes one row per HT market_type per match.

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
import logging
import os
import sys
from pathlib import Path

LOGGER = logging.getLogger("train_halftime_dixon_coles")


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
    ORDER BY m.match_date ASC
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/app/models/staging-ht/dixon_coles_ht_soccer/1.0.0"),
        help="Where to write model.bin (default: staging; promote with scripts/ops/promote_ht_model.sh).",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help="Diagnostic only: hold out the newest fraction and report its metrics. 0 (default) fits everything — "
        "the gated configuration was measured as a fit on all matches before each eval month.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s - %(message)s",
    )
    if not args.database_url:
        LOGGER.error("DATABASE_URL not set.")
        return 1

    sys.path.insert(0, "/app/services/ml-models/src")
    from dataclasses import replace

    import numpy as np
    from predictors.market_derivation import MAX_GOALS_DERIVE, build_dc_matrix
    from predictors.model_config import DIXON_COLES_CONFIG
    from predictors.poisson_models import DixonColesPredictor
    from utils.training_data import load_training_frame

    LOGGER.info("Loading halftime training frame...")
    frame = load_training_frame(database_url=args.database_url, query=HT_TRAINING_QUERY)
    LOGGER.info(
        "Loaded %d halftime matches (%s..%s).",
        len(frame),
        frame["match_date"].min(),
        frame["match_date"].max(),
    )

    # Fit everything by default. The query's ORDER BY keeps match-date order,
    # so a holdout (diagnostic only) is the temporally latest slice.
    n = len(frame)
    if not 0.0 <= args.holdout_fraction < 1.0:
        LOGGER.error("--holdout-fraction must be in [0, 1)")
        return 1
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
    model = DixonColesPredictor(replace(DIXON_COLES_CONFIG, name="dixon_coles_ht_soccer"))
    result = model.train(train_df, val_df=val_df)
    LOGGER.info("Train result: %s", {k: v for k, v in result.items() if k not in ("team_attack", "team_defense")})
    if result.get("iterations") is None:
        LOGGER.error("The fit did not converge — refusing to write an artifact")
        return 1
    # Every training fixture must produce a valid served matrix.
    pairs = train_df[["home_team", "away_team", "league_id"]].drop_duplicates()
    for home, away, league in pairs.itertuples(index=False):
        h_lam, a_lam = model.lambdas_for_match(home, away, league=league)
        P = build_dc_matrix(float(h_lam), float(a_lam), float(model.rho), max_goals=MAX_GOALS_DERIVE)
        if not (np.isfinite(P).all() and abs(float(P.sum()) - 1.0) < 1e-6):
            LOGGER.error(
                "Invalid scoreline matrix for %s v %s (%s) — refusing to write an artifact", home, away, league
            )
            return 1
    LOGGER.info("Checked %d distinct fixtures: every served matrix is finite and normalised", len(pairs))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.bin"
    model.save(str(model_path))
    LOGGER.info("Saved halftime Dixon-Coles to %s", model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
