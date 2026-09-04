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
  * Train + save the model to
    /app/models/production/dixon_coles_ht_soccer/1.0.0/model.bin.
  * `scripts/precompute_predictions.py` loads this artifact (if
    present) and writes one row per HT market_type per match.

Run on prod via:
    docker compose exec api python /app/scripts/train_halftime_dixon_coles.py
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
        default=Path("/app/models/production/dixon_coles_ht_soccer/1.0.0"),
        help="Where to write the model artifact (model.bin).",
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

    # 80/20 train/val by date. Match-date sort is preserved by the
    # query's ORDER BY, so the last 20% are the temporally latest
    # matches — clean walk-forward eval.
    n = len(frame)
    cutoff = int(n * 0.8)
    train_df = frame.iloc[:cutoff].copy()
    val_df = frame.iloc[cutoff:].copy()
    LOGGER.info("Train=%d Val=%d", len(train_df), len(val_df))

    model = DixonColesPredictor(DIXON_COLES_CONFIG)
    result = model.train(train_df, val_df=val_df)
    LOGGER.info("Train result: %s", {k: v for k, v in result.items() if k not in ("team_attack", "team_defense")})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.bin"
    model.save(str(model_path))
    LOGGER.info("Saved halftime Dixon-Coles to %s", model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
