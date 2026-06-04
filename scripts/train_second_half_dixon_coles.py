"""Train a SECOND HALF (2H) Dixon-Coles model for soccer.

Sibling of `train_halftime_dixon_coles.py`. Where that script trains
on `(half_time_home, half_time_away)`, this one trains on
`(home_score - half_time_home, away_score - half_time_away)` — i.e.
goals scored AFTER halftime. The two models together enable the
HT/FT joint distribution: convolving P_HT (this match's HT scoreline)
with P_2H (this match's 2H scoreline) gives P_FT, and aggregating
the joint mass by (HT-outcome, FT-outcome) gives the HT/FT
double-result market.

Why a separate 2H model:
  * 1H and 2H goal rates differ in real soccer — fatigue, tactical
    changes, and 2H "open" play push the 2H goal rate ~15% higher
    than 1H on average. Using the FT or HT model alone as a proxy
    miscalibrates the 2H distribution.
  * Team-level attack/defense strengths can shift between halves
    (some teams start strong, others wear opponents down) — Dixon-
    Coles fits these per-half effectively from the same training
    rows.

Score column rename pattern mirrors the HT trainer: we rename
`(ft - ht)` into `home_score`/`away_score` so the existing
DixonColesPredictor code path works unchanged.

Run on prod via:
    docker compose exec api python /app/scripts/train_second_half_dixon_coles.py
"""

import argparse
import logging
import os
import sys
from pathlib import Path

LOGGER = logging.getLogger("train_second_half_dixon_coles")


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/app/models/production/dixon_coles_h2_soccer/1.0.0"),
        help="Where to write the model artifact (model.bin).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
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

    LOGGER.info("Loading second-half training frame...")
    frame = load_training_frame(database_url=args.database_url, query=SECOND_HALF_TRAINING_QUERY)
    LOGGER.info(
        "Loaded %d second-half matches (%s..%s).",
        len(frame),
        frame["match_date"].min(),
        frame["match_date"].max(),
    )

    n = len(frame)
    cutoff = int(n * 0.8)
    train_df = frame.iloc[:cutoff].copy()
    val_df = frame.iloc[cutoff:].copy()
    LOGGER.info("Train=%d Val=%d", len(train_df), len(val_df))

    model = DixonColesPredictor(DIXON_COLES_CONFIG)
    result = model.train(train_df, val_df=val_df)
    LOGGER.info("Train result: %s", {
        k: v for k, v in result.items()
        if k not in ("team_attack", "team_defense")
    })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.bin"
    model.save(str(model_path))
    LOGGER.info("Saved second-half Dixon-Coles to %s", model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
