"""Train a Dixon-Coles model for NHL.

Hockey scorelines are low-discrete just like soccer (modal ~3-2,
median total ~5.5), so the same Dixon-Coles + Poisson machinery
works directly. This trainer mirrors train_halftime_dixon_coles.py
but pulls the NHL goals (matches.home_score / away_score, including
OT/SO) instead of soccer FT/HT scores.

The resulting model.bin enables analytic market derivation for
NHL via `derive_hockey_markets` in market_derivation.py
(registered under sport="nhl"), giving us moneyline_inc_ot,
total_goals exact buckets, double_chance (with OT redistribution),
and clean_sheet/win_to_nil without retraining a per-market ML
model for each.

Saves to /app/models/production/dixon_coles_nhl/1.0.0/model.bin
(parallel to the soccer DC artifact directory).

Run on prod via:
    docker compose exec api python /app/scripts/train_hockey_dixon_coles.py
"""

import argparse
import logging
import os
import sys
from pathlib import Path

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

from utils.training_data import preseason_exclusion_sql  # noqa: E402

LOGGER = logging.getLogger("train_hockey_dixon_coles")


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/app/models/production/dixon_coles_nhl/1.0.0"),
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

    LOGGER.info("Loading NHL Dixon-Coles training frame...")
    frame = load_training_frame(
        database_url=args.database_url,
        query=NHL_DC_TRAINING_QUERY,
    )
    LOGGER.info(
        "Loaded %d NHL matches (%s..%s).",
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
    LOGGER.info(
        "Train result: %s",
        {k: v for k, v in result.items() if k not in ("team_attack", "team_defense")},
    )

    # NHL goal scoring doesn't have soccer's low-score correlation
    # pattern (the 0-0/1-1/0-1/1-0 cluster the DC `tau` correction
    # is designed for), so leaving rho to the optimizer produces
    # absurd values (~1e27 observed during initial training). Force
    # rho=0 — reduces to pure Poisson, which IS the right baseline
    # for hockey. The team strength fit + lambda machinery still
    # benefits from the DC config; only the correlation correction
    # is dropped.
    LOGGER.info(
        "Forcing rho=0 for hockey (optimizer produced %s, hockey doesn't "
        "have soccer's low-score correlation pattern).",
        model.rho,
    )
    model.rho = 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.bin"
    model.save(str(model_path))
    LOGGER.info("Saved NHL Dixon-Coles to %s", model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
