"""A/B gate for the corpus 3-4x expansion (audit §3.6): does training on
the 7x corpus (41 leagues / 20 seasons, 168k matches) hold the line on
held-out TOP-5-LEAGUE Brier vs the original corpus (6 leagues / 10
seasons)?

Shared test set: the latest ~15% of top-5-league rows (never trained on
by either model). Baseline trains on the original corpus only; expanded
trains on everything before the test window. Secondary readout: both
models scored on the same-window SUMMER-league rows (MLS, Allsvenskan,
Brasileirao...) — the leagues generating live recs with zero history
under the old corpus.

Gate: expanded top-5 Brier must NOT regress beyond the noise floor
(SE ~ 0.009 at n~3.5k). If it holds AND summer-league rows improve,
the expansion ships (production loader already reads the full corpus —
a FAIL here means constraining the loader before Sunday's retrain).

    python /app/scripts/ab_soccer_corpus.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, "/app/services/ml-models/src")

TOP5 = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]
ORIGINAL = TOP5 + ["Championship"]
SUMMER = ["MLS", "Allsvenskan", "Eliteserien", "Veikkausliiga", "Brasileirão Série A", "Brasileirao Serie A"]

LEAGUE_SQL = """
    SELECT m.id::text AS match_id, l.name AS league_name
    FROM matches m JOIN leagues l ON l.id = m.league_id
    WHERE l.sport = 'soccer'
"""


def _xgb_eval(train, val, test, feature_cols, target="match_outcome"):
    import xgboost as xgb

    def _xy(df):
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        return X, df[target].astype(int).values

    Xtr, ytr = _xy(train)
    Xva, yva = _xy(val)
    Xte, yte = _xy(test)
    clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=40,
        random_state=42,
        n_jobs=4,
    )
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    def _score(df):
        if not len(df):
            return {"n": 0}
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        y = df[target].astype(int).values
        p = clf.predict_proba(X)
        oh = np.zeros_like(p)
        oh[np.arange(len(y)), y] = 1.0
        return {
            "n": int(len(y)),
            "brier": round(float(np.mean(np.sum((p - oh) ** 2, axis=1))), 5),
            "accuracy": round(float((np.argmax(p, axis=1) == y).mean()), 5),
        }

    return _score(test), _score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    args = ap.parse_args()

    from utils.training_data import get_feature_columns, load_training_frame

    frame = load_training_frame(database_url=args.database_url)
    with psycopg2.connect(args.database_url) as conn:
        leagues = pd.read_sql(LEAGUE_SQL, conn)
    frame = frame.merge(leagues, on="match_id", how="left")
    feature_cols = get_feature_columns(frame)
    frame = frame.sort_values("match_date").reset_index(drop=True)

    top5 = frame[frame["league_name"].isin(TOP5)]
    test_start = top5["match_date"].quantile(0.85)
    test = top5[top5["match_date"] >= test_start]
    pre = frame[frame["match_date"] < test_start]

    def _split(train_frame):
        n = len(train_frame)
        return train_frame.iloc[: int(n * 0.85)], train_frame.iloc[int(n * 0.85) :]

    base_pool = pre[pre["league_name"].isin(ORIGINAL)]
    # Original corpus was also ~10 seasons deep, not 20.
    base_pool = base_pool[base_pool["match_date"] >= (test_start - pd.Timedelta(days=3650))]
    exp_pool = pre

    summer_test = frame[(frame["league_name"].isin(SUMMER)) & (frame["match_date"] >= test_start)]

    print(
        f"frame={len(frame)}  features={len(feature_cols)}  top5_test={len(test)}  "
        f"baseline_pool={len(base_pool)}  expanded_pool={len(exp_pool)}  summer_test={len(summer_test)}"
    )

    btr, bva = _split(base_pool)
    base_top5, base_scorer = _xgb_eval(btr, bva, test, feature_cols)
    etr, eva = _split(exp_pool)
    exp_top5, exp_scorer = _xgb_eval(etr, eva, test, feature_cols)

    base_summer = base_scorer(summer_test)
    exp_summer = exp_scorer(summer_test)

    d = exp_top5["brier"] - base_top5["brier"]
    verdict = "SHIP" if d <= 0.005 else "FAIL — constrain the training loader before Sunday's retrain"
    print(
        json.dumps(
            {
                "top5_baseline": base_top5,
                "top5_expanded": exp_top5,
                "delta_top5_brier_expanded_minus_baseline": round(d, 5),
                "noise_floor_se": 0.009,
                "summer_baseline": base_summer,
                "summer_expanded": exp_summer,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
