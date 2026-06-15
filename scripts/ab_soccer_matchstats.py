"""A/B: do rolling match_stats features (shots / SoT / corners / possession)
improve the soccer 1X2 model on a held-out test set?

Isolated experiment BEFORE productionizing (compute_features.py +
features_cache backfill). Pulls the exact baseline feature set the real
model trains on (features_cache feature_set='baseline'), adds
leakage-safe rolling team match_stats, and compares an XGBoost trained
on baseline vs baseline+matchstats on a temporal 70/15/15 split.

Leakage guard: a match's rolling stat is the mean over the team's PRIOR
finished matches only (groupby team, sort by date, shift(1) then rolling
window) — the current match is never included.

xG is excluded on purpose: expected_goals is NULL for 100% of rows.

    python /app/scripts/ab_soccer_matchstats.py --window 5
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

WINDOW = 5

# One row per (team, match): the team's own-match stats + opponent's
# (for "against" metrics). Used to build the rolling history.
TEAM_MATCH_STATS_SQL = """
    SELECT
        m.id::text   AS match_id,
        m.match_date,
        ms.team_id::text AS team_id,
        ms.shots, ms.shots_on_target, ms.corners, ms.possession, ms.fouls,
        opp.shots AS shots_against, opp.shots_on_target AS sot_against
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    JOIN match_stats ms ON ms.match_id = m.id
    LEFT JOIN match_stats opp ON opp.match_id = m.id AND opp.team_id <> ms.team_id
    WHERE l.sport = 'soccer' AND m.status = 'finished'
"""

STAT_COLS = ["shots", "shots_on_target", "corners", "possession", "fouls", "shots_against", "sot_against"]


def _load(database_url: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Use the REAL training loader so the baseline == what the soccer
    # model actually trains on (odds subqueries + flattened features).
    from utils.training_data import load_training_frame

    frame = load_training_frame(database_url=database_url)
    with psycopg2.connect(database_url) as conn:
        tms = pd.read_sql(TEAM_MATCH_STATS_SQL, conn)
    return frame, tms


def _rolling_team_history(tms: pd.DataFrame, window: int) -> pd.DataFrame:
    """For each (team, match) return the team's mean stats over its PRIOR
    `window` finished matches (shift(1) excludes the current match)."""
    tms = tms.sort_values(["team_id", "match_date"]).reset_index(drop=True)
    out = tms[["match_id", "team_id"]].copy()
    for c in STAT_COLS:
        tms[c] = pd.to_numeric(tms[c], errors="coerce")
        rolled = (
            tms.groupby("team_id")[c]
            .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        out[f"roll_{c}"] = rolled
    return out


def _build_matchstats_features(frame: pd.DataFrame, tms: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Add home/away rolling stats + their diffs for each window."""
    for window in windows:
        hist = _rolling_team_history(tms, window)
        roll_cols = [c for c in hist.columns if c.startswith("roll_")]
        suf = f"_w{window}"
        home = hist.rename(
            columns={"team_id": "home_team_id", **{c: f"home_{c}{suf}" for c in roll_cols}}
        )
        away = hist.rename(
            columns={"team_id": "away_team_id", **{c: f"away_{c}{suf}" for c in roll_cols}}
        )
        frame = frame.merge(home, on=["match_id", "home_team_id"], how="left")
        frame = frame.merge(away, on=["match_id", "away_team_id"], how="left")
        for c in roll_cols:
            frame[f"ms_diff_{c}{suf}"] = frame[f"home_{c}{suf}"] - frame[f"away_{c}{suf}"]
    return frame


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
    p = clf.predict_proba(Xte)
    oh = np.zeros_like(p)
    oh[np.arange(len(yte)), yte] = 1.0
    brier = float(np.mean(np.sum((p - oh) ** 2, axis=1)))
    ll = float(-np.mean(np.log(np.clip(p[np.arange(len(yte)), yte], 1e-12, 1))))
    acc = float((np.argmax(p, axis=1) == yte).mean())
    return {"brier": round(brier, 5), "log_loss": round(ll, 5), "accuracy": round(acc, 5), "n_test": int(len(yte))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    ap.add_argument("--windows", default="5", help="comma-separated rolling windows, e.g. 3,5,10")
    args = ap.parse_args()
    windows = [int(w) for w in args.windows.split(",") if w.strip()]

    from utils.training_data import get_feature_columns

    frame, tms = _load(args.database_url)
    base_cols = get_feature_columns(frame)  # the REAL baseline feature set
    frame = _build_matchstats_features(frame, tms, windows)
    frame = frame.sort_values("match_date").reset_index(drop=True)
    ms_cols = [c for c in frame.columns if c.startswith("ms_diff_")]

    n = len(frame)
    tr_end, va_end = int(n * 0.70), int(n * 0.85)
    train, val, test = frame.iloc[:tr_end], frame.iloc[tr_end:va_end], frame.iloc[va_end:]

    print(f"n={n}  windows={windows}  base_features={len(base_cols)}  matchstats_features={len(ms_cols)}")
    print(f"match_stats coverage on frame: {frame[ms_cols[0]].notna().mean():.1%}" if ms_cols else "no ms cols")

    baseline = _xgb_eval(train, val, test, base_cols)
    augmented = _xgb_eval(train, val, test, base_cols + ms_cols)

    d_brier = augmented["brier"] - baseline["brier"]
    d_ll = augmented["log_loss"] - baseline["log_loss"]
    verdict = "SHIP" if d_brier <= -0.002 else ("MARGINAL" if d_brier <= -0.0005 else "DROP")
    print(json.dumps({"baseline": baseline, "augmented": augmented, "delta_brier": round(d_brier, 5),
                      "delta_log_loss": round(d_ll, 5), "verdict": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
