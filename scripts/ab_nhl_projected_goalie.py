"""Paired A/B: does serving the PROBABLE starting goalie (instead of the
neutral goalie defaults every upcoming NHL game got until 2026-10) improve
the CURRENT production NHL ensembles' held-out predictions?

For every finished, non-preseason NHL game in the evaluation window the
feature row is rebuilt three times with compute_features_nhl's real code,
differing ONLY in the goalie source:

  none       goalie features at NEUTRAL_DEFAULTS (what serving did)
  projected  probable starter from strictly-earlier starts (what serving
             can know before puck drop — the new default)
  actual     the recorded starter (oracle ceiling, unattainable live)

then scored by the loaded production ensembles (same path as
precompute_predictions_nhl). Reported per market: multiclass Brier
(sum of squares, the repo convention) and the paired delta vs `none`
with its standard error. This is a serving-feature change on a fixed
model, so the §4.3 retrain gate does not apply; the delta is still
quoted with its SE and nothing ships if it is worse.

Run inside the api container (reads prod, writes nothing):

    python /app/scripts/ab_nhl_projected_goalie.py --since 2025-11-15

Result 2026-10-26 (after the 2025-26 goalie + 5v5 backfill), Δ vs none:

    window                     n     moneyline         regulation        puck_line         total
    held-out 2025-11-15..now 1300  proj -0.00090±112  -0.00045±136      -0.00243±179      +0.00120±68
                                   act  -0.00072±111  +0.00015±148      -0.00182±179      +0.00106±75
    in-sample 2024-25        1398  proj -0.00406±112  -0.00281±131      -0.00285±146      -0.00326±65
                                   act  -0.00715±116  -0.00675±135      -0.00767±146      -0.00566±75
    (± in units of 1e-5)

Held out, neither the projected NOR the true starter helps the current
model; the in-sample gain is fit. Verdict: projection stays OFF
(compute_features_nhl.DEFAULT_GOALIE_MODE = "actual"); re-run after a
retrain on the restored goalie data.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("ab_nhl_projected_goalie")

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_cfn", _HERE / "compute_features_nhl.py")
assert _spec and _spec.loader
cfn = importlib.util.module_from_spec(_spec)
sys.modules["_cfn"] = cfn
_spec.loader.exec_module(cfn)

MODES = ("none", "projected", "actual")


def outcome_label(market: str, home: int, away: int, metadata: dict) -> str | None:
    """Realized label for a market, in the TaskSpec label vocabulary."""
    if market == "moneyline":
        return "home" if home > away else "away"
    if market == "regulation":
        rw = (metadata or {}).get("regulation_winner")
        return "tie" if rw == "draw" else rw
    if market == "puck_line":
        return "cover" if home - away >= 2 else "no_cover"
    if market == "total":
        return "over" if home + away >= 6 else "under"
    return None


def brier(probs: list[float], labels: list[str], y: str) -> float:
    return sum((p - (1.0 if lab == y else 0.0)) ** 2 for p, lab in zip(probs, labels))


def mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else float("nan"), float("nan"))
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="2025-11-15", help="first match_date in the evaluation window")
    ap.add_argument("--until", default=None)
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    args = ap.parse_args(argv)

    for p in ("/app/services/api/src", "/app/services/ml-models/src"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import numpy as np  # noqa: F401
    import pandas as pd
    from services.prediction_service import load_models_into_process, tasks_for_sport  # type: ignore

    models = load_models_into_process()
    tasks = [t for t in tasks_for_sport("nhl") if f"{t.sport}:{t.market}" in models]
    logger.info("tasks: %s", [t.market for t in tasks])

    with psycopg2.connect(args.database_url) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT m.id::text AS id, m.match_date, m.home_score, m.away_score, m.metadata,
                   ht.name AS home_team, at.name AS away_team
            FROM matches m
            JOIN leagues l ON l.id = m.league_id
            JOIN teams ht ON ht.id = m.home_team_id
            JOIN teams at ON at.id = m.away_team_id
            WHERE l.sport = 'nhl' AND m.status = 'finished'
              AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
              AND m.match_date >= %s AND (%s::timestamptz IS NULL OR m.match_date < %s::timestamptz)
              AND {cfn.NOT_PRESEASON_SQL}
            ORDER BY m.match_date
            """,
            (args.since, args.until, args.until),
        )
        games = cur.fetchall()
        logger.info("games in window: %d", len(games))

        scores: dict[str, dict[str, list[float]]] = {t.market: {mo: [] for mo in MODES} for t in tasks}
        goalie_known = {"projected": 0, "actual": 0}
        for i, g in enumerate(games, start=1):
            preds: dict[str, dict[str, list[float]]] = {}
            for mode in MODES:
                feats = cfn.compute_match_features(cur, g["id"], goalie_mode=mode)
                if feats is None:
                    break
                if mode != "none" and feats.get("home_goalie_roll_save_pct") is not None:
                    goalie_known[mode] += 1
                feats = cfn._with_defaults(feats)
                X: dict = {}
                for k, v in feats.items():
                    X[k] = v
                    X.setdefault(k if k.startswith("feature__") else f"feature__{k}", v)
                X["home_team"], X["away_team"] = g["home_team"], g["away_team"]
                frame = pd.DataFrame([X])
                for t in tasks:
                    proba = models[f"{t.sport}:{t.market}"].predict_proba(frame)[0]
                    preds.setdefault(t.market, {})[mode] = [float(x) for x in proba]
            else:
                for t in tasks:
                    y = outcome_label(t.market, g["home_score"], g["away_score"], g["metadata"])
                    if y not in t.labels:
                        continue
                    for mode in MODES:
                        scores[t.market][mode].append(brier(preds[t.market][mode], t.labels, y))
            if i % 200 == 0:
                logger.info("progress %d/%d", i, len(games))

    print(f"\nwindow {args.since}..{args.until or 'now'}  games={len(games)}  goalie-known rows: {goalie_known}")
    print(f"{'market':12s} {'n':>5s} {'brier_none':>10s} {'proj-none':>18s} {'actual-none':>18s}")
    for t in tasks:
        s = scores[t.market]
        n = len(s["none"])
        if n == 0:
            continue
        d_proj = [a - b for a, b in zip(s["projected"], s["none"])]
        d_act = [a - b for a, b in zip(s["actual"], s["none"])]
        mp, sp = mean_se(d_proj)
        ma, sa = mean_se(d_act)
        print(f"{t.market:12s} {n:5d} {sum(s['none']) / n:10.5f} {mp:+.5f} ± {sp:.5f} {ma:+.5f} ± {sa:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
