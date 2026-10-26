"""A/B: anchor the served soccer scoreline to the MARKET over/under 2.5.

WHAT IS BEING TESTED
--------------------
Production derives ~15 soccer markets from one Dixon-Coles scoreline matrix
reconciled to the ensemble 1x2 (``derive_from_lambdas(..., target_1x2=...)``
in scripts/precompute_predictions.py). The totals SHAPE of that matrix comes
only from the fitted team/league goal rates, and live (2026-07..10) the
derived over-2.5 scored +0.0108 +/- 0.0043 Brier WORSE than the closing
over/under market (n=1,302; audit §7 2026-10-25). The change under test adds a
second IPF constraint whenever a real market over/under-2.5 price exists:
``reconcile_matrix_to_1x2_and_totals`` (predictors/market_derivation.py)
matches the ensemble 1x2 AND the de-vigged market P(over 2.5) jointly, so every
other derived market moves consistently with that market input.

ARMS (identical folds, identical 1x2 target, identical matrix)
--------------------------------------------------------------
* ``served``   — reconcile to the 1x2 only (production today).
* ``anchored`` — reconcile to the 1x2 AND the market over/under 2.5.

Both arms start from the SAME Dixon-Coles matrix per match, so the paired
delta isolates the anchor. Two bases are scored:

* ``team``  — the current fitter (DIXON_COLES_CONFIG), refit MONTHLY
  walk-forward on matches strictly before the eval month (team-aware).
* ``blind`` — the same fit queried with unknown team names, i.e. each match
  gets its LEAGUE baseline lambdas. Live, the served artifact knows both teams
  in only a small share of fixtures (the open team-identity decision), so this
  stratum is the closer proxy for what production serves today.

The 1x2 target is the de-vigged market 1x2 — the ensemble is statistically
indistinguishable from it live (+0.0020 +/- 0.0016, n=1,754) and holding it
identical across arms keeps the comparison paired.

SCORING
-------
Per-match Brier as the repo defines it (sum of squared errors over a market's
selections; binary markets therefore score 2*(p-y)^2), realized from the final
score, for every derived market the anchor can move: over/under 0.5..5.5, BTTS,
team totals, correct score (full grid), total-goals bands, winning margin,
result+O/U, result+BTTS, odd/even, clean sheet, win to nil, and the integer /
half Asian handicap lines beyond +/-0.5. Markets fixed by the 1x2 (double
chance, draw no bet, AH +/-0.5) are scored as an invariance check. Standard
errors are CLUSTERED by eval month (every match in a month shares one fit).

Gate (§4.3): over-2.5 must improve beyond its noise floor and no derived market
may degrade beyond its noise floor (z > 2). Brier wins do not un-gate the
over_under recommendation stream — that needs realized ROI/CLV.

USAGE
-----
    python scripts/ab_soccer_ou_anchor.py --frame soccer_feat.parquet

``--frame`` is the engineered frame from scripts/ab_soccer_feature_screen.py
(or its raw export CSV): it needs match_date, league, home_team, away_team,
home_score, away_score, oh/od/oa (1x2 decimal odds) and o_over25/o_under25.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "services", "ml-models", "src"))

from predictors.market_derivation import (  # noqa: E402
    build_dc_matrix,
    reconcile_matrix_to_1x2,
    reconcile_matrix_to_1x2_and_totals,
    valid_ou_anchor,
)
from predictors.model_config import DIXON_COLES_CONFIG  # noqa: E402
from predictors.poisson_models import DixonColesPredictor  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("ab_soccer_ou_anchor")

REQUIRED = ["match_date", "league", "home_team", "away_team", "home_score", "away_score", "oh", "od", "oa"]
OU_LINES = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
TT_LINES = (0.5, 1.5, 2.5)
AH_SCORED = (-2.0, -1.5, -1.0, 1.0, 1.5, 2.0)
AH_INVARIANT = (-0.5, 0.5)
# Totals-anchor methods scored against the 1x2-only served arm.
ARMS = ("tilt", "step")
_N = 11  # build_dc_matrix(max_goals=MAX_GOALS_DERIVE=10)
OVER25_MASK = np.add.outer(np.arange(_N), np.arange(_N)) > 2.5


def load_frame(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED + ["o_over25", "o_under25"] if c not in df.columns]
    if missing:
        raise SystemExit(f"frame is missing columns: {missing}")
    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True, errors="coerce")
    for c in ["home_score", "away_score", "oh", "od", "oa", "o_over25", "o_under25"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["match_date", "home_score", "away_score"]).sort_values("match_date").reset_index(drop=True)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    # De-vigged market 1x2 (proportional) and over-2.5.
    inv = 1.0 / df[["oh", "od", "oa"]].where(df[["oh", "od", "oa"]] > 1.0)
    s = inv.sum(axis=1, min_count=3)
    df["t_home"], df["t_draw"], df["t_away"] = inv["oh"] / s, inv["od"] / s, inv["oa"] / s
    df.loc[~((s - 1.0).between(0.0, 0.35)), ["t_home", "t_draw", "t_away"]] = np.nan
    io = 1.0 / df["o_over25"].where(df["o_over25"] > 1.0)
    iu = 1.0 / df["o_under25"].where(df["o_under25"] > 1.0)
    df["p_over_mkt"] = io / (io + iu)
    df.loc[~((io + iu - 1.0).between(0.0, 0.25)), "p_over_mkt"] = np.nan
    return df


def _onehot_brier(p: np.ndarray, k: int) -> float:
    """Sum-of-squares Brier for a categorical prob vector and realized index k."""
    y = np.zeros_like(p)
    y[k] = 1.0
    return float(((p - y) ** 2).sum())


def score_matrix(P: np.ndarray, hs: int, as_: int) -> Dict[str, float]:
    """Per-market Brier of one reconciled matrix against a realized score."""
    N = P.shape[0]
    i, j = np.indices((N, N))
    tot, mar = i + j, i - j
    h = min(hs, N - 1)
    a = min(as_, N - 1)
    T, M = hs + as_, hs - as_
    out: Dict[str, float] = {}
    for line in OU_LINES:
        p = float(P[tot > line].sum())
        out[f"ou_{line}"] = 2.0 * (p - float(T > line)) ** 2
    p = float(P[(i >= 1) & (j >= 1)].sum())
    out["btts"] = 2.0 * (p - float(hs >= 1 and as_ >= 1)) ** 2
    ph, pa = P.sum(axis=1), P.sum(axis=0)
    k = np.arange(N)
    for line in TT_LINES:
        out[f"tt_home_{line}"] = 2.0 * (float(ph[k > line].sum()) - float(hs > line)) ** 2
        out[f"tt_away_{line}"] = 2.0 * (float(pa[k > line].sum()) - float(as_ > line)) ** 2
    # Correct score on the full grid: sum_cells (P - 1[actual])^2.
    out["correct_score"] = float((P**2).sum() - 2.0 * P[h, a] + 1.0)
    out["correct_score_logloss"] = float(-np.log(max(P[h, a], 1e-12)))
    bands = np.array([float(P[tot == t].sum()) for t in range(6)] + [float(P[tot >= 6].sum())])
    out["total_goals"] = _onehot_brier(bands, min(T, 6))
    wm = np.array(
        [
            float(P[mar == 1].sum()),
            float(P[mar == 2].sum()),
            float(P[mar >= 3].sum()),
            float(P[mar == 0].sum()),
            float(P[mar == -1].sum()),
            float(P[mar == -2].sum()),
            float(P[mar <= -3].sum()),
        ]
    )
    wm_k = {1: 0, 2: 1}.get(M, 2) if M > 0 else (3 if M == 0 else {-1: 4, -2: 5}.get(M, 6))
    out["winning_margin"] = _onehot_brier(wm, wm_k)
    regions = [i > j, i == j, i < j]
    res_k = 0 if M > 0 else (1 if M == 0 else 2)
    over = tot > 2.5
    both = (i >= 1) & (j >= 1)
    rou = np.array([float(P[r & c].sum()) for r in regions for c in (over, ~over)])
    out["result_over_under"] = _onehot_brier(rou, 2 * res_k + (0 if T > 2.5 else 1))
    rb = np.array([float(P[r & c].sum()) for r in regions for c in (both, ~both)])
    out["result_btts"] = _onehot_brier(rb, 2 * res_k + (0 if (hs >= 1 and as_ >= 1) else 1))
    out["odd_even"] = 2.0 * (float(P[(tot % 2) == 1].sum()) - float(T % 2 == 1)) ** 2
    out["clean_sheet_home"] = 2.0 * (float(pa[0]) - float(as_ == 0)) ** 2
    out["clean_sheet_away"] = 2.0 * (float(ph[0]) - float(hs == 0)) ** 2
    out["win_to_nil_home"] = 2.0 * (float(P[(i > j) & (j == 0)].sum()) - float(M > 0 and as_ == 0)) ** 2
    out["win_to_nil_away"] = 2.0 * (float(P[(i < j) & (i == 0)].sum()) - float(M < 0 and hs == 0)) ** 2
    for line in AH_SCORED + AH_INVARIANT:
        adj = mar + line
        v = np.array([float(P[adj > 0].sum()), float(P[adj < 0].sum()), float(P[adj == 0].sum())])
        r = M + line
        key = "ah_" + (f"{line:+g}") + ("_invariant" if line in AH_INVARIANT else "")
        out[key] = _onehot_brier(v, 0 if r > 0 else (1 if r < 0 else 2))
    dc = np.array([float(P[regions[0] | regions[1]].sum()), float(P[regions[1] | regions[2]].sum())])
    out["double_chance_invariant"] = float(((dc - np.array([float(M >= 0), float(M <= 0)])) ** 2).sum())
    return out


def run(
    df: pd.DataFrame,
    start: str,
    end: str | None,
    train_years: float,
) -> pd.DataFrame:
    months = pd.date_range(
        pd.Timestamp(start, tz="UTC"),
        (df["match_date"].max() if end is None else pd.Timestamp(end, tz="UTC")),
        freq="MS",
    )
    rows: List[dict] = []
    for m0 in months:
        m1 = m0 + pd.offsets.MonthBegin(1)
        ev = df[(df.match_date >= m0) & (df.match_date < m1)]
        ev = ev[ev[["t_home", "t_draw", "t_away"]].notna().all(axis=1) & ev["p_over_mkt"].map(valid_ou_anchor)]
        if ev.empty:
            continue
        tr = df[(df.match_date < m0) & (df.match_date >= m0 - pd.Timedelta(days=int(365.25 * train_years)))]
        t0 = time.time()
        dc = DixonColesPredictor(DIXON_COLES_CONFIG)
        dc.train(tr[["match_date", "league", "home_team", "away_team", "home_score", "away_score"]].copy())
        rho = float(dc.rho or 0.0)
        for r in ev.itertuples(index=False):
            target = (r.t_home, r.t_draw, r.t_away)
            for base in ("team", "blind"):
                if base == "team":
                    lam = dc.lambdas_for_match(r.home_team, r.away_team, league=r.league)
                else:
                    lam = dc.lambdas_for_match("__unknown_home__", "__unknown_away__", league=r.league)
                P = build_dc_matrix(lam[0], lam[1], rho)
                Ps = reconcile_matrix_to_1x2(P, target)
                arms = {"served": Ps}
                for method in ARMS:
                    arms[method] = reconcile_matrix_to_1x2_and_totals(Ps, target, r.p_over_mkt, method=method)
                row = {"month": m0.strftime("%Y-%m"), "base": base, "league": r.league, "p_over_mkt": r.p_over_mkt}
                row["over25"] = float(r.home_score + r.away_score > 2.5)
                for arm, M in arms.items():
                    row[f"p_over|{arm}"] = float(M[OVER25_MASK].sum())
                    for mk, v in score_matrix(M, r.home_score, r.away_score).items():
                        row[f"{arm}|{mk}"] = v
                rows.append(row)
        logger.warning(
            "%s: fit n=%d in %.1fs, eval n=%d, rho=%.3f", m0.strftime("%Y-%m"), len(tr), time.time() - t0, len(ev), rho
        )
    return pd.DataFrame(rows)


# ── LIVE served-path check (read-only, run inside the api container) ────────
#
# The walk-forward above refits a team-aware Dixon-Coles. Production serves the
# on-disk artifact, which knows both teams in only a minority of live fixtures,
# and reconciles to the ENSEMBLE 1x2 it actually stored. --export-live rebuilds
# exactly that object for every finished live match: the stored ensemble 1x2
# (newest pre-kickoff row), the production artifact's lambdas for the fixture's
# own team names + league_id, and the closing over/under-2.5 consensus from
# odds_snapshots (last pre-kickoff quote per book, proportional de-vig, >= 3
# books — the serve rule's floor). --live-csv then scores it locally.
LIVE_SQL = """
WITH pred AS (
  SELECT DISTINCT ON (p.match_id) p.match_id, p.probabilities
  FROM predictions p
  WHERE p.model_name = 'ensemble' AND p.prediction_type = 'match_result'
  ORDER BY p.match_id, p.created_at DESC
), last_q AS (
  SELECT DISTINCT ON (s.match_id, s.bookmaker, s.selection)
         s.match_id, s.bookmaker, s.selection, s.odds_decimal
  FROM odds_snapshots s JOIN matches m ON m.id = s.match_id
  WHERE s.market_type = 'over_under' AND s.line = 2.5 AND s.captured_at < m.match_date
    AND s.odds_decimal > 1.0
  ORDER BY s.match_id, s.bookmaker, s.selection, s.captured_at DESC
), per_book AS (
  SELECT o.match_id, o.bookmaker,
         (1.0 / o.odds_decimal) / ((1.0 / o.odds_decimal) + (1.0 / u.odds_decimal)) AS p_over
  FROM last_q o JOIN last_q u
    ON u.match_id = o.match_id AND u.bookmaker = o.bookmaker AND u.selection = 'under'
  WHERE o.selection = 'over'
), ou AS (
  SELECT match_id, AVG(p_over) AS p_over_mkt, COUNT(*) AS ou_books FROM per_book GROUP BY match_id
)
SELECT m.id::text AS match_id, m.match_date, m.league_id::text AS league_id, l.name AS league,
       ht.name AS home_team, at.name AS away_team, m.home_score, m.away_score,
       (pred.probabilities->>'home')::float AS t_home, (pred.probabilities->>'draw')::float AS t_draw,
       (pred.probabilities->>'away')::float AS t_away, ou.p_over_mkt, ou.ou_books
FROM matches m
JOIN leagues l ON l.id = m.league_id
JOIN teams ht ON ht.id = m.home_team_id
JOIN teams at ON at.id = m.away_team_id
JOIN pred ON pred.match_id = m.id
JOIN ou ON ou.match_id = m.id
WHERE l.sport = 'soccer' AND m.status = 'finished' AND m.home_score IS NOT NULL
  AND m.match_date >= %(since)s AND ou.ou_books >= 3
ORDER BY m.match_date
"""
LIVE_ARTIFACT = "/app/models/production/dixon_coles_soccer_match_result/1.0.0/model.bin"


def export_live(database_url: str, since: str, out_csv: str, artifact: str = LIVE_ARTIFACT) -> int:
    """Read-only: write the live served-path inputs (one row per match) to CSV."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    dc = DixonColesPredictor(DIXON_COLES_CONFIG)
    dc.load(artifact)
    with psycopg2.connect(database_url) as conn:
        conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(LIVE_SQL, {"since": since})
            rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        r["lam_h"], r["lam_a"] = dc.lambdas_for_match(r["home_team"], r["away_team"], league=r["league_id"])
        r["rho"] = float(dc.rho or 0.0)
        r["home_known"] = r["home_team"] in dc.team_attack
        r["away_known"] = r["away_team"] in dc.team_attack
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"exported {len(rows)} live matches -> {out_csv}")
    return len(rows)


def run_live(df: pd.DataFrame) -> pd.DataFrame:
    """Score the exported live rows: served (1x2-only) vs each anchor arm."""
    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True)
    rows: List[dict] = []
    for r in df.itertuples(index=False):
        if not valid_ou_anchor(r.p_over_mkt) or not np.isfinite([r.t_home, r.t_draw, r.t_away]).all():
            continue
        P = build_dc_matrix(r.lam_h, r.lam_a, r.rho)
        target = (r.t_home, r.t_draw, r.t_away)
        Ps = reconcile_matrix_to_1x2(P, target)
        arms = {"served": Ps}
        for method in ARMS:
            arms[method] = reconcile_matrix_to_1x2_and_totals(Ps, target, r.p_over_mkt, method=method)
        known = int(bool(r.home_known)) + int(bool(r.away_known))
        row = {
            "month": r.match_date.strftime("%Y-%m"),
            "base": f"live_known{known}" if known < 2 else "live_known2",
            "league": r.league,
            "p_over_mkt": r.p_over_mkt,
            "over25": float(r.home_score + r.away_score > 2.5),
        }
        for arm, M in arms.items():
            row[f"p_over|{arm}"] = float(M[OVER25_MASK].sum())
            for mk, v in score_matrix(M, int(r.home_score), int(r.away_score)).items():
                row[f"{arm}|{mk}"] = v
        rows.append(row)
    R = pd.DataFrame(rows)
    if not R.empty:
        allr = R.copy()
        allr["base"] = "live_all"
        R = pd.concat([allr, R], ignore_index=True)
    return R


def clustered(d: pd.Series, groups: pd.Series) -> tuple[float, float, float]:
    """Mean, iid SE and month-clustered SE of a paired delta."""
    n = len(d)
    mean = float(d.mean())
    iid = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    resid = (d - mean).groupby(groups.values).sum()
    g = len(resid)
    cl = float(np.sqrt((resid**2).sum() * g / max(g - 1, 1)) / n)
    return mean, iid, cl


def report(R: pd.DataFrame) -> pd.DataFrame:
    out = []
    markets = [c.split("|", 1)[1] for c in R.columns if c.startswith("served|")]
    for base, arm, s in [(b, a, R[R.base == b]) for b in R["base"].unique() for a in ARMS]:
        for mk in markets:
            served, anchored = s[f"served|{mk}"], s[f"{arm}|{mk}"]
            d = anchored - served
            mean, iid, cl = clustered(d, s["month"])
            yearly = d.groupby(s["month"].str[:4]).mean()
            out.append(
                dict(
                    base=base,
                    arm=arm,
                    market=mk,
                    n=len(s),
                    brier_served=round(float(served.mean()), 5),
                    brier_anchored=round(float(anchored.mean()), 5),
                    delta=round(mean, 5),
                    se_iid=round(iid, 5),
                    se_cluster=round(cl, 5),
                    z=round(mean / cl, 2) if cl > 0 and abs(mean) > 1e-9 else 0.0,
                    years_better=f"{int((yearly < 0).sum())}/{len(yearly)}",
                )
            )
    return pd.DataFrame(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", help="parquet/csv with scores, 1x2 odds and over/under-2.5 odds (walk-forward)")
    ap.add_argument("--export-live", metavar="CSV", help="read-only: export live served-path inputs (api container)")
    ap.add_argument("--live-since", default="2026-07-13", help="first live match date for --export-live")
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    ap.add_argument("--live-csv", help="score a --export-live CSV (served artifact + stored ensemble 1x2)")
    ap.add_argument("--start", default="2019-07-01", help="first eval month")
    ap.add_argument("--end", default=None, help="last eval month (default: end of frame)")
    ap.add_argument("--train-years", type=float, default=6.0, help="fit window before each eval month")
    ap.add_argument("--out", default=None, help="optional CSV for the per-market report")
    args = ap.parse_args(argv)

    if args.export_live:
        if not args.database_url:
            ap.error("--export-live needs --database-url or DATABASE_URL")
        export_live(args.database_url, args.live_since, args.export_live)
        return 0
    if args.live_csv:
        R = run_live(pd.read_csv(args.live_csv))
    elif args.frame:
        R = run(load_frame(args.frame), args.start, args.end, args.train_years)
    else:
        ap.error("one of --frame, --live-csv or --export-live is required")
    if R.empty:
        print("no eval rows")
        return 1
    rep = report(R)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    for base in R["base"].unique():
        for arm in ARMS:
            print(f"\n=== base={base} arm={arm}  (delta = {arm} - served; negative = anchor better)")
            print("    se_cluster = clustered by eval month; live has few months, read se_iid there")
            sub = rep[(rep.base == base) & (rep.arm == arm)].drop(columns=["base", "arm"])
            print(sub.to_string(index=False))
    # Sanity: the anchored over-2.5 IS the market price, and its Brier is the market's.
    for base, s in R.groupby("base"):
        mkt_b = float((2.0 * (s["p_over_mkt"] - s["over25"]) ** 2).mean())
        srv_b = float((2.0 * (s["p_over|served"] - s["over25"]) ** 2).mean())
        gaps = {arm: float((s[f"p_over|{arm}"] - s["p_over_mkt"]).abs().max()) for arm in ARMS}
        print(f"[{base}] max |arm - market| P(over 2.5) = {gaps}; Brier market {mkt_b:.5f} vs served {srv_b:.5f}")
    if args.out:
        rep.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
