"""One-off A/B (2026-09-21): is the consensus's DE-VIG METHOD the source of
the favourite-longshot bias the model monitor flagged?

Trigger: monitor_models warned `horse_racing/win [market_consensus_v1]
MCE 0.181 [bucket 0.6-0.7, n=28: predicted 0.640, actual 0.821]`.
Diagnostics on the 365-day graded corpus (~190k entrants, both the
live-scored and the SP-backfilled halves) showed a MONOTONE reliability
curve: longshots (<0.15) over-priced by ~0.4pt, the 0.20-0.50 range
under-priced by 1-5pts, favourites (0.5+) under-priced by 2-10pts. That is
the textbook signature of proportional ("multiplicative") de-vigging on a
book whose overround is 10-58% (p50 by field size: 1.09 / 1.15 / 1.23 /
1.36): the method spreads the overround evenly across the field, but
bookmakers load it onto the longshots.

This harness recomputes the consensus FROM THE SAME ODDS under:

  * proportional — exactly what precompute_predictions_horse_racing.devig
                   ships (imported, not re-implemented)
  * power        — p_i = pi_i ** k, k solved per race so the field sums to 1
  * shin         — Shin (1993) insider-trading inversion, z solved per race

None of these fits anything on outcomes, so the "bias is non-stationary"
finding that closed the isotonic lever (audit §4, memory
horse-racing-baseline) does not apply: there is no train/test split to
overfit. All three are order-preserving within a race, so top-1 accuracy
must be identical (asserted).

Reports, per method:
  * Brier + log-loss, with the RACE-CLUSTERED SE of the paired delta vs
    proportional (entrants in one race are not independent)
  * reliability table on the monitor's buckets; ECE/MCE (monitor definition)
    on the full window and on the last 30 days the page was computed on
  * per-quarter stability of the delta
  * rec replay A — every SETTLED win rec at its recorded odds: would the
    rec still clear the engine's EV>=0.05 / prob>=0.10 / odds<12 / 5-7 runner
    gate under each method, and what did the kept vs dropped sets return
  * rec replay B — the engine's policy applied to the live corpus with the
    STARTING PRICE as the executable price, flat and quarter-Kelly stakes,
    race-cluster bootstrap CI on ROI

Decision rule (as ab_horse_racing_consensus_isotonic.py): ΔBrier <= -0.001
vs proportional AND clear of the clustered SE AND the replay must not give
back money the proportional policy makes.

Read-only. Run on prod via:
    docker compose exec -T -e ENABLE_TELEGRAM_NOTIFICATIONS=false api \\
        python /app/scripts/ab_horse_racing_devig.py --days 365
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from precompute_predictions_horse_racing import MIN_PRICED_ENTRANTS, _uniform_prob, devig  # noqa: E402

LOGGER = logging.getLogger("ab_horse_racing_devig")

CORPUS_QUERY = """
    SELECT DISTINCT ON (rp.race_id, rp.entrant_id)
           rp.race_id::text        AS race_id,
           rp.entrant_id::text     AS entrant_id,
           r.race_date             AS race_date,
           rp.confidence::float    AS served_prob,
           rp.actual_outcome::float AS actual,
           re.morning_line_odds::float AS morning_line_odds,
           re.starting_price::float    AS starting_price,
           (rp.created_at < r.race_date) AS live,
           COALESCE(re.scratched, false) AS scratched
    FROM race_predictions rp
    JOIN races r          ON r.id = rp.race_id
    JOIN race_entrants re ON re.id = rp.entrant_id
    WHERE rp.model_name = 'market_consensus_v1'
      AND rp.prediction_type = 'win'
      AND rp.actual_outcome IS NOT NULL
      AND rp.confidence IS NOT NULL
      AND r.status = 'finished'
      AND r.race_date >= NOW() - (%(days)s || ' days')::interval
    ORDER BY rp.race_id, rp.entrant_id, rp.created_at DESC
"""

RANKER_QUERY = """
    SELECT DISTINCT ON (rp.race_id, rp.entrant_id)
           rp.race_id::text AS race_id, rp.entrant_id::text AS entrant_id,
           rp.confidence::float AS ranker_prob
    FROM race_predictions rp
    JOIN races r ON r.id = rp.race_id
    WHERE rp.model_name = 'lightgbm_ranker_v1' AND rp.prediction_type = 'win'
      AND r.status = 'finished'
      AND r.race_date >= NOW() - (%(days)s || ' days')::interval
    ORDER BY rp.race_id, rp.entrant_id, rp.created_at DESC
"""

SETTLED_RECS_QUERY = """
    SELECT rr.race_id::text AS race_id, rr.entrant_id::text AS entrant_id,
           rr.odds_at_recommendation::float AS odds, rr.status,
           rr.recommended_stake::float AS stake, rr.profit_loss::float AS pl,
           rp.confidence::float AS rec_prob, rr.created_at
    FROM race_recommendations rr
    JOIN race_predictions rp ON rp.id = rr.race_prediction_id
    WHERE rr.bet_type = 'win' AND rr.status IN ('won', 'lost')
"""

# The engine's policy (generate_recommendations_horse_racing + rec_gating).
EV_THRESHOLD = 0.05
PROB_FLOOR = 0.10
MAX_ODDS = 12.0  # rec_gating: odds_above_max rejects odds >= max
MAX_EV = 1.0  # rec_gating: ev_above_max rejects ev >= max
SUPPRESSED_FIELD_SIZES = range(5, 8)
RANKER_TOP_N = 3
KELLY_FRACTION = 0.25
MAX_STAKE_FRACTION = 0.025

BUCKET_EDGES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 1.0001]
BUCKET_LABELS = ["00-05", "05-10", "10-15", "15-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70+"]
MIN_BUCKET_N_FOR_MCE = 20  # calibration_metrics.MIN_BUCKET_N_FOR_MCE


# ── de-vig methods ─────────────────────────────────────────────────────


def _signal_odds(row) -> float | None:
    ml = row["morning_line_odds"]
    if ml is not None and not (isinstance(ml, float) and math.isnan(ml)) and ml > 1.0:
        return float(ml)
    sp = row["starting_price"]
    if sp is not None and not (isinstance(sp, float) and math.isnan(sp)) and sp > 1.0:
        return float(sp)
    return None


def power_devig(pi: np.ndarray) -> np.ndarray:
    """p_i = pi_i ** k with k solved so the field sums to 1 (bisection)."""
    if pi.sum() <= 1.0:
        return pi / pi.sum()
    lo, hi = 0.05, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.power(pi, mid).sum() > 1.0:
            lo = mid
        else:
            hi = mid
    p = np.power(pi, 0.5 * (lo + hi))
    return p / p.sum()


def shin_devig(pi: np.ndarray) -> np.ndarray:
    """Shin (1993) inversion as implemented by Štrumbelj (2014)."""
    beta = pi.sum()
    if beta <= 1.0:
        return pi / beta

    def probs(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / beta) - z) / (2.0 * (1.0 - z))

    lo, hi = 0.0, 0.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if probs(mid).sum() > 1.0:
            lo = mid
        else:
            hi = mid
    p = probs(0.5 * (lo + hi))
    return p / p.sum()


def alt_devig(entrants: list[dict], method) -> dict[str, float]:
    """Same unpriced handling as the shipped devig(); only the priced-subset
    transform differs."""
    field_size = len(entrants)
    # sig_odds arrives as a float column, so an unpriced entrant is NaN (not
    # None) by the time it gets here; 1/NaN would poison the whole race.
    priced = [
        (e["entrant_id"], 1.0 / e["sig_odds"])
        for e in entrants
        if e["sig_odds"] is not None and not math.isnan(e["sig_odds"])
    ]
    if len(priced) < MIN_PRICED_ENTRANTS:
        u = _uniform_prob(field_size)
        return {e["entrant_id"]: u for e in entrants}
    pi = np.array([p for _, p in priced], dtype=float)
    p = method(pi)
    out = {eid: float(v) for (eid, _), v in zip(priced, p)}
    unpriced = [e["entrant_id"] for e in entrants if e["entrant_id"] not in out]
    if unpriced:
        u = _uniform_prob(field_size)
        for eid in unpriced:
            out[eid] = u
        s = sum(out.values())
        for eid in out:
            out[eid] /= s
    return out


def recompute(df: pd.DataFrame) -> pd.DataFrame:
    """Add prob_prop / prob_power / prob_shin columns, computed per race from
    the same signal odds the shipped devig() would read today."""
    df = df.copy()
    df["sig_odds"] = df.apply(_signal_odds, axis=1)
    prop, powr, shin = {}, {}, {}
    for race_id, g in df.groupby("race_id", sort=False):
        ents = [
            {
                "entrant_id": eid,
                "morning_line_odds": ml,
                "starting_price": sp,
                "sig_odds": so,
            }
            for eid, ml, sp, so in zip(g["entrant_id"], g["morning_line_odds"], g["starting_price"], g["sig_odds"])
        ]
        # The shipped function reads morning_line_odds / starting_price itself
        # (NaN-safe: pandas NaN > 1.0 is False, so it falls through like None).
        prop.update(devig(ents))
        powr.update(alt_devig(ents, power_devig))
        shin.update(alt_devig(ents, shin_devig))
    df["prob_prop"] = df["entrant_id"].map(prop)
    df["prob_power"] = df["entrant_id"].map(powr)
    df["prob_shin"] = df["entrant_id"].map(shin)
    # Race integrity: a NaN anywhere (a served row stored as float NaN, which
    # passes IS NOT NULL) drops the WHOLE race, so top-1 and the per-race
    # clustering stay well-defined. Report it — silent drops hide defects.
    prob_cols = ["served_prob", "prob_prop", "prob_power", "prob_shin"]
    bad = df[prob_cols].isna().any(axis=1)
    if bad.any():
        per_col = {c: int(df[c].isna().sum()) for c in prob_cols}
        bad_races = set(df.loc[bad, "race_id"])
        print(
            f"WARNING: {int(bad.sum())} rows with NaN probabilities {per_col} → dropping "
            f"{len(bad_races)} races ({int(df['race_id'].isin(bad_races).sum())} rows) to keep race integrity"
        )
        df = df[~df["race_id"].isin(bad_races)].reset_index(drop=True)
    return df


# ── metrics ────────────────────────────────────────────────────────────


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p - y) ** 2


def logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def clustered_delta(diff: np.ndarray, cluster: np.ndarray) -> tuple[float, float]:
    """Mean of per-row diff with a cluster-robust SE (ratio-estimator form)."""
    n = len(diff)
    mean = diff.sum() / n
    s = pd.DataFrame({"d": diff, "c": cluster}).groupby("c")["d"].agg(["sum", "count"])
    resid = s["sum"].to_numpy() - s["count"].to_numpy() * mean
    se = math.sqrt((resid**2).sum()) / n
    return float(mean), float(se)


@dataclass
class MethodStats:
    name: str
    brier: float
    logloss: float
    d_brier: float
    d_brier_se: float
    d_logloss: float
    d_logloss_se: float
    ece: float
    mce: float
    mce_bucket: str
    top1: float


def reliability(p: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    b = pd.cut(p, BUCKET_EDGES, labels=BUCKET_LABELS, right=False, include_lowest=True)
    t = (
        pd.DataFrame({"b": b, "p": p, "y": y})
        .groupby("b", observed=False)
        .agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
    )
    t["gap"] = t["actual"] - t["pred"]
    return t


def ece_mce(p: np.ndarray, y: np.ndarray) -> tuple[float, float, str]:
    """Monitor definition: 10 equal-width bins, ECE population-weighted, MCE
    is the worst |gap| over bins with n >= MIN_BUCKET_N_FOR_MCE."""
    idx = np.minimum((p * 10).astype(int), 9)
    n = len(p)
    ece, mce, worst = 0.0, 0.0, "-"
    for b in range(10):
        m = idx == b
        k = int(m.sum())
        if k == 0:
            continue
        gap = abs(y[m].mean() - p[m].mean())
        ece += gap * k / n
        if k >= MIN_BUCKET_N_FOR_MCE and gap > mce:
            mce, worst = gap, f"{b / 10:.1f}-{(b + 1) / 10:.1f} n={k} pred={p[m].mean():.3f} act={y[m].mean():.3f}"
    return ece, mce, worst


def top1(df: pd.DataFrame, col: str) -> float:
    d = df.dropna(subset=[col])
    idx = d.groupby("race_id")[col].idxmax()
    return float(d.loc[idx, "actual"].mean())


def method_table(df: pd.DataFrame, cols: dict[str, str]) -> list[MethodStats]:
    y = df["actual"].to_numpy(dtype=float)
    cl = df["race_id"].to_numpy()
    base_b = brier(df["prob_prop"].to_numpy(dtype=float), y)
    base_l = logloss(df["prob_prop"].to_numpy(dtype=float), y)
    out = []
    for name, col in cols.items():
        p = df[col].to_numpy(dtype=float)
        b, ll = brier(p, y), logloss(p, y)
        db, dbse = clustered_delta(b - base_b, cl)
        dl, dlse = clustered_delta(ll - base_l, cl)
        e, m, w = ece_mce(p, y)
        out.append(MethodStats(name, float(b.mean()), float(ll.mean()), db, dbse, dl, dlse, e, m, w, top1(df, col)))
    return out


def print_methods(title: str, rows: list[MethodStats]) -> None:
    print(f"\n=== {title} ===")
    print(
        f"{'method':<12}{'brier':>9}{'ΔBrier':>10}{'±SE':>9}{'logloss':>9}{'Δll':>9}{'±SE':>9}"
        f"{'ECE':>7}{'MCE':>7}  {'top1':>6}  worst bucket"
    )
    for r in rows:
        print(
            f"{r.name:<12}{r.brier:>9.5f}{r.d_brier:>+10.5f}{r.d_brier_se:>9.5f}{r.logloss:>9.5f}{r.d_logloss:>+9.5f}"
            f"{r.d_logloss_se:>9.5f}{r.ece:>7.4f}{r.mce:>7.4f}  {r.top1:>6.4f}  {r.mce_bucket}"
        )


# ── rec replays ────────────────────────────────────────────────────────


def runner_counts(df: pd.DataFrame) -> pd.Series:
    return df[~df["scratched"]].groupby("race_id").size()


def kelly_stake(prob: float, odds: float) -> float:
    k = (prob * odds - 1.0) / (odds - 1.0)
    return min(max(k, 0.0) * KELLY_FRACTION, MAX_STAKE_FRACTION)


def passes_policy(prob: float, odds: float, runners: int | None) -> bool:
    if prob < PROB_FLOOR or odds <= 1.0 or odds >= MAX_ODDS:
        return False
    ev = prob * odds - 1.0
    if ev < EV_THRESHOLD or ev >= MAX_EV:
        return False
    if runners is not None and runners in SUPPRESSED_FIELD_SIZES:
        return False
    return True


def replay_settled(recs: pd.DataFrame, df: pd.DataFrame) -> None:
    print("\n=== Rec replay A — settled WIN recs at their recorded odds, re-gated under each de-vig ===")
    runners = runner_counts(df)
    m = recs.merge(
        df[["race_id", "entrant_id", "prob_prop", "prob_power", "prob_shin"]], on=["race_id", "entrant_id"], how="inner"
    )
    if m.empty:
        print("no settled recs inside the corpus window")
        return
    m["runners"] = m["race_id"].map(runners)
    m["won"] = (m["status"] == "won").astype(int)
    m["flat_pl"] = np.where(m["won"] == 1, m["odds"] - 1.0, -1.0)
    print(
        f"settled win recs in window: {len(m)} (of {len(recs)} total), "
        f"actual P&L {m['pl'].sum():+.0f} on {m['stake'].sum():.0f} staked"
    )
    print(f"{'policy':<14}{'kept':>6}{'hit%':>7}{'flat ROI':>10}{'kelly ROI':>11}   dropped: n / hit% / flat ROI")
    for name, col in (
        ("served", "rec_prob"),
        ("proportional", "prob_prop"),
        ("power", "prob_power"),
        ("shin", "prob_shin"),
    ):
        keep = m.apply(lambda r: passes_policy(r[col], r["odds"], r["runners"]), axis=1)
        k, d = m[keep], m[~keep]
        ks = k.apply(lambda r: kelly_stake(r[col], r["odds"]), axis=1)
        kelly_roi = (ks * k["flat_pl"]).sum() / ks.sum() if ks.sum() > 0 else float("nan")
        print(
            f"{name:<14}{len(k):>6}{100 * k['won'].mean() if len(k) else 0:>7.1f}"
            f"{100 * k['flat_pl'].mean() if len(k) else 0:>+10.1f}{100 * kelly_roi:>+11.1f}   "
            f"{len(d)} / {100 * d['won'].mean() if len(d) else 0:.1f}% / "
            f"{100 * d['flat_pl'].mean() if len(d) else 0:+.1f}%"
        )
    print("\nodds band of the recs, mean prob under each method vs realised hit rate:")
    m["band"] = pd.cut(m["odds"], [1, 4, 6, 9, 12, 1000], labels=["<4", "4-6", "6-9", "9-12", "12+"], right=False)
    t = m.groupby("band", observed=False).agg(
        n=("won", "size"),
        hit=("won", "mean"),
        served=("rec_prob", "mean"),
        prop=("prob_prop", "mean"),
        power=("prob_power", "mean"),
        shin=("prob_shin", "mean"),
        flat_roi=("flat_pl", "mean"),
    )
    print(t.round(3).to_string())


def bootstrap_roi(
    pl: np.ndarray, stake: np.ndarray, cluster: np.ndarray, n_boot: int = 1000, seed: int = 7
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    g = pd.DataFrame({"pl": pl, "st": stake, "c": cluster}).groupby("c").sum()
    pls, sts = g["pl"].to_numpy(), g["st"].to_numpy()
    R = len(pls)
    rois = []
    for _ in range(n_boot):
        idx = rng.integers(0, R, R)
        s = sts[idx].sum()
        rois.append(pls[idx].sum() / s if s > 0 else 0.0)
    return float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))


def replay_at_sp(df: pd.DataFrame, ranker: pd.DataFrame) -> None:
    print("\n=== Rec replay B — engine policy on the LIVE corpus, executed at starting price ===")
    live = df[df["live"] & (df["starting_price"] > 1.0) & ~df["scratched"]].copy()
    if live.empty:
        print("no live rows with a starting price")
        return
    live = live.merge(ranker, on=["race_id", "entrant_id"], how="left")
    runners = runner_counts(df)
    live["runners"] = live["race_id"].map(runners)
    # Ranker top-N filter, applied exactly as the engine does: only in races
    # where the ranker scored, keep its top-N candidates.
    live["rank"] = live.groupby("race_id")["ranker_prob"].rank(ascending=False, method="first")
    has_ranker = live.groupby("race_id")["ranker_prob"].transform(lambda s: s.notna().any())
    eligible = (~has_ranker) | (live["rank"] <= RANKER_TOP_N)
    live = live[eligible]
    print(f"universe: {len(live)} entrants / {live['race_id'].nunique()} races (after ranker top-{RANKER_TOP_N})")
    print(
        f"{'policy':<14}{'recs':>6}{'races':>7}{'hit%':>7}{'flat ROI':>10}{'95% CI':>18}{'kelly ROI':>11}{'95% CI':>18}"
    )
    for name, col in (("proportional", "prob_prop"), ("power", "prob_power"), ("shin", "prob_shin")):
        sel = live[live.apply(lambda r: passes_policy(r[col], r["starting_price"], r["runners"]), axis=1)]
        if sel.empty:
            print(f"{name:<14}{0:>6}")
            continue
        flat_pl = np.where(sel["actual"] == 1, sel["starting_price"] - 1.0, -1.0)
        ks = sel.apply(lambda r: kelly_stake(r[col], r["starting_price"]), axis=1).to_numpy()
        cl = sel["race_id"].to_numpy()
        flo, fhi = bootstrap_roi(flat_pl, np.ones(len(sel)), cl)
        klo, khi = bootstrap_roi(ks * flat_pl, ks, cl)
        print(
            f"{name:<14}{len(sel):>6}{sel['race_id'].nunique():>7}{100 * sel['actual'].mean():>7.1f}"
            f"{100 * flat_pl.mean():>+10.1f}{f'[{100 * flo:+.1f}, {100 * fhi:+.1f}]':>18}"
            f"{100 * (ks * flat_pl).sum() / ks.sum():>+11.1f}{f'[{100 * klo:+.1f}, {100 * khi:+.1f}]':>18}"
        )
        band = pd.cut(sel["starting_price"], [1, 4, 6, 9, 12], labels=["<4", "4-6", "6-9", "9-12"], right=False)
        t = (
            pd.DataFrame({"band": band, "won": sel["actual"], "pl": flat_pl, "p": sel[col]})
            .groupby("band", observed=False)
            .agg(n=("won", "size"), hit=("won", "mean"), prob=("p", "mean"), flat_roi=("pl", "mean"))
        )
        print(t.round(3).to_string())


# ── main ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--json-out", default=None, help="write the headline numbers as JSON")
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    if not args.database_url:
        LOGGER.error("DATABASE_URL not set")
        return 2

    conn = psycopg2.connect(args.database_url, connect_timeout=10)
    conn.set_session(readonly=True, autocommit=True)
    try:
        df = pd.read_sql(CORPUS_QUERY, conn, params={"days": str(args.days)})
        ranker = pd.read_sql(RANKER_QUERY, conn, params={"days": str(args.days)})
        recs = pd.read_sql(SETTLED_RECS_QUERY, conn)
    finally:
        conn.close()
    LOGGER.info(
        "corpus: %d entrants / %d races; ranker rows %d; settled win recs %d",
        len(df),
        df["race_id"].nunique(),
        len(ranker),
        len(recs),
    )

    df = recompute(df)
    df["race_date"] = pd.to_datetime(df["race_date"], utc=True)

    # Replication of the shipped probabilities from today's odds columns.
    rep = (df["prob_prop"] - df["served_prob"]).abs()
    for label, mask in (
        ("all", np.ones(len(df), bool)),
        ("live", df["live"].to_numpy()),
        ("backfill", ~df["live"].to_numpy()),
    ):
        r = rep[mask]
        print(
            f"replication [{label:<8}] n={int(mask.sum()):>7}  mean|Δ|={r.mean():.5f}  "
            f"within 0.005: {100 * (r < 0.005).mean():.1f}%  within 0.02: {100 * (r < 0.02).mean():.1f}%"
        )
    print(
        "(live rows re-read morning_line_odds, which the racecard upsert overwrites on every pass — "
        "the served row was built from an EARLIER price. All method comparisons below use the same, current odds.)"
    )

    cols = {"served": "served_prob", "proportional": "prob_prop", "power": "prob_power", "shin": "prob_shin"}
    stats_all = method_table(df, cols)
    print_methods(f"Full window ({args.days}d): n={len(df)} entrants, {df['race_id'].nunique()} races", stats_all)
    assert (
        len({round(s.top1, 6) for s in stats_all if s.name != "served"}) == 1
    ), "order-preserving transforms must share top-1"

    for label, mask in (("live-scored", df["live"]), ("SP-backfilled", ~df["live"])):
        sub = df[mask]
        if len(sub):
            print_methods(f"{label}: n={len(sub)}", method_table(sub, cols))

    last30 = df[df["race_date"] >= df["race_date"].max() - pd.Timedelta(days=30)]
    print_methods(f"Last 30 days (the monitor's window): n={len(last30)}", method_table(last30, cols))

    print("\n=== Reliability, full window (actual − predicted per bucket) ===")
    tabs = {name: reliability(df[col].to_numpy(float), df["actual"].to_numpy(float)) for name, col in cols.items()}
    hdr = f"{'bucket':<8}{'n':>8}" + "".join(f"{name + ' pred':>14}{'gap':>8}" for name in cols)
    print(hdr)
    for b in BUCKET_LABELS:
        line = f"{b:<8}{int(tabs['proportional'].loc[b, 'n']):>8}"
        for name in cols:
            t = tabs[name]
            line += f"{t.loc[b, 'pred']:>14.3f}{t.loc[b, 'gap']:>+8.3f}" if t.loc[b, "n"] > 0 else f"{'-':>14}{'-':>8}"
        print(line)
    print("(n column is the proportional bucket count; each method re-buckets on its own probabilities)")

    print("\n=== Per-quarter stability of ΔBrier vs proportional (clustered SE) ===")
    df["quarter"] = df["race_date"].dt.to_period("Q").astype(str)
    y = df["actual"].to_numpy(float)
    base = brier(df["prob_prop"].to_numpy(float), y)
    for q, g in df.groupby("quarter"):
        idx = g.index.to_numpy()
        line = f"{q}  n={len(g):>6}"
        for name in ("power", "shin"):
            d, se = clustered_delta(
                brier(g[f"prob_{name}"].to_numpy(float), y[idx]) - base[idx], g["race_id"].to_numpy()
            )
            line += f"   {name}: {d:+.5f} ± {se:.5f}"
        print(line)

    replay_settled(recs, df)
    replay_at_sp(df, ranker)

    if args.json_out:
        payload = {
            s.name: {
                "brier": s.brier,
                "d_brier": s.d_brier,
                "d_brier_se": s.d_brier_se,
                "ece": s.ece,
                "mce": s.mce,
                "top1": s.top1,
            }
            for s in stats_all
        }
        with open(args.json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
