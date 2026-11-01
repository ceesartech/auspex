"""Soccer asian-handicap (AH) replay: does a monthly refit of the full-time
Dixon-Coles change the AH recommendations, and is there a Brier case for it?

THE QUESTION (operator, 2026-11-01)
  "For the derived-market soccer model, is there a way we can make this
  streamlined for the asian handicap alone?" The derived-market gate
  (scripts/ab_soccer_ft_dixon_coles.py, audit §4 "Soccer full-time
  Dixon-Coles that serves the derived markets") was DO-NOT-SHIP: a monthly
  refit improved the served over_2.5 by only -0.00375 (live regime), short of
  the -0.005 house gate. asian_handicap is the one rec-emitting soccer market
  that artifact shapes, so the question is whether an AH-only deployment (a
  side artifact used for the AH lines only) could be justified on its own.

  This is a READ-ONLY analysis of the gate's own window. Choosing the AH
  market after the gate's per-unit results were read is POST HOC: nothing
  here can ship a model. A forward, pre-registered gate is the only route
  (scripts/ab_soccer_ah_forward_gate.py, a PROPOSAL).

WHAT THE ARTIFACT CAN MOVE ON AH
  precompute_predictions reconciles the Dixon-Coles scoreline to the served
  1x2 (one IPF pass, or the 1x2 + O/U-2.5 tilt when >= 3 fresh books quote
  the 2.5 line), then stores asian_handicap at 17 home-perspective lines as
  effective home / away / push masses (a quarter line splits the stake). After
  the IPF, AH 0, +-0.25 and +-0.5 are functions of the 1x2 alone, so the two
  arms serve them identically; the shaped lines +-0.75..+-2 depend on how the
  artifact spreads mass INSIDE the home-win / draw / away-win regions (e.g.
  P(margin >= 2 | home win)).

INPUTS
  --ftdc-dir   the --out-dir of a run of scripts/ab_soccer_ft_dixon_coles.py
               (per_match.csv.gz: per eval match the served 1x2 target, both
               anchors, every arm's lambdas and rho; scores.npz: every unit's
               per-row Brier). Every arm's AH probabilities are re-derived
               from them through market_derivation.derive_from_lambdas,
               exactly as precompute_predictions calls it.
  the database (a replica of a read-only prod export, or prod itself, read
  only): finished soccer matches, their pre-match AH odds (prematch_odds), the
  odds_snapshots price history, the stored ensemble predictions, the
  production AH recommendations and the bankroll.

PART A — BRIER (context, post hoc; live regime; refit - served)
  * every AH line (12 shaped + 5 invariant) as the gate scored it: the 3-way
    Brier over the effective home / away / push masses;
  * the pooled shaped lines: per match, the MEAN of the 12 shaped-line
    Briers (a mean, so the scale stays a single 3-way Brier; a SUM over 12
    correlated lines would inflate the delta twelvefold without adding
    information);
  * a goal-difference ranked probability score (RPS) over the 7 ordered
    margin classes <=-3, -2, -1, 0, +1, +2, >=+3 — exactly the information
    every AH line in +-2 reads — normalised by K-1 = 6;
  * the "shift ceiling": E[sum_k (p_refit,k - p_served,k)^2], which is the
    Brier gain refit would earn if its own probabilities were the truth. A
    -0.005 gain needs a root-mean-square shift of about 0.07 per class.
  Re-derived matrices must reproduce scores.npz's AH Briers to 1e-12 (the run
  aborts otherwise).

PART B — MONEY: the AH recommendation logic replayed per arm
  Population: every finished soccer match kicking off in [--start, --end)
  with at least one pre-match (prematch_odds) AH odds row. Recommendations
  only exist while the match is upcoming, and the generator deletes and
  re-writes a match's pending recs every 15-minute tick, so the set that
  stands is the one written at the LAST tick before kickoff. That tick is
  replayed: T = the last quarter-hour tick t with t + TICK_LAG_SECONDS before
  kickoff (production's AH recs since 2026-09-05 were written 12.8-14.1 min
  before kickoff on the hour, 3-4 min before kickoffs at :05).
  The odds state at T: a row exists when COALESCE(created_at, timestamp) <=
  T; its price at T is the last odds_snapshots price captured <= T (snapshots
  are written on every price change since 2026-07-13), else the row's own
  price when its timestamp <= T, else the row is dropped and counted; its age
  is T - COALESCE(last_seen_at, timestamp) when that is <= T, else 0 (it was
  still being quoted after T, counted). Then, per arm, exactly the soccer
  generator's path for market asian_handicap, through its own helpers
  (generate_recommendations.selection_value / kelly_fraction /
  display_selection / is_stale_odds, rec_gating.passes_gate / cap_stake):
    eligibility    both teams >= --min-team-history finished matches before T
    best price     per (selection, line): fresh first, then the highest price
                   (best_odds' DISTINCT ON order; ties broken by bookmaker)
    freshness      age <= rec_gating.MAX_ODDS_AGE_HOURS
    EV / floor     EV = win*O + push - 1 >= --ev-threshold, the no-push win
                   probability >= --prob-floor
    gate           ("soccer","asian_handicap"): odds < 4.0, EV <= 0.30,
                   model - market <= 0.10 with the market = the >= 3-book
                   de-vigged consensus at T (power de-vig; proportional before
                   POWER_DEVIG_SINCE in the reproduction check only)
    stake          quarter Kelly on the no-push conditional, capped at 2.5% of
                   the bankroll
  Settlement: grading_outcomes.grade_rec_selection (quarter lines and pushes
  included, 90-minute score) and rec_status_and_pl. Flat P/L stakes 1 unit.
  CLV: (i) same book: odds taken / the last pre-kickoff snapshot at the same
  book - 1 (vw_rec_clv's definition; near 0 by construction, since recs are
  re-priced up to the last tick); (ii) fair close: odds taken x the
  >= 3-book power-de-vigged consensus at kickoff - 1 (the no-push edge
  against the closing market).
  Arms:
    served           the gate's served arm: the 2026-11-01 artifact, the
                     current 1x2 incumbent's target, the live-replayed anchor
    refit            the gate's deciding refit (DIXON_COLES_CONFIG on every
                     finished match before the month), same target and anchor
    stored           production's stored asian_handicap probabilities: the
                     newest row (model_name 'ensemble') created before T
    refit_on_stored  refit reconciled to the stored 1x2 (the match_result row
                     of the stored row's model_version) and the stored anchor
                     (metadata.reconcile '1x2+ou2.5' -> market_over_2.5)
  Comparisons (PAIRS): served vs refit over the whole population (the
  window: today's generator logic applied to every match), and stored vs
  refit_on_stored where production stored predictions (the
  production-faithful pair; the forward gate's pair); both again from
  REPRO_START, where the stored arm is past the 2026-06-15..07-13
  constant-prior incident and production ran today's logic. served vs refit
  over the window is the headline readout. Nothing here decides a ship. Per
  arm: rec count, the overlap (same match and selection), the disagreement
  set (recs only one arm makes; the same side on a different line),
  stake-weighted and flat P/L, CLV. Uncertainty: match-clustered bootstrap
  (resampling matches, paired), and the per-match SE of the paired per-match
  difference.

PART C — REPRODUCTION
  The stored arm, replayed with the logic era's de-vig, against production's
  actual AH recs (betting_recommendations, every status) on matches kicking
  off from REPRO_START (the 2026-09-04 gate + freshness-guard deploy): the
  share of production recs reproduced with the same match, selection and line,
  and of those the same price, book and stake. The served arm is compared the
  same way (its gap is model-input drift: each week's 1x2 and Dixon-Coles
  artifact differ from the 2026-11-01 ones).

PART D — POWER
  From this window's per-match variances: the n (and the months at the
  current monthly volume) a forward AH-only gate needs to detect the measured
  effect with 80% power at two-sided 5%, for each Brier primary and for the
  money / CLV primaries.

THE RUN (fixed with this design, committed before the full run)
  --ftdc-dir = a re-run of the committed scripts/ab_soccer_ft_dixon_coles.py
  (4b8d768; --end 2026-10-31, --served-sha256 02141a6a..., --live-check-model-
  version served) on a local replica of the read-only REPEATABLE READ prod
  export of 2026-11-01 21:09:21Z (the soccer repair runbook had not started:
  no process, no state file), model files copied byte-identical (sha256
  checked); it reproduced the recorded gate exactly (n = 25,604, live
  -0.00375, every AH unit). This replay: --start 2026-05-01 --end
  2026-11-01 on the same replica. Before this commit the replay ran only as
  mechanics smokes (--mechanics, --limit 40 and 600): counts and fidelity
  lines, no P/L, CLV or Brier delta. (The 600-match smoke printed per-arm rec
  counts on its first 146 eligible matches, June-July 2026: served 8, refit
  9, stored 7, refit_on_stored 8.)

Read-only: SELECTs only. Usage (local replica):
    python scripts/ab_soccer_ah_recs_replay.py --ftdc-dir <ft gate --out-dir> \\
        --database-url postgresql://localhost/<replica> --end 2026-11-01 --out-json ah.json
Mechanics smoke (no deltas, no money numbers): add --mechanics --limit 40.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_soccer_ensemble_members as members  # noqa: E402
import ab_soccer_ft_dixon_coles as ftdc  # noqa: E402
import generate_recommendations as gr  # noqa: E402
import grading_outcomes as go  # noqa: E402
import rec_gating  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("ab_soccer_ah_recs_replay")

SPORT = "soccer"
MARKET = "asian_handicap"
TICK_MINUTES = 15  # auspex_pipeline runs */15
TICK_LAG_SECONDS = 60  # the soccer chain reaches generate_recommendations about a minute after the tick
# Commit 1c6e307 (2026-10-25T19:53:33-06:00): the market consensus switched to
# the power de-vig. Before it the gap cap read a proportional de-vig.
POWER_DEVIG_SINCE = pd.Timestamp("2026-10-26T01:53:33Z")
# Commits dc93233 / 2ec19b8 (2026-09-04): the per-stream gate, the AH caps
# and the freshness guard on the live quote. The reproduction check starts
# with the first full day after them.
REPRO_START = pd.Timestamp("2026-09-05T00:00:00Z")
SHAPED_LINES: Tuple[float, ...] = tuple(x for x in ftdc.AH_LINES if x not in ftdc.AH_INVARIANT_LINES)
SHAPED_UNITS: Tuple[str, ...] = tuple(f"ah_{ftdc.fmt_line(x)}" for x in SHAPED_LINES)
ALL_AH_UNITS: Tuple[str, ...] = tuple(f"ah_{ftdc.fmt_line(x)}" for x in ftdc.AH_LINES)
GD_CLASSES = ("<=-3", "-2", "-1", "0", "+1", "+2", ">=+3")
REGIME = "live"
ARMS = ("served", "refit", "stored", "refit_on_stored")
# (a, b, first kickoff or None): the window pair, the production-faithful
# pair, and both again over the current-logic era (from REPRO_START), where
# the stored arm is no longer the 2026-06-15..07-13 constant-prior incident.
PAIRS: Tuple[Tuple[str, str, Optional[pd.Timestamp]], ...] = (
    ("served", "refit", None),
    ("stored", "refit_on_stored", None),
    ("served", "refit", REPRO_START),
    ("stored", "refit_on_stored", REPRO_START),
)
REPRO_ARMS = ("stored", "served")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261101
RECONSTRUCT_ATOL = 1e-12
Z_ALPHA = 1.959963984540054  # two-sided 5%
Z_BETA = 0.8416212335729143  # 80% power

AH_ODDS_SQL = """
    SELECT
        o.match_id::text AS match_id,
        o.bookmaker,
        lower(o.selection) AS selection,
        o.line::float AS line,
        o.odds_decimal::float AS odds_decimal,
        o."timestamp" AS ts,
        o.last_seen_at,
        COALESCE(o.created_at, o."timestamp") AS created,
        o.is_live
    FROM odds o
    JOIN matches m ON m.id = o.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
    WHERE m.status = 'finished'
      AND m.match_date >= %(start)s
      AND m.match_date <  %(end)s
      AND o.market_type = 'asian_handicap'
"""

SNAPSHOTS_SQL = """
    SELECT s.match_id::text AS match_id, s.bookmaker, lower(s.selection) AS selection, s.line::float AS line,
           s.odds_decimal::float AS odds_decimal, s.captured_at
    FROM odds_snapshots s
    JOIN matches m ON m.id = s.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
    WHERE m.status = 'finished'
      AND m.match_date >= %(start)s
      AND m.match_date <  %(end)s
      AND s.market_type = 'asian_handicap'
"""

MATCHES_SQL = """
    SELECT m.id::text AS match_id, m.match_date, m.home_team_id::text AS home_team_id,
           m.away_team_id::text AS away_team_id, m.home_score, m.away_score, m.metadata,
           l.name AS league_name
    FROM matches m
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
    WHERE m.status = 'finished'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
      AND m.match_date >= %(start)s
      AND m.match_date <  %(end)s
"""

# generate_recommendations.list_upcoming counts every finished match of a team.
HISTORY_SQL = """
    SELECT m.home_team_id::text AS home_team_id, m.away_team_id::text AS away_team_id, m.match_date
    FROM matches m
    WHERE m.status = 'finished'
"""

STORED_SQL = """
    SELECT p.match_id::text AS match_id, p.model_version, p.prediction_type, p.probabilities, p.metadata,
           p.created_at
    FROM predictions p
    JOIN matches m ON m.id = p.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
    WHERE p.model_name = 'ensemble'
      AND p.prediction_type IN ('asian_handicap', 'match_result')
      AND m.status = 'finished'
      AND m.match_date >= %(start)s
      AND m.match_date <  %(end)s
"""

# The forward Brier gate's population: finished matches with a stored AH row.
STORED_AH_PER_MONTH_SQL = """
    SELECT to_char(date_trunc('month', m.match_date), 'YYYY-MM') AS month, count(DISTINCT p.match_id) AS n
    FROM predictions p
    JOIN matches m ON m.id = p.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
    WHERE p.model_name = 'ensemble'
      AND p.prediction_type = 'asian_handicap'
      AND m.status = 'finished'
      AND p.created_at < m.match_date
      AND m.match_date >= %(start)s
      AND m.match_date <  %(end)s
    GROUP BY 1
"""

RECS_SQL = """
    SELECT br.id::text AS rec_id, br.match_id::text AS match_id, br.selection,
           br.odds_at_recommendation::float AS odds_decimal, br.bookmaker,
           br.recommended_stake::float AS stake, br.expected_value::float AS ev, br.status, br.actual_result,
           br.profit_loss::float AS profit_loss, br.created_at
    FROM betting_recommendations br
    JOIN matches m ON m.id = br.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
    WHERE br.bet_type = 'asian_handicap'
      AND m.match_date >= %(start)s
      AND m.match_date <  %(end)s
"""


# ── Pure: the tick and the odds state ────────────────────────────────


def _ts(value: Any) -> Optional[pd.Timestamp]:
    return ftdc._ts(value)


def last_tick_before(
    kickoff: Any, tick_minutes: int = TICK_MINUTES, lag_seconds: int = TICK_LAG_SECONDS
) -> pd.Timestamp:
    """The moment the last generator run before kickoff executed: the latest
    quarter-hour tick t with t + lag strictly before kickoff, plus the lag
    (list_upcoming keeps a match only while match_date >= NOW())."""
    ko = _ts(kickoff)
    if ko is None:
        raise ValueError(f"kickoff {kickoff!r} is not a timestamp")
    step = pd.Timedelta(minutes=tick_minutes)
    lag = pd.Timedelta(seconds=lag_seconds)
    t = (ko - lag).floor(f"{tick_minutes}min")
    if t + lag >= ko:
        t -= step
    return t + lag


@dataclass(frozen=True)
class Quote:
    """One book's AH quote as it stood at a moment."""

    bookmaker: str
    selection: str  # 'home' / 'away'
    line: float  # the side's own signed line
    odds: float
    age_hours: Optional[float]
    moved_at: Optional[pd.Timestamp]


STATE_DROP_REASONS = ("live", "not_yet_listed", "price_unknown_at_tick", "seen_after_tick_kept_fresh")


def snapshot_index(
    snaps: Iterable[Dict[str, Any]],
) -> Dict[Tuple[str, str, str, float], List[Tuple[pd.Timestamp, float]]]:
    """{(match, book, selection, line): [(captured_at, price), ...] sorted by time}."""
    idx: Dict[Tuple[str, str, str, float], List[Tuple[pd.Timestamp, float]]] = {}
    for s in snaps:
        at = _ts(s["captured_at"])
        if at is None:
            raise ValueError(f"snapshot without captured_at: {s}")
        key = (str(s["match_id"]), str(s["bookmaker"]), str(s["selection"]).lower(), round(float(s["line"]), 4))
        idx.setdefault(key, []).append((at, float(s["odds_decimal"])))
    for v in idx.values():
        v.sort(key=lambda x: x[0])
    return idx


def quote_state(
    rows: Sequence[Dict[str, Any]],
    snaps: Dict[Tuple[str, str, str, float], List[Tuple[pd.Timestamp, float]]],
    at: Any,
    drops: Optional[Dict[str, int]] = None,
) -> List[Quote]:
    """Every AH quote of one match as it stood at `at` (see the docstring:
    exists when created <= at; price = the last snapshot <= at, else the row's
    own price if it last moved <= at; age from COALESCE(last_seen_at,
    timestamp), 0 when that is after `at`)."""
    t = _ts(at)
    if t is None:
        raise ValueError(f"{at!r} is not a timestamp")

    def drop(reason: str) -> None:
        if drops is not None:
            drops[reason] = drops.get(reason, 0) + 1

    out: List[Quote] = []
    for r in rows:
        if bool(r.get("is_live")):
            drop("live")
            continue
        created = _ts(r.get("created"))
        if created is None or created > t:
            drop("not_yet_listed")
            continue
        key = (str(r["match_id"]), str(r["bookmaker"]), str(r["selection"]).lower(), round(float(r["line"]), 4))
        hist = [h for h in snaps.get(key, []) if h[0] <= t]
        moved = _ts(r.get("ts"))
        if hist:
            moved_at, price = hist[-1]
        elif moved is not None and moved <= t:
            moved_at, price = moved, float(r["odds_decimal"])
        else:
            drop("price_unknown_at_tick")
            continue
        seen = _ts(r.get("last_seen_at")) or moved
        if seen is None:
            age = None
        elif seen > t:
            drop("seen_after_tick_kept_fresh")
            age = 0.0
        else:
            age = (t - seen).total_seconds() / 3600.0
        out.append(Quote(str(r["bookmaker"]), str(r["selection"]).lower(), float(r["line"]), price, age, moved_at))
    return out


def is_fresh(q: Quote, max_age_hours: float = rec_gating.MAX_ODDS_AGE_HOURS) -> bool:
    return q.age_hours is not None and q.age_hours <= max_age_hours


def best_quotes(state: Sequence[Quote], max_age_hours: float = rec_gating.MAX_ODDS_AGE_HOURS) -> List[Quote]:
    """generate_recommendations.best_odds on the state: per (selection, line)
    the fresh quotes first (unknown age last), then the highest price; ties go
    to the bookmaker that sorts first. Ordered like the SQL (selection, line)."""
    by_key: Dict[Tuple[str, float], List[Quote]] = {}
    for q in state:
        by_key.setdefault((q.selection, round(q.line, 4)), []).append(q)
    out = []
    for key in sorted(by_key):
        qs = sorted(by_key[key], key=lambda q: (0 if is_fresh(q, max_age_hours) else 1, -q.odds, q.bookmaker))
        out.append(qs[0])
    return out


def price_ties(state: Sequence[Quote], max_age_hours: float = rec_gating.MAX_ODDS_AGE_HOURS) -> int:
    """How many (selection, line) best prices are shared by two or more fresh books."""
    by_key: Dict[Tuple[str, float], List[float]] = {}
    for q in state:
        if is_fresh(q, max_age_hours):
            by_key.setdefault((q.selection, round(q.line, 4)), []).append(q.odds)
    return sum(1 for v in by_key.values() if len(v) > 1 and sorted(v)[-1] == sorted(v)[-2])


def consensus_prob(
    state: Sequence[Quote],
    selection: str,
    line: float,
    devig: Callable[[Dict[str, float], float], Optional[Dict[str, float]]] = rec_gating.devig_power,
    max_age_hours: float = rec_gating.MAX_ODDS_AGE_HOURS,
    min_books: int = rec_gating.MIN_CONSENSUS_BOOKS,
) -> Optional[float]:
    """rec_gating.market_consensus_prob for one AH selection on the state:
    fresh quotes of the selection at `line` and of the other side at -line,
    per book de-vigged (`devig`), averaged over >= min_books books. A test
    pins it to market_consensus_prob itself for the power de-vig."""
    sel = selection.strip().lower()
    by_book: Dict[str, Dict[str, float]] = {}
    for q in state:
        if not is_fresh(q, max_age_hours):
            continue
        wanted = line if q.selection == sel else -line
        if abs(q.line - wanted) > 1e-9:
            continue
        by_book.setdefault(q.bookmaker.strip().lower(), {})[q.selection] = q.odds
    vals = []
    for prices in by_book.values():
        if len(prices) < 2 or sel not in prices or any(p <= 0.0 for p in prices.values()):
            continue
        fair = devig(prices, 1.0)
        if fair is not None:
            vals.append(fair[sel])
    if len(vals) < min_books:
        return None
    return sum(vals) / len(vals)


# ── Pure: the generator's pricing path ───────────────────────────────


@dataclass
class Rec:
    match_id: str
    selection: str  # display selection, e.g. 'home_-0.75'
    side: str
    line: float
    odds: float
    bookmaker: str
    ev: float
    p_raw: float
    p_kelly: float
    market_prob: Optional[float]
    kelly_stake: float
    stake: float
    age_hours: Optional[float]
    extra: Dict[str, Any] = field(default_factory=dict)


def replay_match(
    match_id: str,
    probs: Dict[str, float],
    state: Sequence[Quote],
    bankroll: float,
    ev_threshold: float,
    prob_floor: float,
    devig: Callable[[Dict[str, float], float], Optional[Dict[str, float]]] = rec_gating.devig_power,
    max_age_hours: float = rec_gating.MAX_ODDS_AGE_HOURS,
) -> Tuple[List[Rec], Dict[str, int]]:
    """generate_recommendations.recommend_for_match for the asian_handicap
    rows of one match, on the replayed odds state, with the arm's stored-form
    AH probabilities. Returns (recs, suppressed by reason)."""
    suppressed: Dict[str, int] = {}
    recs: List[Rec] = []
    if not probs:
        return recs, suppressed
    for q in best_quotes(state, max_age_hours):
        if q.odds <= 1.0:
            continue
        if gr.is_stale_odds(q.age_hours, max_age_hours):
            suppressed[gr.STALE_ODDS_REASON] = suppressed.get(gr.STALE_ODDS_REASON, 0) + 1
            continue
        valued = gr.selection_value(MARKET, probs, q.selection, q.line, q.odds)
        if valued is None:
            continue
        p_kelly, ev, p_raw = valued
        if ev < ev_threshold or p_kelly < prob_floor:
            continue
        gate = rec_gating.gate_for(SPORT, MARKET)
        market_prob = None
        if gate.enabled and gate.max_gap is not None:
            market_prob = consensus_prob(state, q.selection, q.line, devig, max_age_hours)
        allowed, reason = rec_gating.passes_gate(
            SPORT, MARKET, odds=q.odds, ev=ev, model_prob=p_raw, market_prob=market_prob
        )
        if not allowed:
            key = reason if str(reason).startswith("stream_disabled") else f"{MARKET}:{reason}"
            suppressed[key] = suppressed.get(key, 0) + 1
            continue
        f = gr.kelly_fraction(p_kelly, q.odds)
        if f <= 0:
            continue
        recs.append(
            Rec(
                match_id=match_id,
                selection=gr.display_selection(MARKET, q.selection, q.line),
                side=q.selection,
                line=q.line,
                odds=q.odds,
                bookmaker=q.bookmaker,
                ev=round(ev, 4),
                p_raw=p_raw,
                p_kelly=p_kelly,
                market_prob=market_prob,
                kelly_stake=round(bankroll * f, 2),
                stake=rec_gating.cap_stake(bankroll * f * 0.25, bankroll),
                age_hours=q.age_hours,
            )
        )
    return recs, suppressed


def settle(selection: str, odds: float, stake: float, home: int, away: int, metadata: Optional[dict]):
    """(verdict, stake P/L, flat P/L) through grading_outcomes, or Nones when ungradable."""
    verdict = go.grade_rec_selection(
        MARKET, selection, int(home), int(away), sport=SPORT, model_name="ensemble", metadata=metadata or {}
    )
    if verdict is None:
        return None, None, None
    _, pl = go.rec_status_and_pl(verdict, stake, odds)
    _, flat = go.rec_status_and_pl(verdict, 1.0, odds)
    return verdict, float(pl), float(flat)


def closing_price(
    snaps: Dict[Tuple[str, str, str, float], List[Tuple[pd.Timestamp, float]]],
    rows: Sequence[Dict[str, Any]],
    match_id: str,
    bookmaker: str,
    selection: str,
    line: float,
    kickoff: Any,
) -> Optional[float]:
    """vw_rec_clv's closing price: the last snapshot strictly before kickoff at
    the same book; for a key with no snapshot, the row's own price when it
    last moved before kickoff."""
    ko = _ts(kickoff)
    key = (str(match_id), str(bookmaker), selection.lower(), round(float(line), 4))
    hist = [h for h in snaps.get(key, []) if ko is not None and h[0] < ko]
    if hist:
        return hist[-1][1]
    for r in rows:
        if (str(r["bookmaker"]), str(r["selection"]).lower(), round(float(r["line"]), 4)) == key[1:]:
            moved = _ts(r.get("ts"))
            if moved is not None and ko is not None and moved < ko and not snaps.get(key):
                return float(r["odds_decimal"])
    return None


# ── Pure: comparisons and statistics ─────────────────────────────────


def compare_recs(a: Sequence[Rec], b: Sequence[Rec]) -> Dict[str, Any]:
    """Overlap of two arms' rec sets on the same matches. same: identical
    (match, selection incl. line); moved_line: the same match and side on a
    different line (paired one-to-one per match and side); a_only / b_only:
    the rest."""
    ka = {(r.match_id, r.selection): r for r in a}
    kb = {(r.match_id, r.selection): r for r in b}
    same = sorted(set(ka) & set(kb))
    rest_a = [r for k, r in ka.items() if k not in kb]
    rest_b = [r for k, r in kb.items() if k not in ka]
    moved: List[Tuple[Rec, Rec]] = []
    pool_b: Dict[Tuple[str, str], List[Rec]] = {}
    for r in rest_b:
        pool_b.setdefault((r.match_id, r.side), []).append(r)
    a_only = []
    for r in rest_a:
        cand = pool_b.get((r.match_id, r.side))
        if cand:
            moved.append((r, cand.pop(0)))
        else:
            a_only.append(r)
    b_only = [r for v in pool_b.values() for r in v]
    return {"same": same, "moved_line": moved, "a_only": a_only, "b_only": b_only}


def money_summary(recs: Sequence[Rec]) -> Dict[str, Any]:
    """Counts, stake, stake-weighted and flat P/L and ROI, verdicts and CLV of
    settled recs (ungradable ones are counted, never scored)."""
    settled = [r for r in recs if r.extra.get("verdict") is not None]
    stake = sum(r.stake for r in settled)
    pl = sum(r.extra["pl"] for r in settled)
    flat = sum(r.extra["flat"] for r in settled)
    clv_s = [r.extra["clv_same"] for r in recs if r.extra.get("clv_same") is not None]
    clv_f = [r.extra["clv_fair"] for r in recs if r.extra.get("clv_fair") is not None]
    clv_f_w = [(r.extra["clv_fair"], r.stake) for r in recs if r.extra.get("clv_fair") is not None]
    verdicts: Dict[str, int] = {}
    for r in settled:
        verdicts[r.extra["verdict"]] = verdicts.get(r.extra["verdict"], 0) + 1
    return {
        "recs": len(recs),
        "settled": len(settled),
        "ungradable": len(recs) - len(settled),
        "matches": len({r.match_id for r in recs}),
        "stake": stake,
        "pl": pl,
        "roi_stake": pl / stake if stake > 0 else None,
        "flat_pl": flat,
        "roi_flat": flat / len(settled) if settled else None,
        "mean_odds": float(np.mean([r.odds for r in recs])) if recs else None,
        "mean_ev_claimed": float(np.mean([r.ev for r in recs])) if recs else None,
        "verdicts": verdicts,
        "clv_same_mean": float(np.mean(clv_s)) if clv_s else None,
        "clv_same_n": len(clv_s),
        "clv_fair_mean": float(np.mean(clv_f)) if clv_f else None,
        "clv_fair_stake_weighted": (
            float(sum(c * s for c, s in clv_f_w) / sum(s for _, s in clv_f_w))
            if clv_f_w and sum(s for _, s in clv_f_w) > 0
            else None
        ),
        "clv_fair_n": len(clv_f),
    }


def per_match_totals(recs: Sequence[Rec], match_ids: Sequence[str]) -> Dict[str, np.ndarray]:
    """Per match (aligned to match_ids): stake, P/L, flat count and P/L of settled
    recs, and the sum / count of fair-close CLV."""
    pos = {m: i for i, m in enumerate(match_ids)}
    n = len(match_ids)
    out = {k: np.zeros(n) for k in ("stake", "pl", "flat_n", "flat_pl", "clv_sum", "clv_n", "clv_w", "stake_clv")}
    for r in recs:
        i = pos.get(r.match_id)
        if i is None:
            raise KeyError(f"rec on match {r.match_id} outside the compared population")
        if r.extra.get("verdict") is not None:
            out["stake"][i] += r.stake
            out["pl"][i] += r.extra["pl"]
            out["flat_n"][i] += 1
            out["flat_pl"][i] += r.extra["flat"]
        if r.extra.get("clv_fair") is not None:
            out["clv_sum"][i] += r.extra["clv_fair"]
            out["clv_n"][i] += 1
            out["clv_w"][i] += r.extra["clv_fair"] * r.stake
            out["stake_clv"][i] += r.stake
    return out


def _ratio(num: np.ndarray, den: np.ndarray) -> float:
    d = float(den.sum())
    return float(num.sum()) / d if d > 0 else float("nan")


def paired_bootstrap(
    ta: Dict[str, np.ndarray], tb: Dict[str, np.ndarray], draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> Dict[str, Any]:
    """Match-clustered paired bootstrap of each arm's ROI (stake, flat), fair-CLV
    mean, and the differences b - a (ROI, total P/L per match). Returns the
    point estimate, the bootstrap SE and the 2.5 / 97.5 percentiles."""
    n = len(ta["pl"])
    if n == 0:
        return {}
    rng = np.random.default_rng(seed)
    stats: Dict[str, Callable[[np.ndarray], float]] = {
        "a_roi_stake": lambda ix: _ratio(ta["pl"][ix], ta["stake"][ix]),
        "b_roi_stake": lambda ix: _ratio(tb["pl"][ix], tb["stake"][ix]),
        "a_roi_flat": lambda ix: _ratio(ta["flat_pl"][ix], ta["flat_n"][ix]),
        "b_roi_flat": lambda ix: _ratio(tb["flat_pl"][ix], tb["flat_n"][ix]),
        "a_clv_fair": lambda ix: _ratio(ta["clv_sum"][ix], ta["clv_n"][ix]),
        "b_clv_fair": lambda ix: _ratio(tb["clv_sum"][ix], tb["clv_n"][ix]),
        "diff_roi_stake": lambda ix: _ratio(tb["pl"][ix], tb["stake"][ix]) - _ratio(ta["pl"][ix], ta["stake"][ix]),
        "diff_roi_flat": lambda ix: _ratio(tb["flat_pl"][ix], tb["flat_n"][ix])
        - _ratio(ta["flat_pl"][ix], ta["flat_n"][ix]),
        "diff_pl_total": lambda ix: float(tb["pl"][ix].sum() - ta["pl"][ix].sum()),
        "diff_flat_pl_total": lambda ix: float(tb["flat_pl"][ix].sum() - ta["flat_pl"][ix].sum()),
        "diff_clv_fair": lambda ix: _ratio(tb["clv_sum"][ix], tb["clv_n"][ix])
        - _ratio(ta["clv_sum"][ix], ta["clv_n"][ix]),
    }
    full = np.arange(n)
    point = {k: f(full) for k, f in stats.items()}
    boot = {k: np.empty(draws) for k in stats}
    for b in range(draws):
        ix = rng.integers(0, n, n)
        for k, f in stats.items():
            boot[k][b] = f(ix)
    out = {}
    for k in stats:
        v = boot[k][np.isfinite(boot[k])]
        out[k] = {
            "estimate": point[k],
            "se": float(v.std(ddof=1)) if v.size > 1 else None,
            "lo95": float(np.percentile(v, 2.5)) if v.size else None,
            "hi95": float(np.percentile(v, 97.5)) if v.size else None,
        }
    out["matches"] = n
    out["draws"] = draws
    return out


def per_match_diff(x: np.ndarray) -> Dict[str, Any]:
    """Mean, SD and per-match SE of a per-match paired difference."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else None, "sd": None, "se": None}
    sd = float(x.std(ddof=1))
    return {"n": n, "mean": float(x.mean()), "sd": sd, "se": sd / math.sqrt(n)}


def n_for_power(sd: Optional[float], delta: Optional[float], z_alpha: float = Z_ALPHA, z_beta: float = Z_BETA):
    """Units needed for a two-sided 5% test of a paired mean to have 80% power
    against `delta`, given the per-unit SD; None when undefined (delta 0)."""
    if sd is None or delta is None or not math.isfinite(sd) or not math.isfinite(delta) or delta == 0:
        return None
    return int(math.ceil(((z_alpha + z_beta) * sd / abs(delta)) ** 2))


# ── Pure: Brier readouts ─────────────────────────────────────────────


def gd_probabilities(P: np.ndarray) -> np.ndarray:
    """(m, 7) P(margin class) from served matrices (m, n, n), classes GD_CLASSES."""
    P = np.asarray(P, dtype=float)
    n = P.shape[1]
    i, j = np.indices((n, n))
    mar = i - j
    masks = [mar <= -3, mar == -2, mar == -1, mar == 0, mar == 1, mar == 2, mar >= 3]
    return np.stack([np.einsum("rij,ij->r", P, m.astype(float)) for m in masks], axis=1)


def gd_class(home: np.ndarray, away: np.ndarray) -> np.ndarray:
    mar = np.asarray(home, dtype=int) - np.asarray(away, dtype=int)
    return np.clip(mar, -3, 3) + 3


def rps_rows(p: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """Ranked probability score per row over ordered classes, / (K - 1)."""
    p = np.asarray(p, dtype=float)
    k = p.shape[1]
    y = np.zeros_like(p)
    y[np.arange(len(p)), np.asarray(cls, dtype=int)] = 1.0
    return np.sum((np.cumsum(p, axis=1)[:, :-1] - np.cumsum(y, axis=1)[:, :-1]) ** 2, axis=1) / (k - 1)


def handicap_probs(P: np.ndarray, line: float, weights: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """(m, 3) effective home / away / push masses of one AH line (the gate's readout)."""
    w = (weights or ftdc.unit_weights())[f"ah_{ftdc.fmt_line(line)}"]
    return np.einsum("rij,kij->rk", np.asarray(P, dtype=float), w)


# ── Data ─────────────────────────────────────────────────────────────


def load_ftdc(ftdc_dir: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    pm = pd.read_csv(os.path.join(ftdc_dir, "per_match.csv.gz"), low_memory=False)
    with np.load(os.path.join(ftdc_dir, "scores.npz"), allow_pickle=False) as z:
        scores = {k: z[k] for k in z.files}
    if list(scores["match_id"].astype(str)) != list(pm["match_id"].astype(str)):
        raise RuntimeError("per_match.csv.gz and scores.npz are not row-aligned")
    pm["match_date"] = pd.to_datetime(pm["match_date"], utc=True)
    return pm, scores


def load_db(database_url: str, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Any]:
    from utils.training_data import prematch_odds  # type: ignore

    params = {"start": start.to_pydatetime(), "end": end.to_pydatetime()}
    odds = members._read_sql(database_url, prematch_odds(AH_ODDS_SQL), params)
    snaps = members._read_sql(database_url, SNAPSHOTS_SQL, params)
    matches = members._read_sql(database_url, MATCHES_SQL, params)
    history = members._read_sql(database_url, HISTORY_SQL)
    stored = members._read_sql(database_url, STORED_SQL, params)
    recs = members._read_sql(database_url, RECS_SQL, params)
    per_month = members._read_sql(database_url, STORED_AH_PER_MONTH_SQL, params)
    import psycopg2
    from psycopg2.extras import RealDictCursor

    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            bankroll = gr.get_bankroll(cur)
    for df, col in ((matches, "match_date"), (history, "match_date"), (stored, "created_at"), (recs, "created_at")):
        df[col] = pd.to_datetime(df[col], utc=True)
    return {
        "odds": odds,
        "snaps": snaps,
        "matches": matches,
        "history": history,
        "stored": stored,
        "recs": recs,
        "bankroll": float(bankroll),
        "stored_ah_per_month": {str(k): int(v) for k, v in zip(per_month["month"], per_month["n"])},
    }


class TeamHistory:
    """Finished-match counts per team strictly before a moment (list_upcoming's
    team_history at the tick)."""

    def __init__(self, history: pd.DataFrame):
        dates: Dict[str, List[pd.Timestamp]] = {}
        for h, a, d in zip(history["home_team_id"], history["away_team_id"], history["match_date"]):
            dates.setdefault(str(h), []).append(d)
            dates.setdefault(str(a), []).append(d)
        self._dates = {k: np.sort(np.asarray([x.value for x in v], dtype=np.int64)) for k, v in dates.items()}

    def count_before(self, team_id: str, at: pd.Timestamp) -> int:
        arr = self._dates.get(str(team_id))
        if arr is None:
            return 0
        return int(np.searchsorted(arr, at.value, side="left"))


def stored_inputs(stored: Optional[pd.DataFrame], match_id: str, before: pd.Timestamp) -> Optional[Dict[str, Any]]:
    """The newest stored asian_handicap row of `match_id` created before
    `before`, with the match_result row of the same model_version and the
    stored anchor; None when there is no such AH row. Raises on an AH row with
    no 1x2 sibling (precompute writes both in one transaction)."""
    if stored is None or stored.empty:
        return None
    rows = stored[(stored["match_id"] == match_id) & (stored["created_at"] < before)]
    ah = rows[rows["prediction_type"] == MARKET]
    if ah.empty:
        return None
    row = ah.sort_values("created_at").iloc[-1]
    mr = rows[(rows["prediction_type"] == "match_result") & (rows["model_version"] == row["model_version"])]
    if mr.empty:
        raise RuntimeError(f"stored AH row of {match_id} ({row['model_version']}) has no match_result sibling")
    meta = row["metadata"] or {}
    probs1x2 = mr.iloc[-1]["probabilities"] or {}
    anchor = float(meta["market_over_2.5"]) if meta.get("reconcile") == "1x2+ou2.5" else None
    return {
        "model_version": row["model_version"],
        "probabilities": {str(k): float(v) for k, v in (row["probabilities"] or {}).items()},
        "target": tuple(float(probs1x2[k]) for k in ("home", "draw", "away")),
        "anchor": anchor,
        "reconcile": meta.get("reconcile"),
    }


def derive_ah(
    h_lam: float, a_lam: float, rho: float, target: Sequence[float], anchor: Optional[float]
) -> Dict[str, float]:
    """The stored-form asian_handicap dict precompute_predictions would write."""
    from predictors.market_derivation import derive_from_lambdas  # type: ignore

    mk = derive_from_lambdas(
        float(h_lam),
        float(a_lam),
        float(rho),
        target_1x2=tuple(float(x) for x in target),
        target_over_25=anchor,
    )
    return {str(k): float(v) for k, v in mk[MARKET].items()}


def _anchor(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ── Part A: Brier ────────────────────────────────────────────────────


def brier_part(pm: pd.DataFrame, scores: Dict[str, np.ndarray], mechanics: bool, limit: int = 0) -> Dict[str, Any]:
    """AH Brier readouts (refit - served, live regime) from the gate's outputs,
    plus the GD RPS and the shift ceiling from re-derived matrices."""
    folds = np.asarray(scores["fold"]).astype(str)
    idx = np.arange(len(pm)) if not limit else np.arange(min(limit, len(pm)))
    weights = ftdc.unit_weights()
    mats: Dict[str, np.ndarray] = {}
    for arm in ("served", "refit"):
        lams = pm[[f"lam_h::{arm}", f"lam_a::{arm}"]].to_numpy(dtype=float)[idx]
        rho = pm[f"rho::{arm}"].to_numpy(dtype=float)[idx]
        tgt = pm[["target::home", "target::draw", "target::away"]].to_numpy(dtype=float)[idx]
        anc = pm["anchor_live"].to_numpy(dtype=float)[idx]
        mats[arm] = np.stack(
            [
                ftdc.served_matrix(
                    lams[r, 0], lams[r, 1], rho[r], tgt[r], None if not math.isfinite(anc[r]) else anc[r]
                )
                for r in range(len(idx))
            ]
        )
    h = pm["s_home"].to_numpy(dtype=int)[idx]
    a = pm["s_away"].to_numpy(dtype=int)[idx]
    truth = ftdc.vector_truth(h, a)
    worst = 0.0
    for unit in ALL_AH_UNITS:
        for arm in ("served", "refit"):
            p = np.einsum("rij,kij->rk", mats[arm], weights[unit])
            mine = ftdc.brier_rows(p, truth[unit])
            ref = scores[f"brier::{arm}::{REGIME}::{unit}"][idx]
            gap = float(np.max(np.abs(mine - ref))) if len(idx) else 0.0
            if not math.isfinite(gap) or gap > RECONSTRUCT_ATOL:
                raise RuntimeError(f"re-derived {arm} {unit} Brier differs from scores.npz by {gap:.2e}")
            worst = max(worst, gap)
    out: Dict[str, Any] = {"n": int(len(idx)), "reconstruct_max_abs": worst}
    if mechanics:
        return out
    f = folds[idx]

    def b(arm: str, unit: str) -> np.ndarray:
        return scores[f"brier::{arm}::{REGIME}::{unit}"][idx]

    out["units"] = {u: ftdc.pair(b("refit", u), b("served", u), f) for u in ALL_AH_UNITS}
    pooled_s = np.mean(np.stack([b("served", u) for u in SHAPED_UNITS]), axis=0)
    pooled_r = np.mean(np.stack([b("refit", u) for u in SHAPED_UNITS]), axis=0)
    out["pooled_shaped_mean"] = ftdc.pair(pooled_r, pooled_s, f)
    out["pooled_shaped_mean"]["sd_per_match"] = per_match_diff(pooled_r - pooled_s)["sd"]
    cls = gd_class(h, a)
    gd = {arm: gd_probabilities(mats[arm]) for arm in mats}
    rps = {arm: rps_rows(gd[arm], cls) for arm in gd}
    out["gd_rps"] = ftdc.pair(rps["refit"], rps["served"], f)
    out["gd_rps"]["sd_per_match"] = per_match_diff(rps["refit"] - rps["served"])["sd"]
    out["gd_rps"]["served"] = float(rps["served"].mean())
    out["gd_brier_7way"] = ftdc.pair(
        ftdc.brier_rows(gd["refit"], np.eye(7)[cls]), ftdc.brier_rows(gd["served"], np.eye(7)[cls]), f
    )
    shift = {}
    for u in ALL_AH_UNITS:
        ps = np.einsum("rij,kij->rk", mats["served"], weights[u])
        pr = np.einsum("rij,kij->rk", mats["refit"], weights[u])
        shift[u] = {
            "mean_sq_shift": float(np.mean(np.sum((pr - ps) ** 2, axis=1))),
            "rms_per_class": float(np.sqrt(np.mean((pr - ps) ** 2))),
            "max_abs": float(np.max(np.abs(pr - ps))),
        }
    shift["gd_7way"] = {
        "mean_sq_shift": float(np.mean(np.sum((gd["refit"] - gd["served"]) ** 2, axis=1))),
        "rms_per_class": float(np.sqrt(np.mean((gd["refit"] - gd["served"]) ** 2))),
        "max_abs": float(np.max(np.abs(gd["refit"] - gd["served"]))),
    }
    out["shift_ceiling"] = shift
    best = min(SHAPED_UNITS, key=lambda u: out["units"][u]["delta"])
    out["best_single_line"] = {"unit": best, **out["units"][best]}
    out["best_single_line"]["sd_per_match"] = per_match_diff(b("refit", best) - b("served", best))["sd"]
    out["verdicts"] = {
        name: ftdc.ship_strict(p)
        for name, p in (
            ("best_single_line", out["best_single_line"]),
            ("pooled_shaped_mean", out["pooled_shaped_mean"]),
            ("gd_rps", out["gd_rps"]),
        )
    }
    return out


# ── Part B / C: the replay ───────────────────────────────────────────


def run(args: argparse.Namespace) -> Dict[str, Any]:
    members._add_src_paths()
    pm, scores = load_ftdc(args.ftdc_dir)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    db = load_db(args.database_url, start, end)
    report: Dict[str, Any] = {
        "window": {"start": str(start), "end": str(end)},
        "mechanics": bool(args.mechanics),
        "bankroll": db["bankroll"],
        "generator": {
            "ev_threshold": args.ev_threshold,
            "prob_floor": args.prob_floor,
            "min_team_history": args.min_team_history,
            "max_odds_age_hours": rec_gating.MAX_ODDS_AGE_HOURS,
            "gate": dict(rec_gating.gate_for(SPORT, MARKET).__dict__),
            "max_stake_fraction": rec_gating.MAX_STAKE_FRACTION,
            "min_consensus_books": rec_gating.MIN_CONSENSUS_BOOKS,
        },
        "stored_ah_matches_per_month": db["stored_ah_per_month"],
    }
    report["brier"] = brier_part(pm, scores, args.mechanics, args.limit)

    odds = db["odds"]
    report["ah_odds"] = {
        "rows": int(len(odds)),
        "matches": int(odds["match_id"].nunique()) if len(odds) else 0,
        "first_created": str(pd.to_datetime(odds["created"], utc=True).min()) if len(odds) else None,
        "books": sorted(odds["bookmaker"].unique().tolist()) if len(odds) else [],
        "snapshots": int(len(db["snaps"])),
    }
    rows_by_match: Dict[str, List[Dict[str, Any]]] = {}
    for rec in odds.to_dict("records"):
        rows_by_match.setdefault(str(rec["match_id"]), []).append(rec)
    snaps = snapshot_index(db["snaps"].to_dict("records"))
    matches = db["matches"].set_index("match_id")
    pm_by = pm.set_index("match_id")
    hist = TeamHistory(db["history"])
    stored_by = {str(k): g for k, g in db["stored"].groupby("match_id")}
    cand = sorted((m for m in rows_by_match if m in matches.index), key=lambda m: (matches.loc[m, "match_date"], m))
    if args.limit:
        cand = cand[: int(args.limit)]
    bankroll = db["bankroll"]

    state_drops: Dict[str, int] = {k: 0 for k in STATE_DROP_REASONS}
    recs: Dict[str, List[Rec]] = {arm: [] for arm in ARMS}
    repro_recs: List[Rec] = []
    suppressed: Dict[str, Dict[str, int]] = {arm: {} for arm in ARMS}
    defined: Dict[str, List[str]] = {arm: [] for arm in ARMS}
    status: Dict[str, str] = {}
    counts: Dict[str, Any] = {
        "candidates": len(cand),
        "ineligible": 0,
        "not_in_gate_eval": 0,
        "no_stored": 0,
        "price_ties": 0,
    }
    stored_vs_served_ah: List[float] = []
    shaped_keys = [f"{ftdc.fmt_line(x)}_{s}" for x in SHAPED_LINES for s in ("home", "away", "push")]
    for mid in cand:
        mrow = matches.loc[mid]
        ko = mrow["match_date"]
        tick = last_tick_before(ko)
        if (
            hist.count_before(mrow["home_team_id"], tick) < args.min_team_history
            or hist.count_before(mrow["away_team_id"], tick) < args.min_team_history
        ):
            counts["ineligible"] += 1
            status[mid] = "ineligible"
            continue
        status[mid] = "eligible"
        rows = rows_by_match[mid]
        state = quote_state(rows, snaps, tick, state_drops)
        close_state = quote_state(rows, snaps, ko - pd.Timedelta(microseconds=1))
        counts["price_ties"] += price_ties(state)
        meta = mrow["metadata"] if isinstance(mrow["metadata"], dict) else {}
        arm_probs: Dict[str, Dict[str, float]] = {}
        g = pm_by.loc[mid] if mid in pm_by.index else None
        if g is None:
            counts["not_in_gate_eval"] += 1
        else:
            tgt = (float(g["target::home"]), float(g["target::draw"]), float(g["target::away"]))
            for arm in ("served", "refit"):
                arm_probs[arm] = derive_ah(
                    g[f"lam_h::{arm}"], g[f"lam_a::{arm}"], g[f"rho::{arm}"], tgt, _anchor(g["anchor_live"])
                )
        st = stored_inputs(stored_by.get(mid), mid, tick)
        if st is None:
            counts["no_stored"] += 1
        else:
            arm_probs["stored"] = st["probabilities"]
            if g is not None:
                arm_probs["refit_on_stored"] = derive_ah(
                    g["lam_h::refit"], g["lam_a::refit"], g["rho::refit"], st["target"], st["anchor"]
                )
                gaps = [abs(st["probabilities"][k] - arm_probs["served"][k]) for k in shaped_keys]
                stored_vs_served_ah.append(max(gaps))
            if ko >= REPRO_START:
                devig = rec_gating.devig_power if tick >= POWER_DEVIG_SINCE else rec_gating.devig_proportional
                rr, _ = replay_match(
                    mid, st["probabilities"], state, bankroll, args.ev_threshold, args.prob_floor, devig
                )
                repro_recs.extend(rr)
        for arm, probs in arm_probs.items():
            defined[arm].append(mid)
            rr, sup = replay_match(mid, probs, state, bankroll, args.ev_threshold, args.prob_floor)
            for k, v in sup.items():
                suppressed[arm][k] = suppressed[arm].get(k, 0) + v
            for r in rr:
                verdict, pl, flat = settle(r.selection, r.odds, r.stake, mrow["home_score"], mrow["away_score"], meta)
                close = closing_price(snaps, rows, mid, r.bookmaker, r.side, r.line, ko)
                p_close = consensus_prob(close_state, r.side, r.line)
                r.extra.update(
                    {
                        "kickoff": str(ko),
                        "tick": str(tick),
                        "verdict": verdict,
                        "pl": pl,
                        "flat": flat,
                        "clv_same": (r.odds / close - 1.0) if close else None,
                        "clv_fair": (r.odds * p_close - 1.0) if p_close is not None else None,
                        "p_close": p_close,
                    }
                )
            recs[arm].extend(rr)
    counts["state_drops"] = state_drops
    counts["defined"] = {arm: len(v) for arm, v in defined.items()}
    counts["stored_vs_served_max_abs_shaped"] = (
        {
            "n": len(stored_vs_served_ah),
            "median": float(np.median(stored_vs_served_ah)),
            "p90": float(np.percentile(stored_vs_served_ah, 90)),
        }
        if stored_vs_served_ah
        else None
    )
    report["replay"] = counts
    report["suppressed"] = suppressed
    by_month: Dict[str, int] = {}
    for mid, st_ in status.items():
        if st_ == "eligible":
            key = matches.loc[mid, "match_date"].strftime("%Y-%m")
            by_month[key] = by_month.get(key, 0) + 1
    report["eligible_ah_matches_by_month"] = dict(sorted(by_month.items()))
    recent = [v for k, v in by_month.items() if "2026-08" <= k <= "2026-10"]
    report["ah_odds_matches_per_month"] = float(np.mean(recent)) if recent else None
    if args.mechanics:
        report["rec_counts"] = {arm: len(v) for arm, v in recs.items()}
        return report
    if args.out_csv:
        write_recs_csv(args.out_csv, recs)

    report["money"] = {arm: money_summary(v) for arm, v in recs.items()}
    report["pairs"] = {
        pair_name(a, b, since): compare_pair(recs, defined, matches, a, b, since, args.bootstrap_draws)
        for a, b, since in PAIRS
    }
    report["reproduction"] = reproduction(db["recs"], repro_recs, recs["served"], defined, status, matches)
    report["power"] = power(report)
    return report


def pair_name(a: str, b: str, since: Optional[pd.Timestamp]) -> str:
    return f"{a} vs {b}" + (f" from {since.date()}" if since is not None else "")


def compare_pair(
    recs: Dict[str, List[Rec]],
    defined: Dict[str, List[str]],
    matches: pd.DataFrame,
    a: str,
    b: str,
    since: Optional[pd.Timestamp],
    draws: int,
) -> Dict[str, Any]:
    """Arms a and b on the matches both are defined for (kicking off from
    `since`): overlap, disagreement set, money, paired bootstrap."""
    common = sorted(set(defined[a]) & set(defined[b]))
    if since is not None:
        common = [m for m in common if matches.loc[m, "match_date"] >= since]
    keep = set(common)
    ra = [r for r in recs[a] if r.match_id in keep]
    rb = [r for r in recs[b] if r.match_id in keep]
    cmp_ = compare_recs(ra, rb)
    ka = {(r.match_id, r.selection): r for r in ra}
    kb = {(r.match_id, r.selection): r for r in rb}
    ta, tb = per_match_totals(ra, common), per_match_totals(rb, common)
    return {
        "matches": len(common),
        "matches_with_any_rec": int(((ta["flat_n"] + tb["flat_n"]) > 0).sum()),
        "a": money_summary(ra),
        "b": money_summary(rb),
        "same": len(cmp_["same"]),
        "same_price_and_stake": sum(
            1 for k in cmp_["same"] if ka[k].odds == kb[k].odds and abs(ka[k].stake - kb[k].stake) < 0.005
        ),
        "moved_line": len(cmp_["moved_line"]),
        "a_only": money_summary(cmp_["a_only"]),
        "b_only": money_summary(cmp_["b_only"]),
        "moved_line_a": money_summary([x for x, _ in cmp_["moved_line"]]),
        "moved_line_b": money_summary([y for _, y in cmp_["moved_line"]]),
        "bootstrap": paired_bootstrap(ta, tb, draws),
        "per_match_pl_diff": per_match_diff(tb["pl"] - ta["pl"]),
        "per_match_flat_diff": per_match_diff(tb["flat_pl"] - ta["flat_pl"]),
        "per_match_clv_sum_diff": per_match_diff(tb["clv_sum"] - ta["clv_sum"]),
        "a_shaped_share": _shaped_share(ra),
        "b_shaped_share": _shaped_share(rb),
    }


def _shaped_share(recs: Sequence[Rec]) -> Optional[float]:
    if not recs:
        return None
    return sum(1 for r in recs if abs(r.line) >= 0.75 - 1e-9) / len(recs)


def reproduction(
    prod: pd.DataFrame,
    stored_recs: Sequence[Rec],
    served_recs: Sequence[Rec],
    defined: Dict[str, List[str]],
    status: Dict[str, str],
    matches: pd.DataFrame,
) -> Dict[str, Any]:
    """Production's AH recs on matches kicking off from REPRO_START against
    the stored-arm replay (logic-era de-vig) and the served-arm replay. Every
    production rec is accounted for: compared, on a match the replay found
    ineligible, on a match the arm is not defined for, or written after
    kickoff (excluded)."""
    prod = prod.copy()
    prod["kickoff"] = [matches.loc[m, "match_date"] if m in matches.index else pd.NaT for m in prod["match_id"]]
    unknown = int(prod["kickoff"].isna().sum())
    prod = prod[prod["kickoff"].notna()]
    prod = prod[prod["kickoff"] >= REPRO_START]
    after = prod["created_at"] >= prod["kickoff"]
    pre = prod[~after]
    out: Dict[str, Any] = {
        "production_recs": int(len(prod)),
        "production_written_after_kickoff": int(after.sum()),
        "production_recs_on_unfinished_or_unknown_matches": unknown,
        "production_matches": int(pre["match_id"].nunique()),
        "production_by_status": {str(k): int(v) for k, v in pre["status"].value_counts().items()},
    }
    pk = {(str(r.match_id), str(r.selection)): r for r in pre.itertuples()}
    for name, rr in (("stored", stored_recs), ("served", served_recs)):
        pool = {m for m in defined[name] if matches.loc[m, "match_date"] >= REPRO_START}
        mine = {(r.match_id, r.selection): r for r in rr if r.match_id in pool}
        prod_in = {k: v for k, v in pk.items() if k[0] in pool}
        outside: Dict[str, int] = {}
        for k in pk:
            if k[0] in pool:
                continue
            why = status.get(k[0], "no_prematch_ah_odds_in_export")
            why = "arm_undefined" if why == "eligible" else why
            outside[why] = outside.get(why, 0) + 1
        both = sorted(set(mine) & set(prod_in))
        po = {k: float(prod_in[k].odds_decimal) for k in both}
        out[name] = {
            "matches_in_pool": len(pool),
            "production_recs_in_pool": len(prod_in),
            "production_recs_outside_pool": outside,
            "replay_recs": len(mine),
            "both": len(both),
            "production_only": len(set(prod_in) - set(mine)),
            "replay_only": len(set(mine) - set(prod_in)),
            "both_same_price": sum(1 for k in both if abs(po[k] - mine[k].odds) < 1e-9),
            "both_price_within_0.05": sum(1 for k in both if abs(po[k] - mine[k].odds) <= 0.05 + 1e-12),
            "both_same_book": sum(1 for k in both if str(prod_in[k].bookmaker) == mine[k].bookmaker),
            "both_same_stake": sum(1 for k in both if abs(float(prod_in[k].stake) - mine[k].stake) < 0.005),
            "recall": len(both) / len(prod_in) if prod_in else None,
            "precision": len(both) / len(mine) if mine else None,
        }
    return out


def power(report: Dict[str, Any]) -> Dict[str, Any]:
    """n (and months at the current volume) to detect each measured effect with
    80% power at two-sided 5%."""
    br = report["brier"]
    per_month = report.get("stored_ah_matches_per_month") or {}
    recent = [v for k, v in per_month.items() if k >= "2026-08"]
    stored_rate = float(np.mean(recent)) if recent else None
    pr = report["pairs"][pair_name("served", "refit", None)]
    out: Dict[str, Any] = {"stored_ah_matches_per_month_since_2026_08": stored_rate}
    for name in ("best_single_line", "pooled_shaped_mean", "gd_rps"):
        p = br[name]
        n = n_for_power(p.get("sd_per_match"), p.get("delta"))
        out[f"brier_{name}"] = {
            "delta": p.get("delta"),
            "sd_per_match": p.get("sd_per_match"),
            "n_matches_80pct": n,
            "months_at_current_volume": (n / stored_rate) if n and stored_rate else None,
            "house_gate": (
                "unreachable at this effect size (|delta| < 0.005)"
                if p.get("delta") is None or p["delta"] > -0.005
                else "the point estimate clears -0.005"
            ),
        }
    ah_rate = report.get("ah_odds_matches_per_month")
    for key, label in (
        ("per_match_pl_diff", "money_stake_pl"),
        ("per_match_flat_diff", "money_flat_pl"),
        ("per_match_clv_sum_diff", "clv_fair_sum"),
    ):
        d = pr[key]
        n = n_for_power(d.get("sd"), d.get("mean"))
        out[label] = {
            "mean_per_match": d.get("mean"),
            "sd_per_match": d.get("sd"),
            "n_matches_80pct": n,
            "months_at_current_volume": (n / ah_rate) if n and ah_rate else None,
        }
    out["ah_odds_matches_per_month_since_2026_08"] = ah_rate
    return out


def write_recs_csv(path: str, recs: Dict[str, List[Rec]]) -> None:
    rows = []
    for arm, rr in recs.items():
        for r in rr:
            rows.append(
                {
                    "arm": arm,
                    "match_id": r.match_id,
                    "selection": r.selection,
                    "odds": r.odds,
                    "bookmaker": r.bookmaker,
                    "ev": r.ev,
                    "p_raw": r.p_raw,
                    "p_kelly": r.p_kelly,
                    "market_prob": r.market_prob,
                    "stake": r.stake,
                    "age_hours": r.age_hours,
                    **r.extra,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


# ── Formatting ───────────────────────────────────────────────────────


def _f(v: Any, spec: str = "+.5f") -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, spec)


def _money_line(name: str, m: Dict[str, Any]) -> str:
    return (
        f"  {name:<22} recs {m['recs']:>4} (settled {m['settled']}, {m['matches']} matches)  stake {m['stake']:>9.2f}"
        f"  P/L {m['pl']:>+9.2f}  ROI {_f(m['roi_stake'], '+.3f')}  flat {m['flat_pl']:>+7.2f}"
        f" ROI {_f(m['roi_flat'], '+.3f')}  odds {_f(m['mean_odds'], '.2f')}  CLV same {_f(m['clv_same_mean'], '+.4f')}"
        f" fair {_f(m['clv_fair_mean'], '+.4f')} (n={m['clv_fair_n']})"
    )


def format_report(r: Dict[str, Any]) -> str:
    rp = r["replay"]
    br = r["brier"]
    L = [
        f"Soccer AH replay: window {r['window']['start']}..{r['window']['end']}; bankroll {r['bankroll']}",
        f"AH odds: {r['ah_odds']['rows']} pre-match rows on {r['ah_odds']['matches']} finished matches, first created "
        f"{r['ah_odds']['first_created']}; books {r['ah_odds']['books']}; {r['ah_odds']['snapshots']} snapshots",
        f"candidates {rp['candidates']}: ineligible {rp['ineligible']}, outside the gate's eval set "
        f"{rp['not_in_gate_eval']}, no stored prediction {rp['no_stored']}; arms defined {rp['defined']}",
        f"odds state at the tick: {rp['state_drops']}; best prices shared by 2+ books {rp['price_ties']}",
        f"stored vs served shaped AH max |d| per match: {rp['stored_vs_served_max_abs_shaped']}",
        f"Brier re-derivation vs scores.npz: max |d| {br['reconstruct_max_abs']:.1e} over {br['n']} rows",
    ]
    if r["mechanics"]:
        L.append(f"MECHANICS ONLY (no deltas, no money): rec counts {r['rec_counts']}")
        return "\n".join(L)
    L += ["", "PART A — Brier, refit - served, live regime (post hoc; context only):"]
    for u in ALL_AH_UNITS:
        L.append(ftdc._pair_line(u, br["units"][u]))
    for name in ("best_single_line", "pooled_shaped_mean", "gd_rps", "gd_brier_7way"):
        L.append(ftdc._pair_line(name, br[name]))
    for name, v in br["verdicts"].items():
        L.append(f"  house gate on {name}: {'PASS' if v['ship_strict'] else 'FAIL'} — {v['line']}")
    L.append("  shift ceiling (Brier gain refit would earn if it were the truth; rms shift per class):")
    for u, s in br["shift_ceiling"].items():
        L.append(
            f"    {u:<12} E[sum (dp)^2] {s['mean_sq_shift']:.5f}  rms {s['rms_per_class']:.4f}  max {s['max_abs']:.3f}"
        )
    L += ["", "PART B — money (replayed at the last tick; settled by grading_outcomes):"]
    for arm, m in r["money"].items():
        L.append(_money_line(arm, m))
    for name, p in r["pairs"].items():
        bs = p["bootstrap"]
        L += [
            "",
            f"  [{name}] {p['matches']} matches ({p['matches_with_any_rec']} with a rec in either arm); same rec "
            f"{p['same']} (same price and stake {p['same_price_and_stake']}), same side moved line {p['moved_line']}",
            _money_line("a", p["a"]),
            _money_line("b", p["b"]),
            _money_line("a only", p["a_only"]),
            _money_line("b only", p["b_only"]),
            _money_line("moved line, a", p["moved_line_a"]),
            _money_line("moved line, b", p["moved_line_b"]),
            f"    shaped-line share a {_f(p['a_shaped_share'], '.3f')} b {_f(p['b_shaped_share'], '.3f')}",
        ]
        for k in (
            "a_roi_stake",
            "b_roi_stake",
            "diff_roi_stake",
            "a_roi_flat",
            "b_roi_flat",
            "diff_roi_flat",
            "diff_pl_total",
            "diff_flat_pl_total",
            "a_clv_fair",
            "b_clv_fair",
            "diff_clv_fair",
        ):
            v = bs.get(k) or {}
            L.append(
                f"    {k:<20} {_f(v.get('estimate'), '+.4f')}  bootstrap SE {_f(v.get('se'), '.4f')}  95% "
                f"[{_f(v.get('lo95'), '+.4f')}, {_f(v.get('hi95'), '+.4f')}]"
            )
        for k in ("per_match_pl_diff", "per_match_flat_diff", "per_match_clv_sum_diff"):
            d = p[k]
            L.append(f"    {k:<24} mean {_f(d['mean'], '+.4f')}  sd {_f(d['sd'], '.4f')}  SE {_f(d['se'], '.4f')}")
    rep = r["reproduction"]
    L += [
        "",
        f"PART C — reproduction vs production's AH recs from {REPRO_START.date()}: {rep['production_recs']} recs "
        f"({rep['production_written_after_kickoff']} written after kickoff, excluded) on {rep['production_matches']} "
        "matches",
    ]
    for name in REPRO_ARMS:
        v = rep[name]
        L.append(
            f"  {name:<7} production recs in pool {v['production_recs_in_pool']}, replay {v['replay_recs']}, both "
            f"{v['both']} (recall {_f(v['recall'], '.3f')}, precision {_f(v['precision'], '.3f')}); of both: same "
            f"price {v['both_same_price']}, within 0.05 {v['both_price_within_0.05']}, same book "
            f"{v['both_same_book']}, same stake {v['both_same_stake']}; production recs outside the pool "
            f"{v['production_recs_outside_pool']}"
        )
    pw = r["power"]
    L += [
        "",
        f"PART D — power (80%, two-sided 5%); AH-odds matches per month since 2026-08: "
        f"{_f(pw['ah_odds_matches_per_month_since_2026_08'], '.0f')}",
    ]
    for k, v in pw.items():
        if isinstance(v, dict):
            L.append(f"  {k:<32} {v}")
    return "\n".join(L)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    defaults = gr.parse_args([])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ftdc-dir", required=True, help="the --out-dir of scripts/ab_soccer_ft_dixon_coles.py")
    p.add_argument("--start", default="2026-05-01", help="first kickoff day (AH odds exist from 2026-05-30)")
    p.add_argument("--end", default="2026-11-01", help="kickoffs before this day (the gate's window ends 2026-10-31)")
    p.add_argument("--ev-threshold", type=float, default=defaults.ev_threshold)
    p.add_argument("--prob-floor", type=float, default=defaults.prob_floor)
    p.add_argument("--min-team-history", type=int, default=defaults.min_team_history)
    p.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    p.add_argument("--limit", type=int, default=0, help="cap the candidate matches and Brier rows (smoke)")
    p.add_argument("--mechanics", action="store_true", help="print only counts and fidelity lines, never a delta")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None, help="every replayed rec, per arm")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    try:
        report = run(args)
    except Exception as exc:
        logger.error("ab_soccer_ah_recs_replay failed: %s", exc, exc_info=True)
        return 1
    if args.out_json and not args.mechanics:
        Path(args.out_json).write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False))
    print(format_report(report))
    return 0


def _jsonable(obj: Any) -> Any:
    from ab_halftime_dixon_coles import clean_json

    return clean_json(obj)


if __name__ == "__main__":
    sys.exit(main())
