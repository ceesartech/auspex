"""Generate horse racing value-bet recommendations.

Horse racing is structurally different from the team / 1v1 sports
in the rec engine:
  * A race has N entrants, not a 2-way market. Each runner gets a
    devigged win probability (from race_predictions) and the
    recommendation candidate is per-(race, entrant).
  * Bookmaker odds DO NOT live in the `odds` table — that's keyed
    on match_id (sport=soccer/NFL/NBA/...). For horse racing they
    live in race_entrants.metadata.bookmaker_odds, captured by
    load_racing_api.upsert_entrant from the racecard's `odds[]`
    array.
  * The market-consensus baseline IS the devigged morning line.
    Comparing it against the SAME morning-line decimal would give
    zero EV by construction — so value bets only exist where a
    specific bookmaker offers LONGER odds than the consensus
    (best-of-N pricing). Without the bookmaker_odds capture, no
    recommendations can ever fire.

Same EV / Kelly math as the team-sport engines:
  - Quarter Kelly stake sizing
  - 5% minimum EV threshold (default; tunable)
  - 0.10 minimum raw model probability (default; tunable). Horse
    racing fields are wide so even 10% win prob is a credible
    candidate (5-1 odds-on favorites land here).
  - No prob-cap: devigged probs in fields of 8+ rarely exceed 60%
    even for prohibitive favorites, so the soccer/tennis cap isn't
    load-bearing here.

Writes to the race_recommendations table (not betting_recommendations
— horse racing has its own schema for multi-runner shapes; see
migration 013 for the rationale).

Idempotent: drops pending win-market recs for the race before
re-inserting.

Usage (inside the api container):
    python /app/scripts/generate_recommendations_horse_racing.py
    python /app/scripts/generate_recommendations_horse_racing.py \
        --days 2 --ev-threshold 0.05 --prob-floor 0.10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

# Reuse the EV/Kelly + bankroll helpers from the soccer engine. Same
# math; horse racing doesn't need a different formulation.
sys.path.insert(0, os.path.dirname(__file__))
import rec_gating  # noqa: E402
from generate_recommendations import (  # noqa: E402
    MAX_ODDS_AGE_HOURS,
    STALE_ODDS_REASON,
    confidence_rating,
    expected_value,
    get_bankroll,
    is_stale_odds,
    kelly_fraction,
    odds_age_hours,
    with_odds_age,
)
from telegram_notify import Alert, enqueue_alerts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("generate_recommendations_horse_racing")

KELLY_FRACTION = 0.25

# Model precedence (prob source): consensus-only for EV math. The
# LambdaMART ranker has BETTER top-1 accuracy (+2.0pts on the 13k-
# race corpus) but WORSE test Brier (0.1050 vs 0.0831), and the
# recs engine consumes probabilities directly (EV = prob × odds).
# A worse-calibrated model inflates the value-bet count by 17x —
# verified empirically (memory: horse-racing-ml-ranker-v1).
#
# Listing the ranker even as a FALLBACK is wrong: every race the
# ranker scores but consensus hasn't yet (e.g., races added between
# the two precompute tasks' runs) silently picks up the ranker's
# probabilities and the ~30-picks/day budget blows out again. Keep
# this list consensus-only for PROBABILITIES.
MODEL_PRECEDENCE: list[tuple[str, str]] = [
    ("market_consensus_v1", "1.0.0"),
]

# HYBRID mode: while consensus owns the PROBABILITY side of the EV
# math, the ranker's RANK is genuinely better (+2.0pts top-1 on the
# 13k-race held-out test). The hybrid filter uses the ranker to
# narrow each race down to its top-RANKER_TOP_N entrants, then
# evaluates EV on those using consensus probs. This combines what
# each model is good at:
#   - ranker: WHICH horses are likely to win (rank order)
#   - consensus: HOW LIKELY each is (calibrated probability)
#
# When RANKER_TOP_N is None, the filter is disabled and the recs
# engine considers every entrant — the prior consensus-only
# behaviour. When the ranker hasn't scored a race (e.g., because
# its precompute task ran AFTER the race-card list), the filter
# falls back to "consider all entrants" for that race so consensus-
# only races still get full coverage.
RANKER_MODEL_NAME = "lightgbm_ranker_v1"
RANKER_MODEL_VERSION = "1.0.0"
RANKER_TOP_N: Optional[int] = 3

# Field-size band with the worst settled record in the 2026-09 audit:
# WIN recs in 5-7 runner fields went n=182, ROI -57.1%, bootstrap CI
# [-76.1%, -35.4%] — the ONLY band whose CI excludes zero, and it is
# negative. Small fields price the favourite tightly and our devigged
# consensus has nothing left to beat, so we suppress win recs there
# (place recs in the same fields are handled by their own gate — place
# under 6.0 is the one profitable racing band).
# The band is measured on the race's RUNNER count (non-scratched
# entrants), so it is binned on load_runner_count(), NOT on the number of
# entrants we happened to predict.
SUPPRESSED_WIN_FIELD_SIZES = range(5, 8)
SUPPRESSED_WIN_FIELD_REASON = "win:field_5_7_runners"


# ── Odds freshness (audit 2026-09) ─────────────────────────────────
#
# Racing is held to the SAME bound as the team sports
# (generate_recommendations.MAX_ODDS_AGE_HOURS) but cannot use the same SQL:
# its prices are not in the `odds` table at all (module docstring), they are
# the metadata->'bookmaker_odds' array on race_entrants. Each entry carries
# an `updated` string from the upstream feed, but it is unvalidated,
# unparsed and its timezone is undocumented, so it is NOT the signal to
# trust. The entrant row's own updated_at is: load_racing_api.upsert_entrant
# rewrites metadata (`metadata || EXCLUDED.metadata`) and sets
# `updated_at = NOW()` on EVERY racecard pass — changed prices or not — and
# that pass runs on the 15-minute auspex_pipeline tick. So this is a
# last-CONFIRMED time, exactly like the odds.last_seen_at the team sports age
# off (see the note on rec_gating.MAX_ODDS_AGE_HOURS) — and NOT like
# odds."timestamp", which only advances when a price MOVES. Racing could
# justify a far tighter bound than the team sports need; it shares the 6 h one
# anyway, because one number for "too old to bet on" is worth more than a
# per-sport optimum, and at 24 racecard cycles of slack the bound still only
# fires when the feed has stopped covering the race entirely.
ENTRANT_ODDS_AGE_SELECT_SQL = "EXTRACT(EPOCH FROM (NOW() - e.updated_at)) / 3600.0 AS odds_age_hours"


# ── Best-of-N pricing across bookmakers ────────────────────────────


def best_decimal(bookmaker_odds: list[dict]) -> Optional[dict]:
    """Pick the bookmaker offering the longest decimal odds for this
    horse — that's the bettor's most favourable price, and the only
    one that can carry positive EV vs the devigged consensus across
    the same array.

    Returns {bookmaker, decimal} for the winner, or None if the input
    array is empty / malformed.
    """
    if not bookmaker_odds:
        return None
    best = None
    for entry in bookmaker_odds:
        if not isinstance(entry, dict):
            continue
        decimal = entry.get("decimal")
        if decimal is None:
            continue
        try:
            d = float(decimal)
        except (TypeError, ValueError):
            continue
        if d <= 1.0:
            continue
        if best is None or d > best["decimal"]:
            best = {"bookmaker": entry.get("bookmaker"), "decimal": d}
    return best


# ── Place pilot (audit §3 rank 7 — market-side, no model change) ────────
# UK/IRE each-way terms by field size: (places paid, fraction of win odds).
# <5 runners: win only. 5-7: 2 places at 1/4. 8+: 3 places at 1/5.
# (16+ handicaps pay 4 at 1/4 — we don't know handicap status, so we use
# the conservative standard terms; noted in the audit entry.)


def ew_terms(field_size: int) -> tuple[int, float]:
    if field_size <= 4:
        return 0, 0.0
    if field_size <= 7:
        return 2, 0.25
    return 3, 0.2


def place_probability(win_probs: dict[str, float], eid: str, places: int) -> float:
    """Harville top-k probability for entrant `eid` from the field's
    devigged WIN probabilities (sequential conditional model — the
    standard closed-form for deriving place chances from win chances)."""
    p = dict(win_probs)
    pi = p.get(eid, 0.0)
    if places <= 0 or pi <= 0:
        return 0.0
    total = pi  # finish 1st
    if places >= 2:
        for j, pj in p.items():
            if j == eid or 1.0 - pj <= 0:
                continue
            total += pj * pi / (1.0 - pj)
    if places >= 3:
        for j, pj in p.items():
            if j == eid:
                continue
            for k, pk in p.items():
                if k in (eid, j):
                    continue
                d1 = 1.0 - pj
                d2 = 1.0 - pj - pk
                if d1 <= 0 or d2 <= 0:
                    continue
                total += pj * (pk / d1) * (pi / d2)
    return min(1.0, total)


# ── DB I/O ─────────────────────────────────────────────────────────


def list_upcoming_races(cur, days: int) -> list[dict]:
    """Scheduled races in the next N days that have at least one
    market_consensus_v1 prediction. Returns the race_id alongside
    the track + race_date the Telegram alert needs to identify the
    race — skips races without predictions so we don't waste cycles."""
    eligible_model_names = [m for m, _ in MODEL_PRECEDENCE]
    cur.execute(
        """
        SELECT DISTINCT
            r.id::text   AS race_id,
            r.track_name,
            r.race_date,
            r.race_number
        FROM races r
        JOIN race_predictions rp ON rp.race_id = r.id
        WHERE r.status = 'scheduled'
          AND r.race_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
          AND rp.model_name = ANY(%s)
        ORDER BY r.race_date ASC, r.id::text
        """,
        (str(days), eligible_model_names),
    )
    return list(cur.fetchall())


def load_race_candidates(cur, race_id: str) -> list[dict]:
    """Per-entrant pricing + prediction context for one race. Returns
    list of {entrant_id, prediction_id, confidence, bookmaker_odds,
    horse_name, model_name, ranker_confidence, odds_age_hours}.

    odds_age_hours is how long ago the feed last confirmed this entrant's
    price array (see ENTRANT_ODDS_AGE_SELECT_SQL); recommend_for_race
    refuses to price a bet off a cold one.

    The DISTINCT ON + ORDER BY (entrant_id, precedence) picks the
    highest-precedence available model per entrant for the
    consensus-driven probability. Separately we join in the LATEST
    ranker prediction per entrant (when one exists) so the hybrid
    filter can rank candidates by the LambdaMART score while still
    using consensus_prob for EV — see the RANKER_TOP_N block in
    recommend_for_race for how the two are combined."""
    eligible: list[str] = [m for m, _ in MODEL_PRECEDENCE]
    # Build the CASE WHEN ladder dynamically so this works with any
    # number of models. The ELSE branch puts unknown models at the
    # bottom, but the ANY(%s) filter excludes them anyway — defensive.
    when_clauses = " ".join(f"WHEN {ghr_arg(i)} THEN {i + 1}" for i in range(len(eligible)))
    case_sql = f"CASE rp.model_name {when_clauses} ELSE {len(eligible) + 1} END"
    cur.execute(
        f"""
        SELECT DISTINCT ON (entrant_id)
            entrant_id,
            prediction_id,
            confidence,
            bookmaker_odds,
            horse_name,
            model_name,
            ranker_confidence,
            odds_age_hours
        FROM (
            SELECT
                e.id::text          AS entrant_id,
                rp.id::text         AS prediction_id,
                rp.confidence       AS confidence,
                e.metadata->'bookmaker_odds' AS bookmaker_odds,
                h.name              AS horse_name,
                rp.model_name       AS model_name,
                rrk.confidence      AS ranker_confidence,
                {ENTRANT_ODDS_AGE_SELECT_SQL},
                {case_sql}          AS precedence
            FROM race_entrants e
            JOIN race_predictions rp ON rp.entrant_id = e.id
            JOIN horses h ON h.id = e.horse_id
            LEFT JOIN race_predictions rrk
                ON rrk.entrant_id = e.id
               AND rrk.model_name = %s
               AND rrk.model_version = %s
               AND rrk.prediction_type = 'win'
            WHERE e.race_id = %s
              AND NOT e.scratched
              AND rp.model_name = ANY(%s)
              AND rp.prediction_type = 'win'
        ) ranked
        ORDER BY entrant_id, precedence ASC
        """,
        (*eligible, RANKER_MODEL_NAME, RANKER_MODEL_VERSION, race_id, eligible),
    )
    return list(cur.fetchall())


def load_runner_count(cur, race_id: str) -> Optional[int]:
    """True non-scratched runner count for one race.

    NOT the same as len(load_race_candidates(...)): that only returns
    entrants which ALSO carry a win prediction from an eligible model, so it
    is a predicted-entrant count. On prod the two disagree for 3,250 of
    20,102 races. SUPPRESSED_WIN_FIELD_SIZES is an audit band measured on the
    RACE's field, so it must bin on this number — otherwise races get their
    win recs suppressed under a band never measured on them, and true 5-7
    fields escape it.

    Returns None when the count is unavailable, leaving the fallback to the
    caller rather than silently inventing a field size."""
    cur.execute(
        """
        SELECT COUNT(*) AS runners
        FROM race_entrants
        WHERE race_id = %s
          AND NOT scratched
        """,
        (race_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    runners = row["runners"] if isinstance(row, dict) else row[0]
    return int(runners) if runners is not None else None


def ghr_arg(_i: int) -> str:
    """Return a literal positional placeholder for the dynamic CASE
    WHEN clause. Kept as a tiny helper so the call site reads as a
    single f-string template — substituting %s directly inside the
    f-string would be ambiguous with the outer execute() params."""
    return "%s"


def delete_pending(cur, race_id: str) -> None:
    """Drop still-actionable win-market recs for this race before
    re-inserting. Never touches recs the user placed or that settled."""
    cur.execute(
        """
        DELETE FROM race_recommendations
        WHERE race_id = %s
          AND status = 'pending'
          AND bet_type IN ('win', 'place')
        """,
        (race_id,),
    )


def insert_recommendation(cur, rec: dict) -> None:
    cur.execute(
        """
        INSERT INTO race_recommendations
          (race_prediction_id, race_id, entrant_id, bet_type, selection,
           odds_at_recommendation, bookmaker, confidence_rating,
           expected_value, kelly_stake, recommended_stake, reasoning,
           risk_factors)
        VALUES
          (%(prediction_id)s, %(race_id)s, %(entrant_id)s, %(bet_type)s,
           %(selection)s, %(odds)s, %(bookmaker)s, %(conf)s, %(ev)s,
           %(kelly_stake)s, %(rec_stake)s, %(reasoning)s, %(risk)s::jsonb)
        """,
        rec,
    )


def _risk_factors(prob: float, odds_decimal: float, field_size: int) -> list[str]:
    risks: list[str] = []
    if odds_decimal >= 10.0:
        # Decimal 10+ = 9/1 or longer. Longshots are statistically
        # noisier — variance dominates EV math at this end.
        risks.append("longshot")
    if prob < 0.15:
        # Below 15% even after devig — the consensus considers this
        # horse unlikely. The recommendation is essentially a bet
        # AGAINST the consensus's confidence ordering.
        risks.append("low_consensus_probability")
    if field_size >= 14:
        # Large fields amplify variance and break-up traffic patterns.
        # Even a sharp pick can lose to bad luck in a 16-runner sprint.
        risks.append("large_field")
    return risks


# ── Alert factory ───────────────────────────────────────────────────


def horse_racing_alert(
    *,
    track_name: str,
    race_date,
    race_number: Optional[int],
    horse_name: str,
    odds_decimal: float,
    bookmaker: Optional[str],
    confidence: float,
    expected_value: float,
    recommended_stake: float,
) -> Alert:
    """Translate a horse racing value bet into the shared Alert shape.
    The Alert dataclass was designed for 2-team / 1v1 sports, so the
    fit isn't 1:1:
      * `home_team` carries the horse name (the actual selection).
      * `away_team` is "Field" — the consensus implied prob is
        derived ACROSS the field, so the contrast is horse-vs-field
        not horse-vs-horse.
      * `league_name` carries the track name (Newton Abbot, Curragh,
        ...) plus race number when known.
      * Setting `expected_value` flips the digest formatter into
        value-bet mode (odds + EV + stake instead of probability
        breakdown). See telegram_notify._format_alert_line."""
    race_label = f"{track_name}"
    if race_number is not None:
        race_label += f" R{race_number}"
    return Alert(
        sport="horse_racing",
        league_name=race_label,
        home_team=horse_name,
        away_team="Field",
        match_date=race_date,
        market_label="Win",
        predicted_outcome=horse_name,
        confidence=float(confidence),
        probabilities={"win": float(confidence)},
        odds_decimal=float(odds_decimal),
        expected_value=float(expected_value),
        recommended_stake=float(recommended_stake),
        bookmaker=bookmaker,
    )


# ── Recommendation orchestration ────────────────────────────────────


def _format_suppressions(counts: dict[str, int]) -> str:
    """Render the per-reason suppression tally deterministically
    (biggest bucket first, then alphabetically) for the summary log."""
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def recommend_for_race(
    cur,
    race: dict,
    bankroll: float,
    ev_threshold: float,
    prob_floor: float,
    suppressed: Optional[dict[str, int]] = None,
    max_odds_age_hours: float = MAX_ODDS_AGE_HOURS,
) -> list[Alert]:
    """Generate (and DB-insert) value-bet recs for one race; return
    the matching Alert objects ready for Redis. Returns [] when the
    race has no qualifying picks.

    `suppressed` (when given) is a caller-owned counter that this
    function increments per gate-rejection reason, keyed
    "<bet_type>:<reason>", so run() can log ONE loud summary of the
    volume the per-stream gate (scripts/rec_gating.py) refused. Picks
    refused by the odds-freshness guard land in the same counter under the
    shared STALE_ODDS_REASON key.

    Hybrid filter (RANKER_TOP_N): when the LambdaMART ranker has
    scored this race, narrow the candidate set to the top-N entrants
    by ranker confidence before evaluating EV. Consensus probability
    still drives the math; the ranker just picks which horses to
    consider. For races without ranker scores, every entrant is
    evaluated as before."""
    counts = suppressed if suppressed is not None else {}

    def _suppress(bet_type: str, reason: Optional[str]) -> None:
        # "stream_disabled:..." already names the sport+market; the cap
        # reasons don't, so prefix those with the leg they fired on.
        key = str(reason) if str(reason).startswith("stream_disabled") else f"{bet_type}:{reason}"
        counts[key] = counts.get(key, 0) + 1

    race_id = race["race_id"]
    candidates = load_race_candidates(cur, race_id)
    if not candidates:
        return []
    delete_pending(cur, race_id)
    field_size = len(candidates)
    # The 5-7 band is a property of the RACE (its runner count), not of how
    # many entrants we happened to predict, so it bins on the true
    # non-scratched field. field_size (predicted entrants) keeps driving
    # ew_terms and the risk factors exactly as before.
    runner_count = load_runner_count(cur, race_id)
    if runner_count is None:
        logger.warning(
            "Race %s: runner count unavailable; binning the win-suppression band on %d predicted entrant(s)",
            race_id,
            field_size,
        )
        runner_count = field_size
    # Harville needs the WHOLE field's win distribution — capture before
    # the ranker filter narrows the candidate list.
    field_win_probs = {c["entrant_id"]: float(c["confidence"]) for c in candidates}
    places, ew_fraction = ew_terms(field_size)

    # Hybrid filter: if RANKER_TOP_N is set AND every candidate has a
    # ranker_confidence, narrow to the top-N by ranker rank. The
    # "every candidate has a ranker_confidence" guard handles the
    # transient state where the ranker precompute hasn't run yet for
    # this race — fall back to "consider all entrants" so consensus-
    # only races still get full coverage.
    if RANKER_TOP_N is not None and RANKER_TOP_N > 0:
        if all(c.get("ranker_confidence") is not None for c in candidates):
            candidates = sorted(
                candidates,
                key=lambda c: float(c["ranker_confidence"]),
                reverse=True,
            )[:RANKER_TOP_N]

    alerts: list[Alert] = []

    for cand in candidates:
        prob = float(cand["confidence"])
        if prob < prob_floor:
            continue
        bookmaker_odds = cand["bookmaker_odds"]
        if isinstance(bookmaker_odds, str):
            # psycopg2 returns JSONB as str unless cursor configured
            # otherwise — handle both shapes defensively.
            try:
                bookmaker_odds = json.loads(bookmaker_odds)
            except (TypeError, ValueError):
                bookmaker_odds = None
        best = best_decimal(bookmaker_odds or [])
        if not best:
            continue
        odds_decimal = best["decimal"]
        # FRESHNESS GUARD (audit 2026-09). A price the feed has not
        # confirmed inside the bound is not a price we can claim was
        # available, so it may not become a rec — win OR place, since the
        # place leg is derived from this same win quote.
        #
        # Unknown age is treated differently here than in the team-sport
        # generators, where an odds row with no timestamp is a broken row
        # and is refused. race_entrants.updated_at is DEFAULT NOW() and set
        # explicitly on every upsert, so it cannot be NULL for a row this
        # query returned: a candidate with no age means the caller supplied
        # rows this projection did not build. Refusing those would take the
        # whole racing book off the board over a code shape rather than
        # over a stale price, so we log loudly and carry on; a REPORTED age
        # past the bound is refused and counted.
        age_hours = odds_age_hours(cand)
        if age_hours is None:
            logger.warning(
                "Race %s entrant %s: candidate row carries no odds age — "
                "the freshness guard cannot run on this pick",
                race_id,
                cand["entrant_id"],
            )
        elif is_stale_odds(age_hours, max_odds_age_hours):
            counts[STALE_ODDS_REASON] = counts.get(STALE_ODDS_REASON, 0) + 1
            continue
        ev = expected_value(prob, odds_decimal)
        if ev < ev_threshold:
            continue
        # Per-stream gate (audit 2026-09). Racing passes market_prob=None
        # deliberately: its "consensus" is a single bookmaker today, so the
        # racing gates set max_gap=None and nothing here can be rejected
        # for a missing consensus.
        allowed, reason = rec_gating.passes_gate(
            "horse_racing",
            "win",
            odds=odds_decimal,
            ev=ev,
            model_prob=prob,
            market_prob=None,
        )
        # The 5-7 runner band is a racing-specific suppression on top of the
        # shared gate, binned on the race's TRUE runner count (see
        # load_runner_count — len(candidates) is a predicted-entrant count and
        # is not what the audit measured). Note we do NOT `continue` — the
        # place leg below is a separately-gated market and stays eligible.
        win_allowed = allowed and runner_count not in SUPPRESSED_WIN_FIELD_SIZES
        if not allowed:
            _suppress("win", reason)
        elif not win_allowed:
            counts[SUPPRESSED_WIN_FIELD_REASON] = counts.get(SUPPRESSED_WIN_FIELD_REASON, 0) + 1

        k = kelly_fraction(prob, odds_decimal)
        uncapped_stake = round(bankroll * k * KELLY_FRACTION, 2)
        stake = rec_gating.cap_stake(bankroll * k * KELLY_FRACTION, bankroll)

        # Display label: "consensus" when this came through the
        # consensus-only path; "hybrid" when the ranker also scored
        # the race and filtered this pick into the top-N. Useful for
        # post-hoc analysis (does ranker filtering improve win rate?).
        had_ranker_score = cand.get("ranker_confidence") is not None
        label = "hybrid" if (had_ranker_score and RANKER_TOP_N) else "consensus"
        capped_note = (
            f" Stake capped at {rec_gating.MAX_STAKE_FRACTION:.1%} of bankroll "
            f"(quarter-Kelly asked ${uncapped_stake:.2f})."
            if stake < uncapped_stake
            else ""
        )
        reasoning = (
            f"Win: {cand['horse_name']} — {label} {prob:.0%}, "
            f"book {1/odds_decimal:.0%} (@ {odds_decimal:.2f} on "
            f"{best['bookmaker']}) → EV {ev:+.1%}, quarter-Kelly "
            f"stake ${stake:.2f}.{capped_note}"
        )
        rec = {
            "prediction_id": cand["prediction_id"],
            "race_id": race_id,
            "entrant_id": cand["entrant_id"],
            "bet_type": "win",
            "selection": cand["horse_name"],
            "odds": odds_decimal,
            "bookmaker": best["bookmaker"],
            "conf": confidence_rating(ev, prob),
            "ev": ev,
            "kelly_stake": k,
            "rec_stake": stake,
            "reasoning": reasoning,
            # odds_at_recommendation is `odds_decimal`, the price the
            # freshness guard accepted; its age rides along in risk_factors
            # so CLV can be audited after the fact.
            "risk": json.dumps(with_odds_age(_risk_factors(prob, odds_decimal, field_size), age_hours)),
        }
        if win_allowed:
            insert_recommendation(cur, rec)
            alerts.append(
                horse_racing_alert(
                    track_name=race["track_name"],
                    race_date=race["race_date"],
                    race_number=race.get("race_number"),
                    horse_name=cand["horse_name"],
                    odds_decimal=odds_decimal,
                    bookmaker=best["bookmaker"],
                    confidence=prob,
                    expected_value=ev,
                    recommended_stake=stake,
                )
            )

        # ── Place pilot: derived each-way place bet on the same horse ──
        # Place odds = 1 + (win odds - 1) x EW fraction; probability from
        # Harville over the full field. Same EV gate + quarter-Kelly.
        if places >= 2 and ew_fraction > 0:
            p_place = place_probability(field_win_probs, cand["entrant_id"], places)
            place_odds = round(1.0 + (odds_decimal - 1.0) * ew_fraction, 4)
            ev_place = expected_value(p_place, place_odds)
            if ev_place >= ev_threshold and p_place >= prob_floor:
                # Same gate, this leg's own bet_type: place odds 12+ went
                # 0/13, 6-12 -11.9%, under 6 +31.9% (audit 2026-09), so the
                # place gate caps odds at 6.0. market_prob=None again —
                # the place gate sets max_gap=None.
                place_allowed, place_reason = rec_gating.passes_gate(
                    "horse_racing",
                    "place",
                    odds=place_odds,
                    ev=ev_place,
                    model_prob=p_place,
                    market_prob=None,
                )
                if not place_allowed:
                    _suppress("place", place_reason)
                k_place = kelly_fraction(p_place, place_odds)
                uncapped_place = round(bankroll * k_place * KELLY_FRACTION, 2)
                stake_place = rec_gating.cap_stake(bankroll * k_place * KELLY_FRACTION, bankroll)
                place_capped_note = (
                    f" Stake capped at {rec_gating.MAX_STAKE_FRACTION:.1%} of bankroll "
                    f"(quarter-Kelly asked ${uncapped_place:.2f})."
                    if stake_place < uncapped_place
                    else ""
                )
                if place_allowed and stake_place > 0:
                    insert_recommendation(
                        cur,
                        {
                            "prediction_id": cand["prediction_id"],
                            "race_id": race_id,
                            "entrant_id": cand["entrant_id"],
                            "bet_type": "place",
                            "selection": cand["horse_name"],
                            "odds": place_odds,
                            "bookmaker": best["bookmaker"],
                            "conf": confidence_rating(ev_place, p_place),
                            "ev": ev_place,
                            "kelly_stake": k_place,
                            "rec_stake": stake_place,
                            "reasoning": (
                                f"Place ({places} places @ {ew_fraction:.0%} odds): "
                                f"{cand['horse_name']} — Harville {p_place:.0%} "
                                f"@ {place_odds:.2f} → EV {ev_place:+.1%}, "
                                f"stake ${stake_place:.2f}.{place_capped_note}"
                            ),
                            "risk": json.dumps(with_odds_age(_risk_factors(prob, odds_decimal, field_size), age_hours)),
                        },
                    )

    return alerts


def run(
    database_url: str,
    days: int,
    ev_threshold: float,
    prob_floor: float,
    notify: bool,
) -> dict:
    counts = {"races_processed": 0, "recommendations": 0, "alerts_queued": 0, "queue_depth": 0, "suppressed": 0}
    suppressed: dict[str, int] = {}
    all_alerts: list[Alert] = []
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            bankroll = get_bankroll(cur)
            logger.info("Horse racing bankroll for sizing: $%.2f", bankroll)
            races = list_upcoming_races(cur, days)
            if not races:
                logger.info("No upcoming races with predictions in next %d days", days)
                return counts
            logger.info("Generating horse racing recommendations for %d races", len(races))
            for race in races:
                alerts = recommend_for_race(cur, race, bankroll, ev_threshold, prob_floor, suppressed)
                counts["races_processed"] += 1
                counts["recommendations"] += len(alerts)
                all_alerts.extend(alerts)
            conn.commit()

    queue_depth = enqueue_alerts(all_alerts) if (notify and all_alerts) else 0
    counts["alerts_queued"] = len(all_alerts) if notify else 0
    counts["queue_depth"] = queue_depth
    logger.info(
        "Wrote %d horse racing value-bet recommendations across %d races; " "queued %d alerts (queue depth now %d)",
        counts["recommendations"],
        counts["races_processed"],
        counts["alerts_queued"],
        queue_depth,
    )
    # Loud, never silent: the volume the per-stream gate refused is the
    # whole point of the audit remediation, so it gets its own DAG log line.
    counts["suppressed"] = sum(suppressed.values())
    if suppressed:
        logger.info(
            "gating: suppressed %d candidate rec(s): %s",
            counts["suppressed"],
            _format_suppressions(suppressed),
        )
    return counts


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=2, help="Lookahead window in days (default 2).")
    p.add_argument(
        "--ev-threshold",
        type=float,
        default=0.05,
        help="Minimum positive EV (default 0.05 = 5%%).",
    )
    p.add_argument(
        "--prob-floor",
        type=float,
        default=0.10,
        help="Minimum consensus probability (default 0.10).",
    )
    p.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip the Telegram-queue enqueue (DB writes still happen).",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    run(args.database_url, args.days, args.ev_threshold, args.prob_floor, not args.no_notify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
