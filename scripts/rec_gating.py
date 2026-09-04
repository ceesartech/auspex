"""Single source of truth for WHICH recommendation streams are live, and why.

Every recommendation generator (`generate_recommendations*.py`) imports this
module and asks it two questions before writing a row to
`betting_recommendations`:

    1. Is this (sport, bet_type) stream allowed to emit at all?
    2. If it is, does this individual pick sit inside the stream's bounds
       (max price, max claimed edge, max model-vs-market disagreement)?

It also owns MAX_ODDS_AGE_HOURS — how old a quote may be before it can no
longer price a bet. That belongs here for the same reason the caps do (it is a
money-path policy, and one number is the only safe number), and because the
consensus this module builds for the gap cap has to obey exactly the same
bound as the price queries in the generators.

Why this exists (see docs/SYSTEM_AUDIT_AND_ROADMAP.md — the 2026-09 prod
audit): several streams were shipping recommendations from models with *no
measurable skill* (MMA and tennis moneyline both scored worse than the
bookmakers' own prices; the soccer over_under derivation ran on a constant
prior for 96% of live matches), and the NFL/NBA/NHL corpus is missing the
entire 2025-26 season. The one honest model in the system — soccer 1x2 — is
statistically indistinguishable from the closing market, and its recs whose
edge *survives* repricing at the real closing price LOSE money: at current
model quality, model-versus-market disagreement is anti-predictive. So the
surviving streams are additionally bounded by odds / EV / gap caps, and every
stake is capped at MAX_STAKE_FRACTION of bankroll (quarter-Kelly on uncapped
probabilities once staked 23% of bankroll on a single bet).

Design rules this module obeys:
  * Predictions keep being generated and graded for EVERY sport — that is how
    we measure whether a disabled stream has recovered. Only the *recommendation
    emission* is gated.
  * A gated stream logs loudly (INFO, with the reason and the suppressed count)
    in the caller; nothing here raises and nothing here fails silently.
  * No I/O at import time. This module is imported by seven generators and by
    the unit tests; importing it must never touch the database.

Re-enabling a stream is a deliberate act: flip `enabled` here, in one place,
after the evidence in the `note` has been overturned by the §4.3 validation
gate.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Per-bet cap on stake, as a fraction of bankroll. The audit found
# quarter-Kelly on uncapped model probabilities staking up to 23% of bankroll
# on ONE bet (p90 10.8%), and horse-racing place stakes averaging 44% of
# bankroll per day. 2.5% is the hard ceiling until the models earn more.
MAX_STAKE_FRACTION: float = 0.025

# A de-vigged "consensus" needs real multi-book disagreement to mean anything.
# Below this many books pricing the same market group we report None (unknown),
# and a gate carrying a max_gap treats unknown as a reject.
MIN_CONSENSUS_BOOKS: int = 3

# ── Odds freshness bound (audit 2026-09, defect 1) ────────────────────────
#
# THE one definition of "too old to bet on". It lives here, in the module
# every generator already imports, because it is a money-path policy exactly
# like the caps above — and because the consensus query BELOW has to obey the
# same bound as the price queries in generate_recommendations*.py. (It cannot
# live there and be imported here: generate_recommendations imports this
# module, so the arrow only points one way.)
#
# Why it exists: the `odds` table holds ONE row per (match, bookmaker, market,
# selection, line) and is refreshed in place by scripts/fetch_live_odds.py, so
# a book that stops quoting leaves its last price behind forever. "Best price"
# then means "the highest number anyone ever showed", not "the best price you
# can take now". The 2026-09 audit measured live recommendation prices
# averaging 178 h old on soccer 1x2, 149 h on over_under, 89 h on
# asian_handicap and 197 h on MMA — 48% of settled soccer 1x2 recs FAILED the
# 5% EV gate at the price actually available when they were written (mean
# claimed EV 0.132 -> 0.079, realised ROI +14.1% -> +4.4%).
#
# WHAT is measured: `last_seen_at` — when the ingest last SAW the book quote
# this price, stamped on every observation (migration 024). NOT `timestamp`,
# which advances only when the price MOVED: a book that pulled its market 23 h
# after its last tick would read as fresh off that column, which is the whole
# failure this bound exists to stop. Rows written before migration 024 have a
# NULL last_seen_at and fall back to `timestamp`, i.e. they read as older than
# they are — the safe direction, and self-healing after one ingest cycle.
#
# Why 6 hours: the odds DAG runs every 90 minutes
# (services/data-ingestion/dags/fetch_live_odds.py) and refreshes every
# configured sport on every run, so a book still on offer is re-seen ~4x
# within the bound. 6 h therefore survives two consecutively failed DAG runs
# (each of which already retries once) before it starts refusing live markets,
# and refuses every case in the finding above by a factor of 15-30. Widening
# it to keep a stream alive is not a fix — it is the bug.
#
# Markets we only ingest opt-in (btts / double_chance / draw_no_bet and the HT
# family need fetch_live_odds --additional-markets) fail this bound whenever
# that flag is not part of the running schedule. That is intended: those are
# exactly the prices the audit found rotting in the table.
MAX_ODDS_AGE_HOURS: float = 6.0

# Age of an `odds` row in hours, computed DB-side so the comparison can never
# straddle an app-vs-database clock skew. "timestamp" is quoted because it is
# also a type name in Postgres. Requires migration 024 (last_seen_at).
ODDS_AGE_HOURS_SQL: str = 'EXTRACT(EPOCH FROM (NOW() - COALESCE(odds.last_seen_at, odds."timestamp"))) / 3600.0'

# Matches a trailing signed line on a display selection: 'over_2.5' -> 2.5,
# 'home_-0.5' -> -0.5. The generators embed the line in the selection they
# store, while the `odds` table keeps the bare selection plus a numeric `line`.
_LINE_SUFFIX_RE = re.compile(r"^(?P<sel>.+?)_(?P<line>[+-]?\d+(?:\.\d+)?)$")


@dataclass(frozen=True)
class MarketGate:
    """One stream's emission policy.

    enabled   -- False disables recommendation emission entirely (predictions
                 and grading continue; only recs are suppressed).
    max_odds  -- reject if odds_decimal >= max_odds, i.e. max_odds is the
                 first REJECTED price. The racing caps come straight off
                 audit bands whose losing side INCLUDES the boundary ("win
                 recs at odds 12+", "place 6-12"), and 12.00 / 6.00 are
                 round fractional prices that occur constantly, so the
                 boundary itself must fall on the reject side.
    max_ev    -- reject if expected_value > max_ev. An implausible edge is a
                 model error, not value.
    max_gap   -- reject if (model_prob - market_consensus_prob) > max_gap.
    note      -- one-line evidence citation, shown in logs.
    """

    enabled: bool
    max_odds: float | None
    max_ev: float | None
    max_gap: float | None
    note: str


# Unknown streams behave exactly as they did before this module existed.
DEFAULT_GATE = MarketGate(
    enabled=True,
    max_odds=None,
    max_ev=None,
    max_gap=None,
    note="No audit finding for this stream; unbounded (default).",
)

# ── The gate table ────────────────────────────────────────────────────────
# Exact (sport, bet_type) matches. The `note` carries the evidence verbatim
# from the 2026-09 audit; it is what a future reader needs in order to decide
# whether the finding has been overturned.

_SOCCER_DISAGREEMENT_NOTE = (
    "Soccer ensemble is statistically indistinguishable from the closing market "
    "(paired gap +0.0026 +/- 0.0033, n=702); recs that still PASS the 5% EV gate when "
    "repriced at the real closing price LOSE (n=57, stake-weighted ROI -26.7%) while recs "
    "that FAIL it (model ~ market) WON (+38.7%) — disagreement is anti-predictive at "
    "current model quality."
)

_NO_SEASON_NOTE = (
    "Entire 2025-26 season missing from the corpus (0 finished games vs 335/1403/1398 in "
    "2024-25); NFL week-1 rolling-form windows are 61% August-2026 preseason games and the "
    "model picked UNDER on 15 of 16 week-1 totals. Re-enable after the season backfill + "
    "preseason filter land."
)

GATES: dict[tuple[str, str], MarketGate] = {
    ("mma", "moneyline"): MarketGate(
        enabled=False,
        max_odds=None,
        max_ev=None,
        max_gap=None,
        note=(
            "Held-out Brier 0.50254 (n=269) = coin flip; odds features are the hard-coded "
            "default in 1676/1676 training rows (LightGBM importance 0.0); market Brier 0.357 "
            "vs model 0.517 on the same 68 fights; settled recs -46% ROI (n=22). Re-enable only "
            "after the model has real market inputs and clears the audit §4.3 gate."
        ),
    ),
    ("tennis", "moneyline"): MarketGate(
        enabled=False,
        max_odds=None,
        max_ev=None,
        max_gap=None,
        note=(
            "Model Brier 0.489 vs market 0.333 (n=97); 24,446/24,446 training rows carry the "
            "default odds 1.54/2.60 (zero market information); live logloss 0.6929 ~ ln2; the "
            "fixture-identity bug leaves 1,745 recs unsettleable. Re-enable after the identity "
            "fix + real odds in training."
        ),
    ),
    ("soccer", "over_under"): MarketGate(
        enabled=False,
        max_odds=None,
        max_ev=None,
        max_gap=None,
        note=(
            "Dixon-Coles serves a constant prior on 96% of live matches (corr(expected total, "
            "actual total) = 0.036); settled recs ROI -22.4% CI [-40.6%, -4.2%]; the derived "
            "over-2.5 market has no discriminative skill (Brier 0.2464 vs 0.2471 for a constant "
            "base rate), in both the covered and uncovered segments."
        ),
    ),
    ("soccer", "1x2"): MarketGate(
        enabled=True,
        max_odds=8.0,
        max_ev=0.40,
        max_gap=0.10,
        note=_SOCCER_DISAGREEMENT_NOTE,
    ),
    ("soccer", "asian_handicap"): MarketGate(
        enabled=True,
        max_odds=4.0,
        max_ev=0.30,
        max_gap=0.10,
        note=(
            "The headline +34% ROI came from 28 recs graded on sign-flipped BetUS lines the book "
            "never offered; ex-BetUS the stream is -0.9% with CI [-20.9%, +20.0%]. Keep small, "
            "bounded exposure while the pricing bugs are fixed."
        ),
    ),
    ("horse_racing", "win"): MarketGate(
        enabled=True,
        max_odds=12.0,
        max_ev=1.0,
        max_gap=None,
        note=(
            "Win recs at odds 12+ n=138 ROI -22.2%; win recs in 5-7 runner fields n=182 ROI "
            "-57.1% CI [-76.1%, -35.4%]; win EVs average +0.84 and are inflated by first-book "
            "devigging. max_gap is None deliberately: racing 'consensus' is a single bookmaker "
            "today (a separate bug), so a gap cap there would be meaningless. max_ev=1.0 is NOT "
            "the audit's EV>40% band: racing EVs live on a different scale (mean +0.84 vs 0.40 "
            "cap in soccer), so a 0.40 cap here would suppress most of the stream on a number "
            "we know is inflated rather than on evidence. Racing is bounded instead by max_odds "
            "and the 5-7 runner suppression in generate_recommendations_horse_racing.py; max_ev "
            "only clips the implausible tail. Revisit once the multi-book consensus bug is fixed "
            "and racing EVs are on a comparable scale."
        ),
    ),
    ("horse_racing", "place"): MarketGate(
        enabled=True,
        max_odds=6.0,
        max_ev=1.0,
        max_gap=None,
        note=(
            "Place recs at odds 12+ n=13 won 0 (-100%), 6-12 n=45 -11.9%, under 6 n=240 +31.9%; "
            "place EVs average +0.78 and are inflated. SINGLE-SOURCE: the place pilot's profit is "
            "exchange-price capture and is -3.7% at starting-price-derived odds. max_gap is None "
            "deliberately (single-bookmaker 'consensus'), and max_ev=1.0 clips only the "
            "implausible tail for the same reason as the win gate — the binding constraint here "
            "is max_odds=6.0, which keeps the one profitable band (under 6) and drops both losing "
            "ones."
        ),
    ),
}

# Per-sport fallbacks, used when no exact (sport, bet_type) entry exists.
SPORT_DEFAULTS: dict[str, MarketGate] = {
    "nfl": MarketGate(enabled=False, max_odds=None, max_ev=None, max_gap=None, note=_NO_SEASON_NOTE),
    "nba": MarketGate(enabled=False, max_odds=None, max_ev=None, max_gap=None, note=_NO_SEASON_NOTE),
    "nhl": MarketGate(enabled=False, max_odds=None, max_ev=None, max_gap=None, note=_NO_SEASON_NOTE),
    # Every other soccer market (btts, double_chance, draw_no_bet, the HT
    # family, ...) inherits the 1x2 bounds: same model, same evidence.
    "soccer": MarketGate(enabled=True, max_odds=8.0, max_ev=0.40, max_gap=0.10, note=_SOCCER_DISAGREEMENT_NOTE),
}


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


def gate_for(sport: str, bet_type: str) -> MarketGate:
    """The gate governing one stream.

    Resolution order: exact (sport, bet_type) -> per-sport default ->
    DEFAULT_GATE (enabled, no caps). Lookups are case/whitespace insensitive.
    """
    s, b = _norm(sport), _norm(bet_type)
    gate = GATES.get((s, b))
    if gate is not None:
        return gate
    return SPORT_DEFAULTS.get(s, DEFAULT_GATE)


def split_selection_line(selection: str, line: float | None = None) -> tuple[str, float | None]:
    """Split a generator display selection into (bare_selection, line).

    The soccer generator stores 'home'/'draw'/'away' for 1x2 but 'over_2.5' /
    'home_-0.5' for lined markets, while the `odds` table stores the bare
    selection plus a numeric `line` column. Accept either form: a trailing
    '_<signed number>' is stripped, and the parsed number becomes the line when
    the caller did not pass one explicitly (an explicit `line` always wins).
    """
    sel = _norm(selection)
    m = _LINE_SUFFIX_RE.match(sel)
    if m is None:
        return sel, (float(line) if line is not None else None)
    parsed = float(m.group("line"))
    return m.group("sel"), (float(line) if line is not None else parsed)


# Markets where each side of the SAME group carries its OWN signed point:
# scripts/fetch_live_odds.py map_outcome stores the home side of a -0.5
# handicap at line=-0.5 and the away side at line=+0.5 (soccer 'spreads' ->
# asian_handicap; nhl/nba/nfl 'spreads' -> spread). Grouping on one exact
# line therefore yields ONE selection per bookmaker — nothing to de-vig
# against — so the two sides must be paired across (line, -line).
#
# Known imprecision on WHOLE-number handicaps (0.0, -1.0, ...): the two book
# prices fold the push mass in, while the caller's model_prob for those markets
# is p_raw, which excludes it. The gap is therefore measured slightly LOW on
# those lines, i.e. in the permissive direction; fix it by comparing
# p_raw/(1-push) if the AH stream is ever widened.
SIGN_FLIPPED_LINE_MARKETS: frozenset[str] = frozenset({"asian_handicap", "spread"})

# Markets whose selections OVERLAP instead of partitioning the outcome
# space, mapped to (the complete selection set, the sum the true
# probabilities take). double_chance's 1X / 12 / X2 each cover two of the
# three results, so a book's fair prices sum to 2.0, not 1.0 — normalising
# them to 1.0 would halve every consensus and turn the gap cap into a
# blanket rejection. A book that does not quote the WHOLE set is skipped:
# two of the three prices cannot be de-vigged to a known total.
OVERLAPPING_MARKETS: dict[str, tuple[frozenset[str], float]] = {
    "double_chance": (frozenset({"1x", "12", "x2"}), 2.0),
}

# The consensus obeys the SAME freshness bound as the price queries in the
# generators. Without it a rec is priced on a fresh quote but gated against a
# stale consensus — including books that stopped pricing the match — so the
# max_gap cap could pass a pick it should refuse (or refuse one it should
# pass). Books that fall out of the bound simply stop counting, and if that
# takes the group below MIN_CONSENSUS_BOOKS the answer is None ("we cannot see
# the market"), which passes_gate treats as a reject: fail closed, never
# silently thin.
_CONSENSUS_SQL = f"""
    SELECT DISTINCT ON (bookmaker, lower(selection), line)
           bookmaker, selection, line, odds_decimal
    FROM odds
    WHERE match_id = %(match_id)s
      AND market_type = %(market_type)s
      AND (line IS NOT DISTINCT FROM %(line)s
           OR line IS NOT DISTINCT FROM %(opposite_line)s)
      AND is_live = false
      AND {ODDS_AGE_HOURS_SQL} <= %(max_age_hours)s
    ORDER BY bookmaker, lower(selection), line, timestamp DESC NULLS LAST
"""


def _same_line(a, b) -> bool:
    """Line equality that treats NULL as its own value (SQL IS NOT DISTINCT
    FROM semantics) and tolerates float representation noise."""
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) < 1e-9


def market_consensus_prob(
    cur,
    match_id: str,
    market_type: str,
    selection: str,
    line: float | None = None,
    max_age_hours: float = MAX_ODDS_AGE_HOURS,
) -> float | None:
    """De-vigged multi-book consensus probability for one selection.

    Per bookmaker we take the latest non-live price for every selection in the
    same market group, normalise 1/odds over that group so the book's
    probabilities sum to the group's true total (1.0 for a partition,
    OVERLAPPING_MARKETS' multiplicity otherwise), then average the target
    selection's normalised probability across books.

    For SIGN_FLIPPED_LINE_MARKETS (asian_handicap, spread) the group spans two
    line values: the target selection at `line` and the opposite side at
    `-line`, because the ingest stores each side's own signed point. Rows from
    a book's OTHER handicap lines come back in the same query (a book quoting
    -0.5 usually also quotes +0.5) and are dropped here rather than de-vigged
    against the wrong price.

    Only quotes seen within `max_age_hours` count: the gap cap is a money-path
    safety cap, so it must be measured against the market as it is NOW, not
    against books that stopped pricing the match days ago (audit 2026-09,
    defect 1).

    Returns None — meaning "we cannot see the market" — when fewer than
    MIN_CONSENSUS_BOOKS books price the group. A book is skipped when it prices
    fewer than 2 selections in the group (nothing to de-vig against), quotes a
    non-positive price, is stale, or — for an overlapping market — does not
    quote the whole selection set.

    `cur` is a psycopg2 RealDictCursor-style cursor; exactly one query is run.
    """
    sel, resolved_line = split_selection_line(selection, line)
    market = _norm(market_type)
    sign_flipped = market in SIGN_FLIPPED_LINE_MARKETS and resolved_line is not None
    opposite_line = -resolved_line if sign_flipped else resolved_line
    required, group_total = OVERLAPPING_MARKETS.get(market, (None, 1.0))
    cur.execute(
        _CONSENSUS_SQL,
        {
            "match_id": match_id,
            "market_type": market,
            "line": resolved_line,
            "opposite_line": opposite_line,
            "max_age_hours": max_age_hours,
        },
    )
    rows = cur.fetchall() or []

    by_book: dict[str, dict[str, float]] = {}
    for row in rows:
        row_sel = _norm(row["selection"])
        if sign_flipped:
            # The target side keeps its own point; every other side belongs to
            # the group only at the negated point.
            wanted_line = resolved_line if row_sel == sel else opposite_line
            if not _same_line(row.get("line"), wanted_line):
                continue
        book = _norm(row["bookmaker"])
        price = row["odds_decimal"]
        by_book.setdefault(book, {})[row_sel] = float(price) if price is not None else 0.0

    normalised: list[float] = []
    for book, prices in by_book.items():
        if len(prices) < 2:
            continue
        if any(price <= 0.0 for price in prices.values()):
            logger.debug("Consensus: skipping %s on %s/%s — non-positive price", book, match_id, market_type)
            continue
        if sel not in prices:
            continue
        if required is not None and set(prices) != required:
            logger.debug(
                "Consensus: skipping %s on %s/%s — partial overlapping group %s",
                book,
                match_id,
                market_type,
                sorted(prices),
            )
            continue
        overround = sum(1.0 / price for price in prices.values())
        if overround <= 0.0:
            continue
        normalised.append(group_total * (1.0 / prices[sel]) / overround)

    if len(normalised) < MIN_CONSENSUS_BOOKS:
        return None
    return sum(normalised) / len(normalised)


def passes_gate(
    sport: str,
    bet_type: str,
    *,
    odds: float | None,
    ev: float | None,
    model_prob: float | None,
    market_prob: float | None,
) -> tuple[bool, str | None]:
    """Should this individual pick be emitted?

    Returns (True, None) when the pick clears its stream's gate, otherwise
    (False, "<short machine-readable reason>"). Checks run in the order
    disabled -> max_odds -> max_ev -> max_gap.

    None inputs mean "unknown", and unknown is a REJECT for any cap that is
    set: if we cannot measure the quantity the cap bounds, we cannot claim the
    pick is inside it. In particular a gate with max_gap but market_prob=None
    rejects with "no_market_consensus" — thin markets are exactly where the
    model's errors live. A cap that is None ignores its input entirely, so an
    unbounded gate passes anything. This function never raises.
    """
    s, b = _norm(sport), _norm(bet_type)
    gate = gate_for(s, b)

    if not gate.enabled:
        return False, f"stream_disabled:{s}:{b}"

    if gate.max_odds is not None:
        if odds is None:
            return False, "unknown_odds"
        # max_odds is the first REJECTED price (see MarketGate.max_odds):
        # the racing bands the caps come from are "12+" and "6-12", both of
        # which INCLUDE their boundary, and 12.00 / 6.00 are prices books
        # actually quote.
        if float(odds) >= gate.max_odds:
            return False, "odds_above_max"

    if gate.max_ev is not None:
        if ev is None:
            return False, "unknown_ev"
        if float(ev) > gate.max_ev:
            return False, "ev_above_max"

    if gate.max_gap is not None:
        if market_prob is None:
            return False, "no_market_consensus"
        if model_prob is None:
            return False, "unknown_model_prob"
        if float(model_prob) - float(market_prob) > gate.max_gap:
            return False, "gap_above_max"

    return True, None


_PURGE_PENDING_SQL = """
    DELETE FROM betting_recommendations br
    USING matches m, leagues l
    WHERE br.match_id = m.id
      AND m.league_id = l.id
      AND l.sport = %(sport)s
      AND br.status = 'pending'
      AND br.bet_type = ANY(%(bet_types)s)
      AND m.match_date > NOW()
"""


def purge_pending_recs(cur, sport: str, bet_types) -> int:
    """Delete still-`pending` recs on UPCOMING fixtures of a gated stream.

    A generator that short-circuits on a disabled stream never reaches its
    per-match `delete_pending`, so the recs written by the LAST pre-gate run
    would stay status='pending' forever and keep being served as live,
    actionable picks by `vw_active_recommendations` (status IN ('pending',
    'placed') AND m.match_date > NOW()) and by the Telegram digest. Turning a
    stream off has to remove its live book, not just stop adding to it.

    Scoped to `m.match_date > NOW()` on purpose: pending rows on fixtures that
    have already started belong to grading / scripts/repair_stranded_recs.py,
    which settles them honestly. Only picks still offered as bettable are
    withdrawn. Never touches 'placed' or settled rows.

    Returns the number of rows deleted (0 when the cursor reports no rowcount).
    """
    cur.execute(_PURGE_PENDING_SQL, {"sport": _norm(sport), "bet_types": [_norm(b) for b in bet_types]})
    deleted = getattr(cur, "rowcount", 0)
    return int(deleted) if deleted and deleted > 0 else 0


def purge_pending_recs_for_sport(database_url: str, sport: str, bet_types) -> int:
    """`purge_pending_recs` on its own connection, for the short-circuit path.

    A generator whose every emitted bet_type is disabled returns before it
    opens a connection; this is the one call it still has to make, so it owns
    the connect/commit. Logs the withdrawn count at INFO — a stream that goes
    dark must say what it took off the board. Failures are NOT swallowed: a
    stream we cannot withdraw must fail loudly rather than look withdrawn.
    """
    import psycopg2  # local import: keeps this module importable without a DB driver

    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            deleted = purge_pending_recs(cur, sport, bet_types)
        conn.commit()
    logger.info(
        "gating: withdrew %d pending %s rec(s) on upcoming fixtures (bet_types=%s)",
        deleted,
        _norm(sport),
        ",".join(_norm(b) for b in bet_types),
    )
    return deleted


def cap_stake(stake: float, bankroll: float) -> float:
    """Clamp a stake to MAX_STAKE_FRACTION of bankroll, rounded to 2dp.

    Never returns a negative number: a negative stake (or a negative bankroll)
    is nonsense, and 0.0 is the safe reading of it.
    """
    ceiling = float(bankroll) * MAX_STAKE_FRACTION
    capped = min(float(stake), ceiling)
    if capped <= 0.0:
        return 0.0
    return round(capped, 2)
