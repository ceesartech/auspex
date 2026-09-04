"""Unit tests for scripts/rec_gating.py — the single place that decides which
recommendation streams are live. Pure unit: fake cursor, no DB, no network."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rg = _load("rec_gating", REPO / "scripts" / "rec_gating.py")


class FakeCursor:
    """Serves rows for the one consensus query, recording the params it got."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.params: dict | None = None
        self.executions = 0

    def execute(self, sql, params=None):
        assert "FROM odds" in sql
        assert "is_live = false" in sql
        self.params = params
        self.executions += 1

    def fetchall(self):
        return self.rows


def _odds_row(bookmaker: str, selection: str, odds_decimal: float, line: float | None = None) -> dict:
    return {"bookmaker": bookmaker, "selection": selection, "odds_decimal": odds_decimal, "line": line}


# Three books pricing a 1x2 group, each with its own overround.
THREE_BOOK_1X2 = [
    _odds_row("alpha", "home", 2.00),
    _odds_row("alpha", "draw", 3.50),
    _odds_row("alpha", "away", 4.00),
    _odds_row("bravo", "home", 2.10),
    _odds_row("bravo", "draw", 3.40),
    _odds_row("bravo", "away", 3.80),
    _odds_row("charlie", "home", 1.95),
    _odds_row("charlie", "draw", 3.60),
    _odds_row("charlie", "away", 4.20),
]


# ── gate_for resolution ───────────────────────────────────────────────────


def test_gate_for_exact_match_wins():
    gate = rg.gate_for("soccer", "asian_handicap")
    assert gate.enabled is True
    assert gate.max_odds == 4.0
    assert gate.max_ev == 0.30
    assert "BetUS" in gate.note


def test_gate_for_falls_back_to_per_sport_default():
    # No exact ('soccer', 'btts') entry: inherits the soccer default bounds.
    gate = rg.gate_for("soccer", "btts")
    assert gate is rg.SPORT_DEFAULTS["soccer"]
    assert (gate.enabled, gate.max_odds, gate.max_ev, gate.max_gap) == (True, 8.0, 0.40, 0.10)


def test_gate_for_falls_back_to_global_default():
    gate = rg.gate_for("lottery", "jackpot")
    assert gate is rg.DEFAULT_GATE
    assert gate.enabled is True
    assert (gate.max_odds, gate.max_ev, gate.max_gap) == (None, None, None)


def test_gate_for_is_case_and_whitespace_insensitive():
    assert rg.gate_for("  MMA ", "MoneyLine") is rg.GATES[("mma", "moneyline")]


def test_every_gate_carries_evidence():
    for key, gate in rg.GATES.items():
        assert gate.note.strip(), key
        assert len(gate.note) > 40, key


# ── disabled streams ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sport,bet_type",
    [
        ("mma", "moneyline"),
        ("tennis", "moneyline"),
        ("soccer", "over_under"),
        ("nfl", "total"),
        ("nfl", "spread"),
        ("nba", "moneyline"),
        ("nhl", "puck_line"),
    ],
)
def test_disabled_streams_reject_naming_the_stream(sport, bet_type):
    assert rg.gate_for(sport, bet_type).enabled is False
    ok, reason = rg.passes_gate(
        sport,
        bet_type,
        odds=1.90,
        ev=0.08,
        model_prob=0.55,
        market_prob=0.52,
    )
    assert ok is False
    assert reason == f"stream_disabled:{sport}:{bet_type}"


def test_enabled_streams_are_exactly_the_expected_set():
    disabled = {k for k, g in rg.GATES.items() if not g.enabled}
    assert disabled == {("mma", "moneyline"), ("tennis", "moneyline"), ("soccer", "over_under")}
    assert [s for s, g in rg.SPORT_DEFAULTS.items() if not g.enabled] == ["nfl", "nba", "nhl"]


# ── caps ──────────────────────────────────────────────────────────────────


def _soccer_1x2(**kwargs):
    base = {"odds": 2.0, "ev": 0.10, "model_prob": 0.55, "market_prob": 0.52}
    base.update(kwargs)
    return rg.passes_gate("soccer", "1x2", **base)


def test_max_odds_boundary():
    # max_odds is the FIRST REJECTED price: the racing caps come off bands
    # ("odds 12+", "6-12") that include their boundary, and 12.00 / 6.00 are
    # prices books quote constantly.
    assert _soccer_1x2(odds=7.99) == (True, None)
    assert _soccer_1x2(odds=8.0) == (False, "odds_above_max")
    assert _soccer_1x2(odds=8.01) == (False, "odds_above_max")


def test_max_ev_boundary():
    assert _soccer_1x2(ev=0.39) == (True, None)
    assert _soccer_1x2(ev=0.40) == (True, None)
    assert _soccer_1x2(ev=0.4001) == (False, "ev_above_max")


def test_max_gap_boundary():
    assert _soccer_1x2(model_prob=0.61, market_prob=0.52) == (True, None)  # gap 0.09
    assert _soccer_1x2(model_prob=0.60, market_prob=0.50) == (True, None)  # gap 0.10 exactly
    assert _soccer_1x2(model_prob=0.63, market_prob=0.50) == (False, "gap_above_max")


def test_negative_gap_never_rejects():
    # Model below the market is not "disagreement we cannot bound".
    assert _soccer_1x2(model_prob=0.30, market_prob=0.52) == (True, None)


def test_missing_market_consensus_rejects_when_gap_capped():
    assert _soccer_1x2(market_prob=None) == (False, "no_market_consensus")


def test_unknown_odds_and_ev_reject_when_capped():
    assert _soccer_1x2(odds=None) == (False, "unknown_odds")
    assert _soccer_1x2(ev=None) == (False, "unknown_ev")
    assert _soccer_1x2(model_prob=None) == (False, "unknown_model_prob")


def test_check_order_is_disabled_then_odds_then_ev_then_gap():
    # An out-of-bounds pick on a disabled stream reports the stream, not a cap.
    ok, reason = rg.passes_gate("soccer", "over_under", odds=99.0, ev=9.0, model_prob=0.9, market_prob=None)
    assert (ok, reason) == (False, "stream_disabled:soccer:over_under")
    # Odds beats EV beats gap.
    assert _soccer_1x2(odds=99.0, ev=9.0, market_prob=None) == (False, "odds_above_max")
    assert _soccer_1x2(ev=9.0, market_prob=None) == (False, "ev_above_max")


def test_uncapped_gate_passes_anything_including_unknowns():
    assert rg.gate_for("lottery", "jackpot") is rg.DEFAULT_GATE
    assert rg.passes_gate("lottery", "jackpot", odds=500.0, ev=12.0, model_prob=0.99, market_prob=None) == (True, None)
    assert rg.passes_gate("lottery", "jackpot", odds=None, ev=None, model_prob=None, market_prob=None) == (True, None)


def test_horse_racing_gaps_are_not_capped():
    # Racing "consensus" is a single bookmaker today, so a gap cap is meaningless.
    for bet_type, max_odds in (("win", 12.0), ("place", 6.0)):
        gate = rg.gate_for("horse_racing", bet_type)
        assert gate.max_gap is None
        assert gate.max_odds == max_odds
        args = {"ev": 0.5, "model_prob": 0.9, "market_prob": None}
        assert rg.passes_gate("horse_racing", bet_type, odds=max_odds - 0.01, **args) == (True, None)
        # The boundary price itself sits INSIDE the audit's losing band.
        assert rg.passes_gate("horse_racing", bet_type, odds=max_odds, **args) == (False, "odds_above_max")
        assert rg.passes_gate("horse_racing", bet_type, odds=max_odds + 0.01, **args) == (False, "odds_above_max")


# ── cap_stake ─────────────────────────────────────────────────────────────


def test_cap_stake_clamps_to_max_fraction():
    assert rg.MAX_STAKE_FRACTION == 0.025
    assert rg.cap_stake(500.0, 1000.0) == 25.0


def test_cap_stake_leaves_small_stakes_alone_and_rounds():
    assert rg.cap_stake(12.3456, 1000.0) == 12.35
    assert rg.cap_stake(24.999, 1000.0) == 25.0


def test_cap_stake_rounds_the_ceiling_too():
    assert rg.cap_stake(100.0, 1234.0) == 30.85  # 1234 * 0.025 = 30.85


def test_cap_stake_never_negative():
    assert rg.cap_stake(-5.0, 1000.0) == 0.0
    assert rg.cap_stake(10.0, 0.0) == 0.0
    assert rg.cap_stake(10.0, -1000.0) == 0.0


# ── market_consensus_prob ─────────────────────────────────────────────────


def test_consensus_devigs_three_books_exactly():
    cur = FakeCursor(THREE_BOOK_1X2)
    got = rg.market_consensus_prob(cur, "match-1", "1x2", "home")

    # Hand-computed: per book, normalise 1/odds over the three selections.
    expected = (
        (1 / 2.00) / (1 / 2.00 + 1 / 3.50 + 1 / 4.00)
        + (1 / 2.10) / (1 / 2.10 + 1 / 3.40 + 1 / 3.80)
        + (1 / 1.95) / (1 / 1.95 + 1 / 3.60 + 1 / 4.20)
    ) / 3
    assert got == pytest.approx(expected, abs=1e-12)
    assert got == pytest.approx(0.48068175642238103, abs=1e-12)
    assert cur.executions == 1
    assert cur.params == {
        "match_id": "match-1",
        "market_type": "1x2",
        "line": None,
        "opposite_line": None,
    }


def test_consensus_probabilities_of_a_group_sum_to_one():
    total = sum(
        rg.market_consensus_prob(FakeCursor(THREE_BOOK_1X2), "match-1", "1x2", sel) for sel in ("home", "draw", "away")
    )
    assert total == pytest.approx(1.0, abs=1e-12)


def test_consensus_returns_none_below_three_books():
    two_books = [r for r in THREE_BOOK_1X2 if r["bookmaker"] != "charlie"]
    assert rg.market_consensus_prob(FakeCursor(two_books), "match-1", "1x2", "home") is None


def test_consensus_skips_single_selection_and_bad_price_books():
    rows = THREE_BOOK_1X2 + [
        _odds_row("delta", "home", 2.05),  # only one selection -> nothing to de-vig
        _odds_row("echo", "home", 2.05),
        _odds_row("echo", "draw", 0.0),  # non-positive price -> whole book dropped
        _odds_row("echo", "away", 4.10),
    ]
    with_junk = rg.market_consensus_prob(FakeCursor(rows), "match-1", "1x2", "home")
    clean = rg.market_consensus_prob(FakeCursor(THREE_BOOK_1X2), "match-1", "1x2", "home")
    assert with_junk == pytest.approx(clean, abs=1e-12)


def test_consensus_skips_book_that_does_not_price_the_selection():
    rows = THREE_BOOK_1X2 + [_odds_row("delta", "draw", 3.30), _odds_row("delta", "away", 4.05)]
    got = rg.market_consensus_prob(FakeCursor(rows), "match-1", "1x2", "home")
    assert got == pytest.approx(0.48068175642238103, abs=1e-12)


def test_consensus_ignores_live_rows():
    # The single query filters is_live in SQL — assert the filter is present and
    # that a cursor honouring it (returning only pre-match rows) is what we read.
    cur = FakeCursor(THREE_BOOK_1X2)
    rg.market_consensus_prob(cur, "match-1", "1x2", "home")
    assert cur.executions == 1
    # FakeCursor.execute asserts "is_live = false" appears in the SQL; a live-row
    # source is therefore never reachable through this helper.
    assert "is_live = false" in rg._CONSENSUS_SQL
    assert "IS NOT DISTINCT FROM" in rg._CONSENSUS_SQL


def test_consensus_parses_line_out_of_display_selection():
    rows = [
        _odds_row("alpha", "over", 1.90),
        _odds_row("alpha", "under", 1.95),
        _odds_row("bravo", "over", 1.87),
        _odds_row("bravo", "under", 2.00),
        _odds_row("charlie", "over", 1.92),
        _odds_row("charlie", "under", 1.93),
    ]
    cur = FakeCursor(rows)
    got = rg.market_consensus_prob(cur, "m", "over_under", "over_2.5")
    assert cur.params == {
        "match_id": "m",
        "market_type": "over_under",
        "line": 2.5,
        "opposite_line": 2.5,
    }
    assert got == pytest.approx(0.5081960244750942, abs=1e-12)


def test_consensus_accepts_bare_selection_plus_explicit_line():
    cur = FakeCursor([])
    rg.market_consensus_prob(cur, "m", "asian_handicap", "home", line=-0.5)
    # Handicap sides carry opposite signs, so the group spans (line, -line).
    assert cur.params == {
        "match_id": "m",
        "market_type": "asian_handicap",
        "line": -0.5,
        "opposite_line": 0.5,
    }


def test_explicit_line_wins_over_parsed_suffix():
    cur = FakeCursor([])
    rg.market_consensus_prob(cur, "m", "asian_handicap", "home_-0.5", line=-1.0)
    assert cur.params["line"] == -1.0


def test_consensus_empty_rows_returns_none():
    assert rg.market_consensus_prob(FakeCursor([]), "m", "1x2", "home") is None


# ── selection/line parsing ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "selection,expected",
    [
        ("home", ("home", None)),
        ("draw", ("draw", None)),
        ("over_2.5", ("over", 2.5)),
        ("under_0.5", ("under", 0.5)),
        ("home_-0.5", ("home", -0.5)),
        ("away_+1.5", ("away", 1.5)),
        ("home_-1", ("home", -1.0)),
        ("Over_2.5", ("over", 2.5)),
        ("1X", ("1x", None)),
    ],
)
def test_split_selection_line(selection, expected):
    assert rg.split_selection_line(selection) == expected


# ── sign-flipped handicap lines ───────────────────────────────────────────
#
# fetch_live_odds stores each handicap side's OWN signed point (home at -0.5,
# away at +0.5 for the SAME group), so an exact-line group holds ONE selection
# per book. Before the pairing fix every asian_handicap / spread candidate was
# rejected as "no_market_consensus" and the stream emitted nothing.


def _handicap_rows() -> list[dict]:
    rows: list[dict] = []
    for book, home, away in (
        ("alpha", 1.95, 1.90),
        ("bravo", 2.00, 1.85),
        ("charlie", 1.92, 1.94),
    ):
        rows.append(_odds_row(book, "home", home, line=-0.5))
        rows.append(_odds_row(book, "away", away, line=0.5))
        # The SAME book also quotes the mirrored handicap; those rows come back
        # from the (line, -line) query and must not be de-vigged into the group.
        rows.append(_odds_row(book, "home", 1.40, line=0.5))
        rows.append(_odds_row(book, "away", 2.80, line=-0.5))
    return rows


def test_handicap_consensus_pairs_the_two_signed_sides():
    cur = FakeCursor(_handicap_rows())
    got = rg.market_consensus_prob(cur, "m1", "asian_handicap", "home", line=-0.5)
    expected = (
        (1 / 1.95) / (1 / 1.95 + 1 / 1.90) + (1 / 2.00) / (1 / 2.00 + 1 / 1.85) + (1 / 1.92) / (1 / 1.92 + 1 / 1.94)
    ) / 3
    assert got == pytest.approx(expected, abs=1e-12)


def test_handicap_consensus_sides_sum_to_one():
    home = rg.market_consensus_prob(FakeCursor(_handicap_rows()), "m1", "asian_handicap", "home", line=-0.5)
    away = rg.market_consensus_prob(FakeCursor(_handicap_rows()), "m1", "asian_handicap", "away", line=0.5)
    assert home + away == pytest.approx(1.0, abs=1e-12)


def test_handicap_candidate_is_no_longer_rejected_for_missing_consensus():
    market_prob = rg.market_consensus_prob(FakeCursor(_handicap_rows()), "m1", "asian_handicap", "home", line=-0.5)
    assert market_prob is not None
    assert rg.passes_gate(
        "soccer",
        "asian_handicap",
        odds=2.05,
        ev=0.10,
        model_prob=0.55,
        market_prob=market_prob,
    ) == (True, None)


def test_handicap_selection_with_embedded_line_uses_the_same_group():
    got = rg.market_consensus_prob(FakeCursor(_handicap_rows()), "m1", "asian_handicap", "home_-0.5")
    bare = rg.market_consensus_prob(FakeCursor(_handicap_rows()), "m1", "asian_handicap", "home", line=-0.5)
    assert got == pytest.approx(bare, abs=1e-12)


def test_spread_shares_the_handicap_pairing():
    assert "spread" in rg.SIGN_FLIPPED_LINE_MARKETS
    rows = [
        _odds_row(book, sel, price, line=line)
        for book, home, away in (("alpha", 1.91, 1.91), ("bravo", 1.95, 1.87), ("charlie", 1.89, 1.93))
        for sel, price, line in (("home", home, -3.5), ("away", away, 3.5))
    ]
    assert rg.market_consensus_prob(FakeCursor(rows), "m1", "spread", "home", line=-3.5) is not None


def test_zero_line_handicap_still_groups_both_sides():
    rows = [
        _odds_row(book, sel, price, line=0.0)
        for book, home, away in (("alpha", 2.05, 1.80), ("bravo", 2.00, 1.85), ("charlie", 2.10, 1.78))
        for sel, price in (("home", home), ("away", away))
    ]
    got = rg.market_consensus_prob(FakeCursor(rows), "m1", "asian_handicap", "home", line=0.0)
    assert got is not None and 0.0 < got < 1.0


# ── overlapping (non-partitioning) markets ────────────────────────────────


DOUBLE_CHANCE_ROWS = [
    _odds_row(book, sel, price)
    for book, (one_x, one_two, x_two) in (
        ("alpha", (1.30, 1.25, 1.55)),
        ("bravo", (1.32, 1.24, 1.57)),
        ("charlie", (1.29, 1.26, 1.54)),
    )
    for sel, price in (("1X", one_x), ("12", one_two), ("X2", x_two))
]


def test_double_chance_consensus_sums_to_two():
    total = sum(
        rg.market_consensus_prob(FakeCursor(DOUBLE_CHANCE_ROWS), "m", "double_chance", sel)
        for sel in ("1X", "12", "X2")
    )
    # 1X / 12 / X2 each cover TWO of the three results, so the fair prices sum
    # to 2.0 — normalising them to 1.0 halves every consensus and turns the
    # soccer gap cap into a blanket rejection.
    assert total == pytest.approx(2.0, abs=1e-12)


def test_double_chance_candidate_is_not_rejected_by_a_halved_consensus():
    market_prob = rg.market_consensus_prob(FakeCursor(DOUBLE_CHANCE_ROWS), "m", "double_chance", "1X")
    assert market_prob == pytest.approx(0.6948, abs=1e-3)
    assert rg.passes_gate(
        "soccer",
        "double_chance",
        odds=1.55,
        ev=0.054,
        model_prob=0.70,
        market_prob=market_prob,
    ) == (True, None)


def test_double_chance_skips_books_quoting_a_partial_group():
    partial = [r for r in DOUBLE_CHANCE_ROWS if not (r["bookmaker"] == "charlie" and r["selection"] == "X2")]
    # charlie now prices 2 of 3 — its overround has no known total, so it is
    # dropped, leaving 2 books: below MIN_CONSENSUS_BOOKS.
    assert rg.market_consensus_prob(FakeCursor(partial), "m", "double_chance", "1X") is None


# ── purge_pending_recs ────────────────────────────────────────────────────


class FakePurgeCursor:
    def __init__(self, rowcount: int):
        self.rowcount = rowcount
        self.sql: str | None = None
        self.params: dict | None = None

    def execute(self, sql, params=None):
        self.sql = sql
        self.params = params


def test_purge_pending_recs_scopes_to_upcoming_pending_rows_of_the_sport():
    cur = FakePurgeCursor(17)
    assert rg.purge_pending_recs(cur, "NFL", ("Moneyline", "spread", "total")) == 17
    assert cur.params == {"sport": "nfl", "bet_types": ["moneyline", "spread", "total"]}
    sql = cur.sql or ""
    assert "DELETE FROM betting_recommendations" in sql
    assert "status = 'pending'" in sql  # never touches placed or settled rows
    assert "m.match_date > NOW()" in sql  # started fixtures belong to grading
    assert "l.sport = %(sport)s" in sql


def test_purge_pending_recs_handles_missing_rowcount():
    cur = FakePurgeCursor(-1)  # psycopg2 reports -1 when it has no count
    assert rg.purge_pending_recs(cur, "nfl", ("moneyline",)) == 0
