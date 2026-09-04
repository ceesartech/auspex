"""Unit tests for fetch_live_odds' current-price upsert and its per-bookmaker
side-swap canary (2026-09 prod audit, defects 1 and 2).

Defect 1: `odds` kept the FIRST price ever seen for a key forever, so every
recommendation was priced off a quote up to 8 days stale. insert_odds_row now
UPDATEs the non-opening row in place — advancing `timestamp` — when the book
moves, stamps `last_seen_at` on EVERY observation (so "still on offer" can be
told from "frozen since Tuesday"), leaves the is_opening row alone, and reports
which of insert / update / no-op happened. A post-kickoff quote may never
replace a pre-match price: `odds` is read as the pre-match line by the training
frames, so an in-play observation can only create a key that does not exist.

Defect 2: a bookmaker's home and away sides can be transposed upstream in the
feed (two whole BetUS runs in 2026-08). detect_side_swapped_books rejects a
book whose signed home handicap line opposes the median of >= 2 other books in
the SAME payload AND whose moneyline disagrees with those books about which
team is the favourite; the whole book is then dropped for that match, loudly
(logger.error) and counted, never silently.

Fake cursor throughout — no DB, no network.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


flo = _load("fetch_live_odds", "fetch_live_odds.py")

MATCH = "11111111-1111-1111-1111-111111111111"
HOME, AWAY = "Home FC", "Away FC"


def _norm(sql: str) -> str:
    return " ".join(sql.split())


def _same_line(a, b) -> bool:
    """COALESCE(line, -1) equality, the writers' own key semantics."""
    return (-1.0 if a is None else float(a)) == (-1.0 if b is None else float(b))


class FakeCursor:
    """Stand-in for the `odds` + `odds_snapshots` tables (and the one
    `matches` lookup that decides whether the fixture has kicked off).

    Routes by statement shape exactly as psycopg2 would see it, so a test can
    assert both on the resulting rows and on the SQL text that produced them.
    `timestamp` / `last_seen_at` are a monotonic tick standing in for NOW().
    """

    def __init__(self, rows=None, *, started: bool = False):
        self.rows: list[dict] = [dict(r) for r in (rows or [])]
        self.snapshots: list[tuple] = []
        self.statements: list[tuple[str, tuple]] = []
        self.rowcount = 0
        self.started = started  # has the fixture already commenced?
        self._pending: list = []
        # Monotonic stand-in for NOW(), starting after any seeded row.
        self._tick = max([int(r.get("timestamp") or 0) for r in self.rows] + [0])

    # -- helpers ------------------------------------------------------
    def sql(self, i: int) -> str:
        return self.statements[i][0]

    @property
    def updates(self) -> list[str]:
        return [s for s, _ in self.statements if s.startswith("UPDATE odds")]

    @property
    def price_updates(self) -> list[str]:
        return [s for s in self.updates if "odds_decimal = ROUND" in s]

    @property
    def seen_updates(self) -> list[str]:
        return [s for s in self.updates if "odds_decimal = ROUND" not in s]

    @property
    def inserts(self) -> list[str]:
        return [s for s, _ in self.statements if s.startswith("INSERT INTO odds (")]

    def current_rows(self) -> list[dict]:
        return [r for r in self.rows if not r["is_opening"]]

    def _match(self, match_id, book, market, sel, line):
        return [
            r
            for r in self.rows
            if (not r["is_opening"])
            and r["match_id"] == match_id
            and r["bookmaker"] == book
            and r["market_type"] == market
            and r["selection"] == sel
            and _same_line(r["line"], line)
        ]

    # -- cursor API ---------------------------------------------------
    def execute(self, sql, params=None):
        text = _norm(sql)
        self.statements.append((text, params))
        if text.startswith("SELECT 1 FROM matches"):
            assert "match_date > NOW()" in text, "the kickoff test must be DB-side"
            self._pending = [] if self.started else [(1,)]
            self.rowcount = len(self._pending)
        elif text.startswith("SELECT 1 FROM odds"):
            self._pending = [(1,)] if self._match(*params) else []
            self.rowcount = len(self._pending)
        elif text.startswith("UPDATE odds"):
            self._do_update(text, params)
        elif text.startswith("INSERT INTO odds ("):
            self._do_insert(params)
        elif text.startswith("INSERT INTO odds_snapshots"):
            self.snapshots.append(params)
            self.rowcount = 1
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected SQL: {text[:60]}")

    def fetchone(self):
        return self._pending.pop(0) if self._pending else None

    def _do_update(self, text, params):
        assert "COALESCE(is_opening, false) = false" in text, "the opening row must be out of scope"
        assert "COALESCE(is_live, false) = false" in text, "a live price is a different series"
        if "odds_decimal = ROUND" in text:
            price, match_id, book, market, sel, line, price2 = params
            assert price == price2, "price is bound twice (SET + IS DISTINCT FROM)"
            assert "timestamp = NOW()" in text, "a price MOVE advances timestamp"
            assert "last_seen_at = NOW()" in text, "a price move is also a sighting"
            assert "IS DISTINCT FROM" in text, "an unchanged price must not be rewritten"
            hits = [
                r
                for r in self._match(match_id, book, market, sel, line)
                if round(r["odds_decimal"], 4) != round(price, 4)
            ]
            for row in hits:
                self._tick += 1
                row["odds_decimal"] = round(price, 4)
                row["timestamp"] = self._tick
                row["last_seen_at"] = self._tick
            self.rowcount = len(hits)
            return
        # last-seen-only stamp: the book still shows the same price
        assert "last_seen_at = NOW()" in text
        assert "timestamp" not in text.split("WHERE")[0], "an unchanged price must not churn `timestamp`"
        hits = self._match(*params)
        for row in hits:
            self._tick += 1
            row["last_seen_at"] = self._tick
        self.rowcount = len(hits)

    def _do_insert(self, params):
        (match_id, book, market, sel, price, line, g_match, g_book, g_market, g_sel, g_line) = params
        assert (match_id, book, market, sel, line) == (g_match, g_book, g_market, g_sel, g_line)
        if self._match(match_id, book, market, sel, line):
            self.rowcount = 0
            return
        self._tick += 1
        self.rows.append(
            {
                "match_id": match_id,
                "bookmaker": book,
                "market_type": market,
                "selection": sel,
                "odds_decimal": round(price, 4),
                "line": line,
                "is_opening": False,
                "timestamp": self._tick,
                "last_seen_at": self._tick,
            }
        )
        self.rowcount = 1


def _row(price, *, market="1x2", sel="home", line=None, is_opening=False, book="BookA", ts=0):
    return {
        "match_id": MATCH,
        "bookmaker": book,
        "market_type": market,
        "selection": sel,
        "odds_decimal": price,
        "line": line,
        "is_opening": is_opening,
        "timestamp": ts,
        "last_seen_at": ts,
    }


def _upsert(cur, price, *, market="1x2", sel="home", line=None, book="BookA", pre_match=True):
    return flo.insert_odds_row(cur, MATCH, book, market, sel, price, line, pre_match=pre_match)


# ── A. the odds table holds the CURRENT price ─────────────────────────


@pytest.mark.unit
class TestCurrentPriceUpsert:
    def test_changed_price_updates_in_place_and_advances_timestamp(self):
        cur = FakeCursor([_row(2.10, ts=5)])
        assert _upsert(cur, 2.45) == flo.ODDS_UPDATED
        assert len(cur.rows) == 1, "a price move must overwrite, not append"
        assert cur.rows[0]["odds_decimal"] == 2.45
        assert cur.rows[0]["timestamp"] > 5, "timestamp must advance so movement is measurable"
        assert cur.rows[0]["last_seen_at"] > 5
        # UPDATE first; the INSERT is not even attempted once a row was hit.
        assert len(cur.price_updates) == 1 and cur.seen_updates == [] and cur.inserts == []

    def test_unchanged_price_still_stamps_last_seen(self):
        # THE point of last_seen_at: the price did not move, but the book is
        # still showing it, and only that distinguishes a live-but-static
        # market from a book that pulled the market days ago.
        cur = FakeCursor([_row(2.10, ts=5)])
        assert _upsert(cur, 2.10) == flo.ODDS_UNCHANGED
        row = cur.rows[0]
        assert row["odds_decimal"] == 2.10
        assert row["timestamp"] == 5, "an unchanged price must not churn `timestamp`"
        assert row["last_seen_at"] > 5, "…but we DID just see it on offer"
        # One no-op price UPDATE, then the stamp; no INSERT is attempted.
        assert len(cur.price_updates) == 1 and len(cur.seen_updates) == 1 and cur.inserts == []

    def test_first_sighting_inserts_with_a_last_seen(self):
        cur = FakeCursor([])
        assert _upsert(cur, 2.10) == flo.ODDS_INSERTED
        assert len(cur.current_rows()) == 1
        assert cur.current_rows()[0]["last_seen_at"] is not None
        assert "WHERE NOT EXISTS" in cur.inserts[0], "the insert stays guarded"
        assert "last_seen_at" in cur.inserts[0], "a new row is seen the moment it is written"

    def test_price_rounds_to_the_column_scale_before_comparing(self):
        # DECIMAL(10,4): 1.90909090... stores as 1.9091 and must then compare
        # EQUAL on the next cycle, or every run rewrites the row forever.
        cur = FakeCursor([_row(1.9091, ts=5)])
        assert _upsert(cur, 1.9090909090909092) == flo.ODDS_UNCHANGED
        assert cur.rows[0]["timestamp"] == 5
        assert "ROUND(%s::numeric, 4)" in cur.price_updates[0]

    def test_opening_row_is_never_modified(self):
        opening = _row(2.10, is_opening=True, ts=1)
        cur = FakeCursor([opening])
        assert _upsert(cur, 3.60) == flo.ODDS_INSERTED
        assert cur.rows[0] == _row(2.10, is_opening=True, ts=1), "the opening line is immutable"
        assert [r["odds_decimal"] for r in cur.current_rows()] == [3.60]
        # Every statement is scoped away from the opening row, NULL-safely:
        # is_opening is NULLable, and the same key is what migration 024 makes
        # unique, so guard and index must agree exactly.
        for text in (cur.price_updates[0], cur.inserts[0]):
            assert "COALESCE(is_opening, false) = false" in text
            assert "COALESCE(is_live, false) = false" in text

    def test_opening_row_untouched_when_a_current_row_also_exists(self):
        cur = FakeCursor([_row(2.10, is_opening=True, ts=1), _row(2.20, ts=2)])
        assert _upsert(cur, 2.55) == flo.ODDS_UPDATED
        assert cur.rows[0]["odds_decimal"] == 2.10 and cur.rows[0]["timestamp"] == 1
        assert cur.rows[1]["odds_decimal"] == 2.55

    def test_null_line_round_trips(self):
        cur = FakeCursor([])
        assert _upsert(cur, 2.10, market="1x2", sel="home", line=None) == flo.ODDS_INSERTED
        assert cur.current_rows()[0]["line"] is None
        # Same NULL-line key again: matched, not duplicated.
        assert _upsert(cur, 2.10, market="1x2", sel="home", line=None) == flo.ODDS_UNCHANGED
        assert _upsert(cur, 2.30, market="1x2", sel="home", line=None) == flo.ODDS_UPDATED
        assert len(cur.current_rows()) == 1
        assert cur.current_rows()[0]["odds_decimal"] == 2.30
        # NULL lines only dedupe because the key COALESCEs them.
        assert "COALESCE(line, -1) = COALESCE(%s, -1)" in cur.price_updates[0]
        assert "COALESCE(line, -1) = COALESCE(%s, -1)" in cur.inserts[0]

    def test_distinct_lines_are_distinct_keys(self):
        cur = FakeCursor([])
        assert _upsert(cur, 1.95, market="over_under", sel="over", line=2.5) == flo.ODDS_INSERTED
        assert _upsert(cur, 2.60, market="over_under", sel="over", line=3.5) == flo.ODDS_INSERTED
        assert len(cur.current_rows()) == 2

    def test_returns_the_documented_status_strings(self):
        assert flo.ODDS_WRITTEN == frozenset({flo.ODDS_INSERTED, flo.ODDS_UPDATED})
        assert flo.ODDS_UNCHANGED not in flo.ODDS_WRITTEN
        assert flo.ODDS_POST_COMMENCE not in flo.ODDS_WRITTEN


# ── B. an in-play quote may not overwrite the pre-match price ─────────


@pytest.mark.unit
class TestPostCommenceProtection:
    def test_started_match_never_overwrites_an_existing_price(self):
        cur = FakeCursor([_row(2.10, ts=5)], started=True)
        assert _upsert(cur, 4.50, pre_match=False) == flo.ODDS_POST_COMMENCE
        assert cur.rows[0]["odds_decimal"] == 2.10, "the pre-match price is what training reads"
        assert cur.rows[0]["timestamp"] == 5 and cur.rows[0]["last_seen_at"] == 5
        assert cur.updates == [] and cur.inserts == []

    def test_started_match_can_still_create_a_missing_key(self):
        # This is the historical-backfill path: scripts/backfill_historical_odds.py
        # deliberately snapshots at/after commence, and those rows are the only
        # closing lines we have for finished games.
        cur = FakeCursor([], started=True)
        assert _upsert(cur, 2.10, pre_match=False) == flo.ODDS_INSERTED
        assert len(cur.current_rows()) == 1

    def test_insert_event_odds_asks_the_database_whether_it_kicked_off(self):
        event = _event({"BookA": [_h2h_market()]})
        cur = FakeCursor([], started=True)
        stats: dict = {}
        flo.insert_event_odds(cur, MATCH, event, "soccer", HOME, AWAY, stats=stats)
        assert cur.statements[0][0].startswith("SELECT 1 FROM matches"), "one kickoff lookup per event"
        assert stats[flo.ODDS_INSERTED] == 3

        # Second pass on a started match: prices moved, but nothing is rewritten.
        moved = _event({"BookA": [_h2h_market(home_price=9.9)]})
        stats2: dict = {}
        assert flo.insert_event_odds(cur, MATCH, moved, "soccer", HOME, AWAY, stats=stats2) == 0
        assert stats2[flo.ODDS_POST_COMMENCE] == 3
        assert {r["odds_decimal"] for r in cur.current_rows()} == {2.10, 3.40, 3.60}

    def test_unknown_fixture_is_treated_as_started(self):
        # is_pre_match answers False for a match_date it cannot find: unknown
        # means "do not overwrite".
        cur = FakeCursor([], started=True)
        assert flo.is_pre_match(cur, MATCH) is False


# ── C. the side-swap canary ───────────────────────────────────────────


def _spread_market(home_line: float, price: float = 1.91):
    return {
        "key": "spreads",
        "outcomes": [
            {"name": HOME, "price": price, "point": home_line},
            {"name": AWAY, "price": price, "point": -home_line},
        ],
    }


def _totals_market(line: float):
    return {
        "key": "totals",
        "outcomes": [
            {"name": "Over", "price": 1.90, "point": line},
            {"name": "Under", "price": 1.95, "point": line},
        ],
    }


def _h2h_market(home_price: float = 2.10, away_price: float = 3.60):
    return {
        "key": "h2h",
        "outcomes": [
            {"name": HOME, "price": home_price},
            {"name": "Draw", "price": 3.40},
            {"name": AWAY, "price": away_price},
        ],
    }


def _swapped_h2h():
    """The same prices with home and away transposed — what the feed did."""
    return _h2h_market(home_price=3.60, away_price=2.10)


def _event(books: dict):
    return {"bookmakers": [{"title": title, "markets": markets} for title, markets in books.items()]}


@pytest.mark.unit
class TestSideSwapCanary:
    def test_rejects_the_book_whose_line_and_moneyline_both_oppose_the_market(self):
        event = _event(
            {
                "BetOnline": [_spread_market(-0.5), _h2h_market()],
                "LowVig": [_spread_market(-0.5), _h2h_market()],
                "Bovada": [_spread_market(-0.75), _h2h_market()],
                "BetUS": [_spread_market(+0.5), _swapped_h2h()],
            }
        )
        flagged = flo.detect_side_swapped_books(event, "soccer", HOME, AWAY)
        assert set(flagged) == {"BetUS"}, "only the book opposing the market is rejected"
        detail = flagged["BetUS"]
        assert detail["line"] == 0.5 and detail["line_consensus"] == -0.5 and detail["peers"] == 3
        # the corroborating half: the two disagree about which team is better
        assert detail["h2h_margin"] < 0 < detail["h2h_margin_consensus"]

    def test_keeps_every_book_when_they_agree(self):
        event = _event(
            {
                "BetOnline": [_spread_market(-0.5), _h2h_market()],
                "LowVig": [_spread_market(-0.25), _h2h_market()],
                "Bovada": [_spread_market(-0.75), _h2h_market()],
            }
        )
        assert flo.detect_side_swapped_books(event, "soccer", HOME, AWAY) == {}

    def test_a_line_flip_alone_is_not_enough_and_is_logged(self, caplog):
        # The structural false positive: on a near-even game a minority book
        # genuinely posts the handicap the other way round. Its moneyline
        # agrees with the market about who is favourite, so it is KEPT — and
        # the suspicion is logged so the base rate stays visible.
        event = _event(
            {
                "BetOnline": [_spread_market(-0.5), _h2h_market()],
                "LowVig": [_spread_market(-0.5), _h2h_market()],
                "Bovada": [_spread_market(-0.5), _h2h_market()],
                "MinorityBook": [_spread_market(+0.5), _h2h_market()],
            }
        )
        with caplog.at_level(logging.WARNING, logger="fetch_live_odds"):
            assert flo.detect_side_swapped_books(event, "soccer", HOME, AWAY) == {}
        assert "SIDE-SWAP SUSPECTED BUT NOT CORROBORATED" in caplog.text
        assert "MinorityBook" in caplog.text

    def test_a_pickem_moneyline_does_not_corroborate(self):
        # The two sides are priced within SIDE_SWAP_H2H_MARGIN of each other:
        # there is no favourite to disagree about, so a line flip proves
        # nothing.
        pickem = _h2h_market(home_price=2.00, away_price=2.00)
        event = _event(
            {
                "BetOnline": [_spread_market(-0.5), pickem],
                "LowVig": [_spread_market(-0.5), pickem],
                "Bovada": [_spread_market(-0.5), pickem],
                "BetUS": [_spread_market(+0.5), pickem],
            }
        )
        assert flo.detect_side_swapped_books(event, "soccer", HOME, AWAY) == {}
        assert flo.SIDE_SWAP_H2H_MARGIN == 0.04

    def test_no_moneyline_in_the_payload_means_no_corroboration(self):
        event = _event(
            {
                "BetOnline": [_spread_market(-0.5)],
                "LowVig": [_spread_market(-0.5)],
                "Bovada": [_spread_market(-0.5)],
                "BetUS": [_spread_market(+0.5)],
            }
        )
        assert flo.detect_side_swapped_books(event, "soccer", HOME, AWAY) == {}

    def test_does_not_fire_on_totals(self):
        # Over and under legitimately share one positive line — a sign test
        # there would reject every book on every match.
        event = _event(
            {
                "BetOnline": [_totals_market(2.5), _h2h_market()],
                "LowVig": [_totals_market(2.5), _h2h_market()],
                "Bovada": [_totals_market(3.5), _h2h_market()],
                "BetUS": [_totals_market(2.5), _swapped_h2h()],
            }
        )
        assert flo.detect_side_swapped_books(event, "soccer", HOME, AWAY) == {}
        assert "over_under" not in flo.HANDICAP_MARKET_TYPES
        assert "total" not in flo.HANDICAP_MARKET_TYPES

    def test_does_not_fire_with_fewer_than_two_peers(self):
        event = _event(
            {
                "BetOnline": [_spread_market(-0.5), _h2h_market()],
                "BetUS": [_spread_market(+0.5), _swapped_h2h()],
            }
        )
        assert flo.detect_side_swapped_books(event, "soccer", HOME, AWAY) == {}
        assert flo.MIN_SIDE_SWAP_PEERS == 2

    def test_does_not_fire_on_a_pickem_line(self):
        event = _event(
            {
                "BetOnline": [_spread_market(-0.5), _h2h_market()],
                "LowVig": [_spread_market(-0.5), _h2h_market()],
                "Bovada": [_spread_market(-0.5), _h2h_market()],
                "BetUS": [_spread_market(0.0), _swapped_h2h()],
            }
        )
        assert flo.detect_side_swapped_books(event, "soccer", HOME, AWAY) == {}

    def test_applies_to_team_sport_spreads_too(self):
        # NHL puck line: ±1.5 always, so the sign IS the favourite — which is
        # exactly why the moneyline has to corroborate before we drop a book.
        event = _event(
            {
                "BetOnline": [_spread_market(-1.5), _h2h_market()],
                "LowVig": [_spread_market(-1.5), _h2h_market()],
                "Bovada": [_spread_market(-1.5), _h2h_market()],
                "BetUS": [_spread_market(+1.5), _swapped_h2h()],
            }
        )
        assert set(flo.detect_side_swapped_books(event, "nhl", HOME, AWAY)) == {"BetUS"}
        assert flo.HANDICAP_MARKET_TYPES == frozenset({"asian_handicap", "spread"})
        assert flo.MONEYLINE_MARKET_TYPES == frozenset({"1x2", "moneyline"})

    def test_a_book_with_no_handicap_prices_is_never_flagged(self):
        event = _event(
            {
                "BetOnline": [_spread_market(-0.5), _h2h_market()],
                "LowVig": [_spread_market(-0.5), _h2h_market()],
                "Bovada": [_spread_market(-0.5), _h2h_market()],
                "MoneylineOnly": [_swapped_h2h()],
            }
        )
        assert flo.detect_side_swapped_books(event, "soccer", HOME, AWAY) == {}


@pytest.mark.unit
class TestCanaryWiring:
    """insert_event_odds must drop the flagged book WHOLE, count it and shout."""

    def _swapped_event(self):
        return _event(
            {
                "BetOnline": [_spread_market(-0.5), _h2h_market()],
                "LowVig": [_spread_market(-0.5), _h2h_market()],
                "Bovada": [_spread_market(-0.5), _h2h_market()],
                "BetUS": [_spread_market(+0.5), _swapped_h2h()],
            }
        )

    def test_the_whole_flagged_book_is_dropped_counted_and_shouted(self, caplog):
        cur = FakeCursor([])
        stats: dict = {}
        with caplog.at_level(logging.ERROR, logger="fetch_live_odds"):
            written = flo.insert_event_odds(cur, MATCH, self._swapped_event(), "soccer", HOME, AWAY, stats=stats)

        books = {r["bookmaker"] for r in cur.current_rows()}
        assert books == {"BetOnline", "LowVig", "Bovada"}
        # Not one market of a transposed book survives: its 1x2 price is the
        # same phantom-price bug moved into a LIVE stream (soccer 1x2).
        assert "BetUS" not in books, "the transposed payload must never reach `odds`"
        # ...and it must not poison the CLV series either.
        assert not any(p[1] == "BetUS" for p in cur.snapshots)
        assert stats["side_swap_rejections"] == 1
        assert written == len(cur.current_rows())

        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1
        msg = errors[0].getMessage()
        assert "SIDE-SWAP CANARY" in msg and "BetUS" in msg and MATCH in msg
        assert "+0.50" in msg and "-0.50" in msg, "the line and the consensus are both named"
        assert "p_home-p_away=" in msg, "the corroborating moneyline is named too"

    def test_an_uncorroborated_book_keeps_all_of_its_markets(self):
        cur = FakeCursor([])
        event = _event(
            {
                "BetOnline": [_spread_market(-0.5), _h2h_market()],
                "LowVig": [_spread_market(-0.5), _h2h_market()],
                "Bovada": [_spread_market(-0.5), _h2h_market()],
                "MinorityBook": [_spread_market(+0.5), _h2h_market()],
            }
        )
        stats: dict = {}
        flo.insert_event_odds(cur, MATCH, event, "soccer", HOME, AWAY, stats=stats)
        minority = [r for r in cur.current_rows() if r["bookmaker"] == "MinorityBook"]
        assert {r["market_type"] for r in minority} == {"asian_handicap", "1x2"}
        assert stats.get("side_swap_rejections", 0) == 0

    def test_clean_payload_writes_everything_and_counts_the_split(self):
        cur = FakeCursor([])
        stats: dict = {}
        event = _event({"BetOnline": [_spread_market(-0.5)], "LowVig": [_spread_market(-0.5)]})

        first = flo.insert_event_odds(cur, MATCH, event, "soccer", HOME, AWAY, stats=stats)
        assert first == 4 and stats[flo.ODDS_INSERTED] == 4
        assert stats.get("side_swap_rejections", 0) == 0

        # Same payload again: nothing moved, so nothing is written — but every
        # row's last_seen_at advances, which is what keeps it bettable.
        stats2: dict = {}
        assert flo.insert_event_odds(cur, MATCH, event, "soccer", HOME, AWAY, stats=stats2) == 0
        assert stats2[flo.ODDS_UNCHANGED] == 4
        assert all(r["last_seen_at"] > r["timestamp"] for r in cur.current_rows())

        # A moved price is written as an UPDATE, not a second row.
        stats3: dict = {}
        moved = _event({"BetOnline": [_spread_market(-0.5, price=2.05)], "LowVig": [_spread_market(-0.5)]})
        assert flo.insert_event_odds(cur, MATCH, moved, "soccer", HOME, AWAY, stats=stats3) == 2
        assert stats3[flo.ODDS_UPDATED] == 2 and stats3[flo.ODDS_UNCHANGED] == 2
        assert len(cur.current_rows()) == 4
