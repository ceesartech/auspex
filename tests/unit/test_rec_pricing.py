"""Unit tests for the recommendation-pricing freshness guard (audit 2026-09).

A recommendation may never be priced off a quote that has gone cold. The
`odds` table holds ONE row per (match, bookmaker, market, selection, line),
so a book that stops quoting leaves its last price behind forever — and
before this guard those frozen quotes won the "best price" pick and priced
the live book (prod rec prices averaged 178 h old on soccer 1x2, 197 h on
MMA; 48% of settled 1x2 recs failed their own 5% EV gate at the price
actually available when they were written).

What is asserted here:
  * ONE definition of the bound, imported by all seven generators.
  * Every odds query selects the quote's age and prefers a FRESH quote over
    a merely bigger one.
  * A fresh quote is priced and written; a cold one is refused, counted
    under 'stale_odds', and never inserted.
  * The age lands in the rec's risk_factors JSON (list shape preserved).
  * Milestone A's gating — stream disable, caps, suppression counters, the
    log summary and cap_stake — still behaves exactly as before.

Pure unit: fake cursors, no DB, no network.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"


def _load(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rec_gating = _load("rec_gating", "rec_gating.py")
gr = _load("generate_recommendations", "generate_recommendations.py")
gr_nhl = _load("generate_recommendations_nhl", "generate_recommendations_nhl.py")
gr_nba = _load("generate_recommendations_nba", "generate_recommendations_nba.py")
gr_nfl = _load("generate_recommendations_nfl", "generate_recommendations_nfl.py")
gr_mma = _load("generate_recommendations_mma", "generate_recommendations_mma.py")
gr_tennis = _load("generate_recommendations_tennis", "generate_recommendations_tennis.py")
gr_hr = _load("generate_recommendations_horse_racing", "generate_recommendations_horse_racing.py")

TEAM_MODULES = (gr, gr_nhl, gr_nba, gr_nfl, gr_mma, gr_tennis)
ALL_MODULES = TEAM_MODULES + (gr_hr,)


# ── Fake cursor ───────────────────────────────────────────────────────────


class FakeCursor:
    """RealDictCursor stand-in that answers each query by its SQL text.

    Dispatching on the text (rather than on call order) keeps the tests
    stable when a generator adds or reorders a lookup, and it lets the tests
    assert on the SQL the generators actually sent."""

    def __init__(self, *, predictions=(), odds=(), consensus=(), candidates=(), one=None):
        self.predictions = list(predictions)
        self.odds = list(odds)
        self.consensus = list(consensus)
        self.candidates = list(candidates)
        self.one = one
        self.statements: list[tuple[str, object]] = []
        self.inserts: list[dict] = []
        self.deletes: list[tuple[str, object]] = []
        self._last = ""

    # -- DB API ------------------------------------------------------------
    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        self._last = sql
        if "INSERT INTO" in sql:
            self.inserts.append(params)
        if "DELETE FROM" in sql:
            self.deletes.append((sql, params))

    def fetchall(self):
        sql = self._last
        if "FROM predictions" in sql:
            return list(self.predictions)
        if "DISTINCT ON (bookmaker" in sql:  # rec_gating.market_consensus_prob
            return list(self.consensus)
        if "race_entrants" in sql:
            return list(self.candidates)
        if "FROM odds" in sql:
            return list(self.odds)
        return []

    def fetchone(self):
        return self.one

    # -- helpers -----------------------------------------------------------
    def sql_for(self, needle: str) -> str:
        """The one executed statement containing `needle`."""
        hits = [s for s, _ in self.statements if needle in s]
        assert len(hits) == 1, f"expected exactly one statement containing {needle!r}, got {len(hits)}"
        return hits[0]

    def params_for(self, needle: str):
        hits = [p for s, p in self.statements if needle in s]
        assert len(hits) == 1
        return hits[0]


def odds_row(**over):
    row = {
        "market_type": "1x2",
        "selection": "home",
        "line": None,
        "bookmaker": "alpha",
        "odds_decimal": 2.2,
        "odds_age_hours": 0.5,
    }
    row.update(over)
    return row


def prediction_row(**over):
    row = {
        "prediction_type": "match_result",
        "prediction_id": "pred-1",
        "probabilities": {"home": 0.60, "draw": 0.25, "away": 0.15},
    }
    row.update(over)
    return row


# Three books whose de-vigged home price (~0.567) sits close enough to the
# model's 0.60 to clear the soccer gap cap of 0.10.
CONSENSUS_1X2 = [
    {"bookmaker": b, "selection": sel, "odds_decimal": price, "line": None}
    for b in ("alpha", "bravo", "charlie")
    for sel, price in (("home", 1.7), ("draw", 4.0), ("away", 5.0))
]


def soccer_cursor(odds=None, **kw):
    return FakeCursor(
        predictions=[prediction_row()],
        odds=list(odds if odds is not None else [odds_row()]),
        consensus=CONSENSUS_1X2,
        **kw,
    )


# ── The bound itself ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestBound:
    def test_bound_is_a_documented_number_of_hours(self):
        assert isinstance(gr.MAX_ODDS_AGE_HOURS, float)
        # 6 h = 4 cycles of the 90-minute odds DAG, so a book still on offer
        # is re-seen four times inside the bound and two consecutive failed
        # runs cannot empty the book. It is measurable because the ingest
        # stamps odds.last_seen_at on EVERY observation (migration 024), not
        # only when the price moves. The comment above the constant carries
        # the full reasoning — keep them in step.
        assert gr.MAX_ODDS_AGE_HOURS == 6.0
        assert gr.STALE_ODDS_REASON == "stale_odds"

    def test_the_bound_has_exactly_one_home(self):
        # rec_gating owns it: it is a money-path policy like the stream caps,
        # and the gate's own consensus query has to obey the same bound. The
        # generators re-export rec_gating's value; they never pick their own.
        assert gr.MAX_ODDS_AGE_HOURS is rec_gating.MAX_ODDS_AGE_HOURS
        assert gr.ODDS_AGE_HOURS_SQL is rec_gating.ODDS_AGE_HOURS_SQL

    def test_reasoning_is_written_down_next_to_the_constant(self):
        src = (SCRIPTS / "rec_gating.py").read_text()
        head = src.split("MAX_ODDS_AGE_HOURS: float")[0]
        for marker in ("178 h", "90 minutes", "last_seen_at"):
            assert marker in head, f"the bound's rationale lost its {marker!r} reasoning"

    def test_age_is_measured_off_last_seen_not_last_move(self):
        # odds."timestamp" advances only when the PRICE MOVES, so ageing off
        # it would call a book that pulled its market 5 h ago "fresh". The
        # ingest stamps last_seen_at on every sighting; COALESCE keeps rows
        # written before migration 024 reading as older than they are.
        assert "last_seen_at" in gr.ODDS_AGE_HOURS_SQL
        assert "COALESCE" in gr.ODDS_AGE_HOURS_SQL

    @pytest.mark.parametrize("mod", ALL_MODULES, ids=lambda m: m.__name__)
    def test_every_generator_shares_one_bound(self, mod):
        # Imported, never re-declared (see the source check below): two
        # generators disagreeing about "too old to bet on" is how a stream
        # silently drifts. Compared by value, not identity — other test
        # modules re-exec these files under the same module names, so a
        # given generator can legitimately be holding an earlier load's
        # constant.
        assert mod.MAX_ODDS_AGE_HOURS == gr.MAX_ODDS_AGE_HOURS
        assert mod.STALE_ODDS_REASON == gr.STALE_ODDS_REASON

    def test_no_generator_redefines_the_bound(self):
        assign = re.compile(r"^MAX_ODDS_AGE_HOURS\s*[:=]", re.M)
        for path in sorted(SCRIPTS.glob("generate_recommendations*.py")):
            if path.name == "generate_recommendations.py":
                assert assign.search(path.read_text()), "the shared bound must live here"
                continue
            assert not assign.search(path.read_text()), f"{path.name} re-declares MAX_ODDS_AGE_HOURS"


@pytest.mark.unit
class TestStalenessHelpers:
    def test_fresh_quote_is_not_stale(self):
        assert gr.is_stale_odds(0.0) is False
        assert gr.is_stale_odds(5.9) is False

    def test_boundary_is_inclusive(self):
        # "older than the bound" — a quote exactly at the bound still prices.
        assert gr.is_stale_odds(gr.MAX_ODDS_AGE_HOURS) is False
        assert gr.is_stale_odds(gr.MAX_ODDS_AGE_HOURS + 0.01) is True

    def test_unknown_age_is_stale(self):
        # Same rule rec_gating.passes_gate applies to unknown inputs: we
        # cannot claim a price was available if we cannot date it.
        assert gr.is_stale_odds(None) is True
        assert gr.is_stale_odds("not-a-number") is True

    def test_custom_bound_is_honoured(self):
        assert gr.is_stale_odds(2.0, 1.0) is True
        assert gr.is_stale_odds(2.0, 6.0) is False

    def test_age_is_read_off_the_row(self):
        assert gr.odds_age_hours({"odds_age_hours": "3.5"}) == 3.5
        assert gr.odds_age_hours({"odds_age_hours": None}) is None
        assert gr.odds_age_hours({}) is None
        assert gr.odds_age_hours(None) is None

    def test_age_tag_shape(self):
        assert gr.odds_age_tag(0.5) == "odds_age_hours:0.50"
        assert gr.odds_age_tag(None) == "odds_age_hours:unknown"

    def test_with_odds_age_appends_without_mutating(self):
        risks = ["longshot"]
        out = gr.with_odds_age(risks, 1.25)
        assert out == ["longshot", "odds_age_hours:1.25"]
        assert risks == ["longshot"], "risk_factors list must not be mutated in place"
        assert all(isinstance(tag, str) for tag in out), "risk_factors stays a list of string tags"


# ── The SQL every generator sends ─────────────────────────────────────────


@pytest.mark.unit
class TestOddsQueries:
    """Each odds query must (a) report the quote's age and (b) break ties on
    freshness BEFORE price — otherwise the book that stopped quoting at the
    highest number wins the selection forever and the guard just suppresses
    a market other books are still pricing."""

    def _assert_fresh_first(self, sql: str):
        assert "odds_age_hours" in sql
        fresh = sql.index("<= %s) DESC NULLS LAST")
        price = sql.index("odds_decimal DESC")
        assert fresh < price, "freshness tie-break must precede odds_decimal DESC"

    def test_soccer_best_odds(self):
        cur = FakeCursor(odds=[odds_row()])
        gr.best_odds(cur, "match-1")
        sql = cur.sql_for("DISTINCT ON (market_type")
        self._assert_fresh_first(sql)
        assert cur.params_for("DISTINCT ON (market_type") == ("match-1", gr.MAX_ODDS_AGE_HOURS)

    def test_nhl_best_odds(self):
        cur = FakeCursor(odds=[odds_row()])
        gr_nhl.best_odds(cur, "match-1")
        self._assert_fresh_first(cur.sql_for("DISTINCT ON (market_type"))
        assert cur.params_for("DISTINCT ON (market_type") == ("match-1", gr.MAX_ODDS_AGE_HOURS)

    @pytest.mark.parametrize("mod", (gr_nba, gr_nfl), ids=lambda m: m.__name__)
    def test_lined_sports_both_branches(self, mod):
        unlined = FakeCursor(odds=[odds_row()])
        mod.best_odds_for_market(unlined, "match-1", "moneyline", None)
        self._assert_fresh_first(unlined.sql_for("DISTINCT ON (selection)"))
        assert unlined.params_for("DISTINCT ON (selection)") == ("match-1", "moneyline", gr.MAX_ODDS_AGE_HOURS)

        lined = FakeCursor(odds=[odds_row()])
        mod.best_odds_for_market(lined, "match-1", "spread", -7.0)
        self._assert_fresh_first(lined.sql_for("DISTINCT ON (selection)"))
        assert lined.params_for("DISTINCT ON (selection)") == ("match-1", "spread", -7.0, gr.MAX_ODDS_AGE_HOURS)

    @pytest.mark.parametrize("mod", (gr_mma, gr_tennis), ids=lambda m: m.__name__)
    def test_1v1_sports(self, mod):
        cur = FakeCursor(odds=[odds_row()])
        mod.best_odds_for_market(cur, "match-1", "moneyline")
        self._assert_fresh_first(cur.sql_for("DISTINCT ON (selection)"))
        assert cur.params_for("DISTINCT ON (selection)") == ("match-1", "moneyline", gr.MAX_ODDS_AGE_HOURS)

    @pytest.mark.parametrize("mod", (gr_nba, gr_nfl), ids=lambda m: m.__name__)
    def test_the_target_line_is_averaged_over_fresh_lines_only(self, mod):
        # `line` is part of the odds key, so a book that MOVES its line leaves
        # the abandoned one behind forever. Averaging those drags the target
        # toward lines nobody offers, and best_odds_for_market then admits any
        # price within +/-0.5 of it — a fresh price paired with a model
        # probability conditioned on a line the market has left.
        cur = FakeCursor(one={"avg_line": -7.0})
        mod.closing_line_for_match(cur, "match-1", "spread")
        sql = cur.sql_for("AVG(line)")
        assert "last_seen_at" in sql, "the target line is averaged over stale lines"
        assert cur.params_for("AVG(line)") == ("match-1", "spread", gr.MAX_ODDS_AGE_HOURS)

        tightened = FakeCursor(one={"avg_line": -7.0})
        mod.closing_line_for_match(tightened, "match-1", "total", 2.0)
        assert tightened.params_for("AVG(line)") == ("match-1", "total", 2.0)

    def test_the_gate_consensus_obeys_the_same_bound(self):
        # The gap cap (rec_gating.passes_gate max_gap) is a money-path safety
        # cap, so its consensus must be the market as it is NOW. Pricing a rec
        # off a fresh quote but gating it against books that stopped pricing
        # the match days ago can pass a pick that should be refused.
        cur = FakeCursor(consensus=CONSENSUS_1X2)
        rec_gating.market_consensus_prob(cur, "match-1", "1x2", "home")
        sql = cur.sql_for("DISTINCT ON (bookmaker")
        assert "%(max_age_hours)s" in sql, "the consensus query is unbounded in age"
        assert "last_seen_at" in sql
        assert cur.params_for("DISTINCT ON (bookmaker")["max_age_hours"] == gr.MAX_ODDS_AGE_HOURS

    def test_a_tightened_consensus_bound_reaches_the_sql(self):
        cur = FakeCursor(consensus=CONSENSUS_1X2)
        rec_gating.market_consensus_prob(cur, "match-1", "1x2", "home", max_age_hours=1.0)
        assert cur.params_for("DISTINCT ON (bookmaker")["max_age_hours"] == 1.0

    def test_books_aged_out_of_the_consensus_fail_closed(self):
        # Whatever the bound removes simply stops counting; below
        # MIN_CONSENSUS_BOOKS the answer is None ("we cannot see the market"),
        # which passes_gate treats as a reject rather than as agreement.
        cur = FakeCursor(consensus=CONSENSUS_1X2[:6])  # two books survive
        assert rec_gating.market_consensus_prob(cur, "match-1", "1x2", "home") is None
        ok, reason = rec_gating.passes_gate("soccer", "1x2", odds=2.2, ev=0.10, model_prob=0.60, market_prob=None)
        assert (ok, reason) == (False, "no_market_consensus")

    def test_horse_racing_reports_entrant_age(self):
        # Racing prices live in race_entrants.metadata, not `odds`, so the
        # freshness signal is the entrant row's own updated_at.
        cur = FakeCursor(candidates=[])
        gr_hr.load_race_candidates(cur, "race-1")
        sql = cur.sql_for("race_entrants")
        assert "e.updated_at" in sql and "odds_age_hours" in sql
        assert sql.count("odds_age_hours") == 2, "the age must be projected through BOTH select lists"


# ── Soccer: the money path ────────────────────────────────────────────────


@pytest.mark.unit
class TestSoccerPricing:
    def _run(self, cur, **kw):
        kw.setdefault("bankroll", 1000.0)
        kw.setdefault("ev_threshold", 0.05)
        kw.setdefault("prob_floor", 0.10)
        return gr.recommend_for_match(cur, "match-1", **kw)

    def test_fresh_quote_is_priced_and_written(self):
        cur = soccer_cursor([odds_row(odds_age_hours=0.5)])
        written, suppressed = self._run(cur)
        assert (written, suppressed) == (1, {})
        rec = cur.inserts[0]
        # odds_at_recommendation is the price the guard accepted.
        assert rec["odds"] == 2.2
        assert rec["bookmaker"] == "alpha"
        assert rec["bet_type"] == "1x2"

    def test_age_lands_in_risk_factors(self):
        cur = soccer_cursor([odds_row(odds_age_hours=1.25)])
        self._run(cur)
        risk = json.loads(cur.inserts[0]["risk"])
        assert isinstance(risk, list), "risk_factors keeps its JSON list shape"
        assert "odds_age_hours:1.25" in risk

    def test_stale_quote_is_rejected_counted_and_never_written(self):
        cur = soccer_cursor([odds_row(odds_age_hours=40.0)])
        written, suppressed = self._run(cur)
        assert written == 0
        assert suppressed == {"stale_odds": 1}
        assert cur.inserts == [], "a cold price must never reach betting_recommendations"

    def test_unknown_age_is_rejected(self):
        cur = soccer_cursor([odds_row(odds_age_hours=None)])
        written, suppressed = self._run(cur)
        assert (written, suppressed) == (0, {"stale_odds": 1})

    def test_boundary_quote_still_prices(self):
        cur = soccer_cursor([odds_row(odds_age_hours=gr.MAX_ODDS_AGE_HOURS)])
        written, suppressed = self._run(cur)
        assert (written, suppressed) == (1, {})

    def test_caller_can_tighten_the_bound(self):
        cur = soccer_cursor([odds_row(odds_age_hours=5.0)])
        written, suppressed = self._run(cur, max_odds_age_hours=1.0)
        assert (written, suppressed) == (0, {"stale_odds": 1})
        # …and the tightened bound reaches the SQL that picks the price.
        assert cur.params_for("DISTINCT ON (market_type") == ("match-1", 1.0)

    def test_stale_rows_do_not_stop_a_fresh_one_on_another_market(self):
        cur = soccer_cursor(
            [
                odds_row(odds_age_hours=99.0),
                odds_row(selection="away", odds_decimal=9.0, odds_age_hours=0.25),
            ]
        )
        written, suppressed = self._run(cur)
        # The away price is fresh but priced at 9.00, above the 1x2 gate's
        # max_odds — so it is refused by the GATE, not by the guard, and the
        # two refusals are counted under their own reasons.
        assert written == 0
        assert suppressed == {"stale_odds": 1, "1x2:odds_above_max": 1}


@pytest.mark.unit
class TestMilestoneAStillHolds:
    """The gating behaviour shipped in Milestone A must survive the guard."""

    def test_disabled_stream_is_still_suppressed_with_its_own_reason(self):
        cur = FakeCursor(
            predictions=[prediction_row(prediction_type="over_under", probabilities={"over_2.5": 0.7})],
            odds=[odds_row(market_type="over_under", selection="over", line=2.5, odds_age_hours=0.5)],
        )
        written, suppressed = gr.recommend_for_match(cur, "m", 1000.0, 0.05, 0.10)
        assert written == 0
        assert suppressed == {"stream_disabled:soccer:over_under": 1}
        assert cur.inserts == []

    def test_stake_is_still_capped(self):
        cur = soccer_cursor()
        gr.recommend_for_match(cur, "m", 1000.0, 0.05, 0.10)
        rec = cur.inserts[0]
        assert rec["rec_stake"] == pytest.approx(1000.0 * rec_gating.MAX_STAKE_FRACTION)
        assert "Stake capped" in rec["reasoning"]

    def test_summary_formatter_orders_stale_odds_with_the_rest(self):
        line = gr._format_suppressions({"1x2:odds_above_max": 2, "stale_odds": 7, "unknown_ev": 2})
        assert line == "stale_odds=7, 1x2:odds_above_max=2, unknown_ev=2"


# ── Every other generator refuses a cold price the same way ───────────────


def _enable_everything(monkeypatch):
    """Open every gate so the freshness guard is what is being measured.

    NFL/NBA/NHL/MMA/tennis are all gated OFF today (audit 2026-09), and the
    per-market `gate.enabled` check short-circuits before pricing, so the
    guard below is unreachable without this."""
    monkeypatch.setattr(
        rec_gating,
        "gate_for",
        lambda sport, bet_type: rec_gating.DEFAULT_GATE,
    )


NHL_MATCH = {
    "match_id": "match-1",
    "match_date": "2026-09-05T18:00:00+00:00",
    "home_team": "Home",
    "away_team": "Away",
    "league_name": "NHL",
}


def _nhl_cursor(age):
    return FakeCursor(
        predictions=[prediction_row(prediction_type="moneyline", probabilities={"home": 0.60, "away": 0.40})],
        odds=[odds_row(market_type="moneyline", odds_age_hours=age)],
    )


def _team_cursor(mod, age):
    ensemble = next(iter(mod.NFL_MARKETS if mod is gr_nfl else mod.NBA_MARKETS))
    return FakeCursor(
        predictions=[{"model_name": ensemble, "prediction_id": "p1", "probabilities": {"home": 0.75, "away": 0.25}}],
        odds=[odds_row(market_type="moneyline", odds_age_hours=age)],
    )


def _1v1_cursor(mod, age):
    ensemble = next(iter(mod.MMA_MARKETS if mod is gr_mma else mod.TENNIS_MARKETS))
    return FakeCursor(
        predictions=[{"model_name": ensemble, "prediction_id": "p1", "probabilities": {"home": 0.75, "away": 0.25}}],
        odds=[odds_row(market_type="moneyline", odds_age_hours=age)],
    )


@pytest.mark.unit
class TestOtherGenerators:
    def test_nhl(self, monkeypatch):
        _enable_everything(monkeypatch)
        fresh = _nhl_cursor(0.75)
        suppressed: dict[str, int] = {}
        alerts = gr_nhl.recommend_for_match(fresh, NHL_MATCH, 1000.0, 0.05, 0.10, suppressed)
        assert len(alerts) == 1 and suppressed == {}
        assert "odds_age_hours:0.75" in json.loads(fresh.inserts[0]["risk"])

        stale = _nhl_cursor(80.0)
        suppressed = {}
        assert gr_nhl.recommend_for_match(stale, NHL_MATCH, 1000.0, 0.05, 0.10, suppressed) == []
        assert suppressed == {"stale_odds": 1}
        assert stale.inserts == []

    @pytest.mark.parametrize("mod", (gr_nba, gr_nfl), ids=lambda m: m.__name__)
    def test_nba_nfl(self, mod, monkeypatch):
        _enable_everything(monkeypatch)
        fresh = _team_cursor(mod, 2.0)
        suppressed: dict[str, int] = {}
        assert mod.recommend_for_match(fresh, "match-1", 1000.0, 0.03, 0.40, suppressed) == 1
        assert suppressed == {}
        assert "odds_age_hours:2.00" in json.loads(fresh.inserts[0]["risk"])

        stale = _team_cursor(mod, 200.0)
        suppressed = {}
        assert mod.recommend_for_match(stale, "match-1", 1000.0, 0.03, 0.40, suppressed) == 0
        assert suppressed == {"stale_odds": 1}
        assert stale.inserts == []

    @pytest.mark.parametrize("mod", (gr_mma, gr_tennis), ids=lambda m: m.__name__)
    def test_mma_tennis(self, mod, monkeypatch):
        _enable_everything(monkeypatch)
        fresh = _1v1_cursor(mod, 3.0)
        suppressed: dict[str, int] = {}
        assert mod.recommend_for_match(fresh, "match-1", 1000.0, 0.03, 0.40, suppressed) == 1
        assert suppressed == {}
        assert "odds_age_hours:3.00" in json.loads(fresh.inserts[0]["risk"])

        # 197 h is the mean age the audit measured on live MMA rec prices.
        stale = _1v1_cursor(mod, 197.0)
        suppressed = {}
        assert mod.recommend_for_match(stale, "match-1", 1000.0, 0.03, 0.40, suppressed) == 0
        assert suppressed == {"stale_odds": 1}
        assert stale.inserts == []


# ── Horse racing: same bound, different table ─────────────────────────────


RACE = {
    "race_id": "race-1",
    "track_name": "Newton Abbot",
    "race_date": "2026-09-05T15:00:00+00:00",
    "race_number": 1,
}


def _candidate(age, entrant="A"):
    return {
        "entrant_id": entrant,
        "prediction_id": f"pred-{entrant}",
        "confidence": 0.30,
        "bookmaker_odds": [{"bookmaker": "Test Book", "decimal": 5.0}],
        "horse_name": f"Horse {entrant}",
        "model_name": "market_consensus_v1",
        "ranker_confidence": None,
        "odds_age_hours": age,
    }


def _race_field(age):
    """One live candidate plus seven also-rans, all priced at the same age.

    Eight predicted entrants is what turns the each-way place leg on
    (ew_terms: 8+ runners -> 3 places), so this shape exercises BOTH legs
    off the same bookmaker quote. Only the 0.30 horse clears the EV gate at
    5.00; the also-rans are there for the field, and for Harville."""
    return [_candidate(age, "A")] + [{**_candidate(age, chr(66 + i)), "confidence": 0.10} for i in range(7)]


def _race_cursor(candidates):
    # runners=10 keeps the race clear of the 5-7 win-suppression band.
    return FakeCursor(candidates=candidates, one={"runners": 10})


@pytest.mark.unit
class TestHorseRacingPricing:
    def test_fresh_entrant_is_priced_and_carries_its_age(self):
        cur = _race_cursor(_race_field(0.5))
        suppressed: dict[str, int] = {}
        alerts = gr_hr.recommend_for_race(cur, RACE, 1000.0, 0.05, 0.10, suppressed)
        assert len(alerts) == 1 and suppressed == {}
        win = [r for r in cur.inserts if r["bet_type"] == "win"]
        assert len(win) == 1 and win[0]["odds"] == 5.0
        assert "odds_age_hours:0.50" in json.loads(win[0]["risk"])
        # The derived place leg is priced off the same quote, so it carries
        # the same age.
        place = [r for r in cur.inserts if r["bet_type"] == "place"]
        assert place and "odds_age_hours:0.50" in json.loads(place[0]["risk"])

    def test_stale_entrant_is_rejected_counted_and_writes_nothing(self):
        cur = _race_cursor(_race_field(72.0))
        suppressed: dict[str, int] = {}
        alerts = gr_hr.recommend_for_race(cur, RACE, 1000.0, 0.05, 0.10, suppressed)
        assert alerts == []
        assert suppressed == {"stale_odds": 8}
        # Both legs are gone: the place price is derived from the same
        # bookmaker quote, so a cold win price cannot survive as a place bet.
        assert cur.inserts == []

    def test_missing_age_warns_loudly_but_keeps_the_book_open(self, caplog):
        # race_entrants.updated_at is DEFAULT NOW() and rewritten on every
        # upsert, so a candidate with no age is a caller passing rows this
        # query did not build — not a stale price. Refusing it would take
        # the whole racing book off the board over a code shape.
        cur = _race_cursor([_candidate(None)])
        suppressed: dict[str, int] = {}
        with caplog.at_level("WARNING"):
            alerts = gr_hr.recommend_for_race(cur, RACE, 1000.0, 0.05, 0.10, suppressed)
        assert len(alerts) == 1
        assert suppressed == {}
        assert "no odds age" in caplog.text
        assert "odds_age_hours:unknown" in json.loads(cur.inserts[0]["risk"])

    def test_racing_gate_and_field_suppression_still_fire(self, caplog):
        # A 6-runner race is inside the audit's -57.1% band: the win leg is
        # suppressed for the field size, not for staleness.
        cur = FakeCursor(candidates=[_candidate(0.5)], one={"runners": 6})
        suppressed: dict[str, int] = {}
        alerts = gr_hr.recommend_for_race(cur, RACE, 1000.0, 0.05, 0.10, suppressed)
        assert alerts == []
        assert suppressed == {gr_hr.SUPPRESSED_WIN_FIELD_REASON: 1}
