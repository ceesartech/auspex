"""Unit tests for the soccer asian-handicap recommendation replay: the pieces
that decide whether the replay is the production generator. The tick and the
odds state at it, the best-price pick, the market consensus against
rec_gating itself, the whole per-match pricing path against
generate_recommendations.recommend_for_match on a SQL-emulating cursor, the
settlement through grading_outcomes, the stored-prediction pick, the arm
comparison, the bootstrap, the power formula and the Brier re-derivation
guard."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "services" / "ml-models" / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rp = _load("ab_soccer_ah_recs_replay", "ab_soccer_ah_recs_replay.py")
gr = rp.gr
rec_gating = rp.rec_gating
go = rp.go
ftdc = rp.ftdc

from predictors import market_derivation as md  # noqa: E402

T0 = pd.Timestamp("2026-10-10T14:46:00Z")


def _row(book, sel, line, odds, created, ts=None, seen=None, match="m1", live=False):
    return {
        "match_id": match,
        "bookmaker": book,
        "selection": sel,
        "line": line,
        "odds_decimal": odds,
        "created": created,
        "ts": ts if ts is not None else created,
        "last_seen_at": seen,
        "is_live": live,
    }


def _q(book, sel, line, odds, age):
    return rp.Quote(book, sel, float(line), float(odds), age, None)


# ── The tick ─────────────────────────────────────────────────────────


class TestLastTick:
    def test_on_the_hour_is_the_46_run(self):
        assert rp.last_tick_before("2026-10-10T15:00:00Z") == pd.Timestamp("2026-10-10T14:46:00Z")

    def test_off_quarter_kickoff_uses_the_run_just_before(self):
        assert rp.last_tick_before("2026-10-10T15:05:00Z") == pd.Timestamp("2026-10-10T15:01:00Z")

    def test_a_run_at_the_kickoff_minute_does_not_count(self):
        # list_upcoming needs match_date >= NOW(): a run at 15:01 cannot see a 15:01 kickoff.
        assert rp.last_tick_before("2026-10-10T15:01:00Z") == pd.Timestamp("2026-10-10T14:46:00Z")

    def test_production_lead_times_match(self):
        # Production's AH recs since 2026-09-05: 12.8-14.1 min before on-the-hour kickoffs.
        lead = pd.Timestamp("2026-10-10T19:30:00Z") - rp.last_tick_before("2026-10-10T19:30:00Z")
        assert lead == pd.Timedelta(minutes=14)

    def test_naive_kickoff_is_utc_and_garbage_raises(self):
        assert rp.last_tick_before("2026-10-10 15:00:00") == pd.Timestamp("2026-10-10T14:46:00Z")
        with pytest.raises(ValueError):
            rp.last_tick_before(None)


# ── The odds state at the tick ───────────────────────────────────────


class TestQuoteState:
    def test_a_row_created_after_the_tick_did_not_exist(self):
        drops = {}
        st = rp.quote_state([_row("b", "home", -0.5, 1.9, T0 + pd.Timedelta(minutes=1))], {}, T0, drops)
        assert st == [] and drops == {"not_yet_listed": 1}

    def test_price_at_the_tick_is_the_last_snapshot_before_it(self):
        rows = [_row("b", "home", -0.5, 2.10, T0 - pd.Timedelta(hours=5), ts=T0 + pd.Timedelta(minutes=5))]
        snaps = rp.snapshot_index(
            [
                {"match_id": "m1", "bookmaker": "b", "selection": "home", "line": -0.5, "odds_decimal": 1.95,
                 "captured_at": T0 - pd.Timedelta(hours=5)},
                {"match_id": "m1", "bookmaker": "b", "selection": "home", "line": -0.5, "odds_decimal": 2.00,
                 "captured_at": T0 - pd.Timedelta(hours=1)},
                {"match_id": "m1", "bookmaker": "b", "selection": "home", "line": -0.5, "odds_decimal": 2.10,
                 "captured_at": T0 + pd.Timedelta(minutes=5)},
            ]
        )  # fmt: skip
        st = rp.quote_state(rows, snaps, T0)
        assert len(st) == 1 and st[0].odds == 2.00
        assert st[0].moved_at == T0 - pd.Timedelta(hours=1)

    def test_without_snapshots_the_row_price_holds_only_if_it_last_moved_before(self):
        before = _row("b", "home", -0.5, 1.9, T0 - pd.Timedelta(hours=3))
        after = _row("c", "home", -0.5, 2.3, T0 - pd.Timedelta(hours=3), ts=T0 + pd.Timedelta(minutes=2))
        drops = {}
        st = rp.quote_state([before, after], {}, T0, drops)
        assert [q.bookmaker for q in st] == ["b"] and st[0].odds == 1.9
        assert drops == {"price_unknown_at_tick": 1}

    def test_age_is_measured_at_the_tick(self):
        r = _row("b", "home", -0.5, 1.9, T0 - pd.Timedelta(hours=9), seen=T0 - pd.Timedelta(hours=2))
        (q,) = rp.quote_state([r], {}, T0)
        assert q.age_hours == pytest.approx(2.0)

    def test_null_last_seen_falls_back_to_timestamp(self):
        r = _row("b", "home", -0.5, 1.9, T0 - pd.Timedelta(hours=9), ts=T0 - pd.Timedelta(hours=7))
        (q,) = rp.quote_state([r], {}, T0)
        assert q.age_hours == pytest.approx(7.0)

    def test_seen_after_the_tick_is_fresh_and_counted(self):
        drops = {}
        r = _row("b", "home", -0.5, 1.9, T0 - pd.Timedelta(hours=9), seen=T0 + pd.Timedelta(minutes=10))
        (q,) = rp.quote_state([r], {}, T0, drops)
        assert q.age_hours == 0.0 and drops == {"seen_after_tick_kept_fresh": 1}

    def test_live_rows_never_price(self):
        drops = {}
        assert (
            rp.quote_state([_row("b", "home", -0.5, 1.9, T0 - pd.Timedelta(hours=1), live=True)], {}, T0, drops) == []
        )
        assert drops == {"live": 1}


class TestBestQuotes:
    def test_fresh_beats_a_higher_stale_price(self):
        best = rp.best_quotes([_q("old", "home", -0.5, 2.5, 30.0), _q("new", "home", -0.5, 1.9, 0.5)])
        assert [(q.bookmaker, q.odds) for q in best] == [("new", 1.9)]

    def test_highest_fresh_price_wins_and_ties_go_alphabetically(self):
        best = rp.best_quotes(
            [_q("b", "home", -0.5, 1.95, 0.5), _q("a", "home", -0.5, 1.95, 0.5), _q("c", "home", -0.5, 1.90, 0.1)]
        )
        assert best[0].bookmaker == "a"
        assert rp.price_ties([_q("b", "home", -0.5, 1.95, 0.5), _q("a", "home", -0.5, 1.95, 0.5)]) == 1

    def test_one_pick_per_selection_and_line_in_sql_order(self):
        st = [_q("a", "home", -0.5, 1.9, 0.5), _q("a", "away", 0.5, 1.9, 0.5), _q("a", "home", -1.0, 2.4, 0.5)]
        assert [(q.selection, q.line) for q in rp.best_quotes(st)] == [("away", 0.5), ("home", -1.0), ("home", -0.5)]

    def test_all_stale_still_returns_the_quote(self):
        (q,) = rp.best_quotes([_q("a", "home", -0.5, 1.9, 40.0)])
        assert gr.is_stale_odds(q.age_hours)


# ── The consensus against rec_gating itself ──────────────────────────


class _ConsensusCursor:
    """rec_gating._CONSENSUS_SQL emulated on a quote list: market and line
    filters, the freshness bound, DISTINCT ON (book, lower(selection), line)."""

    def __init__(self, quotes):
        self.quotes = quotes

    def execute(self, sql, params=None):
        assert "DISTINCT ON (bookmaker, lower(selection), line)" in sql
        lines = {params["line"], params["opposite_line"]}
        self.rows = [
            {"bookmaker": q.bookmaker, "selection": q.selection, "line": q.line, "odds_decimal": q.odds}
            for q in self.quotes
            if any(abs(q.line - x) < 1e-9 for x in lines)
            and q.age_hours is not None
            and q.age_hours <= params["max_age_hours"]
        ]

    def fetchall(self):
        return self.rows


def _random_state(rng, books=("a", "b", "c", "d", "e"), lines=(-1.25, -0.75, -0.5, 0.0, 0.25, 1.0)):
    st = []
    for book in books:
        for line in lines:
            if rng.random() < 0.25:
                continue
            base = rng.uniform(1.5, 2.6)
            age = float(rng.choice([0.2, 1.0, 5.9, 6.0, 7.5, 30.0]))
            st.append(_q(book, "home", line, round(base, 2), age))
            if rng.random() < 0.9:
                other = 1.0 / max(0.05, 1.06 - 1.0 / base)
                st.append(_q(book, "away", -line, round(other, 2), age if rng.random() < 0.8 else 9.0))
    return st


class TestConsensus:
    @pytest.mark.parametrize("seed", range(12))
    def test_equals_market_consensus_prob_for_the_power_devig(self, seed):
        rng = np.random.default_rng(seed)
        st = _random_state(rng)
        for q in st:
            mine = rp.consensus_prob(st, q.selection, q.line)
            ref = rec_gating.market_consensus_prob(_ConsensusCursor(st), "m1", "asian_handicap", q.selection, q.line)
            if ref is None:
                assert mine is None
            else:
                assert mine == pytest.approx(ref, abs=1e-15)

    def test_needs_min_consensus_books(self):
        st = [_q(b, "home", -0.5, 1.9, 0.5) for b in "ab"] + [_q(b, "away", 0.5, 1.95, 0.5) for b in "ab"]
        assert rp.consensus_prob(st, "home", -0.5) is None
        st += [_q("c", "home", -0.5, 1.9, 0.5), _q("c", "away", 0.5, 1.95, 0.5)]
        assert rp.consensus_prob(st, "home", -0.5) is not None

    def test_proportional_devig_differs_from_power_on_lopsided_prices(self):
        st = []
        for b in "abc":
            st += [_q(b, "home", -1.5, 3.4, 0.5), _q(b, "away", 1.5, 1.30, 0.5)]
        power = rp.consensus_prob(st, "home", -1.5)
        prop = rp.consensus_prob(st, "home", -1.5, rec_gating.devig_proportional)
        assert power < prop  # the power method takes more margin off the longshot


# ── The pricing path against generate_recommendations itself ─────────


class _GeneratorCursor:
    """Answers the four queries recommend_for_match runs for a soccer match,
    emulating each one's SQL semantics on a quote list."""

    def __init__(self, probs, quotes):
        self.probs = probs
        self.quotes = quotes
        self.inserts = []

    def execute(self, sql, params=None):
        self.sql = sql
        if "FROM predictions" in sql:
            self.rows = [{"prediction_type": "asian_handicap", "prediction_id": "p1", "probabilities": self.probs}]
        elif "DISTINCT ON (market_type, selection, COALESCE(line, -1))" in sql:
            max_age = params[1]
            groups = {}
            for q in self.quotes:
                groups.setdefault((q.selection, q.line), []).append(q)
            self.rows = []
            for key in sorted(groups):
                qs = groups[key]
                fresh = [q for q in qs if q.age_hours is not None and q.age_hours <= max_age]
                pool = fresh or qs
                top = max(pool, key=lambda q: (q.odds, [-ord(c) for c in q.bookmaker]))
                self.rows.append(
                    {
                        "market_type": "asian_handicap",
                        "selection": top.selection,
                        "line": top.line,
                        "bookmaker": top.bookmaker,
                        "odds_decimal": top.odds,
                        "odds_age_hours": top.age_hours,
                    }
                )
        elif "DISTINCT ON (bookmaker, lower(selection), line)" in sql:
            c = _ConsensusCursor(self.quotes)
            c.execute(sql, params)
            self.rows = c.rows
        elif sql.strip().startswith("DELETE"):
            self.rows = []
        elif "INSERT INTO betting_recommendations" in sql:
            self.inserts.append(params)
            self.rows = []
        else:  # pragma: no cover - a new query in the generator must be emulated here
            raise AssertionError(f"unexpected SQL: {sql[:80]}")

    def fetchall(self):
        return self.rows


def _ah_probs(rng):
    h, a = rng.uniform(0.6, 2.6), rng.uniform(0.4, 2.0)
    tgt = tuple(rng.dirichlet([4, 2.5, 3]))
    return {k: float(v) for k, v in md.derive_from_lambdas(h, a, -0.05, target_1x2=tgt)["asian_handicap"].items()}


class TestReplayMatchesTheGenerator:
    @pytest.mark.parametrize("seed", range(25))
    def test_same_recs_as_recommend_for_match(self, seed):
        rng = np.random.default_rng(100 + seed)
        probs = _ah_probs(rng)
        st = _random_state(rng, lines=(-2.0, -1.75, -1.25, -0.75, -0.5, -0.25, 0.0, 0.25, 0.75, 1.0, 1.5))
        cur = _GeneratorCursor(probs, st)
        written, gen_sup = gr.recommend_for_match(cur, "m1", 1000.0, 0.05, 0.10)
        mine, my_sup = rp.replay_match("m1", probs, st, 1000.0, 0.05, 0.10)
        assert written == len(mine) == len(cur.inserts)
        assert my_sup == gen_sup
        for ins, r in zip(cur.inserts, mine):
            assert ins["selection"] == r.selection
            assert ins["odds"] == r.odds
            assert ins["bookmaker"] == r.bookmaker
            assert ins["ev"] == r.ev
            assert ins["rec_stake"] == r.stake
            assert ins["kelly_stake"] == r.kelly_stake

    def test_some_scenarios_write_and_some_are_gated(self):
        wrote = gated = 0
        for seed in range(25):
            rng = np.random.default_rng(100 + seed)
            probs = _ah_probs(rng)
            st = _random_state(rng, lines=(-2.0, -1.75, -1.25, -0.75, -0.5, -0.25, 0.0, 0.25, 0.75, 1.0, 1.5))
            recs, sup = rp.replay_match("m1", probs, st, 1000.0, 0.05, 0.10)
            wrote += len(recs)
            gated += sum(v for k, v in sup.items() if k != gr.STALE_ODDS_REASON)
        assert wrote > 0 and gated > 0, "the parity test must exercise both outcomes"

    def test_no_probabilities_no_recs(self):
        assert rp.replay_match("m1", {}, [_q("a", "home", -0.5, 2.5, 0.5)], 1000.0, 0.05, 0.10) == ([], {})


# ── Settlement, CLV, stored inputs ───────────────────────────────────


class TestSettle:
    def test_quarter_line_half_win_pays_half(self):
        verdict, pl, flat = rp.settle("home_-0.75", 2.0, 10.0, 2, 1, {})
        assert verdict == go.REC_HALF_WON and pl == pytest.approx(5.0) and flat == pytest.approx(0.5)

    def test_integer_line_push_refunds(self):
        verdict, pl, flat = rp.settle("away_1", 1.9, 10.0, 2, 1, {})
        assert verdict == go.REC_PUSH and pl == 0.0 and flat == 0.0

    def test_away_side_uses_its_own_line(self):
        verdict, pl, _ = rp.settle("away_0.25", 2.0, 10.0, 1, 1, {})
        assert verdict == go.REC_HALF_WON and pl == pytest.approx(5.0)

    def test_settles_on_the_90_minute_score(self):
        meta = {go.REGULATION_HOME_KEY: 1, go.REGULATION_AWAY_KEY: 1}
        verdict, _, _ = rp.settle("home_-0.5", 2.0, 10.0, 2, 1, meta)
        assert verdict == go.REC_LOST


class TestClosingPrice:
    def test_last_snapshot_strictly_before_kickoff_at_the_same_book(self):
        ko = T0 + pd.Timedelta(minutes=14)
        snaps = rp.snapshot_index(
            [
                {"match_id": "m1", "bookmaker": "b", "selection": "home", "line": -0.5, "odds_decimal": 1.9,
                 "captured_at": T0},
                {"match_id": "m1", "bookmaker": "b", "selection": "home", "line": -0.5, "odds_decimal": 1.8,
                 "captured_at": ko},
            ]
        )  # fmt: skip
        assert rp.closing_price(snaps, [], "m1", "b", "home", -0.5, ko) == 1.9

    def test_row_price_without_snapshots(self):
        rows = [_row("b", "home", -0.5, 1.85, T0 - pd.Timedelta(hours=2))]
        assert rp.closing_price({}, rows, "m1", "b", "home", -0.5, T0) == 1.85
        assert rp.closing_price({}, rows, "m1", "c", "home", -0.5, T0) is None


def _stored(rows):
    df = pd.DataFrame(
        rows, columns=["match_id", "model_version", "prediction_type", "probabilities", "metadata", "created_at"]
    )
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


class TestStoredInputs:
    def test_newest_row_before_the_tick_with_its_own_versions_1x2(self):
        df = _stored(
            [
                ["m1", "v1", "asian_handicap", {"0_home": 0.4}, {"reconcile": "1x2"}, "2026-10-08T10:00Z"],
                ["m1", "v1", "match_result", {"home": 0.5, "draw": 0.3, "away": 0.2}, {}, "2026-10-08T10:00Z"],
                ["m1", "v2", "asian_handicap", {"0_home": 0.45}, {"reconcile": "1x2+ou2.5", "market_over_2.5": 0.55},
                 "2026-10-09T10:00Z"],
                ["m1", "v2", "match_result", {"home": 0.55, "draw": 0.25, "away": 0.2}, {}, "2026-10-09T10:00Z"],
                ["m1", "v3", "asian_handicap", {"0_home": 0.9}, {}, "2026-10-10T15:00Z"],
            ]
        )  # fmt: skip
        st = rp.stored_inputs(df, "m1", T0)
        assert st["model_version"] == "v2"
        assert st["probabilities"] == {"0_home": 0.45}
        assert st["target"] == (0.55, 0.25, 0.2)
        assert st["anchor"] == 0.55

    def test_stale_anchor_key_is_ignored_when_the_last_tick_was_unanchored(self):
        # precompute MERGES metadata: an earlier anchored tick leaves market_over_2.5 behind.
        df = _stored(
            [
                ["m1", "v1", "asian_handicap", {"0_home": 0.4}, {"reconcile": "1x2", "market_over_2.5": 0.6},
                 "2026-10-08T10:00Z"],
                ["m1", "v1", "match_result", {"home": 0.5, "draw": 0.3, "away": 0.2}, {}, "2026-10-08T10:00Z"],
            ]
        )  # fmt: skip
        assert rp.stored_inputs(df, "m1", T0)["anchor"] is None

    def test_no_row_before_the_tick(self):
        df = _stored([["m1", "v3", "asian_handicap", {"0_home": 0.9}, {}, "2026-10-10T15:00Z"]])
        assert rp.stored_inputs(df, "m1", T0) is None
        assert rp.stored_inputs(None, "m1", T0) is None

    def test_missing_1x2_sibling_is_loud(self):
        df = _stored([["m1", "v1", "asian_handicap", {"0_home": 0.4}, {}, "2026-10-08T10:00Z"]])
        with pytest.raises(RuntimeError, match="no match_result sibling"):
            rp.stored_inputs(df, "m1", T0)


class TestTeamHistory:
    def test_counts_strictly_before(self):
        h = pd.DataFrame(
            {
                "home_team_id": ["A", "B", "A"],
                "away_team_id": ["B", "A", "C"],
                "match_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"], utc=True),
            }
        )
        th = rp.TeamHistory(h)
        assert th.count_before("A", pd.Timestamp("2026-03-01", tz="UTC")) == 2
        assert th.count_before("A", pd.Timestamp("2026-03-02", tz="UTC")) == 3
        assert th.count_before("Z", pd.Timestamp("2026-03-02", tz="UTC")) == 0


def test_derive_ah_is_what_precompute_stores():
    mine = rp.derive_ah(1.6, 1.1, -0.05, (0.5, 0.27, 0.23), 0.56)
    ref = md.derive_from_lambdas(1.6, 1.1, -0.05, target_1x2=(0.5, 0.27, 0.23), target_over_25=0.56)["asian_handicap"]
    assert mine == {k: float(v) for k, v in ref.items()}
    assert "-0.75_home" in mine and "-0.75_push" in mine


# ── Comparison and statistics ────────────────────────────────────────


def _rec(match, sel, odds=2.0, stake=10.0, verdict=go.REC_WON, ev_mkt=0.01):
    side, line = sel.split("_")
    r = rp.Rec(match, sel, side, float(line), odds, "b", 0.06, 0.55, 0.55, 0.5, 40.0, stake, 0.5)
    pl = {go.REC_WON: stake * (odds - 1), go.REC_LOST: -stake}[verdict]
    flat = {go.REC_WON: odds - 1, go.REC_LOST: -1.0}[verdict]
    r.extra.update({"verdict": verdict, "pl": pl, "flat": flat, "ev_mkt": ev_mkt, "clv_same": 0.0})
    return r


class TestCompare:
    def test_classifies_same_moved_and_only(self):
        a = [_rec("m1", "home_-0.75"), _rec("m2", "away_0.5"), _rec("m3", "home_-1")]
        b = [_rec("m1", "home_-0.75"), _rec("m2", "away_0.75"), _rec("m4", "home_0")]
        c = rp.compare_recs(a, b)
        assert c["same"] == [("m1", "home_-0.75")]
        assert [(x.selection, y.selection) for x, y in c["moved_line"]] == [("away_0.5", "away_0.75")]
        assert [r.match_id for r in c["a_only"]] == ["m3"]
        assert [r.match_id for r in c["b_only"]] == ["m4"]

    def test_money_summary(self):
        m = rp.money_summary([_rec("m1", "home_-1"), _rec("m2", "home_-1", verdict=go.REC_LOST, ev_mkt=-0.03)])
        assert m["recs"] == 2 and m["pl"] == pytest.approx(0.0) and m["roi_flat"] == pytest.approx(0.0)
        assert m["verdicts"] == {go.REC_WON: 1, go.REC_LOST: 1}
        assert m["ev_mkt_mean"] == pytest.approx(-0.01) and m["ev_mkt_n"] == 2
        assert "clv_fair_mean" not in m  # the bet-time readout is never called a closing-line value

    def test_rec_outside_the_population_is_loud(self):
        with pytest.raises(KeyError):
            rp.per_match_totals([_rec("mX", "home_-1")], ["m1"])


class TestBootstrap:
    def test_identical_arms_have_zero_difference(self):
        recs = [_rec(f"m{i}", "home_-1", verdict=go.REC_WON if i % 3 else go.REC_LOST) for i in range(30)]
        ids = [f"m{i}" for i in range(30)]
        t = rp.per_match_totals(recs, ids)
        bs = rp.paired_bootstrap(t, t, draws=200)
        assert bs["diff_pl_total"]["estimate"] == 0.0 and bs["diff_pl_total"]["se"] == 0.0
        assert bs["a_roi_flat"]["estimate"] == pytest.approx((20 * 1.0 - 10) / 30)

    def test_bet_time_market_ev_keys(self):
        recs = [_rec(f"m{i}", "home_-1", ev_mkt=-0.03 + 0.001 * i) for i in range(10)]
        t = rp.per_match_totals(recs, [f"m{i}" for i in range(10)])
        bs = rp.paired_bootstrap(t, t, draws=100)
        assert bs["a_ev_mkt"]["estimate"] == pytest.approx(np.mean([-0.03 + 0.001 * i for i in range(10)]))
        assert bs["diff_ev_mkt"]["estimate"] == 0.0 and "diff_clv_fair" not in bs

    def test_deterministic_and_point_estimates_are_direct(self):
        ids = [f"m{i}" for i in range(40)]
        a = [_rec(f"m{i}", "home_-1", verdict=go.REC_LOST if i % 2 else go.REC_WON) for i in range(40)]
        b = [_rec(f"m{i}", "home_-1", verdict=go.REC_WON) for i in range(0, 40, 4)]
        ta, tb = rp.per_match_totals(a, ids), rp.per_match_totals(b, ids)
        one = rp.paired_bootstrap(ta, tb, draws=300)
        two = rp.paired_bootstrap(ta, tb, draws=300)
        assert one == two
        assert one["diff_pl_total"]["estimate"] == pytest.approx(tb["pl"].sum() - ta["pl"].sum())


def test_n_for_power():
    assert rp.n_for_power(1.0, 0.1) == math.ceil(((rp.Z_ALPHA + rp.Z_BETA) * 10) ** 2)
    assert rp.n_for_power(1.0, 0.0) is None and rp.n_for_power(None, 0.1) is None


# ── Brier readouts ───────────────────────────────────────────────────


class TestBrierReadouts:
    def test_gd_classes_sum_to_one_and_match_the_handicap_readout(self):
        rng = np.random.default_rng(3)
        P = np.stack(
            [md.reconcile_matrix_to_1x2(md.build_dc_matrix(rng.uniform(0.5, 2.5), rng.uniform(0.4, 2), -0.05),
                                        tuple(rng.dirichlet([4, 2.5, 3]))) for _ in range(20)]
        )  # fmt: skip
        gd = rp.gd_probabilities(P)
        assert np.allclose(gd.sum(axis=1), 1.0)
        ah = rp.handicap_probs(P, -1.5)
        assert np.allclose(ah[:, 0], gd[:, 5] + gd[:, 6])  # home -1.5 wins on margin >= 2

    def test_rps_is_zero_for_a_perfect_forecast_and_ordered(self):
        p = np.eye(7)[[0, 3, 6]]
        assert np.allclose(rp.rps_rows(p, np.array([0, 3, 6])), 0.0)
        near = np.eye(7)[[4]]
        far = np.eye(7)[[6]]
        assert rp.rps_rows(near, np.array([3]))[0] < rp.rps_rows(far, np.array([3]))[0]
        assert list(rp.gd_class(np.array([5, 0, 1]), np.array([0, 4, 1]))) == [6, 0, 3]

    def _ftdc_outputs(self, tmp_path, n=60, perturb=False):
        rng = np.random.default_rng(9)
        rows, scores = [], {}
        weights = ftdc.unit_weights()
        h = rng.integers(0, 5, n)
        a = rng.integers(0, 4, n)
        truth = ftdc.vector_truth(h, a)
        mats = {}
        for arm in ("served", "refit"):
            lam = np.column_stack([rng.uniform(0.6, 2.4, n), rng.uniform(0.4, 1.8, n)])
            mats[arm] = lam
        tgt = rng.dirichlet([4, 2.5, 3], n)
        anc = np.where(rng.random(n) < 0.3, rng.uniform(0.35, 0.65, n), np.nan)
        for r in range(n):
            row = {"match_id": f"m{r}", "match_date": "2026-10-01T15:00:00+00:00", "s_home": h[r], "s_away": a[r],
                   "target::home": tgt[r, 0], "target::draw": tgt[r, 1], "target::away": tgt[r, 2],
                   "anchor_live": anc[r]}  # fmt: skip
            for arm in ("served", "refit"):
                row[f"lam_h::{arm}"], row[f"lam_a::{arm}"], row[f"rho::{arm}"] = mats[arm][r, 0], mats[arm][r, 1], -0.05
            rows.append(row)
        pm = pd.DataFrame(rows)
        for arm in ("served", "refit"):
            P = np.stack(
                [ftdc.served_matrix(mats[arm][r, 0], mats[arm][r, 1], -0.05, tgt[r],
                                    None if not math.isfinite(anc[r]) else anc[r]) for r in range(n)]
            )  # fmt: skip
            for u in rp.ALL_AH_UNITS:
                scores[f"brier::{arm}::live::{u}"] = ftdc.brier_rows(np.einsum("rij,kij->rk", P, weights[u]), truth[u])
        if perturb:
            scores["brier::refit::live::ah_-1.75"][7] += 1e-9
        scores["fold"] = np.array(["2026-10"] * (n // 2) + ["2026-09"] * (n - n // 2))
        scores["match_id"] = pm["match_id"].to_numpy().astype(str)
        pm.to_csv(tmp_path / "per_match.csv.gz", index=False, compression="gzip")
        np.savez_compressed(tmp_path / "scores.npz", **scores)
        return rp.load_ftdc(str(tmp_path))

    def test_brier_part_reproduces_the_gates_scores_and_reads_the_primaries(self, tmp_path):
        pm, scores = self._ftdc_outputs(tmp_path)
        out = rp.brier_part(pm, scores, mechanics=False)
        assert out["reconstruct_max_abs"] <= rp.RECONSTRUCT_ATOL
        for u in ("ah_-0.5", "ah_0", "ah_0.25"):  # fixed by the 1x2 after the IPF
            assert abs(out["units"][u]["delta"]) < 1e-12
        pooled = np.mean([scores[f"brier::refit::live::{u}"] - scores[f"brier::served::live::{u}"]
                          for u in rp.SHAPED_UNITS], axis=0).mean()  # fmt: skip
        assert out["pooled_shaped_mean"]["delta"] == pytest.approx(pooled, abs=1e-15)
        assert out["best_single_line"]["unit"] in rp.SHAPED_UNITS
        assert set(out["verdicts"]) == {"best_single_line", "pooled_shaped_mean", "gd_rps"}

    def test_brier_part_mechanics_reads_no_delta(self, tmp_path):
        pm, scores = self._ftdc_outputs(tmp_path)
        out = rp.brier_part(pm, scores, mechanics=True)
        assert set(out) == {"n", "reconstruct_max_abs"}

    def test_a_score_that_does_not_re_derive_aborts(self, tmp_path):
        pm, scores = self._ftdc_outputs(tmp_path, perturb=True)
        with pytest.raises(RuntimeError, match="differs from scores.npz"):
            rp.brier_part(pm, scores, mechanics=True)


def test_mechanics_never_writes_the_json(tmp_path, monkeypatch):
    out = tmp_path / "r.json"
    report = {
        "mechanics": True,
        "window": {"start": "a", "end": "b"},
        "bankroll": 1000.0,
        "ah_odds": {"rows": 0, "matches": 0, "first_created": None, "books": [], "snapshots": 0},
        "replay": {
            "candidates": 0,
            "ineligible": 0,
            "not_in_gate_eval": 0,
            "no_stored": 0,
            "defined": {},
            "state_drops": {},
            "price_ties": 0,
            "stored_vs_served_max_abs_shaped": None,
            "quotes_captured_after_tick": 0,
            "consensus_differs_at_kickoff": 0,
            "early_candidates": 0,
            "early_ineligible": 0,
            "early_no_stored": 0,
            "defined_early": {},
            "early_state_drops": {},
        },
        "brier": {"n": 0, "reconstruct_max_abs": 0.0},
        "brier_forward_pair": {"n": 0, "no_stored_before_kickoff": 0, "invariant_max_abs": 0.0},
        "rec_counts": {},
        "early_rec_counts": {},
    }
    monkeypatch.setattr(rp, "run", lambda args: report)
    assert rp.main(["--ftdc-dir", "x", "--database-url", "postgresql://x", "--mechanics", "--out-json", str(out)]) == 0
    assert not out.exists()
    assert json.dumps(report)


# ── Amendments after the run: the scaled-shape bound, ties, reproduction ──


class TestScaledShape:
    def test_recovers_the_true_stretch(self):
        rng = np.random.default_rng(2)
        q = rng.uniform(0.2, 0.5, (500, 3))
        d = rng.normal(0, 0.02, (500, 3))
        for k_true in (1.0, 1.4, 0.6):
            y = q + k_true * d
            e_dd, e_dr = rp.shape_moments(q, q + d, y)
            out = rp.scaled_shape([(e_dd, e_dr)])
            assert out["best_k"] == pytest.approx(k_true)
            assert out["gain_at_best_k"] == pytest.approx(k_true**2 * e_dd)
            assert out["gain_at_refit"] == pytest.approx((2 * k_true - 1) * e_dd)

    def test_gain_at_refit_is_the_brier_difference(self):
        rng = np.random.default_rng(3)
        q = rng.dirichlet([2, 2, 1], 400)
        p = rng.dirichlet([2, 2, 1], 400)
        y = np.eye(3)[rng.integers(0, 3, 400)]
        out = rp.scaled_shape([rp.shape_moments(q, p, y)])

        def brier(x):
            return np.mean(np.sum((x - y) ** 2, axis=1))

        assert out["gain_at_refit"] == pytest.approx(brier(q) - brier(p))

    def test_no_shift_has_no_bound(self):
        q = np.full((3, 3), 1 / 3)
        assert rp.scaled_shape([rp.shape_moments(q, q, np.eye(3))])["best_k"] is None


def test_tied_books_are_the_fresh_books_at_the_best_price():
    st = [
        _q("a", "home", -0.5, 1.95, 0.5),
        _q("b", "home", -0.5, 1.95, 0.5),
        _q("c", "home", -0.5, 1.95, 30.0),
        _q("d", "home", -0.5, 1.90, 0.5),
    ]
    (best,) = rp.best_quotes(st)
    assert rp.tied_books(st, best) == ["a", "b"]


def test_reproduction_accounts_for_every_production_rec():
    matches = pd.DataFrame(
        {"match_date": pd.to_datetime(["2026-09-10T15:00Z", "2026-09-11T15:00Z", "2026-09-12T15:00Z"])},
        index=["m1", "m2", "m3"],
    )
    prod = pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m9", "m0"],
            "selection": ["home_-0.75", "away_0.5", "home_-1", "away_2", "home_0"],
            "odds_decimal": [2.0, 1.9, 2.1, 1.95, 1.9],
            "bookmaker": ["b", "x", "a", "a", "a"],
            "stake": [10.0, 12.0, 8.0, 25.0, 5.0],
            "status": ["won", "lost", "void", "pending", "lost"],
            "profit_loss": [10.0, -12.0, 0.0, None, -5.0],
            "created_at": pd.to_datetime(
                [
                    "2026-09-10T14:46Z",
                    "2026-09-11T14:46Z",
                    "2026-09-12T14:46Z",
                    "2026-09-20T14:46Z",
                    "2026-08-01T14:46Z",
                ]
            ),
            # m9: a stale 'scheduled' row from REPRO_START on; m0: unfinished and before it (not counted).
            "kickoff": pd.to_datetime(
                [
                    "2026-09-10T15:00Z",
                    "2026-09-11T15:00Z",
                    "2026-09-12T15:00Z",
                    "2026-09-20T15:00Z",
                    "2026-08-01T15:00Z",
                ]
            ),
            "match_status": ["finished", "finished", "finished", "scheduled", "scheduled"],
        }
    )
    r1 = rp.Rec("m1", "home_-0.75", "home", -0.75, 2.0, "a", 0.06, 0.5, 0.5, 0.5, 40.0, 10.0, 0.5)
    r1.extra.update({"tie_books": ["a", "b"], "ev_mkt": -0.02})
    r2 = rp.Rec("m2", "away_0.5", "away", 0.5, 1.9, "y", 0.06, 0.5, 0.5, 0.5, 40.0, 11.0, 0.5)
    r2.extra.update({"tie_books": ["y"], "ev_mkt": -0.05})
    status = {"m1": "eligible", "m2": "eligible", "m3": "ineligible"}
    out = rp.reproduction(prod, [r1, r2], [], {"stored": ["m1", "m2"], "served": []}, status, matches, draws=200)
    assert out["production_recs_from_repro_start"] == 4 and out["production_recs_on_unfinished_matches"] == 1
    assert out["production_unfinished_detail"]["match_status"] == {"scheduled": 1}
    assert out["production_recs"] == 3
    s = out["stored"]
    assert (s["production_recs_in_pool"], s["both"], s["production_only"], s["replay_only"]) == (2, 2, 0, 0)
    assert s["both_same_price"] == 2 and s["both_same_stake"] == 1
    assert s["both_same_book"] == 0 and s["both_other_book_tied_at_the_price"] == 1
    assert s["production_recs_outside_pool"] == {"ineligible": 1}
    assert s["both_ev_mkt_stake_weighted"] == pytest.approx((-0.02 * 10 - 0.05 * 12) / 22)
    ps = out["production_settled"]
    assert (ps["recs"], ps["matches"], ps["stake"], ps["pl"]) == (3, 3, 30.0, -2.0)
    assert ps["roi_stake"]["estimate"] == pytest.approx(-2.0 / 30.0) and ps["roi_stake"]["se"] > 0


# ── Amendments 4-9: the bet-time readout, the early bet, the forward pair ──


def test_captured_after_counts_quotes_between_the_tick_and_kickoff():
    ko = pd.Timestamp("2026-10-10T15:00:00Z")
    t = rp.last_tick_before(ko)
    snap = {"match_id": "m1", "bookmaker": "a", "selection": "home", "line": -0.5, "odds_decimal": 1.9}
    rows = [_row("a", "home", -0.5, 1.9, ko - pd.Timedelta(hours=48), ts=ko - pd.Timedelta(hours=2), seen=t)]
    snaps = rp.snapshot_index([{**snap, "captured_at": ko - pd.Timedelta(hours=2)}])
    assert rp.captured_after(rows, snaps, t, ko) == 0  # seen AT the tick is not after it
    at_tick = rp.snapshot_index([{**snap, "captured_at": t}])
    assert rp.captured_after(rows, at_tick, t, ko) == 0  # nor is a snapshot captured at it
    snaps = rp.snapshot_index(
        [{**snap, "captured_at": ko - pd.Timedelta(hours=2)}, {**snap, "captured_at": ko - pd.Timedelta(minutes=5)}]
    )
    assert rp.captured_after(rows, snaps, t, ko) == 1
    rows[0]["last_seen_at"] = ko - pd.Timedelta(minutes=1)
    assert rp.captured_after(rows, snaps, t, ko) == 2
    rows[0]["last_seen_at"] = ko + pd.Timedelta(minutes=1)  # after kickoff: not a pre-kickoff observation
    assert rp.captured_after(rows, snaps, t, ko) == 1
    with pytest.raises(ValueError):
        rp.captured_after(rows, snaps, None, ko)


def _ah(h=1.5, a=1.1, rho=-0.05, target=(0.48, 0.27, 0.25), anchor=None):
    out = md.derive_from_lambdas(h, a, rho, target_1x2=target, target_over_25=anchor)["asian_handicap"]
    return {k: float(v) for k, v in out.items()}


class TestScoreAhPair:
    def test_shaped_lines_and_the_pooled_mean(self):
        sc = rp.score_ah_pair(_ah(), _ah(h=1.7, a=1.0), 2, 0)
        assert set(sc) == {*rp.SHAPED_UNITS, rp.POOLED}
        y = np.array(ftdc.handicap_truth(2, -1.75))
        assert sc["ah_-1.75"][0] == pytest.approx(float(np.sum((rp.handicap_vector(_ah(), -1.75) - y) ** 2)))
        assert sc[rp.POOLED][1] == pytest.approx(np.mean([sc[u][1] for u in rp.SHAPED_UNITS]))

    def test_an_invariant_line_mismatch_aborts(self):
        with pytest.raises(RuntimeError, match="invariant AH line"):
            rp.score_ah_pair(_ah(target=(0.48, 0.27, 0.25)), _ah(target=(0.5, 0.26, 0.24)), 1, 0)


class TestStatistics:
    def test_ratio_bootstrap(self):
        rng = np.random.default_rng(1)
        num, den = rng.normal(-0.03, 0.02, 50), np.ones(50)
        bs = rp.ratio_bootstrap(num, den, num, den, draws=200)
        assert bs["diff"]["estimate"] == 0.0 and bs["diff"]["se"] == 0.0
        assert bs["a"]["estimate"] == pytest.approx(num.mean()) and bs["a"]["se"] > 0
        assert rp.ratio_bootstrap(np.array([]), np.array([]), np.array([]), np.array([])) == {}
        empty = rp.ratio_bootstrap(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), draws=20)
        assert empty["a"]["estimate"] is None  # no recs at all: no mean, not a zero

    def test_between_fold_variance_recovers_tau(self):
        rng = np.random.default_rng(7)
        folds = np.repeat([f"f{i:02d}" for i in range(60)], 400)
        true = np.repeat(rng.normal(-0.001, 0.0005, 60), 400)
        d = true + rng.normal(0, 0.02, len(true))
        out = rp.between_fold_variance(d, folds)
        assert out["n_folds"] == 60 and out["tau2"] == pytest.approx(0.0005**2, rel=0.5)
        assert rp.between_fold_variance(d[:800], folds[:800])["tau2"] is None  # 2 folds: no estimate

    def test_clustered_power_matches_a_simulation_of_the_rule(self):
        from scipy.stats import t as student_t

        delta, sd, tau2, k, n = -0.0012, 0.023, 2.0e-7, 6, 5400
        rng = np.random.default_rng(11)
        fm = rng.normal(delta, math.sqrt(sd * sd * k / n + tau2), (20_000, k))
        upper = fm.mean(axis=1) + student_t.ppf(0.975, k - 1) * fm.std(axis=1, ddof=1) / math.sqrt(k)
        assert rp.clustered_t_power(delta, sd, tau2, k, n) == pytest.approx(float((upper < 0).mean()), abs=0.015)

    def test_clustered_power_needs_folds_not_only_matches(self):
        delta, sd, tau2 = -0.0012, 0.023, 2.0e-7
        assert rp.clustered_t_power(delta, sd, tau2, 6, 6000) < rp.clustered_t_power(delta, sd, tau2, 10, 6000)
        assert rp.clustered_t_power(delta, sd, tau2, 6, 4000) < rp.clustered_t_power(delta, sd, tau2, 6, 8000)
        # With tau2 = 0 and many folds it is the per-match z test.
        n_z = rp.n_for_power(sd, delta)
        assert rp.clustered_t_power(delta, sd, 0.0, 400, n_z) == pytest.approx(0.80, abs=0.01)
        # Between-fold drift alone can cap the power below 80% at any n with too few folds.
        assert rp.n_for_clustered_power(delta, sd, 1e-6, 4) is None
        n8 = rp.n_for_clustered_power(delta, sd, tau2, 8)
        assert rp.clustered_t_power(delta, sd, tau2, 8, n8) >= 0.8 > rp.clustered_t_power(delta, sd, tau2, 8, n8 - 1)
        assert rp.clustered_t_power(0.0, sd, tau2, 8, 1000) == 0.0


# ── run() end to end on a synthetic world ────────────────────────────

KO1 = pd.Timestamp("2026-10-10T15:00:00Z")  # before POWER_DEVIG_SINCE
KO2 = pd.Timestamp("2026-10-28T15:00:00Z")  # after it
KO3 = pd.Timestamp("2026-10-12T15:00:00Z")  # team X has 3 finished matches: ineligible
TARGET = (0.60, 0.22, 0.18)
LAMS = {"served": (2.0, 0.9), "refit": (2.1, 0.85)}
SPEC = [("m1", KO1, "H1", "A1", 2, 0), ("m2", KO2, "H2", "A2", 1, 1), ("m3", KO3, "H3", "X", 0, 2)]
# book: (home -0.5 price history [(hours before kickoff, price)], away +0.5 price)
BOOKS = {"a": ([(48, 1.95), (2, 1.85)], 2.00), "b": ([(48, 1.80)], 2.05), "c": ([(48, 1.80)], 2.05)}


def _world(tmp_path, after_tick=False):
    weights = ftdc.unit_weights()
    h = np.array([s[4] for s in SPEC])
    a = np.array([s[5] for s in SPEC])
    truth = ftdc.vector_truth(h, a)
    pm_rows = []
    for mid, ko, _, _, hs, as_ in SPEC:
        row = {"match_id": mid, "match_date": ko.isoformat(), "s_home": hs, "s_away": as_, "anchor_live": np.nan,
               "target::home": TARGET[0], "target::draw": TARGET[1], "target::away": TARGET[2]}  # fmt: skip
        for arm, (lh, la) in LAMS.items():
            row[f"lam_h::{arm}"], row[f"lam_a::{arm}"], row[f"rho::{arm}"] = lh, la, -0.05
        pm_rows.append(row)
    pm = pd.DataFrame(pm_rows)
    scores = {}
    for arm, (lh, la) in LAMS.items():
        P = np.stack([ftdc.served_matrix(lh, la, -0.05, np.array(TARGET), None) for _ in SPEC])
        for u in rp.ALL_AH_UNITS:
            scores[f"brier::{arm}::live::{u}"] = ftdc.brier_rows(np.einsum("rij,kij->rk", P, weights[u]), truth[u])
    scores["fold"] = np.array(["2026-10"] * len(SPEC))
    scores["match_id"] = pm["match_id"].to_numpy().astype(str)
    pm.to_csv(tmp_path / "per_match.csv.gz", index=False, compression="gzip")
    np.savez_compressed(tmp_path / "scores.npz", **scores)

    odds, snaps, matches, stored = [], [], [], []
    for mid, ko, home, away, hs, as_ in SPEC:
        seen = rp.last_tick_before(ko) - pd.Timedelta(seconds=30)  # the fetch just before the generator
        created = ko - pd.Timedelta(hours=48)
        for book, (hist, away_px) in BOOKS.items():
            last_h, last_px = hist[-1]
            odds.append(_row(book, "home", -0.5, last_px, created, ts=ko - pd.Timedelta(hours=last_h), seen=seen,
                             match=mid))  # fmt: skip
            odds.append(_row(book, "away", 0.5, away_px, created, seen=seen, match=mid))
            for hrs, px in hist:
                snaps.append({"match_id": mid, "bookmaker": book, "selection": "home", "line": -0.5,
                              "odds_decimal": px, "captured_at": ko - pd.Timedelta(hours=hrs)})  # fmt: skip
            snaps.append({"match_id": mid, "bookmaker": book, "selection": "away", "line": 0.5,
                          "odds_decimal": away_px, "captured_at": created})  # fmt: skip
        matches.append({"match_id": mid, "match_date": ko, "home_team_id": home, "away_team_id": away,
                        "home_score": hs, "away_score": as_, "metadata": {}, "league_name": "L"})  # fmt: skip
        made = ko - pd.Timedelta(hours=8)  # before the early tick
        stored.append([mid, "v1", "asian_handicap", _ah(1.9, 0.95, -0.05, TARGET), {"reconcile": "1x2"}, made])
        stored.append([mid, "v1", "match_result", dict(zip(("home", "draw", "away"), TARGET)), {}, made])
        # A row written after kickoff (a later run re-storing the match) must never be read: its 1x2 would
        # gate the -0.5 rec (gap 0.17 > 0.10) and change every Brier.
        late, late_t = ko + pd.Timedelta(hours=1), (0.77, 0.13, 0.10)
        stored.append([mid, "v2", "asian_handicap", _ah(2.4, 0.7, -0.05, late_t), {"reconcile": "1x2"}, late])
        stored.append([mid, "v2", "match_result", dict(zip(("home", "draw", "away"), late_t)), {}, late])
    if after_tick:  # a book moves after the last tick: the market IS observed after it
        snaps.append({"match_id": "m1", "bookmaker": "b", "selection": "home", "line": -0.5, "odds_decimal": 1.70,
                      "captured_at": KO1 - pd.Timedelta(minutes=5)})  # fmt: skip
    history = []
    for team in ("H1", "A1", "H2", "A2", "H3"):
        history += [{"home_team_id": team, "away_team_id": "Z", "match_date": pd.Timestamp("2026-08-01", tz="UTC")
                     + pd.Timedelta(days=i)} for i in range(12)]  # fmt: skip
    history += [{"home_team_id": "X", "away_team_id": "Z", "match_date": pd.Timestamp("2026-08-01", tz="UTC")}] * 3
    recs = pd.DataFrame(
        [
            {"rec_id": "r1", "match_id": "m1", "selection": "home_-0.5", "odds_decimal": 1.85, "bookmaker": "a",
             "stake": 25.0, "ev": 0.11, "status": "won", "actual_result": "won", "profit_loss": 21.25,
             "created_at": KO1 - pd.Timedelta(minutes=14), "kickoff": KO1, "match_status": "finished"},
            {"rec_id": "r9", "match_id": "m9", "selection": "away_2", "odds_decimal": 1.9, "bookmaker": "a",
             "stake": 25.0, "ev": 0.08, "status": "pending", "actual_result": None, "profit_loss": None,
             "created_at": KO1, "kickoff": KO1 + pd.Timedelta(hours=1), "match_status": "scheduled"},
        ]
    )  # fmt: skip
    db = {
        "odds": pd.DataFrame(odds),
        "snaps": pd.DataFrame(snaps),
        "matches": pd.DataFrame(matches),
        "history": pd.DataFrame(history),
        "stored": pd.DataFrame(
            stored, columns=["match_id", "model_version", "prediction_type", "probabilities", "metadata", "created_at"]
        ),
        "recs": recs,
        "bankroll": 1000.0,
        "stored_ah_per_month": {"2026-10": 3},
    }
    return db


class TestRunEndToEnd:
    def _run(self, tmp_path, monkeypatch, extra=(), after_tick=False):
        db = _world(tmp_path, after_tick)
        monkeypatch.setattr(rp, "load_db", lambda url, s, e: db)
        calls = []
        orig = rp.replay_match

        def spy(match_id, probs, state, bankroll, ev, floor, devig=rec_gating.devig_power, **kw):
            calls.append((match_id, devig))
            return orig(match_id, probs, state, bankroll, ev, floor, devig, **kw)

        monkeypatch.setattr(rp, "replay_match", spy)
        argv = ["--ftdc-dir", str(tmp_path), "--database-url", "x", "--start", "2026-10-01", "--end", "2026-11-01",
                "--bootstrap-draws", "50", *extra]  # fmt: skip
        return rp.run(rp.parse_args(argv)), calls, db

    def test_eligibility_devig_era_and_the_bet_time_readout(self, tmp_path, monkeypatch):
        csv = tmp_path / "recs.csv"
        r, calls, db = self._run(tmp_path, monkeypatch, extra=["--out-csv", str(csv)])
        assert r["replay"]["ineligible"] == 1 and r["replay"]["defined"]["served"] == 2
        assert "m3" not in {m for m, _ in calls}  # an ineligible match is never priced
        prop = {m for m, d in calls if d is rec_gating.devig_proportional}
        assert prop == {"m1"}  # the reproduction reads the logic era's de-vig before POWER_DEVIG_SINCE only
        assert r["replay"]["quotes_captured_after_tick"] == 0 and r["replay"]["consensus_differs_at_kickoff"] == 0
        stored_m1 = r["pairs"]["stored vs refit_on_stored"]["a"]
        assert stored_m1["recs"] == 2 and stored_m1["clv_same_mean"] == 0.0
        rows = pd.read_csv(csv)
        at_t = rows[(rows["arm"] == "stored") & (rows["match_id"] == "m1")].iloc[0]
        assert at_t["odds"] == 1.85 and at_t["bookmaker"] == "a"
        state_t = rp.quote_state(
            [o for o in db["odds"].to_dict("records") if o["match_id"] == "m1"],
            rp.snapshot_index(db["snaps"].to_dict("records")),
            rp.last_tick_before(KO1),
        )
        p_t = rp.consensus_prob(state_t, "home", -0.5)
        assert at_t["ev_mkt"] == pytest.approx(1.85 * p_t - 1.0)
        assert pd.isna(rows[rows["arm"] == "stored"]["clv_fair"]).all()  # the at-tick rec has no closing-line value
        # The early bet: priced 6 h out at the price then, valued against the LAST tick's consensus.
        early = rows[(rows["arm"] == "stored@early") & (rows["match_id"] == "m1")].iloc[0]
        t_e = rp.last_tick_before(KO1 - pd.Timedelta(hours=rp.EARLY_HOURS))
        assert pd.Timestamp(early["tick"]) == t_e and early["odds"] == 1.95
        state_e = rp.quote_state([o for o in db["odds"].to_dict("records") if o["match_id"] == "m1"],
                                 rp.snapshot_index(db["snaps"].to_dict("records")), t_e)  # fmt: skip
        p_e = rp.consensus_prob(state_e, "home", -0.5)
        assert early["clv_fair"] == pytest.approx(1.95 * p_t - 1.0)
        assert early["ev_mkt"] == pytest.approx(1.95 * p_e - 1.0)
        assert early["move"] == pytest.approx(1.95 * (p_t - p_e)) and early["move"] > 0
        assert early["clv_same"] == pytest.approx(1.95 / 1.85 - 1.0)
        assert r["early"]["stored"]["recs"] == 2 and r["early"]["stored"]["share_moved_toward"] == 1.0
        assert r["early_pairs"]["stored vs refit_on_stored"]["bootstrap_clv_fair"]["diff"]["estimate"] == 0.0
        # The forward pair covers every gate match with a stored row before kickoff, ineligible or not.
        fw = r["brier_forward_pair"]
        assert fw["n"] == 3 and fw["invariant_max_abs"] < 1e-12
        assert fw["units"][rp.POOLED]["post_incident"]["n"] == 3  # October: after the constant-prior incident
        pre = _ah(1.9, 0.95, -0.05, TARGET)
        want = np.mean(
            [
                rp.score_ah_pair(pre, rp.derive_ah(*LAMS["refit"], -0.05, TARGET, None), hs, as_)[rp.POOLED][0]
                for _, _, _, _, hs, as_ in SPEC
            ]
        )
        assert fw["units"][rp.POOLED]["baseline_brier"] == pytest.approx(want)  # the pre-kickoff stored row
        rep = r["reproduction"]
        assert rep["production_recs_on_unfinished_matches"] == 1 and rep["stored"]["both"] == 1
        assert rep["production_settled"]["pl"] == 21.25
        assert "forward_brier_pooled" in r["power"] and "forward_early_clv_fair" in r["power"]
        text = rp.format_report(r)
        assert "NOT closing-line value" in text and "PART B2" in text and "PART A2" in text

    def test_a_quote_after_the_tick_is_counted(self, tmp_path, monkeypatch):
        r, _, _ = self._run(tmp_path, monkeypatch, after_tick=True)
        assert r["replay"]["quotes_captured_after_tick"] == 1
        assert r["replay"]["consensus_differs_at_kickoff"] >= 1

    def test_mechanics_reads_no_money_and_writes_no_csv(self, tmp_path, monkeypatch):
        csv = tmp_path / "recs.csv"
        r, _, _ = self._run(tmp_path, monkeypatch, extra=["--mechanics", "--out-csv", str(csv)])
        assert not csv.exists()
        assert not {"money", "pairs", "early", "early_pairs", "reproduction", "power"} & set(r)
        assert set(r["brier_forward_pair"]) == {"n", "no_stored_before_kickoff", "invariant_max_abs"}
        assert "MECHANICS ONLY" in rp.format_report(r)


class TestForwardPlanning:
    def test_unequal_folds_reduce_to_equal_and_a_small_fold_costs_power(self):
        d, sd, tau2 = -0.0012, 0.023, 2.0e-7
        assert rp.clustered_t_power_sizes(d, sd, tau2, [900] * 8) == pytest.approx(
            rp.clustered_t_power(d, sd, tau2, 8, 7200)
        )
        assert rp.clustered_t_power_sizes(d, sd, tau2, [900] * 7 + [150]) < rp.clustered_t_power_sizes(
            d, sd, tau2, [900] * 7
        )
        assert rp.clustered_t_power_sizes(d, sd, tau2, [900, 0]) == 0.0

    def test_unequal_folds_match_a_simulation_of_the_rule(self):
        from scipy.stats import t as student_t

        d, sd, tau2 = -0.0012, 0.023, 2.0e-7
        sizes = np.array([830, 650, 600, 825, 860, 1060, 930, 150, 400, 860, 760, 850])
        rng = np.random.default_rng(5)
        fm = rng.normal(d, np.sqrt(sd * sd / sizes + tau2), (20_000, len(sizes)))
        k = len(sizes)
        upper = fm.mean(axis=1) + student_t.ppf(0.975, k - 1) * fm.std(axis=1, ddof=1) / math.sqrt(k)
        assert rp.clustered_t_power_sizes(d, sd, tau2, sizes) == pytest.approx(float((upper < 0).mean()), abs=0.03)

    def test_the_n_floor_is_where_the_scaled_projection_hits_80pct(self):
        d, sd, tau2 = -0.0012, 0.023, 2.0e-7
        sizes = [830, 650, 600, 825, 860, 1060, 930, 150, 400, 860, 760, 850]
        floor = rp.n_floor_for_sizes(d, sd, tau2, sizes)
        scale = floor / sum(sizes)
        assert rp.clustered_t_power_sizes(d, sd, tau2, [n * scale for n in sizes]) >= 0.8
        assert rp.clustered_t_power_sizes(d, sd, tau2, [n * scale * 0.97 for n in sizes]) < 0.8
        assert rp.n_floor_for_sizes(d, sd, 1e-5, sizes[:4]) is None  # drift alone caps the power

    def test_seasonal_sizes_follow_the_calendar_from_november(self):
        window = {}
        for y in (2024, 2025, 2026):
            for m in range(1, 13):
                if (y, m) <= (2026, 10):
                    window[f"{y}-{m:02d}"] = 200 if m == 6 else 1000
        out = rp.seasonal_fold_sizes(window, 900.0, 13)
        assert out[0] == pytest.approx(900.0 * 29 / 30)  # 2026-11, opened on the 2nd
        assert out[7] == pytest.approx(180.0)  # 2027-06: a fifth of an Aug-Oct month
        assert out[12] == pytest.approx(900.0)  # 2027-11, unclipped
