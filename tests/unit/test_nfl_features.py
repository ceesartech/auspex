"""Unit tests for compute_features_nfl — pure helpers + defaults.

NFL feature shape differs from NBA's in three ways tested here:
  * Rolling window is 5 games (NBA's is 10) — NFL has fewer games
    per season so the window is shorter.
  * Schedule context has short_week / long_week flags (Thu games +
    bye-week games), not just back_to_back like NBA.
  * Modern NFL averages (22.5 pts/game/team, ~57% home win rate)
    differ from NBA (114 pts/game/team, ~55%).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fnfl = _load("compute_features_nfl", "compute_features_nfl.py")


# ── Constants lockdown ──────────────────────────────────────────────


class TestConstants:
    def test_feature_set_name(self):
        # Training queries + predict-time lookups all key on this
        # exact pair — locked so a rename forces parallel updates.
        assert fnfl.FEATURE_SET == "nfl_baseline"
        assert fnfl.FEATURE_VERSION == "v1"

    def test_window_is_five(self):
        # 5-game rolling window — standard NFL "last 5" cadence. If
        # bumped to 10 it'd be more than half a season; if dropped
        # to 3 it'd be noisy on opening weeks.
        assert fnfl.WINDOW == 5

    def test_short_week_threshold(self):
        # Short-week threshold catches the Thursday game after a
        # Sunday — 3-4 days rest. NBA-style "back-to-back" doesn't
        # exist in NFL, but the short week is just as impactful.
        assert fnfl.SHORT_WEEK_DAYS == 4.0

    def test_long_week_threshold(self):
        # Long-week threshold catches bye-week games and international
        # series (10+ days rest). Models pick up the rested-team
        # advantage.
        assert fnfl.LONG_WEEK_DAYS == 10.0


# ── Neutral defaults ────────────────────────────────────────────────


class TestNeutralDefaults:
    def test_every_default_is_finite_number(self):
        # If any default sneaks in as None / nan / str, the prediction
        # path will silently break when the model fills missing inputs.
        for k, v in fnfl.NEUTRAL_DEFAULTS.items():
            assert isinstance(v, (int, float)), f"{k} is {type(v).__name__}"
            assert v == v, f"{k} is NaN"  # NaN != NaN

    def test_moneyline_implied_probs_sum_to_one(self):
        # Devigging math invariant — the two implied probs should sum
        # to exactly 1.0 (within float epsilon).
        ph = fnfl.NEUTRAL_DEFAULTS["implied_prob_home_ml"]
        pa = fnfl.NEUTRAL_DEFAULTS["implied_prob_away_ml"]
        assert abs(ph + pa - 1.0) < 1e-9

    def test_home_advantage_higher_than_nba(self):
        # NFL home win rate ≈ 57% (NBA's is 55%). The bigger crowd
        # effect + travel for visiting teams produces a slightly
        # larger home edge in football.
        ph = fnfl.NEUTRAL_DEFAULTS["implied_prob_home_ml"]
        assert 0.55 <= ph <= 0.60

    def test_neutral_spread_is_zero(self):
        # 0 = no favorite. Same convention as NBA.
        assert fnfl.NEUTRAL_DEFAULTS["closing_spread_home"] == 0.0

    def test_neutral_total_is_modern_league_modal(self):
        # 45 is the modern NFL modal total (was ~42 a decade ago —
        # passing offense has crept it up).
        assert 42 <= fnfl.NEUTRAL_DEFAULTS["closing_total_line"] <= 48

    def test_neutral_pts_scored_is_modern_average(self):
        # NFL teams average ~22.5 points/game. Locked so a future
        # tweak to 30 (basketball-style numbers) fires.
        assert 20 <= fnfl.NEUTRAL_DEFAULTS["home_roll_pts_scored"] <= 25

    def test_neutral_wins_is_half_window(self):
        # 5-game window, .500 record → 2.5 wins. Same shape as NBA's
        # 10-game / 5-win neutral.
        assert fnfl.NEUTRAL_DEFAULTS["home_roll_wins"] == 2.5

    def test_neutral_days_rest_is_modal_seven(self):
        # NFL modal cadence is Sunday-to-Sunday = 7 days rest.
        # short_week and long_week flags default to 0 (modal game,
        # no schedule-context advantage).
        assert fnfl.NEUTRAL_DEFAULTS["home_days_rest"] == 7.0
        assert fnfl.NEUTRAL_DEFAULTS["home_short_week"] == 0.0
        assert fnfl.NEUTRAL_DEFAULTS["home_long_week"] == 0.0


# ── _with_defaults: missing-value backfill ──────────────────────────


class TestWithDefaults:
    def test_missing_keys_filled(self):
        out = fnfl._with_defaults({})
        for k in fnfl.NEUTRAL_DEFAULTS:
            assert k in out

    def test_provided_values_override(self):
        out = fnfl._with_defaults({"odds_home_ml": 2.5})
        assert out["odds_home_ml"] == 2.5
        assert out["odds_away_ml"] == fnfl.NEUTRAL_DEFAULTS["odds_away_ml"]

    def test_none_value_is_replaced_with_default(self):
        # A None from a missing odds row should fall back to league
        # average, not propagate as None into the model.
        out = fnfl._with_defaults({"closing_total_line": None})
        assert out["closing_total_line"] == fnfl.NEUTRAL_DEFAULTS["closing_total_line"]

    def test_extra_keys_preserved(self):
        out = fnfl._with_defaults({"experimental_drive_efficiency": 0.42})
        assert out["experimental_drive_efficiency"] == 0.42


# ── Diff helper ────────────────────────────────────────────────────


class TestDiffHelper:
    def test_writes_diff_when_both_values_present(self):
        features = {"home_x": 5.0, "away_x": 3.0}
        fnfl._diff(features, "home_x", "away_x", "x_diff")
        assert features["x_diff"] == 2.0

    def test_skips_when_either_missing(self):
        features = {"home_x": 5.0}
        fnfl._diff(features, "home_x", "away_x", "x_diff")
        assert "x_diff" not in features


# ── Argparse plumbing ──────────────────────────────────────────────


class TestCli:
    def test_default_days_is_seven(self):
        args = fnfl.parse_args(["--database-url", "postgresql://x"])
        assert args.days == 7
        assert args.force is False
        assert args.all_finished is False

    def test_match_ids_parses(self):
        args = fnfl.parse_args(["--match-ids", "a,b,c", "--database-url", "x"])
        assert args.match_ids == "a,b,c"


# ── Cross-book TOTAL features ──────────────────────────────────────
#
# Landed 2026-06-03 after scripts/ab_nfl_cross_book.py verdict
# (ΔBrier -0.0110, ΔECE -0.0189 on 2024-2025 walk-forward).
# SPREAD was tested in the same A/B and DROPPED — these tests
# guard the total-only landing.


class _FakeCursor:
    """Minimal fake cursor for fetch_total_crossbook. The function
    runs one query whose rows have shape {bookmaker, p_line, p_odds,
    c_odds}. Tests seed `rows` and assert on the function's return."""

    def __init__(self, rows):
        self._rows = list(rows)
        self.execute_calls = []

    def execute(self, sql, params=None):
        self.execute_calls.append((sql, params))

    def fetchall(self):
        return list(self._rows)


class TestFetchTotalCrossbookEmpty:
    def test_no_rows_returns_empty_dict(self):
        # Missing market or unrecorded game → empty dict. The defaults
        # path in compute_for_match handles the actual feature values.
        cur = _FakeCursor([])
        out = fnfl.fetch_total_crossbook(cur, "match-1")
        assert out == {}


class TestFetchTotalCrossbookSingleBook:
    def test_single_book_zero_disagreement(self):
        # One book → max-min = 0, std = 0, consensus_mean = the only
        # line, book_count = 1.
        cur = _FakeCursor([
            {"bookmaker": "DraftKings", "p_line": 47.5,
             "p_odds": 1.91, "c_odds": 1.91},
        ])
        out = fnfl.fetch_total_crossbook(cur, "match-1")
        assert out["total_book_count"] == 1.0
        assert out["total_consensus_mean"] == 47.5
        assert out["total_max_minus_min"] == 0.0
        assert out["total_std"] == 0.0
        # Devigged over prob with symmetric -110/-110: 0.5.
        assert abs(out["total_consensus_implied_prob"] - 0.5) < 1e-9


class TestFetchTotalCrossbookMultiBook:
    def test_book_count_and_consensus_mean(self):
        rows = [
            {"bookmaker": "DraftKings", "p_line": 47.5,
             "p_odds": 1.91, "c_odds": 1.91},
            {"bookmaker": "FanDuel", "p_line": 48.0,
             "p_odds": 1.91, "c_odds": 1.91},
            {"bookmaker": "BetMGM", "p_line": 47.0,
             "p_odds": 1.91, "c_odds": 1.91},
        ]
        cur = _FakeCursor(rows)
        out = fnfl.fetch_total_crossbook(cur, "match-1")
        assert out["total_book_count"] == 3.0
        # Mean of 47.5, 48.0, 47.0 = 47.5.
        assert abs(out["total_consensus_mean"] - 47.5) < 1e-9
        assert out["total_max_minus_min"] == 1.0

    def test_population_std_matches_ab_harness(self):
        # ddof=0 (population std) to match the A/B script's
        # np.std(..., ddof=0) — this is load-bearing for the
        # validated metrics to translate to production.
        rows = [
            {"bookmaker": "DK", "p_line": 47.0, "p_odds": 2.0, "c_odds": 2.0},
            {"bookmaker": "FD", "p_line": 48.0, "p_odds": 2.0, "c_odds": 2.0},
            {"bookmaker": "MGM", "p_line": 49.0, "p_odds": 2.0, "c_odds": 2.0},
        ]
        cur = _FakeCursor(rows)
        out = fnfl.fetch_total_crossbook(cur, "match-1")
        # mean=48, sq diffs = 1+0+1 = 2, ddof=0 var=2/3, std≈0.8165.
        assert abs(out["total_std"] - 0.81649658) < 1e-6

    def test_devigged_consensus_prob_handles_asymmetric_odds(self):
        # Over -120 / Under -100 → over implied 0.5455 / 0.5000 raw,
        # devigged ≈ 0.522. Two books at the same prices.
        rows = [
            {"bookmaker": "DK", "p_line": 47.5,
             "p_odds": 1.833, "c_odds": 2.0},
            {"bookmaker": "FD", "p_line": 47.5,
             "p_odds": 1.833, "c_odds": 2.0},
        ]
        cur = _FakeCursor(rows)
        out = fnfl.fetch_total_crossbook(cur, "match-1")
        raw_over = 1.0 / 1.833
        raw_under = 1.0 / 2.0
        expected = raw_over / (raw_over + raw_under)
        assert abs(out["total_consensus_implied_prob"] - expected) < 1e-6


class TestFetchTotalCrossbookDegenerateOdds:
    def test_missing_under_odds_falls_back(self):
        # If c_odds is None, that book's devigged prob is dropped
        # but its LINE still counts toward consensus_mean / std.
        # Critical: don't let a one-sided missing odds null out
        # the whole feature row.
        rows = [
            {"bookmaker": "DK", "p_line": 47.5,
             "p_odds": 1.91, "c_odds": 1.91},
            {"bookmaker": "OnlyOver", "p_line": 48.5,
             "p_odds": 1.91, "c_odds": None},
        ]
        cur = _FakeCursor(rows)
        out = fnfl.fetch_total_crossbook(cur, "match-1")
        assert out["total_book_count"] == 2.0
        assert out["total_consensus_mean"] == 48.0
        # Only DK contributes a devigged prob (OnlyOver dropped).
        assert abs(out["total_consensus_implied_prob"] - 0.5) < 1e-9

    def test_zero_or_negative_odds_dropped_from_devig(self):
        # Defensive: zero/negative odds are impossible from real
        # bookmakers but defend against bad ingest data.
        rows = [
            {"bookmaker": "DK", "p_line": 47.5,
             "p_odds": 1.91, "c_odds": 1.91},
            {"bookmaker": "Bad", "p_line": 47.5,
             "p_odds": 0.0, "c_odds": 1.91},
        ]
        cur = _FakeCursor(rows)
        out = fnfl.fetch_total_crossbook(cur, "match-1")
        # Devig should still produce a value from the one valid book.
        assert "total_consensus_implied_prob" in out
        assert abs(out["total_consensus_implied_prob"] - 0.5) < 1e-9


class TestCrossbookDefaults:
    """The 5 cross-book keys must have neutral defaults so
    _with_defaults can fill them when fetch_total_crossbook
    returned empty (missing market data)."""

    CROSSBOOK_KEYS = (
        "total_book_count",
        "total_consensus_mean",
        "total_max_minus_min",
        "total_std",
        "total_consensus_implied_prob",
    )

    def test_all_crossbook_keys_in_neutral_defaults(self):
        for k in self.CROSSBOOK_KEYS:
            assert k in fnfl.NEUTRAL_DEFAULTS, f"{k} missing from NEUTRAL_DEFAULTS"

    def test_neutral_consensus_mean_matches_closing_total_default(self):
        # Both default to the same modern modal value (45) so a
        # match with no odds at all gets consistent total signals.
        assert fnfl.NEUTRAL_DEFAULTS["total_consensus_mean"] == fnfl.NEUTRAL_DEFAULTS["closing_total_line"]

    def test_neutral_disagreement_is_zero(self):
        # Zero disagreement is the "we have one book" neutral —
        # the model should not see fake disagreement signal when
        # the data is missing.
        assert fnfl.NEUTRAL_DEFAULTS["total_max_minus_min"] == 0.0
        assert fnfl.NEUTRAL_DEFAULTS["total_std"] == 0.0

    def test_neutral_implied_prob_is_half(self):
        # 0.5 = "no information" prior on a binary over/under.
        assert fnfl.NEUTRAL_DEFAULTS["total_consensus_implied_prob"] == 0.5

    def test_empty_fetch_filled_by_with_defaults(self):
        # End-to-end: empty cross-book result + _with_defaults fills
        # every cross-book key with its neutral. Guards the contract
        # that compute_for_match always emits all 5 keys.
        cur = _FakeCursor([])
        empty = fnfl.fetch_total_crossbook(cur, "match-1")
        assert empty == {}
        filled = fnfl._with_defaults(empty)
        for k in self.CROSSBOOK_KEYS:
            assert filled[k] == fnfl.NEUTRAL_DEFAULTS[k]
