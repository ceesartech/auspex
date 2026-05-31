"""Unit tests for the NHL feature pipeline (Phase 2 of the NHL slice).

Covers the pure derivation functions (no DB) and a coverage check that
every feature compute_match_features emits has a neutral default —
without that, downstream predict_proba can silently get a NaN input
and skip the match.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


cfn = _load("compute_features_nhl", "compute_features_nhl.py")


# ── Pure math: devig ─────────────────────────────────────────────────


class TestDevigTwoWay:
    def test_balanced_2way_market(self):
        # 1.95 / 1.95 — bookies' 5% vig split evenly.
        p_a, p_b, margin = cfn._devig_two_way(1.95, 1.95)
        assert p_a == pytest.approx(0.5)
        assert p_b == pytest.approx(0.5)
        assert margin == pytest.approx(2 / 1.95 - 1.0)

    def test_asymmetric_market(self):
        # Heavy home favorite, e.g. NHL 1.50 / 2.80.
        p_home, p_away, margin = cfn._devig_two_way(1.50, 2.80)
        assert p_home > p_away
        assert p_home + p_away == pytest.approx(1.0, abs=1e-9)
        assert margin > 0

    @pytest.mark.parametrize("a, b", [(None, 1.95), (1.95, None), (None, None), (0, 1.95)])
    def test_returns_none_on_bad_input(self, a, b):
        # 0 is falsy in Python so the guard treats it as missing — fine
        # since 0 odds aren't a real market quote anyway.
        assert cfn._devig_two_way(a, b) == (None, None, None)


# ── _with_defaults ───────────────────────────────────────────────────


class TestWithDefaults:
    def test_replaces_none_with_neutral(self):
        out = cfn._with_defaults({"odds_home_ml": None})
        assert out["odds_home_ml"] == cfn.NEUTRAL_DEFAULTS["odds_home_ml"]

    def test_keeps_valid_numeric(self):
        out = cfn._with_defaults({"odds_home_ml": 1.65})
        assert out["odds_home_ml"] == 1.65

    def test_non_numeric_replaced(self):
        # A string sneaking into a numeric slot must be replaced, not
        # passed through — model would crash on str input.
        out = cfn._with_defaults({"odds_home_ml": "bad"})
        assert out["odds_home_ml"] == cfn.NEUTRAL_DEFAULTS["odds_home_ml"]

    def test_extras_pass_through(self):
        # Extra keys aren't in NEUTRAL_DEFAULTS but we keep them in the
        # JSONB so future model versions can pick them up without
        # reingesting history.
        out = cfn._with_defaults({"experimental_feature": 0.123})
        assert out["experimental_feature"] == 0.123

    def test_emits_every_default_even_when_input_empty(self):
        out = cfn._with_defaults({})
        for k, v in cfn.NEUTRAL_DEFAULTS.items():
            assert out[k] == v


# ── Coverage check ──────────────────────────────────────────────────


# Keys compute_match_features always sets (regardless of whether the
# underlying DB lookup returned a value). If any key here is missing
# from NEUTRAL_DEFAULTS, _with_defaults will silently leave it as None
# and the downstream model will see NaN.
EMITTED_KEYS = {
    # Moneyline
    "odds_home_ml",
    "odds_away_ml",
    "implied_prob_home_ml",
    "implied_prob_away_ml",
    "ml_bookie_margin",
    # Puck line
    "odds_home_pl15",
    "odds_away_pl15",
    "implied_prob_home_cover_pl15",
    "implied_prob_away_cover_pl15",
    # Totals
    "odds_over55",
    "odds_under55",
    "implied_prob_over55",
    # Rolling form (home + away × 8 stats)
    "home_roll_goals_for",
    "home_roll_goals_against",
    "home_roll_shots_for",
    "home_roll_shots_against",
    "home_roll_pp_pct",
    "home_roll_pk_pct",
    "home_roll_save_pct",
    "home_roll_standings_pts",
    "away_roll_goals_for",
    "away_roll_goals_against",
    "away_roll_shots_for",
    "away_roll_shots_against",
    "away_roll_pp_pct",
    "away_roll_pk_pct",
    "away_roll_save_pct",
    "away_roll_standings_pts",
    # Schedule context
    "home_days_rest",
    "home_back_to_back",
    "home_games_in_last_7",
    "away_days_rest",
    "away_back_to_back",
    "away_games_in_last_7",
    # Diffs
    "form_diff_goals_for",
    "form_diff_goals_against",
    "form_diff_shots_for",
    "form_diff_shots_against",
    "form_diff_pp_pct",
    "form_diff_pk_pct",
    "form_diff_save_pct",
    "form_diff_standings_pts",
    "rest_diff",
    # ── v2: starting-goalie rolling stats (per team) ──────────────
    "home_goalie_roll_save_pct",
    "home_goalie_roll_gaa",
    "home_goalie_days_rest",
    "home_goalie_back_to_back",
    "away_goalie_roll_save_pct",
    "away_goalie_roll_gaa",
    "away_goalie_days_rest",
    "away_goalie_back_to_back",
    "goalie_save_pct_diff",
    "goalie_gaa_diff",
    # ── v2: pace stats (per-60 normalized) ────────────────────────
    "home_shots_for_per_60",
    "home_shots_against_per_60",
    "home_goals_for_per_60",
    "home_goals_against_per_60",
    "away_shots_for_per_60",
    "away_shots_against_per_60",
    "away_goals_for_per_60",
    "away_goals_against_per_60",
    "pace_shots_diff",
    "pace_goals_diff",
    # ── v3: 5v5 even-strength stats (per team rolling) ────────────
    "home_roll_corsi_for_5v5",
    "home_roll_corsi_against_5v5",
    "home_roll_fenwick_for_5v5",
    "home_roll_fenwick_against_5v5",
    "home_roll_sog_for_5v5",
    "home_roll_sog_against_5v5",
    "home_roll_goals_for_5v5",
    "home_roll_goals_against_5v5",
    "home_roll_xg_for_5v5",
    "home_roll_xg_against_5v5",
    "home_roll_cf_pct_5v5",
    "away_roll_corsi_for_5v5",
    "away_roll_corsi_against_5v5",
    "away_roll_fenwick_for_5v5",
    "away_roll_fenwick_against_5v5",
    "away_roll_sog_for_5v5",
    "away_roll_sog_against_5v5",
    "away_roll_goals_for_5v5",
    "away_roll_goals_against_5v5",
    "away_roll_xg_for_5v5",
    "away_roll_xg_against_5v5",
    "away_roll_cf_pct_5v5",
    # ── v3: 5v5 differentials (home minus away) ───────────────────
    "cf_pct_diff_5v5",
    "xg_diff_5v5",
    "corsi_diff_5v5",
}


class TestNeutralDefaultsCoverage:
    def test_every_emitted_key_has_a_default(self):
        missing = EMITTED_KEYS - set(cfn.NEUTRAL_DEFAULTS.keys())
        assert not missing, (
            f"Features emitted by compute_match_features without a neutral "
            f"default — would feed NaN to the model: {sorted(missing)}"
        )

    def test_no_orphan_defaults(self):
        # Catches the inverse mistake: a default for a feature we no
        # longer emit. Not fatal (just dead config), but worth flagging.
        orphans = set(cfn.NEUTRAL_DEFAULTS.keys()) - EMITTED_KEYS
        assert not orphans, f"NEUTRAL_DEFAULTS keys with no matching emitted feature: {sorted(orphans)}"
