"""Unit tests for the soccer derived-market display rule
(scripts/ab_soccer_derived_vs_constant.py): which stored prediction types
lose to a per-league constant, read off the full-time Dixon-Coles harness's
per-row scores. They pin the type catalogue against the served derivation,
the scores.npz format against the harness's own writer, and the
pre-registered rule: a type hides only as a whole, only when served loses
beyond the noise in both the live and the proxy regime."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "services" / "ml-models" / "src"))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dvc = _load("ab_soccer_derived_vs_constant", "ab_soccer_derived_vs_constant.py")
ftdc = sys.modules["ab_soccer_ft_dixon_coles"]

from predictors import market_derivation as md  # noqa: E402

N_FOLDS = 12
PER_FOLD = 60
ALL_REGIMES = ("live", "proxy", "none")


def _scores(
    effects: Optional[Dict[Tuple[Optional[str], str], float]] = None,
    noise: float = 0.002,
    seed: int = 0,
    draws: float = 0.25,
) -> Dict[str, np.ndarray]:
    """A synthetic harness output: every arm x regime x unit Brier array, the
    constants without a regime. served = base_league + effect + noise, where
    effects[(regime, unit)] (or effects[(None, unit)] for every regime) is
    the true served - constant gap. draw_no_bet is NaN on the 'draw' rows in
    every arm, as the harness writes it."""
    effects = effects or {}
    rng = np.random.default_rng(seed)
    n = N_FOLDS * PER_FOLD
    folds = np.repeat([f"2026-{m:02d}" for m in range(1, N_FOLDS + 1)], PER_FOLD)
    is_draw = rng.random(n) < draws
    s: Dict[str, np.ndarray] = {"fold": folds.astype(object), "match_id": np.array([f"m{i}" for i in range(n)])}
    for unit in ftdc.UNIT_NAMES:
        base = rng.uniform(0.15, 0.35, n)
        if unit in ftdc.CONSTANT_UNITS:
            s[f"brier::base_league::-::{unit}"] = base.copy()
            s[f"brier::base_league_dw::-::{unit}"] = base + rng.normal(0, noise, n)
        for arm in ftdc.DC_ARMS:
            for regime in ALL_REGIMES:
                gap = effects.get((regime, unit), effects.get((None, unit), 0.0)) if arm == "served" else 0.0
                s[f"brier::{arm}::{regime}::{unit}"] = base + gap + rng.normal(0, noise, n)
    for key in [k for k in s if k.endswith("::draw_no_bet")]:
        s[key] = np.where(is_draw, np.nan, s[key])
    return s


def _roundtrip(tmp_path: Path, s: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Through the harness's own writer and this script's loader."""
    paths = ftdc.write_outputs(s, str(tmp_path / "out"))
    return dvc.load_scores(paths["scores_npz"])


# ── The type catalogue ───────────────────────────────────────────────


def test_type_units_are_exactly_the_stored_full_time_derived_types():
    """Every market derive_soccer_markets emits but the 1x2 (stored as the
    ensemble's match_result) is one stored type; the half-time markets come
    from other artifacts and are not judged here."""
    P = md.build_dc_matrix(1.4, 1.1, -0.05, max_goals=md.MAX_GOALS_DERIVE)
    stored = set(md.derive_soccer_markets(P)) - {"1x2"}
    assert set(dvc.TYPE_UNITS) == stored
    assert len(dvc.TYPE_UNITS) == 14
    assert not any(t.endswith("_ht") or t.startswith("ht_") for t in dvc.TYPE_UNITS)


def test_every_scored_unit_belongs_to_exactly_one_type_and_listed_to_none():
    seen = [u for units in dvc.TYPE_UNITS.values() for u in units]
    assert len(seen) == len(set(seen))
    assert set(seen) == set(ftdc.UNIT_NAMES) - {ftdc.LISTED}
    for market, units in dvc.TYPE_UNITS.items():
        assert all(ftdc.UNIT_BY_NAME[u].market == market for u in units)


def test_multi_line_types_are_judged_on_every_line_they_show():
    assert dvc.TYPE_UNITS["over_under"] == tuple(f"over_{x}" for x in ("0.5", "1.5", "2.5", "3.5", "4.5", "5.5"))
    assert len(dvc.TYPE_UNITS["asian_handicap"]) == 17  # the five 1x2-fixed lines are shown too
    assert len(dvc.TYPE_UNITS["team_total"]) == 6
    assert dvc.TYPE_UNITS["double_chance"] == ("double_chance_1X", "double_chance_12", "double_chance_X2")
    assert dvc.TYPE_UNITS["correct_score"] == ("correct_score_grid",)
    for single in ("btts", "draw_no_bet", "odd_even", "winning_margin", "total_goals", "result_btts"):
        assert dvc.TYPE_UNITS[single] == (single,)


def test_the_rule_reads_the_served_arms_own_constant_and_the_gates_deciding_regimes():
    assert dvc.BASELINE == ftdc.matched_constant("served") == "base_league"
    assert dvc.CONTEXT_BASELINE == ftdc.matched_constant("refit")
    assert dvc.DECIDING_REGIMES == ftdc.DECIDING_REGIMES == ("live", "proxy")
    assert dvc.MODEL_NAME == "ensemble"


def test_brier_keys_match_the_harness_layout():
    assert dvc.brier_key("served", "live", "btts") == "brier::served::live::btts"
    assert dvc.brier_key("base_league", "live", "btts") == "brier::base_league::-::btts"
    assert dvc.brier_key("base_league_dw", "proxy", "ah_1") == "brier::base_league_dw::-::ah_1"


# ── The rule ─────────────────────────────────────────────────────────


def test_a_type_that_loses_in_both_regimes_hides_and_one_that_loses_in_one_does_not(tmp_path):
    s = _roundtrip(tmp_path, _scores({(None, "btts"): 0.01, ("live", "odd_even"): 0.01}))
    r = dvc.run(s)
    assert r["hidden_pairs"] == ["ensemble:btts"]
    assert r["types"]["btts"]["hide"] and all(v > 0 for v in r["types"]["btts"]["lower_bounds"].values())
    odd = r["types"]["odd_even"]
    assert not odd["hide"] and "live only" in odd["reason"]
    assert odd["lower_bounds"]["live"] > 0 >= odd["lower_bounds"]["proxy"]


def test_noise_alone_hides_nothing(tmp_path):
    r = dvc.run(_roundtrip(tmp_path, _scores()))
    assert r["hidden_pairs"] == []
    assert all(not t["hide"] for t in r["types"].values())


def test_a_type_hides_only_as_a_whole(tmp_path):
    """over_0.5 alone loses but the other five lines win by more: the stored
    over_under row stays visible (the loser is reported per unit). A
    team_total whose lines lose on the sum hides although one line wins."""
    effects: Dict[Tuple[Optional[str], str], float] = {(None, "over_0.5"): 0.01}
    effects.update({(None, f"over_{x}"): -0.01 for x in ("1.5", "2.5", "3.5", "4.5", "5.5")})
    effects.update({(None, u): 0.01 for u in dvc.TYPE_UNITS["team_total"]})
    effects[(None, "team_total_home_over_0.5")] = -0.01
    r = dvc.run(_roundtrip(tmp_path, _scores(effects)))
    ou = r["types"]["over_under"]
    assert not ou["hide"]
    assert ou["vs_base_league"]["live"]["delta"] < 0
    assert dvc.nhl.lower_bound(ou["per_unit"]["live"]["over_0.5"]) > 0
    assert r["types"]["team_total"]["hide"]
    assert dvc.nhl.lower_bound(r["types"]["team_total"]["per_unit"]["proxy"]["team_total_home_over_0.5"]) < 0
    assert r["hidden_pairs"] == ["ensemble:team_total"]


def test_the_type_score_is_the_sum_of_its_units(tmp_path):
    s = _roundtrip(tmp_path, _scores(seed=3))
    total = dvc.type_score(s, "served", "proxy", "clean_sheet")
    expected = s["brier::served::proxy::clean_sheet_home"] + s["brier::served::proxy::clean_sheet_away"]
    np.testing.assert_allclose(total, expected, rtol=0, atol=0)
    p = dvc.compare(s, "clean_sheet", "proxy")
    direct = ftdc.pair(
        expected, s["brier::base_league::-::clean_sheet_home"] + s["brier::base_league::-::clean_sheet_away"], s["fold"]
    )
    assert p["delta"] == direct["delta"] and p["lower_95_cluster_t"] == direct["lower_95_cluster_t"]
    assert p["n_folds"] == N_FOLDS


def test_hide_verdict_reads_the_a2_lower_bound():
    clustered = {"delta": 0.002, "se": 0.0001, "lower_95_cluster_t": 0.0004}
    assert dvc.hide_verdict("btts", {"live": clustered, "proxy": clustered})["hide"]
    straddles = dict(clustered, lower_95_cluster_t=-0.0001)
    v = dvc.hide_verdict("btts", {"live": clustered, "proxy": straddles})
    assert not v["hide"] and "live only" in v["reason"]
    # No clustered bound: the per-match delta - 2 SE decides (rule A2's fallback).
    per_match = {"delta": 0.002, "se": 0.0009, "lower_95_cluster_t": None}
    assert dvc.hide_verdict("btts", {"live": per_match, "proxy": per_match})["lower_bounds"]["live"] == pytest.approx(
        0.0002
    )
    exactly_zero = {"delta": 0.002, "se": 0.001, "lower_95_cluster_t": None}
    assert not dvc.hide_verdict("btts", {"live": exactly_zero, "proxy": exactly_zero})["hide"]


def test_an_unassessable_type_fails_the_run():
    with pytest.raises(RuntimeError, match="cannot be assessed"):
        dvc.hide_verdict("btts", {"live": {"delta": None}, "proxy": {"delta": 0.001, "se": 0.0001}})
    with pytest.raises(RuntimeError, match="no 95% bound"):
        dvc.hide_verdict("btts", {"live": {"delta": 0.001, "se": None}, "proxy": {"delta": 0.001, "se": 0.0001}})
    with pytest.raises(RuntimeError, match="cannot be assessed"):
        dvc.hide_verdict("btts", {"live": {"delta": 0.001, "se": 0.0001}})  # proxy missing


def test_draw_no_bet_skips_draws_but_a_missing_score_elsewhere_fails(tmp_path):
    s = _roundtrip(tmp_path, _scores(seed=5))
    p = dvc.compare(s, "draw_no_bet", "live")
    assert 0 < p["n"] < N_FOLDS * PER_FOLD
    broken = dict(s)
    broken["brier::served::live::btts"] = s["brier::served::live::btts"].copy()
    broken["brier::served::live::btts"][7] = np.nan
    with pytest.raises(RuntimeError, match="without a score"):
        dvc.compare(broken, "btts", "live")
    misaligned = dict(s)
    misaligned["brier::base_league::-::draw_no_bet"] = np.nan_to_num(s["brier::base_league::-::draw_no_bet"])
    with pytest.raises(RuntimeError, match="grade different rows"):
        dvc.compare(misaligned, "draw_no_bet", "live")


def test_a_scores_file_missing_a_unit_fails_loudly(tmp_path):
    s = _roundtrip(tmp_path, _scores())
    del s["brier::base_league::-::ah_1.75"]
    with pytest.raises(KeyError, match="ah_1.75"):
        dvc.run(s)


def test_the_espn_era_context_covers_folds_from_2026_06(tmp_path):
    r = dvc.run(_roundtrip(tmp_path, _scores()))
    assert r["espn_era_rows"] == 7 * PER_FOLD  # 2026-06 .. 2026-12
    assert r["types"]["btts"]["espn_era"]["live"]["n"] == 7 * PER_FOLD


# ── Provenance and the command line ──────────────────────────────────


def test_a_report_from_another_run_is_refused(tmp_path):
    s = _roundtrip(tmp_path, _scores())
    with pytest.raises(RuntimeError, match="n_eval"):
        dvc.run(s, {"n_eval": len(s["fold"]) + 1, "meta": {}})


def test_load_scores_refuses_a_file_without_folds(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, x=np.zeros(3))
    with pytest.raises(RuntimeError, match="fold labels"):
        dvc.load_scores(str(path))


def test_main_end_to_end(tmp_path, capsys):
    s = _scores({(None, "btts"): 0.01, (None, "correct_score_grid"): 0.01})
    paths = ftdc.write_outputs(s, str(tmp_path / "out"))
    report = {
        "n_eval": len(s["fold"]),
        "meta": {"served_fingerprint": {"sha256": "02141a6a" + "0" * 56}, "live_check": {"matches": 3}},
        "anchored": {"live": 1, "proxy": 2},
        "decision": {"reason": "synthetic"},
    }
    (tmp_path / "report.json").write_text(json.dumps(report))
    out = tmp_path / "result.json"
    rc = dvc.main(["--scores", paths["scores_npz"], "--report", str(tmp_path / "report.json"), "--out-json", str(out)])
    assert rc == 0
    result = json.loads(out.read_text())
    assert result["hidden_pairs"] == ["ensemble:btts", "ensemble:correct_score"]
    assert result["provenance"]["served_sha256"].startswith("02141a6a")
    text = capsys.readouterr().out
    assert "btts" in text and "HIDE" in text and "HIDDEN_PREDICTION_PAIRS to add" in text


def test_main_returns_1_on_a_broken_input(tmp_path):
    assert dvc.main(["--scores", str(tmp_path / "missing.npz")]) == 1
