"""Unit tests for the PROPOSED forward AH-only gate (scripts/
ab_soccer_ah_forward_gate.py): the rules pinned in code (the window, the
unapproved metric, the decision point), the per-match scoring against
grading_outcomes' handicap settlement and the invariant-line guard, the two
Brier decisions and the money decision, and run() end to end on a synthetic
corpus with the database and the fit faked."""

from __future__ import annotations

import importlib.util
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


fg = _load("ab_soccer_ah_forward_gate", "ab_soccer_ah_forward_gate.py")
rp = fg.rp
ftdc = fg.ftdc

from predictors import market_derivation as md  # noqa: E402


def _ah(h=1.5, a=1.1, rho=-0.05, target=(0.48, 0.27, 0.25), anchor=None):
    out = md.derive_from_lambdas(h, a, rho, target_1x2=target, target_over_25=anchor)["asian_handicap"]
    return {k: float(v) for k, v in out.items()}


class TestPinnedRules:
    def test_it_is_a_proposal(self):
        assert fg.APPROVED_GATE_METRIC is None
        assert fg.FORWARD_START == pd.Timestamp("2026-11-02T00:00:00Z")
        assert fg.GATE_METRICS["brier_house"] == {
            "policy_change": False,
            "min_folds": 6,
            "min_n": 2400,
            "population": "stored_ah",
        }
        assert fg.GATE_METRICS["brier_significance"]["policy_change"] is True
        assert fg.GATE_METRICS["money_clv"]["min_folds"] == 12 and fg.GATE_METRICS["money_clv"]["min_n"] == 7000

    def test_the_window_cannot_start_early(self):
        assert fg.check_window(None) == fg.FORWARD_START
        assert fg.check_window("2026-12-01") == pd.Timestamp("2026-12-01T00:00:00Z")
        with pytest.raises(RuntimeError, match="refused"):
            fg.check_window("2026-11-01")
        with pytest.raises(RuntimeError, match="refused"):
            fg.check_window("2026-11-01T23:59:59Z")

    def test_a_match_before_the_start_never_scores(self):
        fg.check_scored([pd.Timestamp("2026-11-02T00:00:00Z")])
        with pytest.raises(RuntimeError, match="reached scoring"):
            fg.check_scored([pd.Timestamp("2026-11-01T20:00:00Z")])

    def test_no_decision_without_an_approved_metric_or_below_the_point(self):
        ok, why = fg.decision_gate(None, 10_000, 12)
        assert not ok and "PROPOSAL" in why
        assert fg.decision_gate("brier_magic", 10_000, 12)[0] is False
        assert fg.decision_gate("brier_house", 2399, 6)[0] is False
        assert fg.decision_gate("brier_house", 2400, 5)[0] is False
        assert fg.decision_gate("brier_house", 2400, 6) == (True, "")
        assert fg.decision_gate("money_clv", 6999, 12)[0] is False
        assert fg.decision_gate("money_clv", 7000, 12) == (True, "")


class TestScoring:
    def test_scores_every_shaped_line_against_the_handicap_settlement(self):
        served, refit = _ah(), _ah(h=1.7, a=1.0)
        sc = fg.score_match(served, refit, 2, 0)
        assert set(sc) == {*fg.SHAPED_UNITS, fg.POOLED}
        y = np.array(ftdc.handicap_truth(2, -1.75))  # home -1.75 on a 2-goal margin: half win
        assert tuple(y) == (0.5, 0.0, 0.5)
        mine = float(np.sum((fg.handicap_vector(served, -1.75) - y) ** 2))
        assert sc["ah_-1.75"][0] == pytest.approx(mine)
        assert sc[fg.POOLED][1] == pytest.approx(np.mean([sc[u][1] for u in fg.SHAPED_UNITS]))

    def test_an_invariant_line_that_differs_aborts(self):
        # Reconciled to a different 1x2, AH 0 / +-0.25 / +-0.5 differ: the arms were not given the same target.
        with pytest.raises(RuntimeError, match="invariant AH line"):
            fg.score_match(_ah(target=(0.48, 0.27, 0.25)), _ah(target=(0.50, 0.26, 0.24)), 1, 0)

    def test_same_target_and_anchor_keep_the_invariant_lines(self):
        fg.score_match(_ah(anchor=0.55), _ah(h=1.2, a=1.3, anchor=0.55), 1, 1)

    def test_a_missing_line_is_loud(self):
        bad = {k: v for k, v in _ah().items() if not k.startswith("-2_")}
        with pytest.raises(RuntimeError, match="lacks line -2"):
            fg.handicap_vector(bad, -2.0)


def _pair(delta, se=0.0002, se_cluster=0.0003, n_folds=6):
    t_up = fg.ftdc.ht.t_upper(delta, se_cluster, n_folds)
    t_lo = fg.nhl.t_lower(delta, se_cluster, n_folds)
    return {
        "n": 5000,
        "delta": delta,
        "se": se,
        "delta_cluster": delta,
        "se_cluster": se_cluster,
        "n_folds": n_folds,
        "upper_95_cluster_t": t_up,
        "lower_95_cluster_t": t_lo,
    }


class TestDecisions:
    def _units(self, pooled, line=-0.001):
        units = {u: _pair(line) for u in fg.SHAPED_UNITS}
        units[fg.POOLED] = _pair(pooled)
        return units

    def test_house_needs_the_threshold_significance_does_not(self):
        u = self._units(-0.0012)
        assert fg.decide_brier(u, "brier_house", None)["ship"] is False
        assert fg.decide_brier(u, "brier_significance", None)["ship"] is True
        u = self._units(-0.0060)
        assert fg.decide_brier(u, "brier_house", None)["ship"] is True

    def test_a_regressing_line_blocks_both(self):
        u = self._units(-0.0060)
        u["ah_2"] = _pair(+0.0020)
        for metric in ("brier_house", "brier_significance"):
            d = fg.decide_brier(u, metric, None)
            assert d["ship"] is False and "ah_2" in d["regressions"]

    def test_disqualified_never_ships(self):
        assert fg.decide_brier(self._units(-0.01), "brier_house", "fold 2026-11: not converged")["ship"] is False
        assert fg.decide_money({"diff_clv_fair": {"lo95": 0.01}}, "x")["ship"] is False

    def test_money_needs_the_clv_lower_bound_above_zero(self):
        assert fg.decide_money({"diff_clv_fair": {"estimate": 0.004, "lo95": 0.001}}, None)["ship"] is True
        assert fg.decide_money({"diff_clv_fair": {"estimate": 0.004, "lo95": -0.001}}, None)["ship"] is False
        assert fg.decide_money({}, None)["ship"] is False


class _FakeDC:
    rho = -0.05

    def lambdas_for_match(self, home, away, league=None):
        return (1.7, 0.9) if home < away else (1.2, 1.3)


def _synthetic(n_days=200):
    rng = np.random.default_rng(4)
    rows, stored, mid = [], [], 0
    teams = [f"T{i}" for i in range(10)]
    for d in pd.date_range("2026-09-01", periods=n_days, freq="D", tz="UTC"):
        for _ in range(3):
            h, a = rng.choice(teams, 2, replace=False)
            hs, as_ = int(rng.poisson(1.5)), int(rng.poisson(1.1))
            ko = d + pd.Timedelta(hours=15)
            rows.append(
                dict(match_id=f"m{mid}", match_date=ko, league_id="L1", league_name="L", home_team_id=h,
                     away_team_id=a, home_team=h, away_team=a, home_score=hs, away_score=as_,
                     regulation_home_score=None, regulation_away_score=None)
            )  # fmt: skip
            tgt = tuple(rng.dirichlet([4, 2.5, 3]))
            anchor = 0.55 if mid % 4 == 0 else None
            meta = {"reconcile": "1x2+ou2.5", "market_over_2.5": anchor} if anchor else {"reconcile": "1x2"}
            created = ko - pd.Timedelta(hours=3)
            stored.append([f"m{mid}", "v1", "asian_handicap", _ah(1.4, 1.2, -0.05, tgt, anchor), meta, created])
            stored.append([f"m{mid}", "v1", "match_result", dict(zip(("home", "draw", "away"), tgt)), {}, created])
            mid += 1
    frame = pd.DataFrame(rows)
    frame["outcome"] = np.where(
        frame.home_score > frame.away_score, 0, np.where(frame.home_score == frame.away_score, 1, 2)
    )
    st = pd.DataFrame(
        stored, columns=["match_id", "model_version", "prediction_type", "probabilities", "metadata", "created_at"]
    )
    return frame, st


class TestSyntheticRun:
    def _setup(self, monkeypatch, n_days=200):
        frame, st = _synthetic(n_days)
        monkeypatch.setattr(fg.ftdc, "load_frame", lambda url: frame.copy())
        empty_odds = pd.DataFrame(columns=["match_id", "bookmaker", "selection", "line", "odds_decimal", "ts",
                                           "last_seen_at", "created", "is_live"])  # fmt: skip
        empty_snaps = pd.DataFrame(
            columns=["match_id", "bookmaker", "selection", "line", "odds_decimal", "captured_at"]
        )
        hist = frame[["home_team_id", "away_team_id", "match_date"]]
        monkeypatch.setattr(
            fg.rp,
            "load_db",
            lambda url, s, e: {"stored": st, "odds": empty_odds, "snaps": empty_snaps, "history": hist,
                               "bankroll": 1000.0},
        )  # fmt: skip
        fits = []

        def fake_fit(fit_df, tmp, tag):
            fits.append((tag, fit_df["match_date"].max()))
            return _FakeDC(), {}

        monkeypatch.setattr(fg.ftdc, "fit_refit", fake_fit)
        monkeypatch.setattr(fg.htrainer, "fit_refusal", lambda res, model, fixtures: None)
        return frame, fits

    def test_unapproved_reads_counts_only(self, monkeypatch):
        frame, fits = self._setup(monkeypatch)
        r = fg.run(fg.parse_args(["--database-url", "x", "--end", "2027-03-15"]))
        md = frame["match_date"]
        n_fwd = int(((md >= fg.FORWARD_START) & (md < pd.Timestamp("2027-03-16", tz="UTC"))).sum())
        assert r["n_scored"] == n_fwd and "per_unit" not in r
        assert r["decision"]["decidable"] is False and "PROPOSAL" in r["decision"]["reason"]
        assert r["folds"][0] == "2026-11"
        for tag, newest in fits:  # every refit stops before its own month
            assert newest < pd.Timestamp(f"{tag}-01", tz="UTC")
        assert "NO DECISION" in fg.format_report(r)

    def test_approved_house_metric_decides_on_forward_matches_only(self, monkeypatch):
        frame, _ = self._setup(monkeypatch)
        monkeypatch.setattr(fg, "APPROVED_GATE_METRIC", "brier_house")
        spec = dict(fg.GATE_METRICS["brier_house"], min_n=100, min_folds=3)
        monkeypatch.setattr(fg, "GATE_METRICS", {**fg.GATE_METRICS, "brier_house": spec})
        r = fg.run(fg.parse_args(["--database-url", "x", "--end", "2027-03-15"]))
        assert r["decision"]["decidable"] is True and "ship" in r["decision"]
        assert r["per_unit"][fg.POOLED]["n"] == r["n_scored"]
        assert r["per_unit"][fg.POOLED]["n_folds"] == len(r["folds"])
        assert "DECISION (brier_house)" in fg.format_report(r)

    def test_mechanics_reads_no_delta(self, monkeypatch):
        self._setup(monkeypatch)
        monkeypatch.setattr(fg, "APPROVED_GATE_METRIC", "brier_house")
        r = fg.run(fg.parse_args(["--database-url", "x", "--end", "2027-03-15", "--mechanics", "--limit", "5"]))
        assert "per_unit" not in r and r["decision"]["reason"] == "mechanics run: no delta read"
        assert r["n_scored"] == 5 * len(r["folds"])

    def test_nothing_after_the_start_yet(self, monkeypatch):
        self._setup(monkeypatch, n_days=40)
        r = fg.run(fg.parse_args(["--database-url", "x"]))
        assert r["n_scored"] == 0 and r["decision"]["decidable"] is False
