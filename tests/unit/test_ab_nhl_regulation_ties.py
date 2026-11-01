"""Unit tests for the NHL regulation-tie gate (scripts/ab_nhl_regulation_ties.py)
and the regulation-scoreline pieces it measures in train_hockey_dixon_coles
and precompute_predictions_nhl: the trainer's regulation frame, refusal and
fit; the serve path's matrices and markets (a legacy artifact serves exactly
as before, a malformed structure is refused); the gate's pure decision; and
run() end to end on a synthetic frame, no DB."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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


_load("telegram_notify", "telegram_notify.py")
ppn = _load("precompute_predictions_nhl", "precompute_predictions_nhl.py")
trainer = _load("train_hockey_dixon_coles", "train_hockey_dixon_coles.py")
gate = _load("ab_nhl_regulation_ties", "ab_nhl_regulation_ties.py")

from predictors import hockey_regulation as hr  # noqa: E402
from predictors.market_derivation import MAX_GOALS_DERIVE, build_dc_matrix, derive_markets  # noqa: E402
from predictors.poisson_models import DixonColesPredictor  # noqa: E402

CAP = MAX_GOALS_DERIVE
TILT = {0: 0.3, 1: -0.6, 2: -0.25}


def nhl_frame(start, end, per_day=4, seed=11, tilt=TILT, ot_share=0.53):
    """Synthetic NHL games: a regulation scoreline from a tilted Poisson
    (ties up, one-goal margins down, as in the real league), each tie
    resolved one goal up; FINAL scores + regulation_winner, as the frame
    carries them. No games in July-September."""
    rng = np.random.default_rng(seed)
    teams = [f"T{k}" for k in range(12)]
    attack = dict(zip(teams, rng.uniform(0.85, 1.15, len(teams))))
    i, j = np.indices((CAP + 1, CAP + 1))
    G = np.exp(sum(t * (np.abs(i - j) == m) for m, t in tilt.items()))
    rows, mid = [], 0
    for d in pd.date_range(start, end, freq="D"):
        if d.month in (7, 8, 9):
            continue
        for _ in range(per_day):
            h, a = rng.choice(teams, 2, replace=False)
            P = build_dc_matrix(3.1 * attack[h], 2.8 * attack[a], 0.0, max_goals=CAP) * G
            cell = rng.choice(P.size, p=(P / P.sum()).ravel())
            hs, as_ = divmod(int(cell), CAP + 1)
            if hs == as_:
                reg = "tie" if rng.random() < 0.9 else "draw"
                if rng.random() < ot_share:
                    hs += 1
                else:
                    as_ += 1
            else:
                reg = "home" if hs > as_ else "away"
            rows.append(
                dict(
                    match_id=f"m{mid}",
                    match_date=d,
                    league_id="NHL",
                    home_team=h,
                    away_team=a,
                    home_score=hs,
                    away_score=as_,
                    match_outcome=0 if hs > as_ else 1,
                    regulation_winner=reg,
                )
            )
            mid += 1
    return pd.DataFrame(rows)


def _cfg():
    return gate.nhl.arm_configs()["refit"]


# ── trainer: the regulation frame, refusal and fit ───────────────────


class TestTrainerRegulation:
    def test_query_adds_only_regulation_winner(self):
        q = trainer.NHL_DC_REGULATION_QUERY
        assert "m.metadata->>'regulation_winner' AS regulation_winner" in q
        assert q.replace(",\n        m.metadata->>'regulation_winner' AS regulation_winner", "") == (
            trainer.NHL_DC_TRAINING_QUERY
        )

    def test_regulation_frame(self):
        f = pd.DataFrame(
            {
                "home_score": [4, 2, 5, 1],
                "away_score": [3, 3, 2, 3],
                "regulation_winner": ["tie", "draw", "home", "away"],
            }
        )
        out = trainer.regulation_frame(f)
        assert out["home_score"].tolist() == [3, 2, 5, 1] and out["away_score"].tolist() == [3, 2, 2, 3]
        assert out["overtime_home_win"].tolist()[:2] == [1.0, 0.0] and out["overtime_home_win"].isna().tolist()[2:] == [
            True,
            True,
        ]

    @pytest.mark.parametrize("label", [None, "overtime", float("nan")])
    def test_an_unknown_label_refuses_the_frame(self, label):
        f = pd.DataFrame({"home_score": [4, 2], "away_score": [3, 1], "regulation_winner": ["tie", label]})
        with pytest.raises(ValueError, match="without a known regulation_winner"):
            trainer.regulation_frame(f)

    def test_a_contradicting_score_refuses_the_frame(self):
        f = pd.DataFrame({"home_score": [5], "away_score": [3], "regulation_winner": ["tie"]})
        with pytest.raises(ValueError, match="not a regulation score"):
            trainer.regulation_frame(f)

    def test_fit_writes_a_structure_that_serves(self, tmp_path):
        frame = nhl_frame("2023-10-01", "2024-04-30")
        model, result, refusal = trainer.fit_regulation_model(frame, _cfg())
        assert refusal is None, refusal
        s = hr.structure_of(model)
        assert s.margin_buckets == trainer.REGULATION_MARGIN_BUCKETS == (0, 1, 2, 3, 4)
        assert result["poisson_iterations"] is not None and result["iterations"] is not None
        assert result["fitted_tie_rate"] == pytest.approx(result["observed_tie_rate"], abs=0.005)
        assert model.rho == 0.0
        path = tmp_path / "m.bin"
        model.save(str(path))
        served = ppn._load_hockey_dixon_coles_artifact(str(path))
        R, F, structured = ppn.hockey_dc_matrices(served, "T0", "T1")
        assert structured and np.trace(F) == 0.0 and np.trace(R) > 0.18

    def test_refusals(self):
        frame = nhl_frame("2023-10-01", "2024-01-31")
        model, result, refusal = trainer.fit_regulation_model(frame, _cfg(), (0,))
        assert refusal is None
        fixtures = [("T0", "T1", None)]
        assert "Poisson start" in trainer.regulation_fit_refusal(
            {**result, "poisson_iterations": None}, model, fixtures
        )
        assert "did not converge" in trainer.regulation_fit_refusal({**result, "iterations": None}, model, fixtures)
        off = {**result, "fitted_mean_total_goals": result["observed_mean_total_goals"] * 1.03}
        assert "outside +/-2%" in trainer.regulation_fit_refusal(off, model, fixtures)
        model.regulation_structure = None
        assert "no regulation structure" in trainer.regulation_fit_refusal(result, model, fixtures)
        model.regulation_structure = {
            "version": 1,
            "margin_buckets": [0],
            "margin_log_factors": [7.0],
            "ot_home_share": 0.5,
        }
        assert "invalid regulation structure" in trainer.regulation_fit_refusal(result, model, fixtures)


# ── serve path ───────────────────────────────────────────────────────


class TestServePath:
    def _legacy(self, tmp_path):
        frame = nhl_frame("2023-10-01", "2024-01-31")
        m = DixonColesPredictor(_cfg())
        m.train(frame[gate.nhl.FIT_COLS])
        m.rho = 0.0
        path = tmp_path / "legacy.bin"
        m.save(str(path))
        return ppn._load_hockey_dixon_coles_artifact(str(path)), path

    def test_a_legacy_artifact_serves_exactly_as_before(self, tmp_path):
        model, _ = self._legacy(tmp_path)
        R, F, structured = ppn.hockey_dc_matrices(model, "T0", "T3")
        assert not structured and R is F
        h, a = model.lambdas_for_match("T0", "T3")
        before = derive_markets("nhl", build_dc_matrix(h, a, 0.0, max_goals=CAP))
        before.pop("moneyline")
        assert ppn.hockey_dc_markets(model, "T0", "T3") == before
        assert set(before) == set(ppn.HOCKEY_DERIVE_TO_PREDICTION)

    def test_a_malformed_structure_is_refused_loudly(self, tmp_path, caplog):
        _, path = self._legacy(tmp_path)
        data = json.loads(path.read_text())
        data["regulation_structure"] = {
            "version": 1,
            "margin_buckets": [0],
            "margin_log_factors": [],
            "ot_home_share": 0.5,
        }
        path.write_text(json.dumps(data))
        with caplog.at_level("ERROR"):
            assert ppn._load_hockey_dixon_coles_artifact(str(path)) is None
        assert "malformed regulation structure" in caplog.text

    def test_a_structure_on_a_nonzero_rho_artifact_is_refused_loudly(self, tmp_path, caplog):
        """fit_structure fits the margin factors on the rho-0 Poisson grid;
        hockey_dc_matrices builds the grid with the artifact's rho. A
        structured artifact with rho != 0 (only a hand edit makes one: the
        trainer refuses it) would serve a regulation matrix the fit never
        produced. A legacy artifact keeps whatever rho it has."""
        frame = nhl_frame("2023-10-01", "2024-01-31")
        model, _, refusal = trainer.fit_regulation_model(frame, _cfg(), (0,))
        assert refusal is None
        path = tmp_path / "structured.bin"
        model.save(str(path))
        assert ppn._load_hockey_dixon_coles_artifact(str(path)) is not None
        data = json.loads(path.read_text())
        for rho in (-0.1, float("nan"), "x"):
            path.write_text(json.dumps({**data, "rho": rho}))
            caplog.clear()
            with caplog.at_level("ERROR"):
                assert ppn._load_hockey_dixon_coles_artifact(str(path)) is None, rho
            assert "regulation structure with rho" in caplog.text
        legacy = {k: v for k, v in data.items() if k != "regulation_structure"}
        path.write_text(json.dumps({**legacy, "rho": -0.1}))
        served = ppn._load_hockey_dixon_coles_artifact(str(path))
        assert served is not None and served.rho == -0.1
        assert not ppn.hockey_dc_matrices(served, "T0", "T1")[2]

    def test_a_missing_artifact_disables_the_markets(self, tmp_path):
        assert ppn._load_hockey_dixon_coles_artifact(str(tmp_path / "absent.bin")) is None


# ── the gate's pure pieces ───────────────────────────────────────────


class TestDecision:
    def test_blocking_units_are_every_unit_but_the_primary_and_the_listed_score(self):
        assert gate.PRIMARY == "regulation_1x2"
        assert set(gate.BLOCKING_UNITS) == set(gate.UNIT_NAMES) - {"regulation_1x2", "correct_score_listed"}
        assert "over_5.5" in gate.BLOCKING_UNITS and "correct_score_grid" in gate.BLOCKING_UNITS
        assert gate.DECIDING_ARM == "reg_margin_frozen"
        assert gate.CANDIDATES["reg_margin_frozen"] == ((0, 1, 2, 3, 4), True)

    def test_decide(self):
        ok = {gate.DECIDING_ARM: {"ship_strict": True}}
        assert gate.decide(ok, {gate.DECIDING_ARM: {}}, {})["ship"]
        r = gate.decide(ok, {gate.DECIDING_ARM: {"over_4.5": "significantly worse"}}, {})
        assert not r["ship"] and "over_4.5" in r["reason"]
        r = gate.decide({gate.DECIDING_ARM: {"ship_strict": False}}, {gate.DECIDING_ARM: {}}, {})
        assert not r["ship"] and "fails ship_strict" in r["reason"]
        r = gate.decide({}, {}, {gate.DECIDING_ARM: {"reason": "refused"}})
        assert not r["ship"] and "disqualified: refused" in r["reason"]

    @staticmethod
    def _pair(delta, se_cluster, n_folds=8):
        """run()'s pair() output for one comparison: per-match SE 0.002, the
        clustered delta equal to the per-match one."""
        return {
            "delta": delta,
            "se": 0.002,
            "delta_cluster": delta,
            "se_cluster": se_cluster,
            "n_folds": n_folds,
            "upper_95_cluster_t": gate.ht.t_upper(delta, se_cluster, n_folds),
        }

    def test_ship_strict_needs_the_t_bound_below_zero(self):
        """Rule (a): ship_verdict's SHIP (threshold + per-match and clustered
        z=2 bounds) is not enough. At 8 folds the t(7) multiplier (2.36) is
        wider than z=2, and a delta whose t bound crosses 0 fails."""
        between = self._pair(-0.010, 0.0045)  # z=2 clustered upper -0.0010, t(7) upper +0.0006
        assert between["delta_cluster"] + 2 * between["se_cluster"] < 0 < between["upper_95_cluster_t"]
        v = gate.strict_verdict(between, "served")
        assert v["ship"] and not v["ship_strict"]
        assert v["upper_95_cluster_t"] == between["upper_95_cluster_t"] and v["baseline"] == "served"

        v = gate.strict_verdict(self._pair(-0.010, 0.0030), "served")  # t(7) upper -0.0029
        assert v["ship"] and v["ship_strict"]
        v = gate.strict_verdict(self._pair(-0.004, 0.0010), "served")  # every bound < 0, threshold missed
        assert v["upper_95_cluster_t"] < 0 and not v["ship"] and not v["ship_strict"]
        v = gate.strict_verdict(self._pair(-0.010, 0.0030, n_folds=1), "served")  # no t bound
        assert v["upper_95_cluster_t"] is None and not v["ship_strict"]

    def test_blocking_regressions_read_the_lower_bound(self):
        per_unit = {u: {"delta": -0.001, "lower_95_cluster_t": -0.002} for u in gate.UNIT_NAMES}
        per_unit["clean_sheet_home"] = {"delta": 0.0004, "lower_95_cluster_t": 0.0001}
        per_unit["regulation_1x2"] = {"delta": 0.05, "lower_95_cluster_t": 0.04}  # the primary never "regresses"
        per_unit["correct_score_listed"] = {"delta": 0.05, "lower_95_cluster_t": 0.04}  # context
        assert set(gate.blocking_regressions(per_unit)) == {"clean_sheet_home"}

    def test_per_match_pair(self):
        p = gate.per_match_pair(np.array([0.2, 0.4, np.nan]), np.array([0.3, 0.3, 0.1]))
        assert p["n"] == 2 and p["delta"] == pytest.approx(0.0) and p["se"] == pytest.approx(0.1)
        assert gate.per_match_pair(np.array([np.nan]), np.array([1.0])) == {"n": 0, "delta": None, "se": None}


# ── run() end to end ─────────────────────────────────────────────────


class TestSyntheticRun:
    @staticmethod
    def _setup(monkeypatch, tmp_path):
        frame = nhl_frame("2023-10-01", "2025-02-28")
        reg = frame[["match_id", "regulation_winner"]]
        monkeypatch.setattr(
            gate.nhl, "load_frame", lambda url: gate.nhl.assemble_frame(frame.drop(columns="regulation_winner"), reg)
        )
        served = DixonColesPredictor(gate.nhl.arm_configs()["pinned"])
        served.train(frame[frame["match_date"] < "2024-03-01"][gate.nhl.FIT_COLS])
        served.rho = 0.0
        path = tmp_path / "served.bin"
        served.save(str(path))
        sha = gate.ht.fingerprint(str(path))["sha256"]
        args = gate.parse_args(
            [
                "--start",
                "2024-11-01",
                "--served-artifact",
                str(path),
                "--served-sha256",
                sha,
                "--database-url",
                "x",
                "--out-rows",
                str(tmp_path / "rows.csv"),
                "--out-grids",
                str(tmp_path / "grids.npz"),
            ]
        )
        return args, frame

    def test_bookkeeping_decision_and_per_game_output(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gate, "PREVIOUS_GATE_NEWEST", "2025-02-20T00:00:00+00:00")
        monkeypatch.setattr(gate, "FRESH_SEASON", 2024)
        args, frame = self._setup(monkeypatch, tmp_path)
        r = gate.run(args)
        json.dumps(gate.ht.clean_json(r), allow_nan=False)
        assert r["folds"] == ["2024-11", "2024-12", "2025-01", "2025-02"]
        assert r["disqualified"] == {}
        md = pd.to_datetime(frame["match_date"], utc=True)
        n = int(((md >= "2024-11-01") & (md < "2025-03-01")).sum())
        assert r["n_eval"] == n
        frozen = {f["arm"]: f for f in r["fits"] if f["fold"] == "frozen@2024-11-01"}
        assert set(frozen) == {"refit_frozen", "reg_poisson_frozen", "reg_diag_frozen", "reg_margin_frozen"}
        assert all(f["n_fit"] == int((md < "2024-11-01").sum()) for f in frozen.values())
        assert frozen["reg_margin_frozen"]["margin_buckets"] == [0, 1, 2, 3, 4]
        assert frozen["reg_poisson_frozen"]["margin_buckets"] == []
        assert sum(1 for f in r["fits"] if f["arm"] == "reg_margin") == 4  # walk-forward: one fit per fold
        for arm in gate.CHALLENGERS:
            assert r["vs_served"][arm]["regulation_1x2"]["n"] == n
            assert r["vs_served"][arm]["regulation_1x2"]["n_folds"] == 4
        assert set(r["verdicts_vs_served"]) == set(gate.CHALLENGERS)
        for arm in gate.CHALLENGERS:  # each verdict is rule (a) on the reported pair, t bound included
            v = r["verdicts_vs_served"][arm]
            assert v == gate.strict_verdict(r["vs_served"][arm]["regulation_1x2"], "served")
            assert v["upper_95_cluster_t"] == r["vs_served"][arm]["regulation_1x2"]["upper_95_cluster_t"]
        # The tilted truth: the structured arms put far more mass on regulation ties than served.
        cal = r["calibration"]
        assert cal["reg_margin_frozen"]["regulation_pred"]["draw"] > cal["served"]["regulation_pred"]["draw"] + 0.03
        assert r["vs_served"]["reg_margin_frozen"]["regulation_1x2"]["delta"] < 0
        assert r["vs_served"]["reg_margin_frozen"]["correct_score_grid"]["delta"] < 0  # no tie cell any more
        assert r["decision"]["arm"] == "reg_margin_frozen"
        fresh = r["freshness"]["after_previous_gate"]
        assert fresh["n"] == int((md > "2025-02-20").sum()) > 0
        assert r["freshness"]["season_2026_27"]["n"] == n  # FRESH_SEASON patched to 2024-25
        assert set(r["primary_discovery_split"]) == {"folds 2026-03.."}  # no discovery fold in this window
        rows = pd.read_csv(tmp_path / "rows.csv")
        assert len(rows) == n
        # The per-game file reproduces the report's deltas.
        delta = (rows["reg_margin_frozen__brier__regulation_1x2"] - rows["served__brier__regulation_1x2"]).mean()
        assert delta == pytest.approx(r["vs_served"]["reg_margin_frozen"]["regulation_1x2"]["delta"])
        grids = np.load(tmp_path / "grids.npz")
        assert grids["reg_margin_frozen"].shape == (n, (CAP + 1) ** 2)
        assert np.allclose(grids["reg_margin_frozen"].reshape(n, CAP + 1, CAP + 1).trace(axis1=1, axis2=2), 0.0)
        text = gate.format_report(r)
        assert "DECISION (pre-registered: reg_margin_frozen must pass ship_strict vs served on regulation_1x2" in text
        assert "FRESHNESS (confirmation, never decides)" in text

    def test_another_incumbent_is_refused(self, monkeypatch, tmp_path):
        args, _ = self._setup(monkeypatch, tmp_path)
        args.served_sha256 = "0" * 64
        with pytest.raises(RuntimeError, match="not the incumbent"):
            gate.run(args)

    def test_a_frame_the_trainer_refuses_disqualifies_the_candidates(self, monkeypatch, tmp_path):
        args, frame = self._setup(monkeypatch, tmp_path)
        frame.loc[5, "regulation_winner"] = None
        reg = frame[["match_id", "regulation_winner"]]
        monkeypatch.setattr(
            gate.nhl, "load_frame", lambda url: gate.nhl.assemble_frame(frame.drop(columns="regulation_winner"), reg)
        )
        r = gate.run(args)
        assert set(r["disqualified"]) == set(gate.CANDIDATES)
        assert "refuse this frame" in r["disqualified"]["reg_margin_frozen"]["reason"]
        assert not r["decision"]["ship"] and "disqualified" in r["decision"]["reason"]
        assert "DO-NOT-SHIP" in gate.format_report(r)
