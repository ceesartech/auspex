"""Unit tests for the NHL Dixon-Coles A/B harness (scripts/ab_nhl_dixon_coles.py)
— the pure pieces that decide whether the experiment is honest: seasons and the
thin-fold pooling, grading through grading_outcomes itself, the serve-path
read-out of all 8 served markets, the constants, the disqualification, the
pre-registered regression rule and decision — plus run() end to end on a
synthetic frame. The DB path is exercised by running the harness on prod."""

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


ab = _load("ab_nhl_dixon_coles", "ab_nhl_dixon_coles.py")


def _matrix(h=3.1, a=2.8):
    from predictors.market_derivation import MAX_GOALS_DERIVE, build_dc_matrix

    return build_dc_matrix(h, a, 0.0, max_goals=MAX_GOALS_DERIVE)


def _ev(rows):
    """Eval rows from (home_score, away_score, regulation_winner) triples."""
    return pd.DataFrame(
        {
            "home_team": [f"H{i}" for i in range(len(rows))],
            "away_team": [f"A{i}" for i in range(len(rows))],
            "home_score": [r[0] for r in rows],
            "away_score": [r[1] for r in rows],
            "regulation_winner": [r[2] for r in rows],
        }
    )


# ── seasons and folds ────────────────────────────────────────────────


class TestSeasons:
    @pytest.mark.parametrize(
        "ts, season",
        [
            ("2024-10-05", 2024),
            ("2025-06-20", 2024),
            ("2021-07-07", 2020),  # the 2020-21 final ran into July
            ("2026-08-01", 2026),
            ("2026-01-15T03:00:00+00:00", 2025),
        ],
    )
    def test_season_is_the_year_it_starts(self, ts, season):
        assert ab.season_of(ts) == season

    def test_label(self):
        assert ab.season_label(2024) == "2024-25" and ab.season_label(2099) == "2099-00"


class TestPoolThinFolds:
    def _folds(self):
        return ab.ft.build_folds("2025-03-01", "2025-10-31")  # Mar..Oct 2025

    def test_playoff_tails_join_april_and_the_offseason_is_skipped(self):
        folds = self._folds()
        counts = [240, 170, 40, 6, 0, 0, 12, 200]  # Mar Apr May Jun Jul Aug Sep Oct
        kept, pooled, skipped = ab.pool_thin_folds(folds, counts)
        assert [f.label for f in kept] == ["2025-03", "2025-04", "2025-10"]
        april = kept[1]
        # Still fitted at April 1 — strictly before every pooled row.
        assert april.fit_cutoff == pd.Timestamp("2025-04-01", tz="UTC")
        assert april.eval_end == pd.Timestamp("2025-07-01", tz="UTC")
        assert pooled == [
            {"fold": "2025-05", "n_eval": 40, "into": "2025-04"},
            {"fold": "2025-06", "n_eval": 6, "into": "2025-04"},
        ]
        # A thin September opens a new season: never pooled across the summer.
        assert skipped == [
            {"fold": "2025-07", "n_eval": 0},
            {"fold": "2025-08", "n_eval": 0},
            {"fold": "2025-09", "n_eval": 12},
        ]

    def test_a_thin_first_fold_is_skipped(self):
        folds = self._folds()[:2]
        kept, pooled, skipped = ab.pool_thin_folds(folds, [10, 170])
        assert [f.label for f in kept] == ["2025-04"] and pooled == [] and skipped[0]["fold"] == "2025-03"

    def test_never_pools_across_a_gap(self):
        folds = ab.ft.build_folds("2025-03-01", "2025-05-31")
        kept, pooled, skipped = ab.pool_thin_folds(folds, [200, 0, 30])
        assert [f.label for f in kept] == ["2025-03"]
        assert [s["fold"] for s in skipped] == ["2025-04", "2025-05"] and pooled == []

    def test_pooled_rows_are_in_the_eval_split_not_the_fit(self):
        folds = self._folds()
        kept, _, _ = ab.pool_thin_folds(folds, [240, 170, 40, 6, 0, 0, 12, 200])
        frame = pd.DataFrame(
            {"match_date": pd.to_datetime(["2025-03-30", "2025-04-02", "2025-05-20", "2025-06-10"], utc=True)}
        )
        fit, ev = ab.ft.split_fold(frame, kept[1])
        assert len(fit) == 1 and len(ev) == 3


# ── grading through grading_outcomes ─────────────────────────────────


class TestGradedVector:
    @pytest.mark.parametrize(
        "market, key, score, expected",
        [
            ("over_under", "over_5.5", (4, 2), 1.0),
            ("over_under", "over_5.5", (3, 2), 0.0),
            ("over_under", "over_4.5", (3, 2), 1.0),
            ("over_under", "over_6.5", (4, 3), 1.0),
            ("puck_line", "-1.5_home", (4, 2), 1.0),
            ("puck_line", "-1.5_home", (3, 2), 0.0),
            ("puck_line", "1.5_home", (2, 3), 1.0),  # home loses by one: covers +1.5
            ("puck_line", "1.5_home", (1, 3), 0.0),
            ("clean_sheet", "home_yes", (2, 0), 1.0),
            ("clean_sheet", "away_yes", (2, 0), 0.0),
            ("clean_sheet", "away_yes", (0, 1), 1.0),
            ("win_to_nil", "home_yes", (3, 0), 1.0),
            ("win_to_nil", "away_yes", (0, 0), 0.0),
            ("win_to_nil", "away_yes", (1, 2), 0.0),
        ],
    )
    def test_final_score_markets(self, market, key, score, expected):
        y = ab.graded_vector(market, (key,), score[0], score[1], "home")
        assert y.tolist() == [expected]

    def test_regulation_markets_read_regulation_winner_not_the_final_score(self):
        # A 3-2 OT win is a regulation TIE: draw, and 1X / X2 cover it, 12 does not.
        assert ab.graded_vector("regulation_1x2", ab.REGULATION_CLASSES, 3, 2, "tie").tolist() == [0, 1, 0]
        assert ab.graded_vector("regulation_1x2", ab.REGULATION_CLASSES, 3, 2, "draw").tolist() == [0, 1, 0]
        for key, won in (("1X", 1.0), ("12", 0.0), ("X2", 1.0)):
            assert ab.graded_vector("double_chance", (key,), 3, 2, "tie").tolist() == [won]
        assert ab.graded_vector("regulation_1x2", ab.REGULATION_CLASSES, 2, 4, "away").tolist() == [0, 0, 1]

    @pytest.mark.parametrize("missing", [None, float("nan")])
    def test_regulation_markets_without_regulation_winner_are_ungradable(self, missing):
        assert ab.graded_vector("regulation_1x2", ab.REGULATION_CLASSES, 3, 2, missing) is None
        assert ab.graded_vector("double_chance", ("12",), 3, 2, missing) is None
        assert ab.graded_vector("over_under", ("over_5.5",), 3, 2, missing).tolist() == [0.0]

    def test_total_goals_top_band_is_eight_plus(self):
        keys = next(u.keys for u in ab.UNITS if u.name == "total_goals")
        assert ab.graded_vector("total_goals", keys, 6, 3, "home").tolist() == [0] * 8 + [1]
        assert ab.graded_vector("total_goals", keys, 4, 3, "home")[7] == 1.0

    def test_correct_score_other_wins_off_the_listed_keys(self):
        probs = {"3-2": 0.1, "2-1": 0.1, "other": 0.8}
        keys = list(probs)
        assert ab.graded_vector("correct_score", keys, 3, 2, "home", probabilities=probs).tolist() == [1, 0, 0]
        assert ab.graded_vector("correct_score", keys, 7, 1, "home", probabilities=probs).tolist() == [0, 0, 1]

    def test_a_partition_without_exactly_one_winner_raises(self):
        with pytest.raises(ValueError, match="graded correct"):
            ab.graded_vector("regulation_1x2", ("home", "away"), 3, 2, "tie")

    def test_fixed_truth_shapes_and_ungradable_rows(self):
        t = ab.fixed_truth(_ev([(4, 2, "home"), (3, 2, None)]))
        assert set(t) == {u.name for u in ab.FIXED_UNITS}
        assert t["over_5.5"].shape == (2, 1) and t["over_5.5"][:, 0].tolist() == [1.0, 0.0]
        assert t["regulation_1x2"].shape == (2, 3) and np.isnan(t["regulation_1x2"][1]).all()
        assert t["regulation_1x2"][0].tolist() == [1, 0, 0]
        assert np.isnan(t["double_chance_12"][1, 0])
        assert t["total_goals"].shape == (2, 9)


# ── the serve-path read-out ──────────────────────────────────────────


class TestServedMarkets:
    def test_units_cover_exactly_what_precompute_predictions_nhl_serves(self):
        pytest.importorskip("psycopg2")
        nhl = _load("precompute_predictions_nhl", "precompute_predictions_nhl.py")
        assert (
            set(ab.SERVED_MARKETS)
            == set(nhl.HOCKEY_DERIVE_TO_PREDICTION)
            == set(nhl.HOCKEY_DERIVE_TO_PREDICTION.values())
        )
        assert {u.market for u in ab.UNITS} == set(ab.SERVED_MARKETS)

    def test_served_markets_drop_the_moneyline_and_carry_every_unit_key(self):
        m = ab.served_markets(_matrix())
        assert "moneyline" not in m and set(m) == set(ab.SERVED_MARKETS)
        for unit in ab.FIXED_UNITS:
            for key in unit.keys:
                assert key in m[unit.market], (unit.name, key)
        assert len(m["correct_score"]) == 13

    def test_readout_matches_the_derivation_and_scores_by_hand(self):
        from predictors.market_derivation import derive_hockey_markets

        P = _matrix()
        ref = derive_hockey_markets(P)
        ev = _ev([(3, 2, "tie")])
        probs = ab.readout([ab.served_markets(P)], ev)
        assert probs["over_5.5"][0, 0] == pytest.approx(ref["over_under"]["over_5.5"])
        assert probs["puck_line_+1.5"][0, 0] == pytest.approx(ref["puck_line"]["1.5_home"])
        assert probs["double_chance_X2"][0, 0] == pytest.approx(ref["double_chance"]["X2"])
        assert probs["regulation_1x2"][0].tolist() == pytest.approx(
            [ref["regulation_1x2"][c] for c in ab.REGULATION_CLASSES]
        )
        # correct_score truth is on the arm's own keys: one winner, 3-2 if listed, else 'other'.
        cs_keys = list(ref["correct_score"])
        y = probs["_cs_truth"][0]
        assert y.sum() == 1.0
        assert cs_keys[int(y.argmax())] == ("3-2" if "3-2" in cs_keys else "other")
        s = ab.score_arm(probs, ab.fixed_truth(ev))
        assert set(s) == set(ab.UNIT_NAMES)
        assert s["over_5.5"][0] == pytest.approx((ref["over_under"]["over_5.5"] - 0.0) ** 2)
        p_reg = np.array([ref["regulation_1x2"][c] for c in ab.REGULATION_CLASSES])
        assert s["regulation_1x2"][0] == pytest.approx(float(np.sum((p_reg - [0, 1, 0]) ** 2)))
        p_cs = np.array([ref["correct_score"][k] for k in cs_keys])
        assert s["correct_score"][0] == pytest.approx(float(np.sum((p_cs - y) ** 2)))

    def test_ungradable_rows_score_nan_for_their_units_only(self):
        ev = _ev([(3, 2, None)])
        s = ab.score_arm(ab.readout([ab.served_markets(_matrix())], ev), ab.fixed_truth(ev))
        assert np.isnan(s["regulation_1x2"][0]) and np.isnan(s["double_chance_1X"][0])
        assert np.isfinite(s["over_5.5"][0]) and np.isfinite(s["correct_score"][0])

    def test_model_matrices_never_pass_a_league(self):
        seen = []

        class Fake:
            rho = None  # an artifact without rho serves rho 0

            def lambdas_for_match(self, home, away, league="NOT PASSED"):
                seen.append(league)
                return 3.0, 2.7

        mats = ab.model_matrices(Fake(), pd.DataFrame({"home_team": ["A"], "away_team": ["B"], "league_id": ["L"]}))
        assert seen == ["NOT PASSED"]
        assert mats[0] == pytest.approx(_matrix(3.0, 2.7))

    def test_expected_totals(self):
        P = np.zeros((3, 3))
        P[2, 1] = P[0, 0] = 0.5
        assert ab.expected_totals([P]).tolist() == [1.5]


# ── constants ────────────────────────────────────────────────────────


class TestConstants:
    def _fit(self):
        return pd.DataFrame(
            {
                "home_score": [4, 2, 3, 1, 5, 2],
                "away_score": [1, 3, 2, 4, 0, 1],
                "regulation_winner": ["home", "away", "tie", "away", "home", None],
                "season": [2023, 2023, 2023, 2024, 2024, 2024],
            }
        )

    def test_regulation_markets_come_from_regulation_winner_rates(self):
        fit = self._fit()
        ev = pd.DataFrame({"season": [2024]})
        (m,), (P,) = ab.constant_markets(fit, ev, None, by_season=False)
        # 5 labelled rows: home 2, tie 1, away 2 (the unlabelled one is ignored).
        assert m["regulation_1x2"] == pytest.approx({"home": 0.4, "draw": 0.2, "away": 0.4})
        assert m["double_chance"] == pytest.approx({"1X": 0.6, "12": 0.8, "X2": 0.6})
        assert P.sum() == pytest.approx(1.0)
        # Final-score markets are the empirical rates: totals 5,5,5,5,5,3 -> 5/6 over 4.5, none over 5.5.
        assert m["over_under"]["over_4.5"] == pytest.approx(5 / 6)
        assert m["over_under"]["over_5.5"] == pytest.approx(0.0)
        assert m["clean_sheet"]["away_yes"] == pytest.approx(0.0)  # home never blanked
        assert m["clean_sheet"]["home_yes"] == pytest.approx(1 / 6)

    def test_per_season_shrinks_toward_the_global_rate(self):
        fit = self._fit()
        ev = pd.DataFrame({"season": [2024, 2099]})
        glob, _ = ab.constant_markets(fit, ev, None, by_season=False)
        seas, _ = ab.constant_markets(fit, ev, None, by_season=True)
        # An unseen season is the global constant; a seen one moves (a little, k=200) toward its own rate.
        assert seas[1]["over_under"]["over_4.5"] == pytest.approx(glob[1]["over_under"]["over_4.5"])
        assert seas[0]["regulation_1x2"]["away"] != pytest.approx(glob[0]["regulation_1x2"]["away"], abs=1e-6)

    def test_scorelines_beyond_the_grid_are_capped(self):
        assert ab.scoreline_index([12, 0], [1, 11], cap=10).tolist() == [10 * 11 + 1, 10]


# ── disqualification, regressions, decision ──────────────────────────


def _valid_probs(n=2):
    m = ab.served_markets(_matrix())
    return ab.readout([m] * n, _ev([(3, 2, "home")] * n))


class TestDisqualification:
    def test_trainer_refusal_disqualifies_a_challenger_even_with_valid_probabilities(self):
        reason = ab.disqualification("refit", probs=_valid_probs(), refusal="the fit did not converge")
        assert reason and "would refuse" in reason and "did not converge" in reason

    def test_valid_challenger_passes(self):
        assert ab.disqualification("pinned", probs=_valid_probs()) is None

    def test_invalid_probabilities_disqualify_a_challenger(self):
        probs = _valid_probs()
        probs["regulation_1x2"][0] = [0.5, 0.5, 0.5]
        assert "regulation_1x2" in ab.disqualification("refit_frozen", probs=probs)
        probs = _valid_probs()
        probs["over_5.5"][1, 0] = np.nan
        assert "over_5.5" in ab.invalid_probabilities(probs)

    def test_served_and_constants_raise_instead(self):
        for arm in ("served", "base_global", "base_season", "base_season_dw"):
            with pytest.raises(RuntimeError, match=arm):
                ab.disqualification(arm, refusal="anything")

    def test_invalid_matrices(self):
        good = _matrix()
        assert ab.invalid_matrices([good]) is None
        assert "1 of 2" in ab.invalid_matrices([good, np.full((3, 3), np.nan)])
        assert "1 of 2" in ab.invalid_matrices([good, good * 2])


class TestRegressionRule:
    def test_above_the_margin_with_a_positive_bound_regresses(self):
        assert "> +0.002" in ab.regression({"delta": 0.003, "se": 0.0005, "upper_95_cluster_t": 0.006})

    def test_at_or_below_the_margin_is_not_a_regression(self):
        assert ab.regression({"delta": 0.002, "se": 0.0001, "upper_95_cluster_t": 0.0025}) is None
        assert ab.regression({"delta": -0.01, "se": 0.001, "upper_95_cluster_t": -0.008}) is None

    def test_a_clustered_bound_below_zero_clears_it(self):
        # pooled per-match delta above the margin, but the fold-clustered bound is negative
        assert ab.regression({"delta": 0.0025, "se": 0.0003, "upper_95_cluster_t": -0.0001}) is None

    def test_falls_back_to_the_per_match_bound(self):
        assert ab.regression({"delta": 0.003, "se": 0.001, "upper_95_cluster_t": None}) is not None
        assert ab.upper_bound({"delta": 0.003, "se": 0.001, "upper_95_cluster_t": None}) == pytest.approx(0.005)

    def test_an_unassessable_unit_blocks(self):
        assert "unassessable" in ab.regression({"delta": None, "se": None, "upper_95_cluster_t": None})


class TestDecision:
    V = {"refit_frozen": {"ship_strict": True}, "refit": {"ship_strict": False}}

    def test_ships_only_on_primary_pass_and_no_regression(self):
        d = ab.decide(self.V, {"refit_frozen": {}}, {})
        assert d["ship"] and d["arm"] == "refit_frozen"

    def test_a_regression_blocks(self):
        d = ab.decide(self.V, {"refit_frozen": {"total_goals": "dBrier +0.003 ..."}}, {})
        assert not d["ship"] and "total_goals" in d["reason"] and d["primary_pass"]

    def test_a_primary_miss_blocks(self):
        d = ab.decide({"refit_frozen": {"ship_strict": False}}, {"refit_frozen": {}}, {})
        assert not d["ship"] and "fails ship_strict" in d["reason"]

    def test_refit_never_decides_and_disqualified_frozen_cannot_ship(self):
        d = ab.decide({"refit": {"ship_strict": True}}, {"refit": {}}, {"refit_frozen": {"reason": "no converge"}})
        assert not d["ship"] and "disqualified: no converge" in d["reason"]


class TestArms:
    def test_pinned_is_exactly_the_trainers_config_and_refit_the_soccer_one(self):
        from predictors.model_config import DIXON_COLES_CONFIG, LEGACY_DIXON_COLES_HYPERPARAMETERS

        c = ab.arm_configs()
        assert c["pinned"] == ab.trainer.nhl_config()
        assert c["pinned"].name == c["refit"].name == "dixon_coles_nhl"
        assert c["pinned"].hyperparameters == {
            **DIXON_COLES_CONFIG.hyperparameters,
            **LEGACY_DIXON_COLES_HYPERPARAMETERS,
        }
        assert c["refit"].hyperparameters == DIXON_COLES_CONFIG.hyperparameters

    def test_decayed_arms_meet_the_decayed_constant(self):
        assert ab.matched_constant("refit") == ab.matched_constant("refit_frozen") == "base_season_dw"
        assert ab.matched_constant("served") == ab.matched_constant("pinned") == "base_season"

    def test_fit_arm_forces_rho_to_zero_and_round_trips(self, tmp_path):
        frame = nhl_frame(pd.Timestamp("2024-10-01", tz="UTC"), pd.Timestamp("2025-01-31", tz="UTC"), seed=3)
        model, result, path, refusal = ab.fit_arm(frame, ab.arm_configs()["pinned"], str(tmp_path), "p")
        assert refusal is None and model.rho == 0.0 and result["rho"] == 0.0
        assert json.loads(Path(path).read_text())["rho"] == 0.0


class TestAssembleFrame:
    def test_join_and_audit(self):
        frame = pd.DataFrame({"match_id": ["a", "b"], "home_score": [3, 2], "away_score": [2, 1]})
        reg = pd.DataFrame({"match_id": ["a", "b", "c"], "regulation_winner": ["tie", None, "home"]})
        out, audit = ab.assemble_frame(frame, reg)
        assert out["regulation_winner"].tolist()[0] == "tie"
        assert audit == {"rows": 2, "without_regulation_winner": 1, "regulation_labels": {"tie": 1}}

    def test_a_frame_row_missing_from_the_regulation_query_raises(self):
        frame = pd.DataFrame({"match_id": ["a", "zz"]})
        reg = pd.DataFrame({"match_id": ["a"], "regulation_winner": ["home"]})
        with pytest.raises(RuntimeError, match="the two queries disagree"):
            ab.assemble_frame(frame, reg)


# ── run() end to end on a synthetic frame ────────────────────────────


def nhl_frame(start, end, seed=0, per_day=3, thin=None):
    """A synthetic NHL frame shaped like NHL_DC_TRAINING_QUERY's plus
    regulation_winner. No games July-September; ``thin`` maps month -> games."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(12)]
    attack = {t: rng.uniform(0.85, 1.15) for t in teams}
    rows = []
    mid = 0
    for d in pd.date_range(start, end, freq="D"):
        if d.month in (7, 8, 9):
            continue
        n = per_day
        if thin and d.month in thin:
            n = 1 if d.day <= thin[d.month] else 0
        for _ in range(n):
            h, a = rng.choice(teams, 2, replace=False)
            hs, as_ = int(rng.poisson(3.2 * attack[h])), int(rng.poisson(2.9 * attack[a]))
            if hs == as_:
                reg = "tie"
                if rng.random() < 0.5:
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
                    regulation_winner=reg if rng.random() > 0.02 else None,
                )
            )
            mid += 1
    return pd.DataFrame(rows)


class TestSyntheticRun:
    """run() on a synthetic frame, no DB: the bookkeeping that makes the paired
    comparisons honest (pooled playoff tails, disqualified arms dropped
    everywhere, the frozen arm equal to the first fold's refit), valid JSON,
    and the pre-registered decision line."""

    @staticmethod
    def _setup(monkeypatch, tmp_path):
        frame = nhl_frame(
            pd.Timestamp("2023-10-01", tz="UTC"), pd.Timestamp("2025-02-28", tz="UTC"), thin={5: 20, 6: 4}
        )
        reg = frame[["match_id", "regulation_winner"]]
        monkeypatch.setattr(
            ab, "load_frame", lambda url: ab.assemble_frame(frame.drop(columns="regulation_winner"), reg)
        )
        from predictors.poisson_models import DixonColesPredictor

        served = DixonColesPredictor(ab.arm_configs()["pinned"])
        served.train(frame[frame["match_date"] < "2024-03-01"][ab.FIT_COLS])
        served.rho = 0.0
        path = tmp_path / "served.bin"
        served.save(str(path))
        return ab.parse_args(["--start", "2024-04-01", "--served-artifact", str(path), "--database-url", "x"])

    def test_bookkeeping_and_decision_line(self, monkeypatch, tmp_path):
        r = ab.run(self._setup(monkeypatch, tmp_path))
        json.dumps(ab.ht.clean_json(r), allow_nan=False)
        assert r["folds"] == ["2024-04", "2024-10", "2024-11", "2024-12", "2025-01", "2025-02"]
        assert [p["fold"] for p in r["pooled_folds"]] == ["2024-05", "2024-06"]
        assert [s["fold"] for s in r["skipped_folds"]] == ["2024-07", "2024-08", "2024-09"]
        assert r["disqualified"] == {}
        assert set(r["verdicts_vs_served"]) == {"pinned", "refit", "refit_frozen"}
        first = r["per_fold_vs_served"]
        assert first["refit_frozen"][0]["delta"] == pytest.approx(first["refit"][0]["delta"], abs=1e-12)
        assert set(r["brier"]["served"]) == set(ab.UNIT_NAMES)
        # regulation units skip the rows without regulation_winner; the rest grade every row.
        assert r["n_gradable"]["regulation_1x2"] < r["n_gradable"]["over_5.5"] == r["n_eval"]
        assert r["decision"]["arm"] == "refit_frozen"
        assert set(r["regressions"]) == {"pinned", "refit", "refit_frozen"}
        assert {"2023-24", "2024-25"} == set(r["by_season"])
        text = ab.format_report(r)
        assert "DECISION (pre-registered: refit_frozen must pass ship_strict vs served on over_5.5" in text

    def test_every_challenger_disqualified_still_reports(self, monkeypatch, tmp_path):
        args = self._setup(monkeypatch, tmp_path)
        orig = ab.fit_arm

        def refused(fit_df, cfg, tmp, tag):
            model, res, path, _ = orig(fit_df, cfg, tmp, tag)
            return model, res, path, "the fit did not converge"

        monkeypatch.setattr(ab, "fit_arm", refused)
        r = ab.run(args)
        assert r["verdicts_vs_served"] == {} and set(r["disqualified"]) == {"pinned", "refit", "refit_frozen"}
        assert not r["decision"]["ship"] and "disqualified" in r["decision"]["reason"]
        text = ab.format_report(r)
        assert "DO-NOT-SHIP" in text and "no challenger survived" in text
        json.dumps(ab.ht.clean_json(r), allow_nan=False)

    def test_the_default_served_path_must_be_the_legacy_fit(self, monkeypatch, tmp_path):
        args = self._setup(monkeypatch, tmp_path)
        monkeypatch.setattr(ab, "SERVED_ARTIFACT", args.served_artifact)  # 12 teams, not the legacy 33
        with pytest.raises(RuntimeError, match="not the 2026-06-04 legacy fit"):
            ab.run(args)
