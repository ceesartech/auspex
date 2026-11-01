"""Unit tests for the full-time soccer Dixon-Coles (derived markets) A/B
harness: the pieces that decide whether the experiment is honest. That covers
the unit catalogue against the served derivation, the graded truth against
grading_outcomes, the O/U-2.5 anchor replay against rec_gating, the served
matrix against derive_from_lambdas, the per-league constants, the fold scorer,
the report's signs and the pre-registered decision rule."""

from __future__ import annotations

import importlib.util
import itertools
import re
import sys
import zlib
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


ab = _load("ab_soccer_ft_dixon_coles", "ab_soccer_ft_dixon_coles.py")

from predictors import market_derivation as md  # noqa: E402


class _FakeDC:
    """A served Dixon-Coles stand-in: fixed lambdas per (home, away)."""

    def __init__(self, lambdas, rho=-0.05, teams=("A", "B", "C", "D")):
        self.lambdas = dict(lambdas)
        self.rho = rho
        self.team_attack = {t: 1.0 for t in teams}

    def lambdas_for_match(self, home, away, league=None):
        return self.lambdas[(home, away)]


def _random_matrices(n=40, seed=0, anchored=False):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        h, a = rng.uniform(0.4, 2.8), rng.uniform(0.3, 2.2)
        tgt = rng.dirichlet([4, 2.5, 3])
        p_over = float(rng.uniform(0.3, 0.7)) if anchored else None
        out.append((h, a, -0.06, tuple(tgt), p_over))
    return out


# ── Units ────────────────────────────────────────────────────────────


class TestUnitCatalogue:
    def test_every_unit_has_exactly_one_role(self):
        names = list(ab.UNIT_NAMES)
        assert len(names) == len(set(names))
        roles = [
            sum(
                [
                    u == ab.PRIMARY,
                    u in ab.BLOCKING_UNITS,
                    u in ab.INVARIANT_UNITS,
                    u in ab.CONTEXT_UNITS,
                ]
            )
            for u in names
        ]
        assert roles == [1] * len(names)
        assert ab.PRIMARY == "over_2.5" and ab.UNIT_BY_NAME[ab.PRIMARY].keys == ("over_2.5",)
        assert ab.DECIDING_ARM == "refit" and ab.DECIDING_REGIMES == ("live", "proxy")

    def test_line_sets_match_the_served_derivation(self):
        assert ab.OU_LINES == md.OU_LINES
        assert ab.AH_LINES == md.AH_LINES
        assert ab.TEAM_TOTAL_LINES == md.TEAM_TOTAL_LINES
        assert ab.GRID_SIDE == md.MAX_GOALS_DERIVE + 1

    def test_every_served_market_type_is_scored(self):
        served = set(md.derive_soccer_markets(md.build_dc_matrix(1.4, 1.1, -0.05))) - {"1x2"}
        assert served == {u.market for u in ab.UNITS}

    def test_unit_probabilities_equal_derive_soccer_markets(self):
        weights = ab.unit_weights()
        for anchored in (False, True):
            mats = [ab.served_matrix(h, a, rho, t, p) for h, a, rho, t, p in _random_matrices(25, 1, anchored)]
            probs = ab.unit_probabilities(np.stack(mats), weights)
            for r, P in enumerate(mats):
                mk = md.derive_soccer_markets(P)
                for unit in ab.UNITS:
                    if unit.kind == "grid":
                        assert probs[unit.name][r] == pytest.approx(P.ravel(), abs=1e-15)
                    elif unit.kind == "listed":
                        keys = [f"{i // 11}-{i % 11}" for i in probs["_listed_order"][r]] + ["other"]
                        assert list(mk["correct_score"]) == keys
                        assert probs[unit.name][r] == pytest.approx([mk["correct_score"][k] for k in keys], abs=1e-15)
                    else:
                        ref = [mk[unit.market][k] for k in unit.keys]
                        assert probs[unit.name][r] == pytest.approx(ref, abs=1e-14), unit.name

    def test_invariant_units_are_functions_of_the_1x2_alone(self):
        weights = ab.unit_weights()
        tgt = (0.47, 0.27, 0.26)
        mats = np.stack(
            [
                ab.served_matrix(h, a, rho, tgt, None)
                for h, a, rho in ((0.7, 0.6, -0.1), (2.6, 2.1, 0.0), (1.3, 1.3, 0.1))
            ]
        )
        probs = ab.unit_probabilities(mats, weights)
        for u in ab.INVARIANT_UNITS:
            assert np.ptp(probs[u], axis=0).max() < 1e-12, u
        assert np.ptp(probs[ab.PRIMARY], axis=0).max() > 0.2  # the primary is NOT pinned by the 1x2


# ── Truth ────────────────────────────────────────────────────────────


class TestTruth:
    def test_vectorised_truth_equals_grading_outcomes(self):
        pairs = list(itertools.product(range(9), range(9))) + [(12, 0), (0, 11)]
        h = np.array([p[0] for p in pairs])
        a = np.array([p[1] for p in pairs])
        vec = ab.vector_truth(h, a)
        graded = ab.graded_truth(h, a)
        assert set(graded) == {u.name for u in ab.UNITS if u.kind not in ("grid", "listed")}
        for u, arr in graded.items():
            assert np.array_equal(np.isnan(arr), np.isnan(vec[u])), u
            assert np.array_equal(np.nan_to_num(arr), np.nan_to_num(vec[u])), u

    def test_handicap_settles_quarter_lines_by_halves(self):
        assert ab.handicap_truth(1, -0.75) == (0.5, 0.0, 0.5)  # home -0.75 wins by one: half won
        assert ab.handicap_truth(0, -0.25) == (0.0, 0.5, 0.5)  # home -0.25 draws: half lost
        assert ab.handicap_truth(-1, 1.0) == (0.0, 0.0, 1.0)  # whole line push
        assert ab.handicap_truth(2, -1.5) == (1.0, 0.0, 0.0)

    def test_draw_no_bet_does_not_grade_a_draw_and_the_grid_caps_each_side(self):
        t = ab.vector_truth(np.array([1, 2, 13]), np.array([1, 0, 2]))
        assert np.isnan(t["draw_no_bet"][0]).all() and t["draw_no_bet"][1].tolist() == [1.0, 0.0]
        assert t[ab.GRID][2].argmax() == 10 * 11 + 2 and t[ab.GRID].sum(axis=1).tolist() == [1.0, 1.0, 1.0]

    def test_listed_truth_is_the_realised_key_or_other(self):
        order = np.array([[0 * 11 + 0, 1 * 11 + 0] + list(range(20, 30))] * 3)
        y = ab.listed_truth(order, np.array([1, 3, 11]), np.array([0, 3, 0]))
        assert y[0, 1] == 1.0 and y[0].sum() == 1.0 and y[0, 12] == 0.0
        assert y[1, 12] == 1.0  # 3-3 not listed
        assert y[2, 12] == 1.0  # a side above the grid is never listed

    def test_brier_rows_sums_keys_and_keeps_nan(self):
        p = np.array([[0.2, 0.8], [0.5, 0.5]])
        y = np.array([[0.0, 1.0], [np.nan, np.nan]])
        out = ab.brier_rows(p, y)
        assert out[0] == pytest.approx(0.08) and np.isnan(out[1])


# ── Anchor ───────────────────────────────────────────────────────────


KO = pd.Timestamp("2026-10-20 15:00", tz="UTC")


def _row(book, sel, price, seen_hours_before=1.0, ts_hours_before=None, line=2.5, live=False, last_seen=True):
    ts = KO - pd.Timedelta(hours=ts_hours_before if ts_hours_before is not None else seen_hours_before + 1)
    return {
        "bookmaker": book,
        "selection": sel,
        "line": line,
        "odds_decimal": price,
        "ts": ts,
        "last_seen_at": (KO - pd.Timedelta(hours=seen_hours_before)) if last_seen else None,
        "is_live": live,
    }


def _book(name, over, under, **kw):
    return [_row(name, "over", over, **kw), _row(name, "under", under, **kw)]


class TestAnchor:
    def test_freshness_aggregates_live_and_line_filters(self):
        rows = (
            _book("BetMGM", 1.9, 1.95, seen_hours_before=5.9)
            + _book("Bovada", 1.8, 2.0, seen_hours_before=6.1)  # stale
            + _book("Average", 1.85, 1.95)  # aggregate
            + _book("LowVig.ag", 1.9, 1.9, live=True)
            + _book("BetUS", 1.9, 1.9, line=3.5)
        )
        kept = ab.consensus_rows(rows, KO)
        assert sorted({r["bookmaker"] for r in kept}) == ["BetMGM"]

    def test_last_seen_null_falls_back_to_timestamp_and_newest_wins(self):
        rows = [
            _row("BetMGM", "over", 1.80, ts_hours_before=3.0, last_seen=False),
            _row("BetMGM", "over", 1.90, ts_hours_before=2.0, last_seen=False),
            _row("BetMGM", "under", 2.0, ts_hours_before=8.0, last_seen=False),  # stale on its timestamp
        ]
        kept = ab.consensus_rows(rows, KO)
        assert [(r["selection"], r["odds_decimal"]) for r in kept] == [("over", 1.90)]

    def test_live_anchor_is_market_consensus_prob_on_three_books(self):
        from rec_gating import devig_power

        books = [("BetMGM", 1.9, 1.95), ("BetRivers", 1.85, 2.0), ("Bovada", 2.0, 1.83)]
        rows = [r for b in books for r in _book(*b)]
        want = np.mean([devig_power({"over": o, "under": u})["over"] for _, o, u in books])
        assert ab.live_anchor(rows, KO) == pytest.approx(want, abs=1e-15)
        assert ab.live_anchor(rows[:4], KO) is None  # two books: below MIN_CONSENSUS_BOOKS

    def test_proxy_prefers_average_then_bet365(self):
        from rec_gating import devig_power

        rows = _book("Bet365", 1.9, 1.9) + _book("Average", 1.8, 2.05)
        assert ab.proxy_anchor(rows) == pytest.approx(devig_power({"over": 1.8, "under": 2.05})["over"])
        only365 = _book("Bet365", 1.7, 2.1) + [_row("Average", "over", 1.8)]
        assert ab.proxy_anchor(only365) == pytest.approx(devig_power({"over": 1.7, "under": 2.1})["over"])
        assert ab.proxy_anchor(_book("BetMGM", 1.9, 1.9)) is None


# ── Matrices ─────────────────────────────────────────────────────────


class TestServedMatrix:
    def test_markets_equal_derive_from_lambdas(self):
        for anchored in (False, True):
            for h, a, rho, tgt, p in _random_matrices(10, 3, anchored):
                mine = md.derive_soccer_markets(ab.served_matrix(h, a, rho, tgt, p))
                ref = md.derive_from_lambdas(h, a, rho, target_1x2=tgt, target_over_25=p)
                for mt, sels in ref.items():
                    for k, v in sels.items():
                        assert mine[mt][k] == pytest.approx(v, abs=1e-15)

    def test_regimes_anchor_only_where_their_anchor_exists(self):
        lams = np.array([[1.4, 1.0], [1.1, 1.3], [2.0, 0.7]])
        tg = np.array([[0.45, 0.28, 0.27], [0.35, 0.3, 0.35], [0.6, 0.22, 0.18]])
        a_live = np.array([0.55, np.nan, np.nan])
        a_proxy = np.array([0.55, 0.48, np.nan])
        mats = ab.regime_matrices(lams, -0.05, tg, a_live, a_proxy)
        over = {r: ab.unit_probabilities(m, ab.unit_weights())[ab.PRIMARY][:, 0] for r, m in mats.items()}
        assert over["live"][0] == pytest.approx(0.55, abs=1e-9) and over["proxy"][1] == pytest.approx(0.48, abs=1e-9)
        assert over["live"][1] == over["none"][1] and over["proxy"][2] == over["none"][2]
        assert over["none"][0] != pytest.approx(0.55, abs=1e-6)


# ── Constants ────────────────────────────────────────────────────────


class TestConstants:
    def test_binary_column_equals_ft_fit_base_rates(self):
        ft = ab.ft
        rng = np.random.default_rng(0)
        n = 500
        leagues = rng.choice(["L1", "L2", "L3"], size=n)
        y = (rng.random(n) < np.where(leagues == "L1", 0.6, 0.45)).astype(float)
        w = rng.uniform(0.2, 1.0, size=n)
        codes = pd.Series(leagues).map({"L1": 0, "L2": 1, "L3": 2}).to_numpy()
        g, by = ab.fit_constants(y[:, None], np.ones((n, 1)), codes, 3, w)
        ref = ft.fit_base_rates(pd.DataFrame({"league": leagues, "y": y}), "y", weights=w)
        assert g[0] == pytest.approx(ref.global_rate, abs=1e-12)
        for lg, k in (("L1", 0), ("L2", 1), ("L3", 2)):
            assert by[k, 0] == pytest.approx(ref.by_league[lg], abs=1e-12)

    def test_partitions_sum_to_one_ungradable_rows_drop_out_unknown_leagues_get_global(self):
        Y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        V = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=float)
        g, by = ab.fit_constants(Y, V, np.array([0, 0, 1, 1]), 2, np.ones(4), k=2.0)
        assert g == pytest.approx([1 / 3, 1 / 3, 1 / 3])
        assert by.sum(axis=1) == pytest.approx([1.0, 1.0])
        pred = ab.predict_constants(g, by, np.array([-1, 1]))
        assert pred[0] == pytest.approx(g) and pred[1] == pytest.approx(by[1])


# ── Statistics and the decision rule ─────────────────────────────────


def _pair_of(delta_per_fold, n_per=200, noise=0.0005, seed=0):
    rng = np.random.default_rng(seed)
    folds, ch, base = [], [], []
    for k, d in enumerate(delta_per_fold):
        b = rng.uniform(0.2, 0.3, n_per)
        folds += [f"f{k:02d}"] * n_per
        base.append(b)
        ch.append(b + d + rng.normal(0, noise, n_per))
    return ab.pair(np.concatenate(ch), np.concatenate(base), np.array(folds, dtype=object))


def _units(primary_delta=-0.008, regress=None, n_folds=12):
    per = {}
    for u in ab.UNIT_NAMES:
        d = primary_delta if u == ab.PRIMARY else -0.0005
        if regress and u in regress:
            d = 0.003
        per[u] = _pair_of([d] * n_folds, seed=zlib.crc32(u.encode()))
    return per


class TestDecision:
    def test_pair_sign_is_challenger_minus_baseline(self):
        worse = _pair_of([0.004] * 10)
        assert worse["delta"] > 0 and worse["lower_95_cluster_t"] > 0
        assert not ab.ship_strict(worse)["ship_strict"]

    def test_ship_strict_needs_the_t_bound(self):
        p = dict(_pair_of([-0.008] * 10))
        assert ab.ship_strict(p)["ship_strict"]
        p["upper_95_cluster_t"] = 0.0001
        assert not ab.ship_strict(p)["ship_strict"]

    def test_both_regimes_must_pass(self):
        good = _units()
        assert ab.decide({"live": good, "proxy": good, "none": good}, {})["ship"]
        weak = _units(primary_delta=-0.002)
        d = ab.decide({"live": good, "proxy": weak, "none": good}, {})
        assert not d["ship"] and "[proxy]" in d["reason"]
        d = ab.decide({"live": weak, "proxy": good}, {})
        assert not d["ship"] and "[live]" in d["reason"]

    def test_a_blocking_regression_blocks_but_invariant_and_context_units_never_do(self):
        assert not ab.decide({"live": _units(regress={"btts"}), "proxy": _units()}, {})["ship"]
        assert not ab.decide({"live": _units(), "proxy": _units(regress={"ah_-1"})}, {})["ship"]
        quiet = _units(regress={"double_chance_1X", "ah_0.5", ab.LISTED})
        assert ab.decide({"live": quiet, "proxy": quiet}, {})["ship"]

    def test_unassessable_blocking_unit_blocks(self):
        per = _units()
        per["btts"] = {"delta": None}
        assert not ab.decide({"live": per, "proxy": _units()}, {})["ship"]

    def test_disqualified_or_missing_regime_never_ships(self):
        good = _units()
        assert not ab.decide({"live": good, "proxy": good}, {"refit": {"reason": "x"}})["ship"]
        assert not ab.decide({"live": good}, {})["ship"]

    def test_invariance_check_refuses_a_difference(self):
        base = {u: np.full((3, len(ab.UNIT_BY_NAME[u].keys)), 0.3) for u in ab.INVARIANT_UNITS}
        other = {u: v.copy() for u, v in base.items()}
        assert ab.check_invariance({"served": base, "refit": other}) == 0.0
        other["ah_0"][1, 0] += 1e-6
        with pytest.raises(RuntimeError, match="ah_0"):
            ab.check_invariance({"served": base, "refit": other})


# ── Fold scorer and report ───────────────────────────────────────────


def _fold_inputs(n=60, seed=0):
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D"]
    homes = rng.choice(teams, n)
    aways = np.array([rng.choice([t for t in teams if t != h]) for h in homes])
    ev = pd.DataFrame(
        {
            "home_team": homes,
            "away_team": aways,
            "league_id": "L1",
            "s_home": rng.poisson(1.5, n),
            "s_away": rng.poisson(1.1, n),
        }
    )
    lam = {(h, a): (float(rng.uniform(0.8, 2.2)), float(rng.uniform(0.6, 1.6))) for h in teams for a in teams if h != a}
    tg = rng.dirichlet([4, 2.6, 3], size=n)
    a_live = np.where(rng.random(n) < 0.3, rng.uniform(0.35, 0.65, n), np.nan)
    a_proxy = np.where(np.isfinite(a_live), a_live, np.where(rng.random(n) < 0.5, rng.uniform(0.35, 0.65, n), np.nan))
    truth = ab.vector_truth(ev["s_home"].to_numpy(), ev["s_away"].to_numpy())
    const = {}
    for cname in ab.CONSTANT_ARMS:
        const[cname] = {
            u: (
                np.tile(
                    np.nan_to_num(truth[u]).mean(axis=0) / max(np.nan_to_num(truth[u]).mean(axis=0).sum(), 1e-12),
                    (n, 1),
                )
                if ab.UNIT_BY_NAME[u].kind != "binary"
                else np.tile(truth[u].mean(axis=0), (n, 1))
            )
            for u in ab.CONSTANT_UNITS
        }
    return ev, lam, tg, a_live, a_proxy, truth, const


class TestScoreFold:
    def test_arms_stay_aligned_and_identical_models_score_identically(self):
        ev, lam, tg, a_live, a_proxy, truth, const = _fold_inputs()
        served = _FakeDC(lam)
        out, disq, stats = ab.score_fold(
            "2026-09",
            ev,
            tg,
            a_live,
            a_proxy,
            truth,
            {"served": served, "refit": _FakeDC(lam)},
            const,
            ab.unit_weights(),
        )
        assert disq == {} and stats["derive"] <= ab.DERIVE_ATOL
        lengths = {k: len(v) for k, v in out.items()}
        assert set(lengths.values()) == {len(ev)}
        for regime in ab.REGIMES:
            for u in ab.UNIT_NAMES:
                a, b = out[f"brier::refit::{regime}::{u}"], out[f"brier::served::{regime}::{u}"]
                assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(np.nan_to_num(a), np.nan_to_num(b))
            assert np.isnan(out[f"brier::refit_frozen::{regime}::{ab.PRIMARY}"]).all()
        lv = np.isfinite(a_live)
        assert out[f"p::served::live::{ab.PRIMARY}"][lv] == pytest.approx(a_live[lv], abs=1e-9)

    def test_a_different_shape_moves_shaped_units_only(self):
        ev, lam, tg, a_live, a_proxy, truth, const = _fold_inputs(seed=1)
        other = {k: (v[0] * 1.3, v[1] * 0.8) for k, v in lam.items()}
        out, disq, _ = ab.score_fold(
            "2026-09",
            ev,
            tg,
            a_live,
            a_proxy,
            truth,
            {"served": _FakeDC(lam), "refit": _FakeDC(other)},
            const,
            ab.unit_weights(),
        )
        assert disq == {}
        for u in ab.INVARIANT_UNITS:
            a, b = out[f"brier::refit::none::{u}"], out[f"brier::served::none::{u}"]
            assert np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-12)
        assert not np.allclose(out["brier::refit::none::btts"], out["brier::served::none::btts"])
        lv = np.isfinite(a_live)
        assert np.allclose(out[f"brier::refit::live::{ab.PRIMARY}"][lv], out[f"brier::served::live::{ab.PRIMARY}"][lv])

    def test_a_broken_challenger_is_disqualified_and_a_broken_served_aborts(self):
        ev, lam, tg, a_live, a_proxy, truth, const = _fold_inputs(seed=2)
        broken = _FakeDC({k: (float("nan"), 1.0) for k in lam})
        out, disq, _ = ab.score_fold(
            "2026-09",
            ev,
            tg,
            a_live,
            a_proxy,
            truth,
            {"served": _FakeDC(lam), "refit": broken},
            const,
            ab.unit_weights(),
        )
        assert "refit" in disq and np.isnan(out[f"brier::refit::live::{ab.PRIMARY}"]).all()
        with pytest.raises(RuntimeError, match="served"):
            ab.score_fold("2026-09", ev, tg, a_live, a_proxy, truth, {"served": broken}, const, ab.unit_weights())


def _stacked(refit_shift, n_folds=12, n_per=120, seed=0):
    """Synthetic stacked arrays as run() collects them: refit's Brier on every
    unit is served's plus refit_shift[unit] (default -0.0005)."""
    rng = np.random.default_rng(seed)
    n = n_folds * n_per
    s = {
        "fold": np.repeat([f"2025-{k + 1:02d}" for k in range(n_folds)], n_per).astype(object),
        "match_id": np.arange(n).astype(str).astype(object),
        "s_home": rng.poisson(1.5, n).astype(float),
        "s_away": rng.poisson(1.1, n).astype(float),
        "anchor_live": np.where(rng.random(n) < 0.1, 0.5, np.nan),
        "anchor_proxy": np.where(rng.random(n) < 0.5, 0.5, np.nan),
        "espn_era": (rng.random(n) < 0.2).astype(float),
        "identity_split": (rng.random(n) < 0.05).astype(float),
    }
    for arm in ab.DC_ARMS:
        s[f"known::{arm}"] = rng.integers(0, 3, n).astype(float)
        s[f"lam_h::{arm}"] = np.ones(n)
        for regime in ab.REGIMES:
            s[f"p::{arm}::{regime}::{ab.PRIMARY}"] = np.full(n, 0.5)
            s[f"p::{arm}::{regime}::btts"] = np.full(n, 0.52)
            s[f"etot::{arm}::{regime}"] = np.full(n, 2.6)
    for regime in ab.REGIMES:
        for u in ab.UNIT_NAMES:
            base = rng.uniform(0.15, 0.35, n)
            s[f"brier::served::{regime}::{u}"] = base
            s[f"brier::refit::{regime}::{u}"] = base + refit_shift.get(u, -0.0005) + rng.normal(0, 0.0004, n)
            s[f"brier::refit_frozen::{regime}::{u}"] = base + 0.0001
    for cname in ab.CONSTANT_ARMS:
        for u in ab.CONSTANT_UNITS:
            s[f"brier::{cname}::-::{u}"] = rng.uniform(0.15, 0.35, n)
        s[f"p::{cname}::-::{ab.PRIMARY}"] = np.full(n, 0.52)
        s[f"p::{cname}::-::btts"] = np.full(n, 0.53)
    return s


META = {"disqualified": {}}


class TestBuildReport:
    def test_a_challenger_worse_by_construction_reads_positive_and_does_not_ship(self):
        r = ab.build_report(_stacked({ab.PRIMARY: 0.006}), dict(META))
        p = r["vs_served"]["live"]["refit"][ab.PRIMARY]
        assert p["delta"] > 0 and not r["decision"]["ship"]

    def test_a_challenger_better_by_construction_ships(self):
        r = ab.build_report(_stacked({ab.PRIMARY: -0.008}), dict(META))
        assert r["vs_served"]["live"]["refit"][ab.PRIMARY]["delta"] < -0.005
        assert r["decision"]["ship"], r["decision"]["reason"]

    def test_a_blocking_regression_in_the_report_blocks(self):
        r = ab.build_report(_stacked({ab.PRIMARY: -0.008, "ah_-1.5": 0.004}), dict(META))
        assert not r["decision"]["ship"] and "ah_-1.5" in r["regressions"]["live"]["refit"]

    def test_a_disqualified_refit_never_ships_and_is_not_paired(self):
        s = _stacked({ab.PRIMARY: -0.008})
        for k in [k for k in s if "::refit::" in k or k.endswith("::refit")]:
            s[k] = np.full(len(s["fold"]), np.nan)
        r = ab.build_report(s, {"disqualified": {"refit": {"fold": "2025-03", "reason": "not converged"}}})
        assert not r["decision"]["ship"] and "refit" not in r["vs_served"]["live"]
        text = ab.format_report({**r, "meta": {**r["meta"], **_format_meta()}})
        assert "DO-NOT-SHIP" in text

    def test_format_report_prints_the_decision(self):
        r = ab.build_report(_stacked({ab.PRIMARY: -0.008}), dict(META))
        text = ab.format_report({**r, "meta": {**r["meta"], **_format_meta()}})
        assert re.search(r"DECISION \(pre-registered: refit, live AND proxy\): SHIP", text)


def _format_meta():
    return {
        "served_fingerprint": {"path": "x", "sha256": "0" * 64, "teams": 951, "league_baselines": 32, "rho": -0.05},
        "served_fit_slice": {},
        "ensemble_artifact": "x",
        "seen_ids": 1,
        "seen_cutoff": "x",
        "eval_rows_without_features": 0,
        "eval_rows_unservable": 0,
        "eval_rows_without_settlement": 0,
        "unseen_rows_inside_seen_span": 0,
        "median_filled_values": 0,
        "fidelity_single_row_max_abs": 0.0,
        "fidelity_single_row_sample": 1,
        "loader_lambda_gap": 0.0,
        "derive_fidelity_max_abs": 0.0,
        "invariance_max_abs": 0.0,
        "anchor_max_abs": 0.0,
        "skipped_folds": [],
        "fits": [],
    }


# ── SQL and the live check ───────────────────────────────────────────


def _where(sql):
    return re.sub(r"\s+", " ", sql.split("WHERE", 1)[1]).strip()


class TestSqlAndLiveCheck:
    def test_frame_population_is_the_members_harness_population(self):
        assert _where(ab.FRAME_SQL) == _where(ab.members.MATCHES_SQL)

    def test_live_check_reproduces_stored_rows_and_refuses_a_difference(self, monkeypatch):
        dc = _FakeDC({("A", "B"): (1.5, 1.1), ("C", "D"): (1.2, 1.4)})
        rows = []
        for mid, (hm, aw), anchor in (("m1", ("A", "B"), None), ("m2", ("C", "D"), 0.56)):
            tgt = {"home": 0.44, "draw": 0.28, "away": 0.28}
            lam_h, lam_a = dc.lambdas_for_match(hm, aw)
            mk = md.derive_from_lambdas(lam_h, lam_a, dc.rho, target_1x2=tuple(tgt.values()), target_over_25=anchor)
            meta = {"reconcile": "1x2+ou2.5", "market_over_2.5": anchor} if anchor else {"reconcile": "1x2"}
            rows.append((mid, "match_result", tgt, {}, hm, aw))
            for ptype in {u.market for u in ab.UNITS}:
                rows.append((mid, ptype, dict(mk[ptype]), meta, hm, aw))
        frame = pd.DataFrame(
            rows, columns=["match_id", "prediction_type", "probabilities", "metadata", "home_team", "away_team"]
        )
        frame["league_id"] = "L1"
        monkeypatch.setattr(ab.members, "_read_sql", lambda url, sql, params=None: frame)
        out = ab.live_check("db", "v1", dc)
        assert out["matches"] == 2 and out["anchored"] == 1
        assert out["max_abs"] == 0.0 and out["max_abs_anchored"] == 0.0
        frame.loc[frame.prediction_type == "btts", "probabilities"] = [{"yes": 0.9, "no": 0.1}] * 2
        with pytest.raises(RuntimeError, match="live check"):
            ab.live_check("db", "v1", dc)
