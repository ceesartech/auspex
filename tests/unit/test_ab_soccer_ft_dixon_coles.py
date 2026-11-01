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


def _row(
    book,
    sel,
    price,
    seen_hours_before=1.0,
    ts_hours_before=None,
    line=2.5,
    live=False,
    last_seen=True,
    created_hours_before=None,
    ko=None,
):
    """One OU_ODDS_SQL row; `created` defaults to the price timestamp (a row
    whose price never moved), and every offset counts back from `ko`."""
    ko = KO if ko is None else ko
    ts = ko - pd.Timedelta(hours=ts_hours_before if ts_hours_before is not None else seen_hours_before + 1)
    return {
        "bookmaker": book,
        "selection": sel,
        "line": line,
        "odds_decimal": price,
        "ts": ts,
        "last_seen_at": (ko - pd.Timedelta(hours=seen_hours_before)) if last_seen else None,
        "created": ts if created_hours_before is None else ko - pd.Timedelta(hours=created_hours_before),
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


# ── Settlement and the constant frame ────────────────────────────────


def _frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="12h", tz="UTC")
    return pd.DataFrame(
        {
            "match_date": dates,
            "league_id": rng.choice(["L1", "L2"], n),
            "home_score": rng.poisson(1.5, n),
            "away_score": rng.poisson(1.1, n),
            "regulation_home_score": None,
            "regulation_away_score": None,
        }
    )


class TestSettlementAndConstantFrame:
    def test_full_time_markets_settle_on_the_90_minute_pair(self):
        f = _frame(4)
        f.loc[0, ["home_score", "away_score", "regulation_home_score", "regulation_away_score"]] = [3, 2, "2", "2"]
        f.loc[1, ["home_score", "away_score", "regulation_home_score", "regulation_away_score"]] = [1, 0, "4", "0"]
        h, a, ok = ab.settlement(f)
        assert (h[0], a[0], ok[0]) == (2, 2, True)  # decided in extra time: graded on 2-2
        assert not ok[1]  # a 90-minute pair above the final score is unusable: not graded
        assert ok[2] and (h[2], a[2]) == (f.loc[2, "home_score"], f.loc[2, "away_score"])

    def test_fold_constants_fit_the_prefix_only(self):
        f = _frame()
        f["s_home"], f["s_away"], f["s_ok"] = ab.settlement(f)
        cf = ab.build_constant_frame(f)
        n_fit = 120
        out = ab.fold_constants(cf, n_fit, ["L1", "L2", "L9"], np.ones(n_fit))
        y = ((f["s_home"] + f["s_away"]) > 2.5).astype(float).to_numpy()[:n_fit]
        lg = f["league_id"].to_numpy()[:n_fit]
        g = y.mean()
        l1 = (y[lg == "L1"].sum() + ab.CONSTANT_SHRINKAGE * g) / ((lg == "L1").sum() + ab.CONSTANT_SHRINKAGE)
        p = out["base_league"][ab.PRIMARY][:, 0]
        assert p[0] == pytest.approx(l1, abs=1e-6) and p[2] == pytest.approx(g, abs=1e-6)  # L9 unseen -> global
        assert out["base_league_dw"][ab.GRID].sum(axis=1) == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)

    def test_an_unsorted_frame_is_refused(self):
        f = _frame(10).iloc[::-1].reset_index(drop=True)
        f["s_home"], f["s_away"], f["s_ok"] = ab.settlement(f)
        with pytest.raises(RuntimeError, match="sorted"):
            ab.build_constant_frame(f)


class TestAnchorGuard:
    def test_a_matrix_that_ignores_its_anchor_aborts_the_fold(self, monkeypatch):
        ev, lam, tg, a_live, a_proxy, truth, const = _fold_inputs(seed=3)
        real = ab.served_matrix
        monkeypatch.setattr(ab, "served_matrix", lambda h, a, rho, t, p: real(h, a, rho, t, None))
        monkeypatch.setattr(ab, "derive_fidelity", lambda *args, **kw: 0.0)  # the sampled check would catch it first
        with pytest.raises(RuntimeError, match="misses its anchor"):
            ab.score_fold("2026-09", ev, tg, a_live, a_proxy, truth, {"served": _FakeDC(lam)}, const, ab.unit_weights())


# ── Guards the decision rests on ─────────────────────────────────────


class TestReplayGuards:
    def test_live_anchor_drops_stale_live_aggregate_and_off_line_rows(self):
        from rec_gating import devig_power

        two = _book("BetMGM", 1.9, 1.95) + _book("BetRivers", 1.85, 2.0)
        for third in (
            _book("Bovada", 2.0, 1.83, seen_hours_before=6.5),  # stale
            _book("Bovada", 2.0, 1.83, live=True),
            _book("Average", 2.0, 1.83),  # a football-data aggregate, not a book
            _book("Bovada", 2.0, 1.83, line=3.5),
        ):
            assert ab.live_anchor(two + third, KO) is None
        books = [("BetMGM", 1.9, 1.95), ("BetRivers", 1.85, 2.0), ("Bovada", 2.0, 1.83)]
        want = np.mean([devig_power({"over": o, "under": u})["over"] for _, o, u in books])
        assert ab.live_anchor(two + _book("Bovada", 2.0, 1.83), KO) == pytest.approx(want, abs=1e-15)

    def test_a_backfilled_trio_created_after_kickoff_never_anchors(self):
        trio = [("BetMGM", 1.9, 1.95), ("BetRivers", 1.85, 2.0), ("Bovada", 2.0, 1.83)]
        years_later = -2 * 365 * 24.0  # hours BEFORE kickoff, i.e. two years after it
        backfill = [
            r
            for b in trio
            for r in _book(*b, ts_hours_before=years_later, created_hours_before=years_later, last_seen=False)
        ]
        drops: dict = {}
        assert ab.consensus_rows(backfill, KO, drops=drops) == []
        assert drops == {"created_at_or_after_kickoff": 6}
        assert ab.live_anchor(backfill, KO) is None
        at_kickoff = [r for b in trio for r in _book(*b, created_hours_before=0.0)]
        assert ab.live_anchor(at_kickoff, KO) is None  # created AT kickoff: never read pre-kickoff
        assert ab.live_anchor([r for b in trio for r in _book(*b)], KO) is not None  # the control

    def test_a_row_last_seen_after_kickoff_is_refused_and_counted(self):
        trio = [("BetMGM", 1.9, 1.95), ("BetRivers", 1.85, 2.0), ("Bovada", 2.0, 1.83)]
        rows = [r for b in trio for r in _book(*b, seen_hours_before=-0.5, created_hours_before=5.0)]
        drops: dict = {}
        assert ab.live_anchor(rows, KO, drops=drops) is None
        assert drops == {"seen_after_kickoff": 6}

    def test_the_odds_query_selects_the_creation_clock_and_is_read_pre_kickoff(self, monkeypatch):
        from utils.training_data import PREMATCH_ODDS_SQL

        assert 'COALESCE(o.created_at, o."timestamp") AS created' in ab.OU_ODDS_SQL
        seen = {}

        def fake(url, sql, params=None):
            seen["sql"], seen["params"] = sql, params
            return pd.DataFrame([{"match_id": "m1", **_row("BetMGM", "over", 1.9)}])

        monkeypatch.setattr(ab.members, "_read_sql", fake)
        out = ab.load_ou_rows("db", pd.Timestamp("2026-09-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC"))
        assert PREMATCH_ODDS_SQL in seen["sql"] and set(out) == {"m1"}


class TestValidityChecks:
    def _probs(self, anchored=False):
        rows = _random_matrices(8, 11, anchored)
        lams = np.array([[h, a] for h, a, *_ in rows])
        tg = np.array([t for *_, t, _p in rows])
        anchors = [p for *_, p in rows]
        mats = ab.arm_matrices(lams, -0.06, tg, anchors)
        return lams, tg, anchors, ab.unit_probabilities(mats, ab.unit_weights())

    def test_invalid_probabilities_rejects_rows_that_are_not_probabilities(self):
        _l, _t, _a, probs = self._probs()
        assert ab.invalid_probabilities(probs, ab.UNIT_NAMES) is None
        for unit, how in (
            ("winning_margin", lambda v: v * 0.9),  # a partition that sums to 0.9
            ("ah_-1.5", lambda v: v + np.array([0.05, 0.0, 0.0])),  # a handicap that sums to 1.05
            (ab.GRID, lambda v: v * 1.01),
            ("btts", lambda v: v + 1.0),  # a binary above 1
            ("over_3.5", lambda v: v * np.nan),
        ):
            bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in probs.items()}
            bad[unit][0] = how(bad[unit][0])
            assert unit in (ab.invalid_probabilities(bad, ab.UNIT_NAMES) or ""), unit

    def test_derive_fidelity_refuses_a_readout_off_by_more_than_the_tolerance(self):
        for anchored in (False, True):
            lams, tg, anchors, probs = self._probs(anchored)
            rows = list(range(len(lams)))
            assert ab.derive_fidelity(lams, -0.06, tg, anchors, probs, rows) <= ab.DERIVE_ATOL
            for unit in ("btts", "ah_1.75", "total_goals"):
                bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in probs.items()}
                bad[unit][3, 0] += 1e-9
                with pytest.raises(RuntimeError, match=unit):
                    ab.derive_fidelity(lams, -0.06, tg, anchors, bad, rows)


class TestFrozenFitAndVersion:
    def test_a_frozen_fit_reaching_the_first_cutoff_is_refused(self):
        folds = ab.ft.build_folds("2024-04-07", "2024-06-30")
        assert folds[0].fit_cutoff == pd.Timestamp("2024-04-01", tz="UTC")
        dates = pd.date_range("2024-01-01", "2024-03-31 23:00", freq="7h", tz="UTC")
        ab.check_frozen_fit(pd.DataFrame({"match_date": dates}), folds)
        leaky = pd.DataFrame({"match_date": dates.append(pd.DatetimeIndex([folds[0].fit_cutoff]))})
        with pytest.raises(RuntimeError, match="refit_frozen"):
            ab.check_frozen_fit(leaky, folds)

    def test_served_model_version_is_the_name_precompute_stores(self, tmp_path):
        import os

        meta = tmp_path / "ensemble_soccer_match_result" / "1.0.0" / "model.bin"
        meta.parent.mkdir(parents=True)
        meta.write_text("{}")
        os.utime(meta, (1793505994, 1793505994))
        assert ab.served_model_version(meta) == "ensemble_soccer_match_result_1.0.0+1793505994"


# ── run() end to end ─────────────────────────────────────────────────


class _Learned:
    """A fitted learned member: a fixed function of one serve-row feature."""

    is_fitted = True
    feature_names = ["feature__f1"]

    def __init__(self, scale):
        self.scale = scale

    def predict_proba(self, X):
        f = pd.to_numeric(X["feature__f1"]).to_numpy(dtype=float)
        raw = np.stack([np.exp(self.scale * f), np.ones_like(f), np.exp(-self.scale * f)], axis=1)
        return raw / raw.sum(axis=1, keepdims=True)


ENSEMBLE_WEIGHTS = {
    "xgboost_soccer_match_result": 0.4484,
    "lightgbm_soccer_match_result": 0.1051,
    "neural_network_soccer_match_result": 0.3545,
    "poisson_soccer_match_result": 0.046,
    "dixon_coles_soccer_match_result": 0.046,
}


def _run_corpus(seed=7):
    """Two matches a day, 2023-01-01 .. 2024-06-30, eight teams of fixed
    strength, one league, with every column FRAME_SQL selects."""
    rng = np.random.default_rng(seed)
    strength = rng.normal(0.0, 0.3, 8)
    dates = pd.date_range("2023-01-01 15:00", "2024-06-30 15:00", freq="12h", tz="UTC")
    rows = []
    for i, d in enumerate(dates):
        h, a = rng.choice(8, 2, replace=False)
        rows.append(
            {
                "match_id": f"m{i:05d}",
                "match_date": d,
                "league_id": "L1",
                "league_name": "League One",
                "home_team_id": f"t{h}",
                "away_team_id": f"t{a}",
                "home_team": f"Team {h}",
                "away_team": f"Team {a}",
                "home_score": int(rng.poisson(1.45 * np.exp(strength[h] - strength[a]))),
                "away_score": int(rng.poisson(1.15 * np.exp(strength[a] - strength[h]))),
                "regulation_home_score": None,
                "regulation_away_score": None,
            }
        )
    frame = pd.DataFrame(rows)
    frame["outcome"] = np.where(
        frame["home_score"] > frame["away_score"], 0, np.where(frame["home_score"] == frame["away_score"], 1, 2)
    )
    return frame


@pytest.fixture(scope="module")
def run_world(tmp_path_factory):
    """The synthetic corpus, the incumbent ensemble (its two scoreline members
    fitted on the oldest 70% of its seen rows, as the trainer does) and the
    path of the served Dixon-Coles' saved artifact. Fitted once per module."""
    from predictors.ensemble import EnsemblePredictor
    from predictors.model_config import ENSEMBLE_CONFIG

    tmp = tmp_path_factory.mktemp("ftdc_registry")
    frame = _run_corpus()
    seen_rows = frame[frame["match_date"] < pd.Timestamp("2024-03-16", tz="UTC")]
    ens = EnsemblePredictor(ENSEMBLE_CONFIG)
    for m, scale in zip(ab.members.LEARNED_MEMBERS, (0.3, 0.5, 0.8)):
        ens.add_model(m, _Learned(scale), weight=ENSEMBLE_WEIGHTS[m])
    for m in ab.members.SCORELINE_MEMBERS:
        model, _ = ab.members.fit_member(m, seen_rows.head(int(0.7 * len(seen_rows))), str(tmp), "incumbent")
        ens.add_model(m, model, weight=ENSEMBLE_WEIGHTS[m])
    dc_path = tmp / f"{ab.SERVED_DC}_incumbent.bin"
    assert dc_path.exists()
    return frame, set(seen_rows["match_id"]), ens, str(dc_path)


def _replay_odds(frame):
    """O/U 2.5 rows for five eval matches, each built to exercise one rule.
    Returns (rows DataFrame, {role: match_id}, expected anchors)."""
    from rec_gating import devig_power

    late = frame[frame["match_date"] >= pd.Timestamp("2024-05-01", tz="UTC")].reset_index(drop=True)
    roles = {name: late.loc[k, "match_id"] for k, name in enumerate(("good", "extreme", "mixed", "backfill", "after"))}
    ko = dict(zip(late["match_id"], late["match_date"]))
    trio = [("BetMGM", 1.9, 1.95), ("BetRivers", 1.85, 2.0), ("Bovada", 2.0, 1.83)]
    later = -2 * 365 * 24.0
    rows = []

    def add(role, recs):
        rows.extend({"match_id": roles[role], **r} for r in recs)

    add("good", [r for b in trio for r in _book(*b, ko=ko[roles["good"]])])
    add(
        "extreme", [r for b in ("BetMGM", "BetRivers", "Bovada") for r in _book(b, 1.02, 15.0, ko=ko[roles["extreme"]])]
    )
    k = ko[roles["mixed"]]
    add(
        "mixed",
        _book("BetMGM", 1.9, 1.95, ko=k)
        + _book("BetRivers", 1.85, 2.0, ko=k)
        + _book("Bovada", 2.0, 1.83, seen_hours_before=7.0, ko=k)
        + _book("FanDuel", 2.0, 1.83, live=True, ko=k)
        + _book("Average", 1.8, 2.05, ko=k),
    )
    k = ko[roles["backfill"]]
    late_kw = dict(ts_hours_before=later, created_hours_before=later, last_seen=False, ko=k)
    add(
        "backfill",
        [r for b in trio for r in _book(*b, **late_kw)]
        + _book("Average", 1.75, 2.1, **late_kw)
        + _book("Bet365", 1.7, 2.2, **late_kw),
    )
    k = ko[roles["after"]]
    add("after", [r for b in trio for r in _book(*b, seen_hours_before=-0.5, created_hours_before=5.0, ko=k)])
    want_live = float(np.mean([devig_power({"over": o, "under": u})["over"] for _, o, u in trio]))
    want = {
        "live": {roles["good"]: want_live},
        "proxy": {
            roles["good"]: want_live,
            roles["mixed"]: devig_power({"over": 1.8, "under": 2.05})["over"],
            roles["backfill"]: devig_power({"over": 1.75, "under": 2.1})["over"],
        },
    }
    return pd.DataFrame(rows), roles, want


class TestRunEndToEnd:
    """run() on a synthetic corpus with only the database reads and the
    registry faked: the serve rows, the anchor replay, the refits and their
    refusal rule, the fold guards, the scoring, the report and the per-match
    CSV are the real code."""

    def _setup(self, run_world, tmp_path, monkeypatch):
        frame, seen, ens, dc_path = run_world
        late = frame[frame["match_date"] >= pd.Timestamp("2024-03-01", tz="UTC")]
        rng = np.random.default_rng(1)
        features = {mid: {"f1": float(rng.normal())} for mid in late["match_id"]}
        no_features = late["match_id"].iloc[-5]
        del features[no_features]
        unservable = late["match_id"].iloc[-9]
        features[unservable] = {"f2": 1.0}  # no f1 at all: production skips the match
        odds, roles, want = _replay_odds(frame)
        calls = []

        def read_sql(url, sql, params=None):
            calls.append((sql, params))
            if "o.market_type = 'over_under'" in sql:
                return odds.copy()
            raise AssertionError(f"unexpected query: {sql[:80]}")

        served = ens.models[ab.SERVED_DC]
        fingerprint = {"path": dc_path, "sha256": "f" * 64, "teams": 8, "league_baselines": 1, "rho": served.rho}
        monkeypatch.setattr(ab, "load_frame", lambda url: frame.copy())
        monkeypatch.setattr(ab.members, "load_ensemble", lambda path: (ens, "fake-ensemble"))
        monkeypatch.setattr(ab.members, "member_fingerprint", lambda path, m: dict(fingerprint))
        monkeypatch.setattr(ab.members, "served_fit_slice", lambda *a: {"checked": False, "reason": "synthetic"})
        monkeypatch.setattr(ab.members, "load_seen_ids", lambda path: set(seen))
        monkeypatch.setattr(ab.members, "load_features", lambda url, start, end: dict(features))
        monkeypatch.setattr(ab.members, "load_medians", lambda url: {"f1": 0.0})
        monkeypatch.setattr(ab.members, "_read_sql", read_sql)
        args = ab.parse_args(
            ["--database-url", "postgresql:///unused", "--merge-map", "", "--out-dir", str(tmp_path / "out")]
        )
        return frame, calls, roles, want, {"no_features": no_features, "unservable": unservable}, args

    def test_runs_and_replays_the_anchor_on_pre_kickoff_rows_only(self, run_world, tmp_path, monkeypatch):
        from utils.training_data import PREMATCH_ODDS_SQL

        frame, calls, roles, want, removed, args = self._setup(run_world, tmp_path, monkeypatch)
        r = ab.run(args)
        m = r["meta"]
        assert r["folds"] == ["2024-04", "2024-05", "2024-06"]
        assert [f["fold"] for f in m["skipped_folds"]] == ["2024-03"]  # 32 rows < MIN_FOLD_EVAL
        # the odds are read once, through prematch_odds, over the whole eval span
        assert len(calls) == 1 and PREMATCH_ODDS_SQL in calls[0][0]
        assert calls[0][1] == {
            "start": pd.Timestamp("2024-03-16", tz="UTC").to_pydatetime(),
            "end": pd.Timestamp("2024-07-01", tz="UTC").to_pydatetime(),
        }
        # rows the serve path could not predict are excluded and counted
        assert m["eval_rows_unservable"] == 1 and m["eval_rows_without_features"] == 1
        scored = frame[frame["match_date"] >= pd.Timestamp("2024-04-01", tz="UTC")]
        df = pd.read_csv(r["outputs"]["per_match_csv"])
        assert r["n_eval"] == len(df) == len(scored) - 2
        assert not set(removed.values()) & set(df["match_id"])
        # the replay: one live anchor; the extreme consensus, the stale / live / aggregate mix, the
        # post-kickoff backfill and the rows seen after kickoff never anchor live
        for regime in ("live", "proxy"):
            got = dict(zip(df["match_id"], df[f"anchor_{regime}"]))
            anchored = {k: v for k, v in got.items() if np.isfinite(v)}
            assert set(anchored) == set(want[regime]), regime
            for mid, v in want[regime].items():
                assert anchored[mid] == pytest.approx(v, abs=1e-12)
        assert m["live_replay"] == {
            "rows_dropped": {
                "live": 2,
                "aggregate": 4,
                "other_line": 0,
                "created_at_or_after_kickoff": 8,
                "seen_after_kickoff": 6,
                "stale": 2,
            },
            "matches_with_rows_after_kickoff": 2,
            "consensus_outside_valid_range": 1,
        }
        # no fit reaches the matches it is scored on, the frozen one included
        frozen = [f for f in m["fits"] if f["arm"] == "refit_frozen"]
        assert len(frozen) == 1 and pd.Timestamp(frozen[0]["fit_newest"]) < pd.Timestamp("2024-03-01", tz="UTC")
        for fit in m["fits"]:
            if fit["arm"] == "refit":
                assert pd.Timestamp(fit["fit_newest"]) < pd.Timestamp(f"{fit['fold']}-01", tz="UTC")
        assert m["disqualified"] == {} and r["decision"]["arm"] == "refit"
        assert m["loader_lambda_gap"] == 0.0 and m["derive_fidelity_max_abs"] <= ab.DERIVE_ATOL
        d = (df["p::refit::live::over_2.5"] - df["p::served::live::over_2.5"]).abs()
        assert d[df["match_id"] == roles["good"]].max() < 1e-9  # an anchored over_2.5 is the market's number
        text = ab.format_report(r)
        assert "live replay refused rows" in text and "DECISION (pre-registered" in text

    def test_a_refit_the_trainer_would_refuse_is_disqualified_and_never_ships(self, run_world, tmp_path, monkeypatch):
        *_, args = self._setup(run_world, tmp_path, monkeypatch)
        real = ab.htrainer.fit_refusal
        calls = {"n": 0}

        def refuse_first(result, model, fixtures, **kw):
            calls["n"] += 1  # per fold: the refit is checked first, then refit_frozen
            return "the fit did not converge" if calls["n"] == 1 else real(result, model, fixtures, **kw)

        monkeypatch.setattr(ab.htrainer, "fit_refusal", refuse_first)
        r = ab.run(args)
        assert set(r["meta"]["disqualified"]) == {"refit"}
        assert r["meta"]["disqualified"]["refit"]["fold"] == "2024-04"
        assert not r["decision"]["ship"] and r["decision"]["reason"].startswith("disqualified")
        assert "refit" not in r["vs_served"]["live"] and "refit_frozen" in r["vs_served"]["live"]
        assert "DO-NOT-SHIP" in ab.format_report(r)
