"""Analytic betting-market derivation from a joint scoreline distribution.

A single ``(N x N)`` matrix ``P[i, j] = P(home scores i, away scores j)`` — as
produced by the Dixon-Coles / Poisson models in :mod:`poisson_models` — is
enough to derive probabilities for a large catalog of betting markets
*analytically*. Every market below is a sum over the relevant cells of ``P``,
so the markets are mutually consistent (no arbitrage between them) and need no
extra training data or labels.

The functions here are PURE (numpy in, plain ``dict`` out), so they are
trivially unit-testable and reusable across the prediction pipeline and any
future serving endpoint. Sport-specific catalogs register in
:data:`MARKET_DERIVERS`; only soccer is wired today, but the framework is
sport-agnostic so NHL (a hockey scoreline matrix) can register later without
touching the soccer math.

Output shape::

    {market_type: {selection: probability, ...}, ...}

For multi-line markets (``over_under``, ``asian_handicap``, ``team_total``) the
line is encoded in the selection key (e.g. ``"over_2.5"``, ``"-0.5_home"``,
``"home_over_1.5"``) so each market stays a flat ``Dict[str, float]`` suitable
for JSONB storage.

Asian handicap note: for each line we emit the RAW outcome masses
``{line}_home`` (home covers), ``{line}_away`` (away covers) and ``{line}_push``
(stake refunded). They sum to 1 per line. Expected value for backing home at
decimal odds ``O`` is ``raw_home * O + push - 1`` — the recommender must use the
raw masses, not a no-push conditional, to price pushes correctly.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .poisson_models import dixon_coles_tau_vec, poisson_pmf_grid

# Goals grid used when building the derivation matrix. The trained model keeps
# max_goals=6, which truncates ~the high-scoring tail; deriving over/under 4.5
# and 5.5 accurately wants a few more rows. N=11 (0..10 goals/side) drives the
# residual tail mass below ~1e-4 for realistic lambdas.
MAX_GOALS_DERIVE = 10

# Line sets offered per market.
OU_LINES: Tuple[float, ...] = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
AH_LINES: Tuple[float, ...] = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
TEAM_TOTAL_LINES: Tuple[float, ...] = (0.5, 1.5, 2.5)
# Halftime totals top out lower than FT — typical HT line is 0.5 or
# 1.5 goals at most. 2.5 included for the rare high-scoring case.
HT_OU_LINES: Tuple[float, ...] = (0.5, 1.5, 2.5)


def _fmt_line(line: float) -> str:
    """Format a betting line for a selection key: ``2.5`` -> ``"2.5"``,
    ``-1.0`` -> ``"-1"``, ``0.0`` -> ``"0"``. Integer lines drop the ``.0`` so
    they read like a bookmaker would quote them.

    Must stay in sync with the line formatter used by the recommendation
    generator (scripts/generate_recommendations.py) when it reconstructs model
    selection keys from ``(selection, line)`` odds rows.
    """
    f = float(line)
    return str(int(f)) if f.is_integer() else str(f)


def build_dc_matrix(
    home_lambda: float,
    away_lambda: float,
    rho: float = 0.0,
    max_goals: int = MAX_GOALS_DERIVE,
) -> np.ndarray:
    """Build a normalized ``(N x N)`` joint scoreline matrix from Dixon-Coles
    expected goals + the low-score correction ``rho``.

    Mirrors :meth:`DixonColesPredictor._score_matrix` but is parameterized on
    ``max_goals`` and reuses the same vectorized helpers (``poisson_pmf_grid``,
    ``dixon_coles_tau_vec``) so the tau math is identical to training. With
    ``rho=0`` this reduces to the independent-Poisson matrix.
    """
    mg = max_goals + 1
    h_lam = float(home_lambda)
    a_lam = float(away_lambda)

    home_pmf = poisson_pmf_grid(np.array([h_lam]), max_goals)[0]  # (mg,)
    away_pmf = poisson_pmf_grid(np.array([a_lam]), max_goals)[0]  # (mg,)
    P = np.outer(home_pmf, away_pmf)  # independent base

    i_idx, j_idx = np.indices((mg, mg))
    tau = dixon_coles_tau_vec(
        i_idx.ravel(),
        j_idx.ravel(),
        np.full(mg * mg, h_lam),
        np.full(mg * mg, a_lam),
        float(rho),
    ).reshape(mg, mg)
    P = P * tau

    s = P.sum()
    return P / s if s > 0 else P


def reconcile_matrix_to_1x2(
    P: np.ndarray,
    target: Sequence[float],
    iters: int = 50,
    tol: float = 1e-12,
) -> np.ndarray:
    """Rescale a scoreline matrix so its 1x2 region-marginals match ``target``
    (home, draw, away) via iterative proportional fitting.

    The ensemble's blended 1x2 is more accurate than raw Dixon-Coles for the
    headline market, so we reshape the matrix to hit it exactly while
    preserving the WITHIN-region shape (relative correct-score / totals
    structure). Because the three regions (home win / draw / away win)
    partition the matrix disjointly, IPF converges in a single pass; the loop
    is defensive. Zero-mass regions are skipped to avoid divide-by-zero.
    """
    P = P.astype(float, copy=True)
    mg = P.shape[0]
    i_idx, j_idx = np.indices((mg, mg))
    regions = [i_idx > j_idx, i_idx == j_idx, i_idx < j_idx]  # home, draw, away

    tgt = np.asarray(target, dtype=float)
    tsum = tgt.sum()
    if tsum <= 0:
        return P
    tgt = tgt / tsum  # normalize defensively

    for _ in range(iters):
        cur = np.array([P[m].sum() for m in regions])
        if np.all(np.abs(cur - tgt) < tol):
            break
        for k, m in enumerate(regions):
            if cur[k] > 0:
                P[m] *= tgt[k] / cur[k]

    s = P.sum()
    return P / s if s > 0 else P


def derive_soccer_markets(P: np.ndarray, top_n_scores: int = 12) -> Dict[str, Dict[str, float]]:
    """Derive the full soccer market catalog from a normalized scoreline matrix.

    Returns ``{market_type: {selection: probability}}``. See the module
    docstring for the selection-key conventions. ``P`` is expected to be
    normalized; a tiny defensive renormalization is applied if it drifts.
    """
    P = np.asarray(P, dtype=float)
    s = P.sum()
    if s > 0 and abs(s - 1.0) > 1e-9:
        P = P / s
    N = P.shape[0]

    i_idx, j_idx = np.indices((N, N))
    total = i_idx + j_idx
    margin = i_idx - j_idx  # home goals - away goals
    home_mask = i_idx > j_idx
    draw_mask = i_idx == j_idx
    away_mask = i_idx < j_idx
    ph = P.sum(axis=1)  # ph[k] = P(home scores k)
    pa = P.sum(axis=0)  # pa[k] = P(away scores k)
    karr = np.arange(N)

    markets: Dict[str, Dict[str, float]] = {}

    # 1x2 — match result.
    p_home = float(P[home_mask].sum())
    p_draw = float(P[draw_mask].sum())
    p_away = float(P[away_mask].sum())
    markets["1x2"] = {"home": p_home, "draw": p_draw, "away": p_away}

    # Double chance — three two-way markets (each in [0,1], do NOT sum to 1).
    markets["double_chance"] = {
        "1X": p_home + p_draw,
        "12": p_home + p_away,
        "X2": p_draw + p_away,
    }

    # Draw no bet — renormalize excluding the draw (stake refunded on draw).
    decisive = p_home + p_away
    if decisive > 0:
        markets["draw_no_bet"] = {"home": p_home / decisive, "away": p_away / decisive}
    else:
        markets["draw_no_bet"] = {"home": 0.0, "away": 0.0}

    # Over/Under total goals — half-integer lines never push, so over+under=1.
    ou: Dict[str, float] = {}
    for line in OU_LINES:
        lbl = _fmt_line(line)
        ou[f"over_{lbl}"] = float(P[total > line].sum())
        ou[f"under_{lbl}"] = float(P[total < line].sum())
    markets["over_under"] = ou

    # Both teams to score.
    btts_yes = float(P[(i_idx >= 1) & (j_idx >= 1)].sum())
    markets["btts"] = {"yes": btts_yes, "no": 1.0 - btts_yes}

    # Correct score — top-N most likely scorelines + an "other" bucket.
    flat = P.ravel()
    order = np.argsort(flat)[::-1][:top_n_scores]
    cs: Dict[str, float] = {}
    acc = 0.0
    for idx in order:
        h, a = divmod(int(idx), N)
        p = float(flat[idx])
        cs[f"{h}-{a}"] = p
        acc += p
    cs["other"] = max(0.0, 1.0 - acc)
    markets["correct_score"] = cs

    # Asian / goal handicap — RAW masses per home-perspective line. Home covers
    # line h when (margin + h) > 0, pushes when == 0 (integer lines only).
    ah: Dict[str, float] = {}
    for line in AH_LINES:
        lbl = _fmt_line(line)
        adj = margin + line
        ah[f"{lbl}_home"] = float(P[adj > 0].sum())
        ah[f"{lbl}_away"] = float(P[adj < 0].sum())
        ah[f"{lbl}_push"] = float(P[adj == 0].sum())
    markets["asian_handicap"] = ah

    # Team totals — per side, per line (each pair sums to 1).
    tt: Dict[str, float] = {}
    for line in TEAM_TOTAL_LINES:
        lbl = _fmt_line(line)
        home_over = float(ph[karr > line].sum())
        away_over = float(pa[karr > line].sum())
        tt[f"home_over_{lbl}"] = home_over
        tt[f"home_under_{lbl}"] = 1.0 - home_over
        tt[f"away_over_{lbl}"] = away_over
        tt[f"away_under_{lbl}"] = 1.0 - away_over
    markets["team_total"] = tt

    # Clean sheet — home keeps it when away scores 0, and vice versa.
    home_cs = float(pa[0])
    away_cs = float(ph[0])
    markets["clean_sheet"] = {
        "home_yes": home_cs,
        "home_no": 1.0 - home_cs,
        "away_yes": away_cs,
        "away_no": 1.0 - away_cs,
    }

    # Win to nil — win while keeping a clean sheet.
    home_wtn = float(P[(i_idx > j_idx) & (j_idx == 0)].sum())
    away_wtn = float(P[(i_idx < j_idx) & (i_idx == 0)].sum())
    markets["win_to_nil"] = {
        "home_yes": home_wtn,
        "home_no": 1.0 - home_wtn,
        "away_yes": away_wtn,
        "away_no": 1.0 - away_wtn,
    }

    # Odd/even total goals.
    odd = float(P[(total % 2) == 1].sum())
    markets["odd_even"] = {"odd": odd, "even": 1.0 - odd}

    # Total goals exact bands.
    tg: Dict[str, float] = {str(k): float(P[total == k].sum()) for k in range(6)}
    tg["6+"] = float(P[total >= 6].sum())
    markets["total_goals"] = tg

    # Winning margin.
    markets["winning_margin"] = {
        "home_1": float(P[margin == 1].sum()),
        "home_2": float(P[margin == 2].sum()),
        "home_3plus": float(P[margin >= 3].sum()),
        "draw": p_draw,
        "away_1": float(P[margin == -1].sum()),
        "away_2": float(P[margin == -2].sum()),
        "away_3plus": float(P[margin <= -3].sum()),
    }

    # Result + BTTS combos.
    both = (i_idx >= 1) & (j_idx >= 1)
    markets["result_btts"] = {
        "home_yes": float(P[home_mask & both].sum()),
        "home_no": float(P[home_mask & ~both].sum()),
        "draw_yes": float(P[draw_mask & both].sum()),
        "draw_no": float(P[draw_mask & ~both].sum()),
        "away_yes": float(P[away_mask & both].sum()),
        "away_no": float(P[away_mask & ~both].sum()),
    }

    # Result + Over/Under 2.5 combos.
    over25 = total > 2.5
    markets["result_over_under"] = {
        "home_over": float(P[home_mask & over25].sum()),
        "home_under": float(P[home_mask & ~over25].sum()),
        "draw_over": float(P[draw_mask & over25].sum()),
        "draw_under": float(P[draw_mask & ~over25].sum()),
        "away_over": float(P[away_mask & over25].sum()),
        "away_under": float(P[away_mask & ~over25].sum()),
    }

    return markets


def derive_soccer_halftime_markets(P: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Derive halftime soccer markets from a HALFTIME scoreline matrix.

    ``P`` is the halftime Dixon-Coles output (P[i,j] = P(HT home=i,
    HT away=j)). Only three markets are derived because that's what
    HT odds actually trade on retail books:

      * ``match_result_ht`` — 3-way home/draw/away at the break.
      * ``over_under_ht`` — HT total goals over/under 0.5 / 1.5 / 2.5.
      * ``btts_ht`` — both teams to score before halftime.

    The selection-key conventions mirror the FT equivalents so the
    recs engine + store_market_predictions can treat them
    identically — only the prediction_type label differs.
    """
    P = np.asarray(P, dtype=float)
    s = P.sum()
    if s > 0 and abs(s - 1.0) > 1e-9:
        P = P / s
    N = P.shape[0]
    i_idx, j_idx = np.indices((N, N))
    total = i_idx + j_idx
    home_mask = i_idx > j_idx
    draw_mask = i_idx == j_idx
    away_mask = i_idx < j_idx

    markets: Dict[str, Dict[str, float]] = {}

    markets["match_result_ht"] = {
        "home": float(P[home_mask].sum()),
        "draw": float(P[draw_mask].sum()),
        "away": float(P[away_mask].sum()),
    }

    ou: Dict[str, float] = {}
    for line in HT_OU_LINES:
        lbl = _fmt_line(line)
        ou[f"over_{lbl}"] = float(P[total > line].sum())
        ou[f"under_{lbl}"] = float(P[total < line].sum())
    markets["over_under_ht"] = ou

    btts_yes = float(P[(i_idx >= 1) & (j_idx >= 1)].sum())
    markets["btts_ht"] = {"yes": btts_yes, "no": 1.0 - btts_yes}

    return markets


def derive_soccer_htft_markets(
    P_HT: np.ndarray,
    P_2H: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Derive the halftime/fulltime joint double-result market.

    Inputs:
        P_HT: HT scoreline matrix from the halftime Dixon-Coles
              model. ``P_HT[i, j] = P(HT home goals = i,
              HT away goals = j)``.
        P_2H: SECOND-HALF scoreline matrix from the 2H Dixon-Coles
              model. ``P_2H[a, b] = P(2H home goals = a,
              2H away goals = b)``.

    The joint FT scoreline matrix is the 2D convolution
    ``P_FT = P_HT * P_2H`` (each (k, l) entry is
    ``sum_{i, j} P_HT[i, j] * P_2H[k - i, l - j]``). The HT/FT
    double result aggregates joint (HT, FT) mass under the 9
    (HT outcome) × (FT outcome) buckets.

    Selection keys: ``{ht_outcome}_{ft_outcome}`` where outcome ∈
    {home, draw, away}. Example: ``home_draw`` = home leading at
    HT and the match ends drawn. The 9 selections sum to 1.
    """
    P_HT = np.asarray(P_HT, dtype=float)
    P_2H = np.asarray(P_2H, dtype=float)
    # Defensive renorm — callers may hand us drifted matrices.
    s_ht = P_HT.sum()
    s_2h = P_2H.sum()
    if s_ht > 0 and abs(s_ht - 1.0) > 1e-9:
        P_HT = P_HT / s_ht
    if s_2h > 0 and abs(s_2h - 1.0) > 1e-9:
        P_2H = P_2H / s_2h

    N_ht = P_HT.shape[0]
    N_2h = P_2H.shape[0]

    out: Dict[str, float] = {
        "home_home": 0.0,
        "home_draw": 0.0,
        "home_away": 0.0,
        "draw_home": 0.0,
        "draw_draw": 0.0,
        "draw_away": 0.0,
        "away_home": 0.0,
        "away_draw": 0.0,
        "away_away": 0.0,
    }

    # Per-HT-cell: distribute P_HT[i,j] across each (i+a, j+b) in
    # the 2H grid, bucketing by both HT result and FT result. This
    # is the convolution + joint-aggregate in one pass.
    for i in range(N_ht):
        for j in range(N_ht):
            p_ht_cell = float(P_HT[i, j])
            if p_ht_cell == 0.0:
                continue
            if i > j:
                ht_outcome = "home"
            elif i < j:
                ht_outcome = "away"
            else:
                ht_outcome = "draw"
            for a in range(N_2h):
                for b in range(N_2h):
                    p_2h_cell = float(P_2H[a, b])
                    if p_2h_cell == 0.0:
                        continue
                    ft_home = i + a
                    ft_away = j + b
                    if ft_home > ft_away:
                        ft_outcome = "home"
                    elif ft_home < ft_away:
                        ft_outcome = "away"
                    else:
                        ft_outcome = "draw"
                    key = f"{ht_outcome}_{ft_outcome}"
                    out[key] += p_ht_cell * p_2h_cell

    # Final defensive renorm — shouldn't be needed if P_HT and P_2H
    # were normalised but caps any rounding drift.
    total = sum(out.values())
    if total > 0 and abs(total - 1.0) > 1e-9:
        out = {k: v / total for k, v in out.items()}
    return {"ht_ft_double_result": out}


# Sport -> deriver. Only soccer is wired today; NHL would register a
# derive_hockey_markets here (same matrix machinery, different line sets).
MARKET_DERIVERS: Dict[str, Callable[..., Dict[str, Dict[str, float]]]] = {
    "soccer": derive_soccer_markets,
    "soccer_halftime": derive_soccer_halftime_markets,
}


def derive_markets(sport: str, P: np.ndarray, **kwargs) -> Dict[str, Dict[str, float]]:
    """Dispatch to the registered market deriver for ``sport``."""
    try:
        deriver = MARKET_DERIVERS[sport]
    except KeyError as exc:
        raise ValueError(f"No market deriver registered for sport {sport!r}") from exc
    return deriver(P, **kwargs)


def derive_from_lambdas(
    home_lambda: float,
    away_lambda: float,
    rho: float = 0.0,
    *,
    target_1x2: Optional[Sequence[float]] = None,
    sport: str = "soccer",
    max_goals: int = MAX_GOALS_DERIVE,
    **kwargs,
) -> Dict[str, Dict[str, float]]:
    """Convenience: expected goals -> (optionally reconciled) markets.

    ``target_1x2`` is the ensemble's ``(home, draw, away)`` — when provided the
    matrix is reconciled to it before derivation so every market's 1x2 split
    matches the headline prediction.
    """
    P = build_dc_matrix(home_lambda, away_lambda, rho, max_goals=max_goals)
    if target_1x2 is not None:
        P = reconcile_matrix_to_1x2(P, target_1x2)
    return derive_markets(sport, P, **kwargs)
