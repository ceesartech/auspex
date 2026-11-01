"""Which stored soccer derived markets lose to a simple per-league constant:
the display rule, read off scripts/ab_soccer_ft_dixon_coles.py's per-row scores.

WHY THIS EXISTS
---------------
The 2026-11-01 gate for the full-time Dixon-Coles that serves the derived
soccer markets (scripts/ab_soccer_ft_dixon_coles.py, DO-NOT-SHIP) found, as
context, that the served derived BTTS is significantly worse than a per-league
constant (live regime: per-match +0.00131 Brier +/- 0.00030; fold-clustered
+0.00139 +/- 0.00033, t(30) lower bound +0.00072) and that its over-2.5 is no
better than one (per-match -0.00043; clustered -0.00024 +/- 0.00044, inside
the noise). It compared only those two markets and the correct-score grid
with a constant.

The NHL precedent (d2246cf, 641f93f): the dixon_coles_nhl markets that lost to
a per-season constant beyond the noise stopped being displayed. They are
listed in services/api prediction_service.HIDDEN_PREDICTION_PAIRS, every
display query filters them out, and precompute still writes them and grading
still grades them, so their record keeps accruing. This script applies that
rule to every soccer market the full-time Dixon-Coles serves. It reads scores
only: it fits nothing and touches no database.

PRE-REGISTERED RULE (fixed 2026-11-01 and committed before the scores it reads
were produced; the harness and its arms are unchanged)
  Input     scores.npz from `ab_soccer_ft_dixon_coles.py --out-dir` (one Brier
            per eval match for every arm, anchor regime and unit, keyed
            brier::<arm>::<regime>::<unit>, plus the fold label), and the
            harness's --out-json report for provenance (served sha256, n_eval,
            folds, the live check). The npz row count must equal the report's
            n_eval.
  Window    the harness's own: the 1x2 incumbent's unseen finished matches in
            monthly folds, run with --end 2026-10-31 on a replica of a fresh
            read-only prod export, with the served artifact pinned by sha256.
  Served    the harness's `served` arm: the production Dixon-Coles read
            through precompute_predictions' own derivation, reconciled to the
            served 1x2, in each anchor regime.
  Constant  base_league: per league, the realised rate of each unit's outcomes
            over every finished match before the fold's month, shrunk toward
            the global rate with k = 200 matches, unweighted (the harness's
            fold_constants; not reconciled, not anchored).
            Why this constant: it is the NHL rule's constant with league as the
            group. ab_nhl_dixon_coles's base_season is the walk-forward
            per-season rate shrunk toward the global one with k = 200,
            unweighted, and NHL is one league, so its season plays the part a
            league plays here. It is also the harness's own matched constant
            for the served arm (matched_constant). base_league_dw carries the
            refit arms' recency weighting and is matched to them; it is
            reported as context.
  Type      one stored predictions.prediction_type. Hiding works per stored
            type (HIDDEN_PREDICTION_PAIRS is "<model_name>:<prediction_type>"),
            and one stored row shows every line of its type together, so a
            type is judged as a whole: its per-match score is the SUM of its
            units' per-match Brier (TYPE_UNITS below):
              over_under      over_0.5 .. over_5.5 (6 binary lines)
              team_total      home and away over 0.5 / 1.5 / 2.5 (6 binary)
              asian_handicap  all 17 lines, each a 3-way home / away / push
                              Brier on the effective masses (the five lines
                              fixed by the 1x2 included: they are shown too)
              double_chance   1X + 12 + X2 (fixed by the 1x2)
              clean_sheet, win_to_nil   home + away
              correct_score   the 121-cell grid (the NHL rule's A1); the
                              listed top-12 is improper across arms and has
                              no constant
              btts, draw_no_bet, odd_even, winning_margin, total_goals,
              result_btts, result_over_under   one unit each
            Scaling the sum (a mean) changes no bound and no verdict.
  Statistic ab_soccer_ft_dixon_coles.pair(served, base_league, folds): the
            paired per-match delta served - constant, its fold-clustered SE
            and the t(n_folds-1) 95% bounds.
  HIDE      a type exactly when its 95% lower bound (rule A2,
            ab_nhl_dixon_coles.lower_bound: the fold-clustered t bound, else
            the per-match delta - 2 SE) is > 0 in BOTH the live and the proxy
            regime. Why both: they are the gate's own deciding regimes, and
            together they bracket how production anchors today. The O/U-2.5
            anchor moves every totals-shaped market. The live replay anchors
            about 2% of this window's rows and the proxy about 49%, while
            production anchored 25-32% of matches in Sep-Oct 2026. A type that
            loses in only one regime stays visible and is reported.
            A type with no delta or no bound in a deciding regime fails the
            run: an unassessable type is a broken run, never a silent keep.
  Context   never decides: each unit on its own (a type hides only as a
            whole, so a losing line inside a type that does not lose is
            reported, not hidden); the none regime; served vs base_league_dw;
            the ESPN-era folds (2026-06 on), the only months the live anchor
            reaches.
  Output    the hide set as HIDDEN_PREDICTION_PAIRS entries "ensemble:<type>":
            precompute_predictions writes every soccer derived row with
            model_name 'ensemble', and on 2026-11-01 no other sport wrote that
            model_name (read-only prod check).
  Not covered: the half-time markets (over_under_ht, match_result_ht, btts_ht,
            ht_ft_double_result). They come from two other artifacts, and the
            served half-time artifact was refitted on every match up to
            2026-10-31, so no historical window is out of sample for it.

RESULT (2026-11-01; the committed rule run once, unamended): hide odd_even only.
  Harness run: ab_soccer_ft_dixon_coles.py --end 2026-10-31 --served-sha256
  02141a6a... --live-check-model-version served, unchanged, on a local replica
  of a read-only REPEATABLE READ prod export taken 21:11:21-21:11:35Z. At the
  export no soccer_repair_runbook.sh or other scripts/ops process was running
  and no runbook state file existed, so this is the pre-runbook corpus. All 11
  soccer model files were copied byte-identical (sha256 checked against the
  api container). The run reproduced the gate's record exactly: n = 25,604,
  31 folds 2024-04..2026-10, refit - served -0.00375 on over_2.5 (live),
  live-anchored 504 rows (2%), proxy-anchored 12,436 (49%); the live check
  re-derived all 371 matches' stored predictions of
  ensemble_soccer_match_result_1.0.0+1793505994 (1x2-only to 2.8e-17, the 69
  anchored to 5.0e-7).
  Per type, served - base_league, in two forms. "Per-match" is the mean
  delta over all matches with its SE (matches treated as independent).
  "Clustered" is the unweighted mean of the 31 monthly fold means, with its
  SE from the spread of those monthly means. The rule reads the 95% interval
  t(30) puts on the clustered numbers (t(30) = 2.04), not the per-match
  pair, so the bound cannot be rebuilt from the per-match numbers.
    HIDE odd_even
      live   per-match +0.00039 +/- 0.00017; clustered +0.00055 +/- 0.00021,
             t(30) [+0.00013, +0.00098]
      proxy  per-match +0.00036 +/- 0.00016; clustered +0.00052 +/- 0.00021,
             t(30) [+0.00010, +0.00094]
      Served's mean P(odd) is 0.488 against 0.499 realised and the
      constant's 0.492. ESPN era (2026-06 on, n = 3,211, 5 folds), live:
      per-match +0.00012 +/- 0.00047; clustered +0.00083 +/- 0.00104, t(4)
      [-0.00206, +0.00372]: too few months to say anything. Its
      recommendations were already switched off.
    keep btts (an operator decision: see the POST-RUN NOTE below)
      live   per-match +0.00131 +/- 0.00030; clustered +0.00139 +/- 0.00033,
             t(30) [+0.00072, +0.00205]
      proxy  per-match -0.00027 +/- 0.00034; clustered +0.00006 +/- 0.00036,
             t(30) [-0.00067, +0.00079]
      Loses in the live regime only, so the committed rule keeps it. ESPN
      era, live: per-match -0.00006 +/- 0.00071; clustered +0.00014 +/-
      0.00058, t(4) [-0.00148, +0.00176]: as uninformative as odd_even's.
      Context: against base_league_dw it loses in both regimes (clustered
      +0.00208 +/- 0.00035 / +0.00075 +/- 0.00035, t(30) lower bounds
      +0.00137 / +0.00003).
    keep over_under
      live   per-match -0.00072 +/- 0.00091; clustered -0.00004 +/- 0.00141,
             t(30) [-0.00291, +0.00283]
      proxy  per-match -0.00942 +/- 0.00113; clustered -0.00731 +/- 0.00178,
             t(30) [-0.01095, -0.00367]
      No line loses on its own in either regime (over_2.5 live -0.00043).
    keep total_goals
      live   per-match +0.00036 +/- 0.00026; clustered +0.00061 +/- 0.00043,
             t(30) [-0.00028, +0.00149]
      proxy  per-match -0.00163 +/- 0.00030; clustered -0.00105 +/- 0.00049,
             t(30) [-0.00205, -0.00006]
    keep, served better beyond the noise in both regimes (live per-match delta):
          double_chance -0.0472, draw_no_bet -0.0749, asian_handicap -0.537
          (17-line sum), team_total -0.0514, clean_sheet -0.0138, win_to_nil
          -0.0150, winning_margin -0.0163, result_btts -0.0233,
          result_over_under -0.0263, correct_score (121-cell grid) -0.0049.
  No unit loses on its own inside a multi-line type that keeps.
  Independent recompute (closeout2 scratch, adhoc/recompute.py): odd_even
  and btts re-derived row by row through derive_from_lambdas from the saved
  lambdas, rho, 1x2 targets and anchors, with the per-league constant rebuilt
  in float64 from the raw export. Every per-row Brier matched scores.npz to
  8.9e-16, and the deltas and bounds matched.

POST-RUN NOTE (added after review, 2026-11-01 evening; the rule above is NOT
amended and its output, odd_even only, stands as the rule's result)
  * The rule combines the two regimes with AND: a type hides only when it
    loses in both. That departs from the house convention it borrows its
    regimes from. In ab_soccer_ft_dixon_coles.decide() a significant
    regression in EITHER deciding regime blocks a ship (rule b, A2). The NHL
    rule (d2246cf) had one regime and hid a whole type when any one of its
    lines lost (over_under, for its 4.5 line alone). BTTS stays visible only
    because of the AND. Under an either-regime reading it hides (live t(30)
    lower bound +0.00072). odd_even hides under either reading. No other
    type changes, and no line loses on its own inside a kept type.
  * The rule text's "production anchored 25-32% of matches in Sep-Oct 2026"
    is the harness's live-replay share (anchor_live per fold: 2026-09 25.4%,
    2026-10 32.2%), not a production measurement. Production only started
    anchoring on 2026-10-26 (15:45Z). A read-only prod check at 2026-11-01
    22:00Z, on the newest stored btts row per match since then (the
    metadata 'reconcile' tag), found 66 of 226 finished matches anchored
    (29%: 47 of 166 in October, 19 of 60 in November). For the 318 upcoming
    matches the share is 51 (16%), because O/U odds arrive close to kickoff.
  * At production's share BTTS loses by more than odd_even. anchor_mix()
    (context, added after the run; run with --per-match) scores the
    anchored share of rows in the proxy regime (the rows the proxy anchors)
    and the rest in the none regime, and mixes the two per-match means. With
    a 4,000-draw bootstrap over the monthly folds (95% interval):
        anchored   btts                            odd_even
        16%        +0.00121 [+0.00047, +0.00193]   +0.00046 [+0.00011, +0.00078]
        29%        +0.00062 [-0.00004, +0.00129]   +0.00042 [+0.00009, +0.00071]
        32%        +0.00048 [-0.00017, +0.00115]   +0.00041 [+0.00008, +0.00069]
        49%        -0.00029 [-0.00096, +0.00034]   +0.00035 [+0.00003, +0.00064]
    At 29% the BTTS lower bound sits at about zero (two independent
    bootstraps gave +0.00002 and, at 28%, +0.00001). total_goals ties the
    constant at 16% and 29% and beats it from 32%; over_under ties it at 16%
    and beats it from 29%; every other type beats it at every share.
    Re-running the script with the context reproduced every other output
    field of the original run to 1.3e-15 relative (float rounding).
  * So BTTS is left to the operator, with a recommendation to hide it (one
    entry in each of the two lists; see the audit-doc entry), unless the
    operator deliberately prefers this rule's conservative reading.
  * For future display rules: say up front whether regimes combine with AND
    or OR, and justify any break from the A2 either-regime convention.

Usage (after the harness run; no database needed):
    python scripts/ab_soccer_derived_vs_constant.py --scores <out-dir>/scores.npz \\
        --report <out-json> --out-json /tmp/derived_vs_constant.json \\
        [--per-match <out-dir>/per_match.csv.gz]   # adds the anchor-mix context
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_halftime_dixon_coles as ht  # noqa: E402
import ab_nhl_dixon_coles as nhl  # noqa: E402
import ab_soccer_ft_dixon_coles as ftdc  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("ab_soccer_derived_vs_constant")

MODEL_NAME = ftdc.MODEL_NAME  # 'ensemble': predictions.model_name of every soccer derived row
SERVED = "served"
BASELINE = "base_league"  # the decision's constant (see the docstring)
CONTEXT_BASELINE = "base_league_dw"
DECIDING_REGIMES: Tuple[str, ...] = ("live", "proxy")
CONTEXT_REGIMES: Tuple[str, ...] = ("none",)
ESPN_ERA_FIRST_FOLD = "2026-06"  # fold labels are "YYYY-MM"
NAN_TYPES = frozenset({"draw_no_bet"})  # the only type that does not grade on every row (a draw)
# Context only (added after the run; never decides): anchored shares for the
# row mix. 0.16 and 0.29 are production's measured shares since the anchor
# shipped on 2026-10-26 (upcoming / finished matches, read-only prod check
# 2026-11-01 22:00Z), 0.32 the live replay's October share, 0.49 the proxy's.
ANCHOR_SHARES: Tuple[float, ...] = (0.16, 0.29, 0.32, 0.49)
ANCHOR_MIX_BOOT = 4000
ANCHOR_MIX_SEED = 20261101

# Every stored full-time derived type -> the harness units summed into its per-match score.
TYPE_UNITS: Dict[str, Tuple[str, ...]] = {
    market: tuple(u.name for u in ftdc.UNITS if u.market == market and u.name != ftdc.LISTED)
    for market in dict.fromkeys(u.market for u in ftdc.UNITS)
}


def brier_key(arm: str, regime: str, unit: str) -> str:
    """The harness's scores.npz key; a constant has no anchor regime ('-')."""
    return f"brier::{arm}::{'-' if arm in ftdc.CONSTANT_ARMS else regime}::{unit}"


def load_scores(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        scores = {k: z[k] for k in z.files}
    if "fold" not in scores:
        raise RuntimeError(f"{path} carries no fold labels: not an ab_soccer_ft_dixon_coles scores.npz")
    return scores


def type_score(scores: Dict[str, np.ndarray], arm: str, regime: str, market: str) -> np.ndarray:
    """Per-match sum of the type's unit Briers for one arm and regime. NaN
    only where the type does not grade (draw_no_bet on a draw)."""
    keys = [brier_key(arm, regime, u) for u in TYPE_UNITS[market]]
    missing = [k for k in keys if k not in scores]
    if missing:
        raise KeyError(f"{market}: scores.npz lacks {missing}")
    total = np.sum([np.asarray(scores[k], dtype=float) for k in keys], axis=0)
    if market not in NAN_TYPES and not np.isfinite(total).all():
        bad = int((~np.isfinite(total)).sum())
        raise RuntimeError(f"{market} [{arm}/{regime}]: {bad} row(s) without a score; only {sorted(NAN_TYPES)} may")
    return total


def compare(
    scores: Dict[str, np.ndarray],
    market: str,
    regime: str,
    baseline: str = BASELINE,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Paired served - constant on the type's per-match score."""
    served = type_score(scores, SERVED, regime, market)
    const = type_score(scores, baseline, regime, market)
    if not np.array_equal(np.isfinite(served), np.isfinite(const)):
        raise RuntimeError(f"{market} [{regime}]: served and {baseline} grade different rows")
    return ftdc.pair(served, const, np.asarray(scores["fold"]), mask)


def compare_unit(scores: Dict[str, np.ndarray], unit: str, regime: str, baseline: str = BASELINE) -> Dict[str, Any]:
    return ftdc.pair(
        np.asarray(scores[brier_key(SERVED, regime, unit)], dtype=float),
        np.asarray(scores[brier_key(baseline, regime, unit)], dtype=float),
        np.asarray(scores["fold"]),
    )


def load_anchor_mask(per_match_csv: str, scores: Dict[str, np.ndarray]) -> np.ndarray:
    """True on the rows the proxy regime anchors (anchor_proxy not null in the
    harness's per_match.csv.gz). Refuses a file whose match_ids are not the
    scores' match_ids in the same order."""
    pm = pd.read_csv(per_match_csv, usecols=["match_id", "anchor_proxy"], dtype={"match_id": str})
    if "match_id" not in scores:
        raise RuntimeError("scores.npz carries no match_id: cannot align it with per_match.csv.gz")
    ids = np.asarray(scores["match_id"]).astype(str)
    if len(pm) != len(ids) or not np.array_equal(pm["match_id"].to_numpy(dtype=str), ids):
        raise RuntimeError(
            f"{per_match_csv} ({len(pm)} rows) does not list the scores' {len(ids)} match_ids in the same order"
        )
    return pm["anchor_proxy"].notna().to_numpy()


def anchor_mix(
    scores: Dict[str, np.ndarray],
    anchored: np.ndarray,
    market: str,
    shares: Sequence[float] = ANCHOR_SHARES,
    n_boot: int = ANCHOR_MIX_BOOT,
    seed: int = ANCHOR_MIX_SEED,
) -> Dict[str, Any]:
    """Context, never decides (added after the run). Served - base_league on
    the type's per-match score if a share f of matches were O/U-anchored: the
    rows the proxy regime anchors are scored in the proxy regime, the rows it
    never anchors in the none regime, and the two per-match means are mixed
    f : 1 - f. The interval resamples the monthly folds with replacement
    (n_boot draws) and takes the 2.5 / 97.5 percentiles."""
    anchored = np.asarray(anchored, dtype=bool)
    if anchored.shape != (len(scores["fold"]),):
        raise RuntimeError(f"anchor mask has {anchored.shape} rows, scores have {len(scores['fold'])}")
    const = type_score(scores, BASELINE, "none", market)
    d_anch = type_score(scores, SERVED, "proxy", market) - const
    d_none = type_score(scores, SERVED, "none", market) - const
    ok = np.isfinite(d_anch) & np.isfinite(d_none)
    a, u = anchored & ok, ~anchored & ok
    if not a.any() or not u.any():
        raise RuntimeError(f"{market}: the anchor mix needs both anchored and unanchored gradable rows")
    labels, inv = np.unique(np.asarray(scores["fold"]).astype(str), return_inverse=True)
    k = len(labels)
    sum_a = np.bincount(inv[a], weights=d_anch[a], minlength=k)
    cnt_a = np.bincount(inv[a], minlength=k).astype(float)
    sum_u = np.bincount(inv[u], weights=d_none[u], minlength=k)
    cnt_u = np.bincount(inv[u], minlength=k).astype(float)
    mean_a, mean_u = float(sum_a.sum() / cnt_a.sum()), float(sum_u.sum() / cnt_u.sum())
    # Each draw is a vector of fold multiplicities (k folds drawn with replacement).
    w = np.random.default_rng(seed).multinomial(k, np.full(k, 1.0 / k), size=n_boot).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot_a, boot_u = (w @ sum_a) / (w @ cnt_a), (w @ sum_u) / (w @ cnt_u)
    out: Dict[str, Any] = {
        "anchored_rows": int(a.sum()),
        "unanchored_rows": int(u.sum()),
        "delta_anchored_proxy": mean_a,
        "delta_unanchored_none": mean_u,
        "shares": {},
    }
    for f in shares:
        draws = f * boot_a + (1.0 - f) * boot_u
        draws = draws[np.isfinite(draws)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        out["shares"][f"{f:.2f}"] = {
            "delta": f * mean_a + (1.0 - f) * mean_u,
            "lower_95_boot": float(lo),
            "upper_95_boot": float(hi),
        }
    return out


def hide_verdict(market: str, per_regime: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The pre-registered rule: hide when the A2 lower bound of served -
    constant is > 0 in every DECIDING_REGIMES regime. Raises when a deciding
    regime has no delta or no bound."""
    bounds: Dict[str, float] = {}
    for regime in DECIDING_REGIMES:
        p = per_regime.get(regime) or {}
        d = p.get("delta")
        if d is None or not math.isfinite(d):
            raise RuntimeError(f"{market} [{regime}]: no gradable paired rows — the type cannot be assessed")
        lb = nhl.lower_bound(p)
        if lb is None:
            raise RuntimeError(f"{market} [{regime}]: dBrier {d:+.5f} has no 95% bound — the type cannot be assessed")
        bounds[regime] = lb
    losing = [r for r in DECIDING_REGIMES if bounds[r] > 0]
    hide = len(losing) == len(DECIDING_REGIMES)
    if hide:
        reason = "served loses to the per-league constant beyond the noise in " + " and ".join(DECIDING_REGIMES)
    elif losing:
        reason = f"loses beyond the noise in {losing[0]} only: stays visible"
    else:
        reason = "not worse than the per-league constant beyond the noise"
    return {"hide": hide, "lower_bounds": bounds, "reason": reason}


def provenance(report: Optional[Dict[str, Any]], scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
    n = int(len(scores["fold"]))
    out: Dict[str, Any] = {"n_rows": n, "folds": list(dict.fromkeys(np.asarray(scores["fold"]).tolist()))}
    if report is None:
        return out
    if int(report.get("n_eval", -1)) != n:
        raise RuntimeError(f"scores.npz has {n} rows but the report's n_eval is {report.get('n_eval')}")
    meta = report.get("meta") or {}
    out.update(
        {
            "served_sha256": (meta.get("served_fingerprint") or {}).get("sha256"),
            "ensemble_artifact": meta.get("ensemble_artifact"),
            "live_check": meta.get("live_check"),
            "anchored": report.get("anchored"),
            "harness_decision": (report.get("decision") or {}).get("reason"),
        }
    )
    return out


def run(
    scores: Dict[str, np.ndarray],
    report: Optional[Dict[str, Any]] = None,
    anchored: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    prov = provenance(report, scores)
    folds = np.asarray(scores["fold"]).astype(str)
    espn = np.array([f >= ESPN_ERA_FIRST_FOLD for f in folds], dtype=bool)
    types: Dict[str, Dict[str, Any]] = {}
    for market, units in TYPE_UNITS.items():
        regimes = {r: compare(scores, market, r) for r in DECIDING_REGIMES + CONTEXT_REGIMES}
        verdict = hide_verdict(market, regimes)
        types[market] = {
            "units": list(units),
            **verdict,
            "vs_" + BASELINE: regimes,
            "vs_" + CONTEXT_BASELINE: {r: compare(scores, market, r, CONTEXT_BASELINE) for r in DECIDING_REGIMES},
            "espn_era": {r: compare(scores, market, r, mask=espn) for r in DECIDING_REGIMES} if espn.sum() > 1 else {},
            "per_unit": (
                {r: {u: compare_unit(scores, u, r) for u in units} for r in DECIDING_REGIMES} if len(units) > 1 else {}
            ),
            "anchor_mix": anchor_mix(scores, anchored, market) if anchored is not None else {},
        }
    hidden = sorted(f"{MODEL_NAME}:{m}" for m, t in types.items() if t["hide"])
    return {"provenance": prov, "espn_era_rows": int(espn.sum()), "types": types, "hidden_pairs": hidden}


# ── Formatting ───────────────────────────────────────────────────────


def _f(v: Any, spec: str = "+.5f") -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, spec)


def _pair_cells(p: Dict[str, Any]) -> str:
    return (
        f"{_f(p.get('delta'))} ±{_f(p.get('se'), '.5f')}  cl {_f(p.get('delta_cluster'))} "
        f"±{_f(p.get('se_cluster'), '.5f')}  t [{_f(p.get('lower_95_cluster_t'))}, {_f(p.get('upper_95_cluster_t'))}]"
    )


def format_report(r: Dict[str, Any]) -> str:
    prov = r["provenance"]
    L: List[str] = [
        f"Soccer derived markets vs a per-league constant ({BASELINE}): {prov['n_rows']} eval matches, "
        f"folds {prov['folds'][0]}..{prov['folds'][-1]} ({len(prov['folds'])}); "
        f"served sha256 {str(prov.get('served_sha256'))[:8]}...",
        "dBrier = served - constant on the type's per-match score (sum of its units); "
        "HIDE when the 95% lower bound is > 0 in live AND proxy.",
        "Columns: per-match delta +/-SE; cl = fold-clustered delta (unweighted mean of the monthly fold means) "
        "+/-SE; t = the t(n_folds-1) 95% interval on the clustered numbers, which the rule reads.",
        "",
    ]
    for market, t in r["types"].items():
        L.append(f"{market:<18} {'HIDE' if t['hide'] else 'keep'}  ({len(t['units'])} unit(s)): {t['reason']}")
        for regime in DECIDING_REGIMES + CONTEXT_REGIMES:
            L.append(
                f"    {regime:<6} {_pair_cells(t['vs_' + BASELINE][regime])}  n={t['vs_' + BASELINE][regime]['n']}"
            )
        for regime, p in t["vs_" + CONTEXT_BASELINE].items():
            L.append(f"    {regime:<6} vs {CONTEXT_BASELINE}: {_pair_cells(p)}")
        for regime, p in t["espn_era"].items():
            L.append(f"    {regime:<6} ESPN era: {_pair_cells(p)}  n={p['n']}")
        mix = t.get("anchor_mix") or {}
        if mix:
            cells = ", ".join(
                f"{float(f):.0%} {_f(v['delta'])} [{_f(v['lower_95_boot'])}, {_f(v['upper_95_boot'])}]"
                for f, v in mix["shares"].items()
            )
            L.append(f"    anchor mix (context; share anchored: delta [fold-bootstrap 95%]): {cells}")
        for regime, per in t["per_unit"].items():
            for unit, p in per.items():
                flag = "  <- this unit alone loses" if (nhl.lower_bound(p) or 0) > 0 else ""
                L.append(f"      {regime:<6} {unit:<28} {_pair_cells(p)}{flag}")
    L += ["", f"HIDDEN_PREDICTION_PAIRS to add: {r['hidden_pairs'] or 'none'}"]
    return "\n".join(L)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", required=True, help="scores.npz from ab_soccer_ft_dixon_coles.py --out-dir")
    p.add_argument("--report", default=None, help="the same run's --out-json (provenance and row-count check)")
    p.add_argument(
        "--per-match",
        default=None,
        help="the same run's per_match.csv.gz: adds the anchor-mix context (never decides)",
    )
    p.add_argument("--out-json", default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = json.loads(Path(args.report).read_text()) if args.report else None
        scores = load_scores(args.scores)
        anchored = load_anchor_mask(args.per_match, scores) if args.per_match else None
        result = run(scores, report, anchored)
    except Exception as exc:
        logger.error("ab_soccer_derived_vs_constant failed: %s", exc, exc_info=True)
        return 1
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(ht.clean_json(result), indent=2, allow_nan=False))
    print(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
