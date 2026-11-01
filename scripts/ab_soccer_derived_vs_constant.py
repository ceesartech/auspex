"""Which stored soccer derived markets lose to a simple per-league constant:
the display rule, read off scripts/ab_soccer_ft_dixon_coles.py's per-row scores.

WHY THIS EXISTS
---------------
The 2026-11-01 gate for the full-time Dixon-Coles that serves the derived
soccer markets (scripts/ab_soccer_ft_dixon_coles.py, DO-NOT-SHIP) found, as
context, that the served derived BTTS is significantly worse than a per-league
constant (live regime +0.00131 Brier, fold-clustered t lower bound +0.00072)
and that its over-2.5 is no better than one (-0.00043, inside the noise). It
compared only those two markets and the correct-score grid with a constant.

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
  Per type, served - base_league (per-match delta +/- SE; fold-clustered t
  95% interval):
    HIDE odd_even  live  +0.00039 +/- 0.00017, t [+0.00013, +0.00098]
                   proxy +0.00036 +/- 0.00016, t [+0.00010, +0.00094]
          Served's mean P(odd) is 0.488 against 0.499 realised and the
          constant's 0.492. ESPN era (n = 3,211): +0.00012 +/- 0.00047, inside
          the noise. Its recommendations were already switched off.
    keep btts      live  +0.00131 +/- 0.00030, t [+0.00072, +0.00205]
                   proxy -0.00027 +/- 0.00034, t [-0.00067, +0.00079]
          Loses in the live regime only, so it stays visible under the rule.
          ESPN era, the months the live anchor actually reaches: live -0.00006
          +/- 0.00071 (a tie). Context: against base_league_dw it loses in both
          regimes (+0.00195 / +0.00037, lower bounds +0.00137 / +0.00003).
    keep over_under   live -0.00072 +/- 0.00091, t [-0.00291, +0.00283];
                      proxy -0.00942, t upper -0.00367. No line loses on its
                      own in either regime (over_2.5 live -0.00043).
    keep total_goals  live +0.00036, t [-0.00028, +0.00149]; proxy -0.00163,
                      t upper -0.00006.
    keep, served better beyond the noise in both regimes (live delta):
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

Usage (after the harness run; no database needed):
    python scripts/ab_soccer_derived_vs_constant.py --scores <out-dir>/scores.npz \\
        --report <out-json> --out-json /tmp/derived_vs_constant.json
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


def run(scores: Dict[str, np.ndarray], report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
    p.add_argument("--out-json", default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = json.loads(Path(args.report).read_text()) if args.report else None
        result = run(load_scores(args.scores), report)
    except Exception as exc:
        logger.error("ab_soccer_derived_vs_constant failed: %s", exc, exc_info=True)
        return 1
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(ht.clean_json(result), indent=2, allow_nan=False))
    print(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
