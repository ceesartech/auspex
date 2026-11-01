"""PROPOSAL — forward-window pre-registration for an asian-handicap-only gate of
a monthly-refit soccer Dixon-Coles side artifact.

STATUS: PROPOSAL, NOT APPROVED. The gate metric needs operator approval.
APPROVED_GATE_METRIC is None, and while it is None this script refuses to
decide or to print any delta. CLAUDE.md non-negotiable 2 fixes the house gate
at dBrier <= -0.005, clear of noise; GATE_METRICS also lists two metrics that
would be a POLICY CHANGE for the operator. Approving means a commit that sets
APPROVED_GATE_METRIC to one key of GATE_METRICS, made before the decision
point below; changing it after any delta of this window has been read voids
the pre-registration.

WHY THIS EXISTS (2026-11-01)
  The derived-market gate (scripts/ab_soccer_ft_dixon_coles.py) was
  DO-NOT-SHIP: a monthly refit improved the served over_2.5 by -0.00375, short
  of -0.005. asian_handicap is the one rec-emitting soccer market that
  artifact shapes, and the refit improved all 12 shaped AH lines. Reading
  the AH lines after that result and gating on them would be post hoc, so a
  fresh, forward window is the only honest route. scripts/
  ab_soccer_ah_recs_replay.py measured what to expect on the old window
  (RESULT there): pooled shaped-line Brier -0.00112 +/- 0.00012 (the best
  single line -0.00212, a goal-difference RPS -0.00047); the replayed AH
  recs of the two arms agree on about 2 picks in 3; and an early bet (6 h
  before kickoff) of either arm finishes -2.3% per rec against the last
  observed fair price (refit - served -0.03 pts, SE 0.25).

FORWARD WINDOW AND DECISION SAMPLE (pinned)
  Only finished soccer matches kicking off at or after FORWARD_START
  (2026-11-02T00:00Z) are scored; an earlier --start is refused, and a guard
  refuses the run if an earlier match reaches scoring. The decision sample
  is EXACTLY the 12 monthly folds DECISION_FOLDS = 2026-11..2027-10: a match
  after 2027-10-31 is never scored, whatever --end says, and no decision is
  read before the data covers 2027-10-31 or on any other fold set (no
  optional stopping: nobody can read at 6 folds, then 7, and stop at the
  first SHIP).

ARMS
  served  production's stored asian_handicap probabilities: the newest
          model_name 'ensemble' asian_handicap row created before kickoff
          (as stored; its anchor, metadata.market_over_2.5 when
          metadata.reconcile is '1x2+ou2.5', is the one production used).
  refit   DIXON_COLES_CONFIG fitted on every finished soccer match before the
          kickoff month's first day (monthly walk-forward, round-tripped
          through save() and the side-artifact loader), its lambdas_for_match
          (home name, away name, league_id) reconciled by
          market_derivation.derive_from_lambdas to the SAME stored 1x2 (the
          match_result row of the served row's model_version) and the SAME
          stored anchor. A fit the half-time trainer's fit_refusal would
          refuse disqualifies the arm (DO-NOT-SHIP).
  Fidelity: AH 0, +-0.25, +-0.5 are functions of the 1x2 after the
  reconcile, so both arms must serve them identically (1e-9), or the run
  aborts (a stored AH row that does not come from its stored 1x2 is a bug,
  not data: all 2,826 stored rows of 2026-06..10 agree to 6.7e-16).

UNITS (Brier)
  The 12 shaped AH lines (+-0.75..+-2), each a 3-way Brier over the effective
  home / away / push masses against grading_outcomes' settlement of the line
  on the 90-minute score; and their per-match MEAN (pooled shaped lines).

GATE_METRICS (the operator picks ONE before the decision)
  brier_house         (no policy change) CLAUDE.md §4.3 on the pooled shaped
                      lines: ft.ship_verdict (dBrier <= -0.005, per-match and
                      fold-clustered z=2 upper bounds < 0) AND the
                      t(n_folds-1) clustered upper bound < 0; AND no shaped
                      line significantly worse under rule A2 (a line regresses
                      when its fold-clustered t lower bound of dBrier is > 0).
                      Expected outcome from the old window: DO-NOT-SHIP (the
                      measured effect is about a quarter of the threshold;
                      the in-sample best rescaling of the refit's correction
                      reaches -0.00118, so no power exists at any n).
  brier_significance  (POLICY CHANGE) the same units and A2 rule, the pooled
                      delta < 0 with the t(n_folds-1) clustered upper bound
                      < 0, and NO -0.005 threshold: "strictly better and
                      1x2-neutral" for a side artifact that never touches the
                      1x2, over/under or any other market.
  NO MONEY METRIC. The 2026-11-01 proposal listed money_clv ("fair-close
  CLV" of the recs replayed at the last tick). Review showed it measures
  nothing: no AH quote is captured after the last tick, so its "close" was
  the bet's own moment and the number was the market-implied EV at bet time.
  The replay's genuine closing-line readout (the early bet, 6 h out, against
  the last tick's consensus) and realised P/L were then measured on THIS
  gate's pair (stored vs refit_on_stored): early-bet CLV differs by -0.0002
  a match (SD 0.0146: 36,500 matches for 80% power, ~9 years), stake P/L by
  +0.006 a match (7.6 million matches). No money metric could decide here,
  so none is offered.

DECISION POINT (planned on the replay's Part A2 / Part D, 2026-11-01)
  Both metrics: the 12 pinned folds complete AND at least MIN_N = 5,300
  scored matches. The plan is THIS gate's pair on the old window — 2,757
  matches, pooled -0.00119 +/- 0.00043, per-match SD 0.0228 — under THIS
  gate's rule, the t(k-1) upper bound of the unweighted mean of the k fold
  means: each fold mean is N(delta, SD^2 / n_fold + tau2) with tau2 = 2.1e-7
  (the window pair's 31 folds; the pair's own from 2026-07-14 on is 0), and
  folds sized by the 2024-26 calendar at 822 matches an Aug-Oct month (the
  June fold ~150 matches, July ~400, Dec/Jan ~600-650). Power at the
  projected sizes: 6 folds (2026-11..2027-04, ~4,800 matches) 70%; 7 folds
  81%, but the small June fold drops 8 folds to 74%; >= 80% from 10 folds
  on; 12 folds (~8,800 matches, a full season, no season mix) 94% — 66% if
  tau2 were the pair's all-fold 1.5e-6, which the constant-prior incident's
  66-match June fold inflates. MIN_N is the 12-fold floor: below 5,224
  matches (the projection scaled down) the power falls under 80%, so a
  shortfall there is NO DECISION, not a weaker test. Decidable from
  2027-11-01. Below the point, or with no approved metric, the run prints
  counts only.

WHAT A SHIP WOULD DEPLOY (the AH-only design; nothing here deploys it)
  1. Artifact. dixon_coles_soccer_ah: DIXON_COLES_CONFIG fitted on every
     finished soccer match before the run (the refit arm above), written by
     a new scripts/train_soccer_ah_dixon_coles.py modelled on
     train_halftime_dixon_coles.py: htrainer.fit_refusal before anything is
     written (unconverged, level bias beyond 2%, |rho| > 0.1, an unservable
     fixture), then model.bin + manifest.json (run id, git sha, rows, newest
     match, convergence, level, rho, sha256) written atomically to
     /app/models/staging-ah/dixon_coles_soccer_ah/1.0.0. Never to
     /app/models/staging (retrain_models' swap_production copies everything
     there into production unchecked) nor to production. The 1x2 ensemble
     and its own dixon_coles member are untouched.
  2. Schedule. services/data-ingestion/dags/retrain_ah_dc.py, the shape of
     retrain_halftime_dc.py: monthly on the 1st (06:30 UTC, after the
     half-time refit at 05:30 and clear of the Sunday retrain_models restart),
     outside retrain_models: train_ah (--run-id <run>) >> promote_ah
     (promote_dc_model.py --model ah --expect-run-id <run>
     --rollback-on-verify-failure), ALL_SUCCESS, Telegram on failure. It
     starts paused; the operator unpauses it after the first manual promote.
  3. Promotion. scripts/promote_dc_model.py REGISTRY gets
     "ah": DCModelSpec(name="dixon_coles_soccer_ah",
         staging_dir=/app/models/staging-ah/dixon_coles_soccer_ah/1.0.0,
         production_dir=/app/models/production/dixon_coles_soccer_ah/1.0.0,
         expected_config_name=<the config name the trainer writes>,
         serve_loader="_load_ah_dixon_coles", smoke_market_group="soccer",
         smoke_market=<a home/draw/away market derive_markets("soccer")
         returns>).
     Its existing checks then apply unchanged: manifest sha256 and run id,
     "nothing to promote" when staged == production, config name, a fit on
     every row, per-league baselines, finite rho, no coverage regression
     (teams >= 98%, leagues >= production's - 1), the same data give or take
     a month (level within 5%, rows >= 99%), a serve-loader smoke; then a
     sha256-verified backup to /app/models/backup-dixon_coles_soccer_ah-
     <UTC stamp>/, an atomic swap, and verification through the serve loader.
  4. Serve path (scripts/precompute_predictions.py). A loader
     _load_ah_dixon_coles() = _load_dixon_coles_artifact("/app/models/
     production/dixon_coles_soccer_ah/1.0.0/model.bin", "AH"), called once per
     run. Per match, the full catalogue is derived from the ensemble member
     exactly as today. When the AH artifact loaded, asian_handicap alone is
     re-derived: derive_from_lambdas(AH lambdas_for_match(home, away,
     league=league_id), AH rho, target_1x2 = the same proba, target_over_25
     = the same p_over_mkt already read for the match)["asian_handicap"]. It
     is stored with a second store_market_predictions call whose metadata
     adds {"ah_source": "dixon_coles_soccer_ah", "ah_sha256": <12 hex>}; every
     other market, the 1x2 and over/under are byte-identical to today.
     Without the artifact the member's asian_handicap is stored as today,
     stamped "ah_source": "ensemble_member", and the loader's ERROR line
     says why. Ship the serve-path change only after the first verified
     promotion, or the loader logs ERROR on every tick until then.
  5. Rollback. The rollback command promote_dc_model.py prints (the last
     line of every promote and of every failure) restores the verified
     backup atomically; the DAG's --rollback-on-verify-failure does it
     unattended when verification fails. Fastest manual off switch: move the
     production file aside; the next 15-minute tick serves the member's
     asian_handicap again, with the ERROR line. No DB change either way:
     pending AH recs re-price on the next tick (generate_recommendations
     deletes and re-writes a match's pending recs), and the stored rows'
     ah_source says which artifact priced each one.

MECHANICS (before each commit)
  d773da4: one smoke (--mechanics --limit 20) on the replica of the
  2026-11-01 21:09:21Z read-only export: 0 forward matches (none had kicked
  off), counts only; --start 2026-10-01 was refused. This revision (review
  fixes, 2026-11-01 ~23:10Z, before any forward match kicked off): the same
  smoke, 0 forward matches. No delta of the forward window exists.

Read-only: SELECTs, plus temp files for the fit round-trip. Usage (api
container, from 2027-11-01 and after an approving commit; --end is not
needed, the window stops at 2027-10-31):
    python /app/scripts/ab_soccer_ah_forward_gate.py --out-json /tmp/ah_forward.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_nhl_dixon_coles as nhl  # noqa: E402
import ab_soccer_ah_recs_replay as rp  # noqa: E402
import ab_soccer_dixon_coles_refit as ft  # noqa: E402
import ab_soccer_ensemble_members as members  # noqa: E402
import ab_soccer_ft_dixon_coles as ftdc  # noqa: E402
import train_halftime_dixon_coles as htrainer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("ab_soccer_ah_forward_gate")

FORWARD_START = pd.Timestamp("2026-11-02T00:00:00Z")
DECISION_FOLDS: Tuple[str, ...] = tuple(
    f"{y}-{m:02d}" for y, m in [(2026, 11), (2026, 12)] + [(2027, i) for i in range(1, 11)]
)
DECISION_LAST_DAY = "2027-10-31"
MIN_N = 5300  # the 12-fold n floor for 80% power (replay Part D: 5,224), rounded up
APPROVED_GATE_METRIC: Optional[str] = None  # PROPOSAL: set by an operator-approved commit, before the decision
GATE_METRICS: Dict[str, Dict[str, Any]] = {
    "brier_house": {"policy_change": False, "decision_folds": DECISION_FOLDS, "min_n": MIN_N},
    "brier_significance": {"policy_change": True, "decision_folds": DECISION_FOLDS, "min_n": MIN_N},
}
HOUSE_THRESHOLD = -0.005
SHAPED_UNITS = rp.SHAPED_UNITS
POOLED = rp.POOLED


# ── Pure: the pinned rules ───────────────────────────────────────────


def check_window(start: Optional[str]) -> pd.Timestamp:
    """The window start: FORWARD_START; any earlier --start is refused."""
    if start is None:
        return FORWARD_START
    t = pd.Timestamp(start)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    if t < FORWARD_START:
        raise RuntimeError(f"the forward window starts at {FORWARD_START}; --start {start} refused")
    return t


def check_scored(kickoffs: Sequence[Any]) -> None:
    """Refuse loudly if a match before FORWARD_START reached scoring."""
    early = [k for k in kickoffs if pd.Timestamp(k) < FORWARD_START]
    if early:
        raise RuntimeError(f"{len(early)} match(es) before {FORWARD_START} reached scoring — refused")


def decision_end(end: Optional[str], newest: pd.Timestamp) -> Tuple[str, bool]:
    """(the last day scored, whether the pinned decision window is complete).
    Scoring never runs past DECISION_LAST_DAY whatever --end says; the window
    is complete only when the scored span AND the data reach that day."""
    last = pd.Timestamp(DECISION_LAST_DAY, tz="UTC")
    want = pd.Timestamp(end, tz="UTC") if end else newest.tz_convert("UTC").normalize()
    day = min(want, last)
    return day.strftime("%Y-%m-%d"), bool(day >= last and newest >= last)


def decision_gate(metric: Optional[str], n: int, folds: Sequence[str], complete: bool = True) -> Tuple[bool, str]:
    """(may decide, why not). Refuses with no approved metric, an unknown
    metric, an incomplete window, a fold set other than the pinned one, or
    fewer scored matches than the pre-registered floor."""
    if metric is None:
        return False, "PROPOSAL: no gate metric is approved (APPROVED_GATE_METRIC is None) — no decision, no delta"
    if metric not in GATE_METRICS:
        return False, f"unknown gate metric {metric!r}"
    spec = GATE_METRICS[metric]
    if not complete:
        return False, f"{metric}: the data do not yet cover {DECISION_LAST_DAY} — no decision, no delta"
    if tuple(folds) != tuple(spec["decision_folds"]):
        return False, (
            f"{metric}: scored folds {list(folds)} are not the pinned {list(spec['decision_folds'])} — "
            "no decision, no delta"
        )
    if n < spec["min_n"]:
        return False, (
            f"{metric}: {n} scored matches < the pre-registered floor {spec['min_n']} (under 80% power) — "
            "no decision, no delta; re-register"
        )
    return True, ""


# The per-match scoring is the replay's (scripts/ab_soccer_ah_recs_replay.py
# Part A2 plans this gate's decision point on exactly this function).
handicap_vector = rp.handicap_vector
score_match = rp.score_ah_pair


def decide_brier(per_unit: Dict[str, Dict[str, Any]], metric: str, disqualified: Optional[str]) -> Dict[str, Any]:
    """brier_house / brier_significance on paired refit - served deltas."""
    if disqualified:
        return {"ship": False, "reason": f"refit disqualified: {disqualified}"}
    p = per_unit[POOLED]
    regs = {u: why for u in SHAPED_UNITS if (why := nhl.regression(per_unit[u])) is not None}
    if metric == "brier_house":
        v = ftdc.ship_strict(p)
        primary_ok, line = bool(v["ship_strict"]), v["line"]
    else:
        t_up = p.get("upper_95_cluster_t")
        primary_ok = p.get("delta") is not None and p["delta"] < 0 and t_up is not None and t_up < 0
        line = f"pooled dBrier {p.get('delta')}, t upper bound {t_up}"
    ship = primary_ok and not regs
    reason = "passes" if ship else ("primary fails: " + line if not primary_ok else f"regressions: {sorted(regs)}")
    return {"ship": ship, "reason": reason, "regressions": regs}


# ── Run ──────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> Dict[str, Any]:
    members._add_src_paths()
    start = check_window(args.start)
    metric = APPROVED_GATE_METRIC
    frame = ftdc.load_frame(args.database_url)
    frame["s_home"], frame["s_away"], frame["s_ok"] = ftdc.settlement(frame)
    end_day, complete = decision_end(args.end, frame["match_date"].max())
    end_excl = pd.Timestamp(end_day, tz="UTC") + pd.Timedelta(days=1)
    counts: Dict[str, Any] = {
        "window": [str(start), end_day],
        "window_complete": complete,
        "metric": metric,
        "no_stored": 0,
        "folds": [],
    }
    if end_excl <= start:
        counts.update(n_scored=0, disqualified=None)
        counts["decision"] = {"decidable": False, "ship": False, "reason": decision_gate(metric, 0, [], False)[1]}
        return counts
    ev_all = frame[(frame["match_date"] >= start) & (frame["match_date"] < end_excl) & frame["s_ok"]]
    db = rp.load_db(args.database_url, start, end_excl)
    stored_by = {str(k): g for k, g in db["stored"].groupby("match_id")}
    rows: List[Dict[str, Any]] = []
    disq: Optional[str] = None
    with tempfile.TemporaryDirectory() as tmp:
        for fold in ft.build_folds(start, end_day):
            fit_df, ev = ft.split_fold(frame, fold)
            ev = ev[ev["match_id"].isin(ev_all["match_id"])]
            if args.limit:
                ev = ev.head(int(args.limit))
            if ev.empty:
                continue
            check_scored(ev["match_date"].tolist())
            members.check_fold_disjoint(fit_df, ev, fold)
            refit, res = ftdc.fit_refit(fit_df, tmp, fold.label)
            why = htrainer.fit_refusal(res, refit, list(zip(ev["home_team"], ev["away_team"], ev["league_id"])))
            if why is not None:
                disq = f"fold {fold.label}: {why}"
                logger.error("refit DISQUALIFIED at %s", disq)
                break
            counts["folds"].append(fold.label)
            rho = float(refit.rho or 0.0)
            for r in ev.itertuples():
                mid = str(r.match_id)
                st = rp.stored_inputs(stored_by.get(mid), mid, r.match_date)
                if st is None:
                    counts["no_stored"] += 1
                    continue
                h_lam, a_lam = refit.lambdas_for_match(r.home_team, r.away_team, league=r.league_id)
                refit_ah = rp.derive_ah(h_lam, a_lam, rho, st["target"], st["anchor"])
                sc = score_match(st["probabilities"], refit_ah, r.s_home, r.s_away)
                row = {"match_id": mid, "fold": fold.label, "kickoff": r.match_date}
                for k, v in sc.items():
                    row[f"{k}::served"], row[f"{k}::refit"] = v
                rows.append(row)
    counts.update(n_scored=len(rows), disqualified=disq)
    if args.mechanics:
        counts["decision"] = {"decidable": False, "ship": False, "reason": "mechanics run: no delta read"}
        return counts
    if disq:  # a refit the trainer would refuse never ships; it is a verdict only once the gate may decide
        counts["decision"] = {
            "decidable": metric in GATE_METRICS and complete,
            "ship": False,
            "reason": f"refit disqualified: {disq}",
        }
        return counts
    ok, why = decision_gate(metric, len(rows), counts["folds"], complete)
    if not ok:
        counts["decision"] = {"decidable": False, "ship": False, "reason": why}
        return counts
    out = dict(counts)
    df = pd.DataFrame(rows)
    check_scored(df["kickoff"].tolist())
    folds_arr = df["fold"].to_numpy()
    out["per_unit"] = {
        u: ftdc.pair(df[f"{u}::refit"].to_numpy(), df[f"{u}::served"].to_numpy(), folds_arr)
        for u in (*SHAPED_UNITS, POOLED)
    }
    out["decision"] = {"decidable": True, **decide_brier(out["per_unit"], str(metric), disq)}
    return out


def format_report(r: Dict[str, Any]) -> str:
    d = r["decision"]
    L = [
        f"Soccer AH forward gate (PROPOSAL; approved metric: {r['metric']}): window {r['window'][0]}..{r['window'][1]}",
        f"scored {r['n_scored']} matches in folds {r['folds'] or 'none'} (pinned {DECISION_FOLDS[0]}.."
        f"{DECISION_FOLDS[-1]}, floor {MIN_N}; window complete: {r['window_complete']}); no stored AH row "
        f"{r['no_stored']}; disqualified {r['disqualified'] or 'no'}",
    ]
    if not d.get("decidable"):
        L.append(f"NO DECISION: {d['reason']}")
        return "\n".join(L)
    for u, p in (r.get("per_unit") or {}).items():
        L.append(ftdc._pair_line(u, p))
    L.append(f"DECISION ({r['metric']}): {'SHIP' if d['ship'] else 'DO-NOT-SHIP'} — {d['reason']}")
    return "\n".join(L)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=None, help=f"refused before {FORWARD_START.date()}")
    p.add_argument(
        "--end",
        default=None,
        help=f"last kickoff day scored (default: newest finished match); never past {DECISION_LAST_DAY}",
    )
    p.add_argument("--limit", type=int, default=0, help="cap matches per fold (mechanics smoke)")
    p.add_argument("--mechanics", action="store_true", help="counts only: never read a delta")
    p.add_argument("--out-json", default=None)
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    try:
        report = run(args)
    except Exception as exc:
        logger.error("ab_soccer_ah_forward_gate failed: %s", exc, exc_info=True)
        return 1
    if args.out_json and report["decision"].get("decidable"):
        from ab_halftime_dixon_coles import clean_json

        Path(args.out_json).write_text(json.dumps(clean_json(report), indent=2, allow_nan=False, default=str))
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
