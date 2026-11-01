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
  single line -0.00212, a goal-difference RPS -0.00047), and the replayed AH
  recs of the two arms agree on ~80% of picks with the same fair-close CLV
  (-3.2% each; a random side at the best price is -3.6%).

FORWARD WINDOW (pinned)
  Only finished soccer matches kicking off at or after FORWARD_START
  (2026-11-02T00:00Z, after this commit) are scored; an earlier --start is
  refused, and a guard refuses the run if an earlier match reaches scoring.

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
                      measured effect is about a fifth of the threshold; the
                      in-sample best rescaling of the refit's correction
                      reaches -0.00118).
  brier_significance  (POLICY CHANGE) the same units and A2 rule, the pooled
                      delta < 0 with the t(n_folds-1) clustered upper bound
                      < 0, and NO -0.005 threshold: "strictly better and
                      1x2-neutral" for a side artifact that never touches the
                      1x2, over/under or any other market.
  money_clv           (POLICY CHANGE) the AH recommendation logic replayed for
                      both arms at the last pipeline tick before kickoff
                      (ab_soccer_ah_recs_replay), on matches with pre-match AH
                      odds that pass the eligibility gate, served = the
                      stored row the generator would have read at that tick
                      (the newest created before it), refit reconciled to
                      that row's 1x2 and anchor: SHIP only if the refit's
                      recs beat the served recs on fair-close CLV per rec
                      (odds taken x the >= 3-book power-de-vigged consensus at
                      kickoff - 1), the match-clustered bootstrap 95% lower
                      bound of the difference > 0. P/L is reported as context.

DECISION POINT (from the 2026-11-01 replay's numbers)
  brier_house / brier_significance: at least 6 monthly folds AND at least
    2,400 scored matches (the old window's per-match SD of the pooled delta,
    0.0189, needs 2,241 matches for 80% power against the measured -0.00112;
    6 folds keep the t(5) multiplier at 2.57). At ~930 matches a month with a
    stored AH row that is the folds 2026-11..2027-04 (~5,500 matches),
    decidable from 2027-05-01.
  money_clv: at least 12 monthly folds AND at least 7,000 matches with
    pre-match AH odds that pass the eligibility gate (the old window needs
    6,934 for the measured stake-P/L difference and 5,146 for the flat one;
    its CLV difference was -0.0004 a match, i.e. none). At 275-460 such
    matches a month that is 15-25 months.
  Below the point, or with no approved metric, the run prints counts only.

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

MECHANICS (before this commit)
  One smoke (--mechanics --limit 20) on the replica of the 2026-11-01
  21:09:21Z read-only export: 0 forward matches (none had kicked off),
  counts only; --start 2026-10-01 was refused. No delta of the forward
  window exists yet.

Read-only: SELECTs, plus temp files for the fit round-trip. Usage (api
container, after the decision point and after an approving commit):
    python /app/scripts/ab_soccer_ah_forward_gate.py --end <last day> --out-json /tmp/ah_forward.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
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
APPROVED_GATE_METRIC: Optional[str] = None  # PROPOSAL: set by an operator-approved commit, before the decision
GATE_METRICS: Dict[str, Dict[str, Any]] = {
    "brier_house": {"policy_change": False, "min_folds": 6, "min_n": 2400, "population": "stored_ah"},
    "brier_significance": {"policy_change": True, "min_folds": 6, "min_n": 2400, "population": "stored_ah"},
    "money_clv": {"policy_change": True, "min_folds": 12, "min_n": 7000, "population": "eligible_ah_odds"},
}
HOUSE_THRESHOLD = -0.005
INVARIANT_ATOL = 1e-9
SHAPED_UNITS = rp.SHAPED_UNITS
INVARIANT_LINES = ftdc.AH_INVARIANT_LINES
POOLED = "pooled_shaped_mean"


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


def decision_gate(metric: Optional[str], n: int, n_folds: int) -> Tuple[bool, str]:
    """(may decide, why not). Refuses with no approved metric, an unknown
    metric, or below the metric's pre-registered folds and n."""
    if metric is None:
        return False, "PROPOSAL: no gate metric is approved (APPROVED_GATE_METRIC is None) — no decision, no delta"
    if metric not in GATE_METRICS:
        return False, f"unknown gate metric {metric!r}"
    spec = GATE_METRICS[metric]
    if n_folds < spec["min_folds"] or n < spec["min_n"]:
        return False, (
            f"{metric}: {n} scored / {n_folds} fold(s) < the pre-registered {spec['min_n']} / {spec['min_folds']} — "
            "no decision, no delta"
        )
    return True, ""


def handicap_vector(probs: Dict[str, float], line: float) -> np.ndarray:
    """(home, away, push) of one line from a stored-form asian_handicap dict."""
    lbl = ftdc.fmt_line(line)
    try:
        return np.array([float(probs[f"{lbl}_{s}"]) for s in ("home", "away", "push")])
    except KeyError as exc:
        raise RuntimeError(f"asian_handicap dict lacks line {lbl}: {exc}") from exc


def score_match(
    served: Dict[str, float], refit: Dict[str, float], home: int, away: int
) -> Dict[str, Tuple[float, float]]:
    """{unit: (served Brier, refit Brier)} for the 12 shaped lines and their
    mean, graded by grading_outcomes' handicap settlement. Raises when an
    invariant line differs between the arms beyond INVARIANT_ATOL."""
    for line in INVARIANT_LINES:
        gap = float(np.max(np.abs(handicap_vector(served, line) - handicap_vector(refit, line))))
        if not math.isfinite(gap) or gap > INVARIANT_ATOL:
            raise RuntimeError(f"invariant AH line {line} differs between served and refit by {gap:.2e}")
    out: Dict[str, Tuple[float, float]] = {}
    for unit, line in zip(SHAPED_UNITS, rp.SHAPED_LINES):
        y = np.array(ftdc.handicap_truth(int(home) - int(away), line))
        out[unit] = (
            float(np.sum((handicap_vector(served, line) - y) ** 2)),
            float(np.sum((handicap_vector(refit, line) - y) ** 2)),
        )
    out[POOLED] = (
        float(np.mean([out[u][0] for u in SHAPED_UNITS])),
        float(np.mean([out[u][1] for u in SHAPED_UNITS])),
    )
    return out


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


def decide_money(boot: Dict[str, Any], disqualified: Optional[str]) -> Dict[str, Any]:
    if disqualified:
        return {"ship": False, "reason": f"refit disqualified: {disqualified}"}
    d = boot.get("diff_clv_fair") or {}
    lo = d.get("lo95")
    ship = lo is not None and lo > 0
    return {"ship": ship, "reason": f"fair-close CLV difference {d.get('estimate')} (95% lower bound {lo})"}


# ── Run ──────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> Dict[str, Any]:
    members._add_src_paths()
    start = check_window(args.start)
    metric = APPROVED_GATE_METRIC
    frame = ftdc.load_frame(args.database_url)
    frame["s_home"], frame["s_away"], frame["s_ok"] = ftdc.settlement(frame)
    end_day = args.end or frame["match_date"].max().strftime("%Y-%m-%d")
    end_excl = pd.Timestamp(end_day, tz="UTC") + pd.Timedelta(days=1)
    counts: Dict[str, Any] = {
        "window": [str(start), end_day],
        "metric": metric,
        "no_stored": 0,
        "folds": [],
        "money_matches": 0,
        "money_ineligible": 0,
    }
    if end_excl <= start:
        counts.update(n_scored=0, disqualified=None)
        counts["decision"] = {"decidable": False, "ship": False, "reason": decision_gate(metric, 0, 0)[1]}
        return counts
    ev_all = frame[(frame["match_date"] >= start) & (frame["match_date"] < end_excl) & frame["s_ok"]]
    db = rp.load_db(args.database_url, start, end_excl)
    stored_by = {str(k): g for k, g in db["stored"].groupby("match_id")}
    odds_by: Dict[str, List[Dict[str, Any]]] = {}
    for rec in db["odds"].to_dict("records"):
        odds_by.setdefault(str(rec["match_id"]), []).append(rec)
    snaps = rp.snapshot_index(db["snaps"].to_dict("records"))
    hist = rp.TeamHistory(db["history"])
    rows: List[Dict[str, Any]] = []
    recs: Dict[str, List[rp.Rec]] = {"served": [], "refit": []}
    money_ids: List[str] = []
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
                h_lam, a_lam = refit.lambdas_for_match(r.home_team, r.away_team, league=r.league_id)
                st = rp.stored_inputs(stored_by.get(mid), mid, r.match_date)
                if st is None:
                    counts["no_stored"] += 1
                else:
                    refit_ah = rp.derive_ah(h_lam, a_lam, rho, st["target"], st["anchor"])
                    sc = score_match(st["probabilities"], refit_ah, r.s_home, r.s_away)
                    row = {"match_id": mid, "fold": fold.label, "kickoff": r.match_date}
                    for k, v in sc.items():
                        row[f"{k}::served"], row[f"{k}::refit"] = v
                    rows.append(row)
                if mid in odds_by:
                    money_match(r, h_lam, a_lam, rho, odds_by[mid], snaps, hist, stored_by, db["bankroll"], recs,
                                money_ids, counts)  # fmt: skip
    n_brier, n_money = len(rows), len(money_ids)
    counts.update(n_scored=n_brier, money_matches=n_money, disqualified=disq)
    n = n_money if metric == "money_clv" else n_brier
    ok, why = decision_gate(metric, n, len(counts["folds"]))
    if args.mechanics or not ok:
        reason = "mechanics run: no delta read" if args.mechanics else why
        counts["decision"] = {"decidable": False, "ship": False, "reason": reason}
        return counts
    out = dict(counts)
    if metric == "money_clv":
        ta = rp.per_match_totals(recs["served"], money_ids)
        tb = rp.per_match_totals(recs["refit"], money_ids)
        boot = rp.paired_bootstrap(ta, tb)
        out["money"] = {"served": rp.money_summary(recs["served"]), "refit": rp.money_summary(recs["refit"])}
        out["bootstrap"] = boot
        out["per_unit"] = {}
        out["decision"] = {"decidable": True, **decide_money(boot, disq)}
        return out
    df = pd.DataFrame(rows)
    check_scored(df["kickoff"].tolist())
    folds_arr = df["fold"].to_numpy()
    out["per_unit"] = {
        u: ftdc.pair(df[f"{u}::refit"].to_numpy(), df[f"{u}::served"].to_numpy(), folds_arr)
        for u in (*SHAPED_UNITS, POOLED)
    }
    out["decision"] = {"decidable": True, **decide_brier(out["per_unit"], str(metric), disq)}
    return out


def money_match(
    r: Any,
    h_lam: float,
    a_lam: float,
    rho: float,
    rows: List[Dict[str, Any]],
    snaps: Dict[Any, Any],
    hist: Any,
    stored_by: Dict[str, pd.DataFrame],
    bankroll: float,
    recs: Dict[str, List[Any]],
    money_ids: List[str],
    counts: Dict[str, Any],
) -> None:
    """money_clv for one match with pre-match AH odds: the generator replayed
    at the last tick for the stored probabilities and for the refit reconciled
    to the stored 1x2 and anchor of the row the generator would have read."""
    mid = str(r.match_id)
    tick = rp.last_tick_before(r.match_date)
    gen = rp.gr.parse_args([])
    if (
        hist.count_before(r.home_team_id, tick) < gen.min_team_history
        or hist.count_before(r.away_team_id, tick) < gen.min_team_history
    ):
        counts["money_ineligible"] += 1
        return
    st = rp.stored_inputs(stored_by.get(mid), mid, tick)
    if st is None:
        return
    state = rp.quote_state(rows, snaps, tick)
    close_state = rp.quote_state(rows, snaps, r.match_date - pd.Timedelta(microseconds=1))
    arms = {"served": st["probabilities"], "refit": rp.derive_ah(h_lam, a_lam, rho, st["target"], st["anchor"])}
    for arm, probs in arms.items():
        out, _ = rp.replay_match(mid, probs, state, bankroll, gen.ev_threshold, gen.prob_floor)
        for rec in out:
            verdict, pl, flat = rp.settle(rec.selection, rec.odds, rec.stake, r.s_home, r.s_away, {})
            p_close = rp.consensus_prob(close_state, rec.side, rec.line)
            rec.extra.update(
                {
                    "verdict": verdict,
                    "pl": pl,
                    "flat": flat,
                    "clv_fair": (rec.odds * p_close - 1.0) if p_close is not None else None,
                }
            )
        recs[arm].extend(out)
    money_ids.append(mid)


def format_report(r: Dict[str, Any]) -> str:
    d = r["decision"]
    L = [
        f"Soccer AH forward gate (PROPOSAL; approved metric: {r['metric']}): window {r['window'][0]}..{r['window'][1]}",
        f"scored {r['n_scored']} matches in folds {r['folds'] or 'none'}; no stored AH row {r['no_stored']}; "
        f"money population {r['money_matches']} (ineligible {r['money_ineligible']}); "
        f"disqualified {r['disqualified'] or 'no'}",
    ]
    if not d.get("decidable"):
        L.append(f"NO DECISION: {d['reason']}")
        return "\n".join(L)
    for u, p in r["per_unit"].items():
        L.append(ftdc._pair_line(u, p))
    for arm, m in (r.get("money") or {}).items():
        L.append(rp._money_line(arm, m))
    L.append(f"DECISION ({r['metric']}): {'SHIP' if d['ship'] else 'DO-NOT-SHIP'} — {d['reason']}")
    return "\n".join(L)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=None, help=f"refused before {FORWARD_START.date()}")
    p.add_argument("--end", default=None, help="last kickoff day (default: newest finished match)")
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
