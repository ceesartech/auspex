"""A/B gate for the FULL-TIME soccer Dixon-Coles that the derived soccer
markets are served from (dixon_coles_soccer_match_result, as
scripts/precompute_predictions.py derives from it).

WHAT IS BEING TESTED
--------------------
services/ml-models/src/training/train_all_models.py fits every soccer base
model on its temporal TRAIN split (the oldest 70% of the frame), and the
weekly retrain_models DAG swaps the result into production every Sunday. So
the served Dixon-Coles is always a fit of the oldest 70%: the 2026-11-01
artifact (sha256 02141a6a...) knows 951 teams and its data ends in 2021-10.
Nothing refits it on recent matches. The 2026-09-04 gate
(scripts/ab_soccer_dixon_coles_refit.py; audit doc "Update (2026-09-04) —
STEP 3"; fitter fix 234bbb1) validated a MONTHLY walk-forward refit with the
fixed fitter on over-2.5 (-0.00698 vs the then-served artifact), but what
serves is the trainer's split-only fit: the validated object was never
deployed. The soccer_members_ab stream measured the refit at -0.0039 on the
unanchored over-2.5 (information only). This harness gates the deployable
object, scored exactly as production serves it.

SERVE PATH (mapped from the code at 5092ab3, 2026-11-01)
  precompute_predictions.run(), every 15-minute pipeline tick, for each
  scheduled soccer match of the next 7 days with a live baseline features_cache
  row:
  1. proba = ensemble.predict_proba(row): the 5-member blend at the production
     weights, no calibrator (the served 1x2, stored as match_result).
  2. dc = ensemble.models["dixon_coles_soccer_match_result"], the 1x2
     ensemble's OWN Dixon-Coles member (blend weight 0.046, the trainer's
     floor), and its lambdas_for_match(home name, away name, league=league_id)
     and rho.
  3. anchor = rec_gating.market_consensus_prob(cur, match, "over_under",
     "over", 2.5): per book, the newest non-live price of each side of the 2.5
     line seen within MAX_ODDS_AGE_HOURS (6 h on last_seen_at, else timestamp),
     de-vigged by the power method, then averaged over books; None unless at
     least MIN_CONSENSUS_BOOKS = 3 books quote both sides. (The task brief said
     ">= 2 books"; the code says 3.) A failed lookup serves the 1x2-only path.
  4. derive_from_lambdas(h, a, rho, target_1x2=proba, target_over_25=anchor):
     build_dc_matrix(max_goals=MAX_GOALS_DERIVE=10); when valid_ou_anchor
     (finite, 0.05..0.95) reconcile_matrix_to_1x2_and_totals(method="tilt"),
     which alternates a goal-rate tilt with the 1x2 region scaling (1x2 last);
     otherwise reconcile_matrix_to_1x2 (one IPF pass over home/draw/away).
  5. derive_soccer_markets(P), stored as 14 market types (over_under 0.5..5.5,
     btts, correct_score top-12 + other, double_chance, draw_no_bet,
     asian_handicap at 17 lines incl. quarters, team_total, clean_sheet,
     win_to_nil, odd_even, winning_margin, total_goals, result_btts,
     result_over_under) with metadata.reconcile "1x2" or "1x2+ou2.5". The
     half-time markets come from two other artifacts and are not touched here.
  What the artifact can move. After the IPF every market that is a function of
  the 1x2 alone is IDENTICAL whichever Dixon-Coles serves: double_chance,
  draw_no_bet and asian_handicap 0 / +-0.25 / +-0.5. Where the anchor applies,
  over_2.5 is the market's number whichever Dixon-Coles serves. Everything
  else (the other totals lines, BTTS, team totals, clean sheet, win to nil,
  odd/even, margins, exact totals, correct score, the two combos and
  asian_handicap +-0.75..+-2) is the artifact's shape.
  Recommendations (scripts/rec_gating.py, generate_recommendations.py). The
  soccer odds feed carries 1x2, over_under and asian_handicap only (prod, the
  last 60 days). ("soccer","over_under") and ("soccer","odd_even") are
  disabled. ("soccer","asian_handicap") is ENABLED (max_odds 4.0, max_ev 0.30,
  max_gap 0.10): the one rec-emitting market served from this artifact. In the
  last 60 days 168 AH recs were written, 106 of them on lines this artifact
  shapes (+-0.75..+-2; away +1 alone is 33). btts / double_chance /
  draw_no_bet / correct_score inherit the enabled soccer default but have no
  odds feed, and the remaining derived markets have no odds mapping at all:
  display-only.

PRE-REGISTERED DESIGN (fixed 2026-11-01 and committed BEFORE the first full run)
  Frame    finished soccer matches with both scores, the same predicates as
           ab_soccer_ensemble_members.MATCHES_SQL (the trainer's population),
           plus the 90-minute pair from matches.metadata that full-time markets
           settle on (grading_outcomes.settlement_scores).
  Window   the matches the 1x2 incumbent never saw (its seen_match_ids:
           train+val), so the served 1x2 target is out of sample; every one
           of them is also out of sample for the served Dixon-Coles (its fit
           ends at the 70% split, years earlier). Monthly folds from the first
           unseen month to --end (ft.build_folds / split_fold); a fit uses
           match_date < the fold's month start only. Rows the serve path could
           not predict (no features row, a missing learned-member feature) and
           rows with no usable settlement score are excluded and counted. A
           fold with fewer than MIN_FOLD_EVAL rows is skipped.
  Target   the served 1x2: ensemble.predict_proba on the row
           precompute_predictions builds (ab_soccer_ensemble_members.serve_row),
           IDENTICAL in every arm. A single-row sample must equal the batch.
  Anchor   three regimes, every Dixon-Coles arm scored in each:
    live    the live rule exactly: pre-kickoff rows only (prematch_odds), the
            newest non-live quote per (book, side) at line 2.5 seen within
            MAX_ODDS_AGE_HOURS of KICKOFF (the last tick is <= 15 min before
            it, so this is up to 15 min permissive), then
            rec_gating.market_consensus_prob itself (power de-vig, >= 3 books).
            The football-data aggregates 'Average' and 'Max' are not books and
            are excluded (they never occur live).
    proxy   live where the live rule anchors; otherwise the football-data
            closing O/U 2.5 ('Average', else 'Bet365'), power de-vigged. In
            this window the live rule can only see the-odds-api books (from
            2026-06), whereas live today the same leagues are quoted by several
            books; proxy is the robustness regime for that under-anchoring.
    none    1x2-only reconcile everywhere (context: the members stream's
            readout).
  Arms     every fitted arm is round-tripped through save() and the side-
           artifact loader precompute_predictions uses
           (_load_dixon_coles_artifact == ht.load_serving).
    served        the production artifact as the ensemble loads it (or
                  --model-path); a side-loader copy must give identical lambdas.
    refit         DIXON_COLES_CONFIG (the served artifact's own config) refit
                  on EVERY finished match before each eval month: the object a
                  monthly refit DAG + checked promoter would serve. DECIDES.
    refit_frozen  the same, fitted ONCE at the first fold's cutoff and scored
                  on every fold (how fast the refit goes stale). Context.
    base_league / base_league_dw
                  per-league constants for every unit (class rates shrunk to
                  the global rate with k = 200 effective matches, walk-forward;
                  _dw with the refit's own recency weighting). Not reconciled,
                  not anchored. Context: refit arms are read against
                  base_league_dw, served against base_league.
  Units    every served derived market, read from each arm's served matrix
           and GRADED as grading_outcomes grades it (on the settlement score).
           A binary proposition is its own unit (binary Brier (p - y)^2, the
           2026-09-04 gate's convention); a partition is one unit (Brier summed
           over its keys); an asian handicap line is one 3-way unit over the
           effective home / away / push masses the recommender prices (quarter
           lines: a half win realises 0.5 home + 0.5 push), its truth from
           grading_outcomes._handicap_outcome.
             over_0.5 .. over_5.5, btts, team_total_{home,away}_over_{0.5,1.5,
             2.5}, clean_sheet_{home,away}, win_to_nil_{home,away}, odd_even,
             winning_margin (7-way), total_goals (7-way), result_btts (6-way),
             result_over_under (6-way), ah_<line> for the 12 shaped lines,
             correct_score_grid (the whole 11x11 served matrix as a 121-way
             Brier on the realised scoreline, each side capped at 10: one
             partition shared by every arm — the NHL gate's A1),
             and, never blocking:
             INVARIANT  double_chance_{1X,12,X2}, draw_no_bet, ah_0, ah_+-0.25,
                        ah_+-0.5: fixed by the 1x2 target, so every
                        Dixon-Coles arm must serve them identically (checked
                        to 1e-9; a violation aborts the run).
             CONTEXT    correct_score_listed (the served top-12 + 'other',
                        graded on each arm's own keys: improper across arms).
  PRIMARY  over_2.5 as served (binary Brier). Why this market: it is the one
           the 2026-09-04 gate used and the promote gate's derived guard
           reads; it is the totals level the Dixon-Coles shape exists to
           carry once the 1x2 is pinned; and it is the derived market with the
           deepest odds feed. Where the anchor applies its delta is exactly 0,
           which is the honest deployable effect.
  Disqualification
           served and the constants must yield finite, valid probabilities on
           every row, or the run aborts. A challenger is DISQUALIFIED (never
           scored again) the first time a fit of it would be refused by the
           half-time trainer's own rule (train_halftime_dixon_coles.fit_refusal:
           not converged, fitted vs observed mean total goals beyond +/-2%,
           |rho| > 0.1, or an invalid matrix for one of the fold's fixtures) or
           it emits an invalid probability: a monthly job would refuse to
           promote that fit.
  Statistics
           paired per-match deltas (challenger - served) with fold-clustered
           SE (ft.paired_delta), t(n_folds-1) bounds.

DECISION RULE (pre-registered 2026-11-01, before any full run)
  SHIP only if refit is not disqualified and, in BOTH the live and the proxy
  regime,
    (a) it passes ship_strict vs served on over_2.5: ft.ship_verdict (dBrier
        <= -0.005, per-match AND fold-clustered z=2 95% upper bounds < 0) AND
        the t(n_folds-1) fold-clustered upper bound < 0; and
    (b) it is significantly worse than served on NO blocking unit (A2): a
        blocking unit regresses when the fold-clustered t(n_folds-1) 95% LOWER
        bound of its dBrier is > 0 (per-match delta - 2 SE when no clustered SE
        exists); a unit with no gradable paired row or no bound blocks too.
  Everything else — refit_frozen, the constants, the none regime, the strata,
  correct_score_listed, the per-fold table — is context and never decides.
  The decision is made on the CURRENT corpus. --simulate-merge-map (below) is
  the corpus sensitivity: if the current corpus reads SHIP and the simulated
  merge reads DO-NOT-SHIP, nothing is deployed before a re-run on the repaired
  corpus.
  SHIP would mean: a separate derived-market artifact refit monthly by its own
  trainer and DAG, promoted by scripts/promote_dc_model.py after a backup,
  outside the weekly retrain_models swap; the 1x2 ensemble member untouched.
  This script does none of that.

CORPUS CAVEAT
  As of 2026-11-01 the soccer corpus is mid-repair (scripts/ops/
  soccer_repair_runbook.sh: a 171-pair team-identity merge, football-data
  kickoffs an hour late, ~98 duplicate rows). --simulate-merge-map applies the
  merge map in memory exactly as ab_soccer_ensemble_members does (twins take
  the kept team's id and name; twin fixtures fold to the row with features).
  It does not rebuild features (runbook step 10), shift kickoffs, or dedupe
  step 4's rows; the comparison is paired, so a duplicate counts in every arm.

CONFOUNDS, PRINTED NOT HIDDEN
  * The served 1x2 target comes from the CURRENT incumbent for the whole
    window; live, each week's match_result came from that week's ensemble.
    The target is shared by every arm, so it cannot favour one.
  * Live-rule anchoring in this window exists only from 2026-06 (the-odds-api
    era) and mostly after 2026-09-04 (migration 024: before it the odds row
    held the FIRST-seen price, so its age fails the 6 h bound). The anchored
    share is printed by era.
  * matches.metadata carried no 90-minute pair on 2026-11-01 (the backfill is
    an operator-run wrapper), so settlement equals the final score today; the
    harness routes through settlement_scores regardless.
  * Team coverage: strata by how many of a match's teams the served artifact
    and the refit know.

THE RUNS (fixed with the rule)
  full    --end 2026-10-31 --served-sha256 02141a6a... on a local replica of the
          read-only REPEATABLE READ prod export of 2026-11-01 06:44Z (odds
          re-read 06:49Z through prematch_odds; the runbook had not started),
          --model-path a byte-identical copy of the production registry, and
          --live-check-model-version for the incumbent's stored predictions.
  merged  the same with --simulate-merge-map scripts/data/soccer_team_merge_map.csv.
  Before this commit the harness ran once on one fold (2026-09) for mechanics
  only: the fidelity checks (derive_from_lambdas vs readout 6.7e-16, the two
  loaders' lambdas identical, invariant units 1.4e-15, anchor hit 1e-10) and
  the live check (370 stored live predictions of the 2026-11-01 incumbent
  re-derived: 1x2-only rows to 1.1e-16, the 43 anchored rows to 4.9e-7 — the
  stored anchor is rounded to 6 dp). No delta of that run was read.

RESULT (2026-11-01; local replica of the 06:44Z read-only prod export, the
pre-runbook corpus; served = the 2026-11-01 weekly-retrain artifact, sha256
02141a6a..., 951 teams, fit ending 2021-10-23): DO-NOT-SHIP.
  n = 25,604 matches, 31 monthly folds 2024-04..2026-10 (the 1x2 incumbent's
  unseen rows from 2024-04-07); excluded 4 without features, 12 unservable, 3
  unseen inside the seen span. No fit refused, no arm disqualified, all 32
  fits converged. Fidelity: derive_from_lambdas vs readout 7.8e-16, the two
  loaders identical, invariant units 1.7e-15, anchor hit 1.0e-10, live check
  370/370 (1x2-only to 1.1e-16; anchored to 4.9e-7, the 6-dp stored anchor).
  PRIMARY over_2.5 as served, refit - served:
    live   -0.00375 +/- 0.00037, fold-clustered -0.00358 +/- 0.00044,
           t(30) upper bound -0.00269: real, but short of -0.005
    proxy  -0.00204 +/- 0.00028, clustered -0.00214 +/- 0.00037, t-bound -0.00139
    none   -0.00378, clustered -0.00361 +/- 0.00042 (the members stream's
           unanchored -0.0039, reproduced on the new artifact)
  No blocking unit regresses in any regime or run. In the live regime every
  shaped unit improves with its t upper bound below zero except odd_even
  (-0.00001, bound +0.00014): BTTS -0.00255, over_3.5 -0.00361, total_goals
  -0.00276, correct_score_grid -0.00148, AH +-0.75..+-2 -0.0002..-0.0021
  (3-way); in the proxy regime ah_1 (-0.00032, bound +0.00004) and odd_even
  are the two not clear of zero.
  Only 504 eval rows (2%) carry a live-rule anchor, all from 2026-07 (196 and
  301 in 2026-09 / 2026-10: 25% and 32% of those months); there the primary
  delta is exactly 0, so live today the deployable effect is smaller still.
  refit_frozen -0.00202 (clustered -0.00174 +/- 0.00033): a one-off refit loses
  half its lead over the 31 months (frozen - refit +0.00174, clustered
  +0.00184 +/- 0.00038), so the deployable object would be the monthly refit.
  Context: served is no better than a per-league constant on over_2.5
  (-0.00043, clustered -0.00024 +/- 0.00044) and WORSE on BTTS (+0.00131, t
  lower bound +0.00072); refit beats the recency-matched constant on both
  (-0.00326 / -0.00060). After the 1x2-only reconcile both arms under-predict
  over_2.5 (served 0.490, refit 0.499, realised 0.524).
  Corpus caveat, --simulate-merge-map (171 pairs, 1,588 rows re-pointed, 171
  twin rows folded, n = 25,434): live -0.00391 (clustered -0.00374 +/- 0.00043,
  t-bound -0.00286), proxy -0.00219, no regression. The merge moves the primary
  0.00016 TOWARD the gate, an order of magnitude short of the 0.0011 a flip
  would need; the unsimulated steps (kickoff shift, ~98-row dedupe, feature
  rebuild) change the shared target or rows every arm scores alike.
  Anchor replay vs production: for the 166 finished eval matches whose stored
  over_under row carries metadata.reconcile (the two previous incumbents,
  2026-10-26..31), production anchored 44 and the replay 47 (41 both, 3
  production only, 6 replay only); on the 41 the replayed anchor equals the
  stored one to the 6-dp rounding on 24 and differs by up to 0.087 on the
  rest. The replay reads the CLOSING prices (the odds row is refreshed in
  place), while production's last tick can come well before kickoff, so
  "the last tick is <= 15 min before kickoff" in the design does not always
  hold. That timing only touches the 504 anchored rows (2%), where the
  primary delta is 0 whichever value anchors them; scoring all 504 unanchored
  gives -0.00378 (the none regime). It cannot move the verdict.
  Every number above recomputes from the run's per_match.csv.gz / scores.npz.

Read-only: SELECTs, plus temp files for the save/load round-trip. Usage (api container):
    python /app/scripts/ab_soccer_ft_dixon_coles.py --end 2026-10-31 \\
        --out-json /tmp/ab_ftdc.json --out-dir /tmp/ab_ftdc
    python /app/scripts/ab_soccer_ft_dixon_coles.py --end 2026-10-31 --simulate-merge-map \\
        /app/scripts/data/soccer_team_merge_map.csv --out-json /tmp/ab_ftdc_merged.json
A local replica works too: point --database-url at it and --model-path at a
byte-identical copy of the production registry (OMP_NUM_THREADS=1 on macOS).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_halftime_dixon_coles as ht  # noqa: E402
import ab_nhl_dixon_coles as nhl  # noqa: E402
import ab_soccer_dixon_coles_refit as ft  # noqa: E402
import ab_soccer_ensemble_members as members  # noqa: E402
import grading_outcomes as go  # noqa: E402
import rec_gating  # noqa: E402
import train_halftime_dixon_coles as htrainer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("ab_soccer_ft_dixon_coles")

SERVED_DC = members.DIXON_COLES  # "dixon_coles_soccer_match_result"
MODEL_NAME = "ensemble"  # predictions.model_name of every derived soccer row (what grading sees)
GRID_SIDE = 11  # market_derivation.MAX_GOALS_DERIVE + 1 (a test pins it)
GRID_CAP = GRID_SIDE - 1
MIN_FOLD_EVAL = 50
CONSTANT_SHRINKAGE = 200.0
ESPN_ERA_START = members.ESPN_ERA_START
INVARIANCE_ATOL = 1e-9
DERIVE_ATOL = 1e-12
DERIVE_SAMPLE = 12  # rows per fold, arm and regime re-derived through derive_from_lambdas
LIVE_CHECK_ATOL = 1e-9  # a 1x2-only row is re-derived from exactly what was stored
# precompute_predictions stores the anchor rounded to 6 dp (metadata
# market_over_2.5), so an anchored row can only be re-derived to about that.
LIVE_CHECK_ANCHORED_ATOL = 5e-6
ANCHOR_LINE = 2.5
AGGREGATE_BOOKS = frozenset({"average", "max"})  # football-data aggregates, not books
PROXY_BOOKS = ("average", "bet365")  # preference order for the proxy regime

DC_ARMS = ("served", "refit", "refit_frozen")
CHALLENGERS = ("refit", "refit_frozen")
DECIDING_ARM = "refit"
CONSTANT_ARMS = ("base_league", "base_league_dw")
DECAYED_ARMS = ("refit", "refit_frozen")
REGIMES = ("live", "proxy", "none")
DECIDING_REGIMES = ("live", "proxy")

OU_LINES = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)  # market_derivation.OU_LINES
TEAM_TOTAL_LINES = (0.5, 1.5, 2.5)  # market_derivation.TEAM_TOTAL_LINES
AH_LINES = (-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
AH_INVARIANT_LINES = (-0.5, -0.25, 0.0, 0.25, 0.5)  # functions of the 1x2 alone
HANDICAP_TRUTH = {
    go.REC_WON: (1.0, 0.0, 0.0),
    go.REC_LOST: (0.0, 1.0, 0.0),
    go.REC_PUSH: (0.0, 0.0, 1.0),
    go.REC_HALF_WON: (0.5, 0.0, 0.5),
    go.REC_HALF_LOST: (0.0, 0.5, 0.5),
}


def fmt_line(line: float) -> str:
    """market_derivation._fmt_line / grading_outcomes._fmt_line: 2.5 -> '2.5', -1.0 -> '-1'."""
    return go._fmt_line(line)


@dataclass(frozen=True)
class Unit:
    """One scored proposition set of a served market. kind: 'binary' (one
    key), 'partition' (exactly one key wins), 'handicap' (effective home /
    away / push masses), 'grid' (the whole served matrix) or 'listed' (the
    served top-12 + 'other', each arm on its own keys)."""

    name: str
    market: str  # the served market_type == predictions.prediction_type
    keys: Tuple[str, ...] = ()
    kind: str = "binary"


def _build_units() -> Tuple[Unit, ...]:
    units: List[Unit] = [Unit(f"over_{fmt_line(x)}", "over_under", (f"over_{fmt_line(x)}",)) for x in OU_LINES]
    units.append(Unit("btts", "btts", ("yes",)))
    units += [Unit(f"double_chance_{k}", "double_chance", (k,)) for k in ("1X", "12", "X2")]
    units.append(Unit("draw_no_bet", "draw_no_bet", ("home", "away"), "partition"))
    for line in AH_LINES:
        lbl = fmt_line(line)
        units.append(Unit(f"ah_{lbl}", "asian_handicap", (f"{lbl}_home", f"{lbl}_away", f"{lbl}_push"), "handicap"))
    for side in ("home", "away"):
        for line in TEAM_TOTAL_LINES:
            key = f"{side}_over_{fmt_line(line)}"
            units.append(Unit(f"team_total_{key}", "team_total", (key,)))
    units += [Unit(f"clean_sheet_{s}", "clean_sheet", (f"{s}_yes",)) for s in ("home", "away")]
    units += [Unit(f"win_to_nil_{s}", "win_to_nil", (f"{s}_yes",)) for s in ("home", "away")]
    units.append(Unit("odd_even", "odd_even", ("odd",)))
    units.append(
        Unit(
            "winning_margin",
            "winning_margin",
            ("home_1", "home_2", "home_3plus", "draw", "away_1", "away_2", "away_3plus"),
            "partition",
        )
    )
    units.append(Unit("total_goals", "total_goals", ("0", "1", "2", "3", "4", "5", "6+"), "partition"))
    rb = tuple(f"{r}_{b}" for r in ("home", "draw", "away") for b in ("yes", "no"))
    units.append(Unit("result_btts", "result_btts", rb, "partition"))
    ro = tuple(f"{r}_{b}" for r in ("home", "draw", "away") for b in ("over", "under"))
    units.append(Unit("result_over_under", "result_over_under", ro, "partition"))
    units.append(Unit("correct_score_grid", "correct_score", (), "grid"))
    units.append(Unit("correct_score_listed", "correct_score", (), "listed"))
    return tuple(units)


UNITS: Tuple[Unit, ...] = _build_units()
UNIT_BY_NAME: Dict[str, Unit] = {u.name: u for u in UNITS}
UNIT_NAMES: Tuple[str, ...] = tuple(u.name for u in UNITS)
PRIMARY = "over_2.5"
GRID = "correct_score_grid"
LISTED = "correct_score_listed"
INVARIANT_UNITS: Tuple[str, ...] = (
    "double_chance_1X",
    "double_chance_12",
    "double_chance_X2",
    "draw_no_bet",
) + tuple(f"ah_{fmt_line(x)}" for x in AH_INVARIANT_LINES)
CONTEXT_UNITS: Tuple[str, ...] = (LISTED,)
BLOCKING_UNITS: Tuple[str, ...] = tuple(
    u for u in UNIT_NAMES if u != PRIMARY and u not in INVARIANT_UNITS and u not in CONTEXT_UNITS
)
# Units read off the matrix as fixed linear functionals (everything but the
# ratio market draw_no_bet, the grid and the per-arm listed keys).
LINEAR_UNITS: Tuple[str, ...] = tuple(u.name for u in UNITS if u.kind in ("binary", "partition", "handicap"))
LINEAR_UNITS = tuple(u for u in LINEAR_UNITS if u != "draw_no_bet")
# Units a per-league constant is fitted for (the listed keys differ per arm).
CONSTANT_UNITS: Tuple[str, ...] = tuple(u for u in UNIT_NAMES if u != LISTED)

FRAME_SQL = """
    SELECT
        m.id::text AS match_id,
        m.match_date,
        m.league_id::text AS league_id,
        l.name AS league_name,
        m.home_team_id::text AS home_team_id,
        m.away_team_id::text AS away_team_id,
        ht.name AS home_team,
        at.name AS away_team,
        m.home_score,
        m.away_score,
        m.metadata->>'regulation_home_score' AS regulation_home_score,
        m.metadata->>'regulation_away_score' AS regulation_away_score
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    WHERE l.sport = 'soccer'
      AND m.status = 'finished'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
    ORDER BY m.match_date ASC
"""

# Every over/under 2.5 row of the eval matches; run() wraps it in
# prematch_odds() (pre-kickoff rows only, or all rows of a match that has none:
# the football-data backfill) before it reaches the database.
OU_ODDS_SQL = """
    SELECT
        o.match_id::text AS match_id,
        o.bookmaker,
        o.selection,
        o.line::float AS line,
        o.odds_decimal::float AS odds_decimal,
        o."timestamp" AS ts,
        o.last_seen_at,
        o.is_live
    FROM odds o
    JOIN matches m ON m.id = o.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
    WHERE m.status = 'finished'
      AND m.match_date >= %(start)s
      AND m.match_date <  %(end)s
      AND o.market_type = 'over_under'
      AND o.line = 2.5
"""

LIVE_CHECK_SQL = """
    SELECT p.match_id::text AS match_id, p.prediction_type, p.probabilities, p.metadata,
           ht.name AS home_team, at.name AS away_team, m.league_id::text AS league_id
    FROM predictions p
    JOIN matches m ON m.id = p.match_id
    JOIN teams ht ON ht.id = m.home_team_id
    JOIN teams at ON at.id = m.away_team_id
    WHERE p.model_name = 'ensemble'
      AND p.model_version = %(version)s
"""


# ── Pure: units ──────────────────────────────────────────────────────


def unit_weights(n: int = GRID_SIDE) -> Dict[str, np.ndarray]:
    """{unit: (k, n, n) weights} so that each key's served probability is
    sum(P * W[key]) — derive_soccer_markets' definitions, cell by cell
    (a test checks them against derive_soccer_markets itself)."""
    i, j = np.indices((n, n))
    tot, mar = i + j, i - j
    both = (i >= 1) & (j >= 1)
    regions = (mar > 0, mar == 0, mar < 0)
    W: Dict[str, List[np.ndarray]] = {}
    for line in OU_LINES:
        W[f"over_{fmt_line(line)}"] = [tot > line]
    W["btts"] = [both]
    W["double_chance_1X"] = [mar >= 0]
    W["double_chance_12"] = [mar != 0]
    W["double_chance_X2"] = [mar <= 0]
    for line in AH_LINES:
        lbl = fmt_line(line)
        if (float(line) * 4) % 2 != 0:  # quarter line: the stake splits over line -/+ 0.25
            lo, hi = mar + line - 0.25, mar + line + 0.25
            full_win = (lo > 0) & (hi > 0)
            full_loss = (lo < 0) & (hi < 0)
            half_win = ((lo == 0) & (hi > 0)) | ((lo > 0) & (hi == 0))
            half_loss = ((lo == 0) & (hi < 0)) | ((lo < 0) & (hi == 0))
            W[f"ah_{lbl}"] = [
                full_win + 0.5 * half_win,
                full_loss + 0.5 * half_loss,
                0.5 * (half_win.astype(float) + half_loss.astype(float)),
            ]
        else:
            adj = mar + line
            W[f"ah_{lbl}"] = [adj > 0, adj < 0, adj == 0]
    for line in TEAM_TOTAL_LINES:
        W[f"team_total_home_over_{fmt_line(line)}"] = [i > line]
        W[f"team_total_away_over_{fmt_line(line)}"] = [j > line]
    W["clean_sheet_home"] = [j == 0]
    W["clean_sheet_away"] = [i == 0]
    W["win_to_nil_home"] = [(mar > 0) & (j == 0)]
    W["win_to_nil_away"] = [(mar < 0) & (i == 0)]
    W["odd_even"] = [(tot % 2) == 1]
    W["winning_margin"] = [mar == 1, mar == 2, mar >= 3, mar == 0, mar == -1, mar == -2, mar <= -3]
    W["total_goals"] = [tot == k for k in range(6)] + [tot >= 6]
    W["result_btts"] = [r & c for r in regions for c in (both, ~both)]
    over = tot > 2.5
    W["result_over_under"] = [r & c for r in regions for c in (over, ~over)]
    missing = set(LINEAR_UNITS) ^ set(W)
    if missing:
        raise RuntimeError(f"unit weights and LINEAR_UNITS disagree: {sorted(missing)}")
    return {u: np.stack([np.asarray(w, dtype=float) for w in ws]) for u, ws in W.items()}


def unit_probabilities(P: np.ndarray, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """{unit: (m, k)} served probabilities from a stack of served matrices
    (m, n, n), plus '_listed_order' (m, 12): the flat indices of each row's
    listed scorelines, in derive_soccer_markets' order."""
    P = np.asarray(P, dtype=float)
    m, n = P.shape[0], P.shape[1]
    out: Dict[str, np.ndarray] = {u: np.einsum("rij,kij->rk", P, w) for u, w in weights.items()}
    i, j = np.indices((n, n))
    ph = np.einsum("rij,ij->r", P, (i > j).astype(float))
    pa = np.einsum("rij,ij->r", P, (i < j).astype(float))
    dec = ph + pa
    with np.errstate(invalid="ignore", divide="ignore"):
        out["draw_no_bet"] = np.where(dec[:, None] > 0, np.stack([ph, pa], axis=1) / dec[:, None], 0.0)
    flat = P.reshape(m, n * n)
    out[GRID] = flat.copy()
    order = np.empty((m, 12), dtype=int)
    listed = np.empty((m, 13))
    for r in range(m):  # per row, exactly as derive_soccer_markets sorts one matrix
        idx = np.argsort(flat[r])[::-1][:12]
        order[r] = idx
        vals = flat[r][idx]
        listed[r, :12] = vals
        listed[r, 12] = max(0.0, 1.0 - float(vals.sum()))
    out[LISTED] = listed
    out["_listed_order"] = order
    return out


def handicap_truth(margin: int, line: float) -> Tuple[float, float, float]:
    """(home, away, push) as the stake actually settles: grading_outcomes'
    own _handicap_outcome, a half win realised as half home + half push."""
    return HANDICAP_TRUTH[go._handicap_outcome(int(margin), float(line))]


def vector_truth(h: np.ndarray, a: np.ndarray) -> Dict[str, np.ndarray]:
    """{unit: (m, k)} realised outcome of every non-listed unit from the
    settlement scores, NaN rows where the unit does not grade (draw_no_bet on
    a draw). Vectorised; graded_truth (grading_outcomes, per row) must agree."""
    h = np.asarray(h, dtype=int)
    a = np.asarray(a, dtype=int)
    m = len(h)
    tot, mar = h + a, h - a
    both = (h >= 1) & (a >= 1)
    regions = (mar > 0, mar == 0, mar < 0)
    out: Dict[str, Any] = {}
    for line in OU_LINES:
        out[f"over_{fmt_line(line)}"] = [tot > line]
    out["btts"] = [both]
    out["double_chance_1X"] = [mar >= 0]
    out["double_chance_12"] = [mar != 0]
    out["double_chance_X2"] = [mar <= 0]
    dnb = np.stack([mar > 0, mar < 0], axis=1).astype(float)
    dnb[mar == 0] = np.nan
    margins = np.unique(mar)
    for line in AH_LINES:
        table = {int(mg): handicap_truth(int(mg), line) for mg in margins}
        out[f"ah_{fmt_line(line)}"] = np.array([table[int(mg)] for mg in mar], dtype=float).reshape(m, 3)
    for line in TEAM_TOTAL_LINES:
        out[f"team_total_home_over_{fmt_line(line)}"] = [h > line]
        out[f"team_total_away_over_{fmt_line(line)}"] = [a > line]
    out["clean_sheet_home"] = [a == 0]
    out["clean_sheet_away"] = [h == 0]
    out["win_to_nil_home"] = [(mar > 0) & (a == 0)]
    out["win_to_nil_away"] = [(mar < 0) & (h == 0)]
    out["odd_even"] = [(tot % 2) == 1]
    out["winning_margin"] = [mar == 1, mar == 2, mar >= 3, mar == 0, mar == -1, mar == -2, mar <= -3]
    out["total_goals"] = [tot == k for k in range(6)] + [tot >= 6]
    out["result_btts"] = [r & c for r in regions for c in (both, ~both)]
    over = tot > 2.5
    out["result_over_under"] = [r & c for r in regions for c in (over, ~over)]
    res: Dict[str, np.ndarray] = {}
    for u, v in out.items():
        res[u] = v if isinstance(v, np.ndarray) else np.stack([np.asarray(c, dtype=float) for c in v], axis=1)
    res["draw_no_bet"] = dnb
    grid = np.zeros((m, GRID_SIDE * GRID_SIDE))
    grid[np.arange(m), np.minimum(h, GRID_CAP) * GRID_SIDE + np.minimum(a, GRID_CAP)] = 1.0
    res[GRID] = grid
    return res


def graded_truth(h: Sequence[int], a: Sequence[int]) -> Dict[str, np.ndarray]:
    """The same outcomes through grading_outcomes, row by row: actual_outcome
    + grade_prediction per key for the binary and partition units (exactly how
    the grading job grades a stored pick of that key), _handicap_outcome for
    the handicap units. NaN row where a unit does not grade."""
    hs = [int(x) for x in h]
    as_ = [int(x) for x in a]
    out: Dict[str, np.ndarray] = {}
    for unit in UNITS:
        if unit.kind in ("grid", "listed"):
            continue
        arr = np.full((len(hs), len(unit.keys)), np.nan)
        for r, (hh, aa) in enumerate(zip(hs, as_)):
            if unit.kind == "handicap":
                line = float(unit.keys[0].rsplit("_", 1)[0])
                arr[r] = handicap_truth(hh - aa, line)
                continue
            vals = []
            for key in unit.keys:
                actual = go.actual_outcome(
                    prediction_type=unit.market,
                    model_name=MODEL_NAME,
                    predicted_outcome=key,
                    home_score=hh,
                    away_score=aa,
                    metadata={},
                )
                verdict = go.grade_prediction(key, actual, unit.market)
                if verdict is None:
                    vals = None
                    break
                vals.append(1.0 if verdict else 0.0)
            if vals is None:
                continue
            if unit.kind == "partition" and sum(vals) != 1.0:
                raise ValueError(f"{unit.name}: {sum(vals)} of {unit.keys} graded correct on {hh}-{aa}")
            arr[r] = vals
        out[unit.name] = arr
    return out


def listed_truth(order: np.ndarray, h: np.ndarray, a: np.ndarray) -> np.ndarray:
    """(m, 13) graded outcome of each row's own listed keys: the realised
    scoreline's key, else 'other' (grading_outcomes.grade_correct_score_other).
    A side above GRID_CAP is never a listed key."""
    h = np.asarray(h, dtype=int)
    a = np.asarray(a, dtype=int)
    real = np.where((h <= GRID_CAP) & (a <= GRID_CAP), h * GRID_SIDE + a, -1)
    y = np.zeros((len(h), 13))
    y[:, :12] = (np.asarray(order) == real[:, None]).astype(float)
    y[:, 12] = 1.0 - y[:, :12].sum(axis=1)
    return y


def brier_rows(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row Brier summed over a unit's keys; NaN where the row does not grade."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.sum((p - y) ** 2, axis=1)
    out[~np.isfinite(y).all(axis=1)] = np.nan
    return out


def invalid_probabilities(probs: Dict[str, np.ndarray], units: Sequence[str]) -> Optional[str]:
    """None when every row of every listed unit is a probability (and a
    partition's / handicap's / the grid's rows sum to 1), else what is wrong."""
    for name in units:
        v = np.asarray(probs[name], dtype=float)
        bad = ~np.isfinite(v).all(axis=1) | (v < -1e-12).any(axis=1) | (v > 1 + 1e-12).any(axis=1)
        if UNIT_BY_NAME[name].kind != "binary":
            bad |= ~np.isclose(v.sum(axis=1), 1.0, atol=1e-6)
        if bad.any():
            return f"{int(bad.sum())} of {len(v)} {name} rows invalid"
    return None


def check_invariance(arm_probs: Dict[str, Dict[str, np.ndarray]], atol: float = INVARIANCE_ATOL) -> float:
    """Every Dixon-Coles arm must serve the 1x2-determined units identically;
    returns the max |difference| or raises."""
    arms = list(arm_probs)
    worst = 0.0
    for u in INVARIANT_UNITS:
        base = np.asarray(arm_probs[arms[0]][u], dtype=float)
        for arm in arms[1:]:
            gap = float(np.max(np.abs(np.asarray(arm_probs[arm][u], dtype=float) - base))) if len(base) else 0.0
            if not math.isfinite(gap) or gap > atol:
                raise RuntimeError(f"invariant unit {u} differs between {arms[0]} and {arm} by {gap:.2e}")
            worst = max(worst, gap)
    return worst


# ── Pure: the O/U 2.5 anchor ─────────────────────────────────────────


def _ts(value: Any) -> Optional[pd.Timestamp]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        out = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out.tz_localize("UTC") if out.tzinfo is None else out.tz_convert("UTC")


def consensus_rows(
    rows: Sequence[Dict[str, Any]], kickoff: Any, max_age_hours: float = rec_gating.MAX_ODDS_AGE_HOURS
) -> List[Dict[str, Any]]:
    """rec_gating._CONSENSUS_SQL replayed at the last tick before kickoff:
    non-live rows at line 2.5 whose COALESCE(last_seen_at, timestamp) is within
    max_age_hours of the kickoff, newest timestamp per (bookmaker, lower
    selection, line), NULLs last. The football-data aggregates are dropped:
    they are not books and never reach the live table. `rows` must already be
    pre-kickoff (prematch_odds)."""
    ko = _ts(kickoff)
    if ko is None:
        raise ValueError(f"kickoff {kickoff!r} is not a timestamp")
    best: Dict[Tuple[str, str, float], Tuple[Optional[pd.Timestamp], Dict[str, Any]]] = {}
    for row in rows:
        if bool(row.get("is_live")):
            continue
        if str(row.get("bookmaker") or "").strip().lower() in AGGREGATE_BOOKS:
            continue
        line = row.get("line")
        if line is None or abs(float(line) - ANCHOR_LINE) > 1e-9:
            continue
        seen = _ts(row.get("last_seen_at"))
        if seen is None:
            seen = _ts(row.get("ts"))
        if seen is None or (ko - seen).total_seconds() / 3600.0 > max_age_hours:
            continue
        key = (str(row["bookmaker"]), str(row["selection"]).strip().lower(), float(line))
        stamp = _ts(row.get("ts"))
        held = best.get(key)
        if held is None or (stamp is not None and (held[0] is None or stamp > held[0])):
            best[key] = (stamp, row)
    return [r for _, r in best.values()]


class _RowsCursor:
    """The one query market_consensus_prob runs, answered from replayed rows."""

    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self._rows = [
            {
                "bookmaker": r["bookmaker"],
                "selection": r["selection"],
                "line": r["line"],
                "odds_decimal": r["odds_decimal"],
            }
            for r in rows
        ]

    def execute(self, sql: str, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params

    def fetchall(self) -> List[Dict[str, Any]]:
        return list(self._rows)


def live_anchor(rows: Sequence[Dict[str, Any]], kickoff: Any, match_id: str = "") -> Optional[float]:
    """The anchor precompute_predictions would have read at the last tick:
    rec_gating.market_consensus_prob itself on the replayed rows (power
    de-vig per book, mean over >= MIN_CONSENSUS_BOOKS books), or None."""
    cur = _RowsCursor(consensus_rows(rows, kickoff))
    p = rec_gating.market_consensus_prob(cur, match_id, "over_under", "over", ANCHOR_LINE)
    return float(p) if p is not None else None


def proxy_anchor(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    """The football-data closing O/U 2.5 of a match ('Average', else
    'Bet365'), power de-vigged, or None."""
    by_book: Dict[str, Dict[str, float]] = {}
    for row in rows:
        line = row.get("line")
        if bool(row.get("is_live")) or line is None or abs(float(line) - ANCHOR_LINE) > 1e-9:
            continue
        price = row.get("odds_decimal")
        if price is None or not float(price) > 1.0:
            continue
        by_book.setdefault(str(row.get("bookmaker") or "").strip().lower(), {})[
            str(row["selection"]).strip().lower()
        ] = float(price)
    for book in PROXY_BOOKS:
        prices = by_book.get(book)
        if prices and {"over", "under"} <= set(prices):
            fair = rec_gating.devig_power({"over": prices["over"], "under": prices["under"]})
            if fair is not None:
                return float(fair["over"])
    return None


# ── Pure: matrices, constants, statistics, the decision ──────────────


def served_matrix(
    h_lam: float, a_lam: float, rho: float, target: Sequence[float], p_over: Optional[float]
) -> np.ndarray:
    """The matrix derive_from_lambdas reads its markets from, by the same
    calls in the same order (a test and the run's fidelity sample check it)."""
    from predictors.market_derivation import (  # type: ignore
        MAX_GOALS_DERIVE,
        build_dc_matrix,
        reconcile_matrix_to_1x2,
        reconcile_matrix_to_1x2_and_totals,
        valid_ou_anchor,
    )

    P = build_dc_matrix(float(h_lam), float(a_lam), float(rho), max_goals=MAX_GOALS_DERIVE)
    tgt = tuple(float(x) for x in target)
    if valid_ou_anchor(p_over):
        return reconcile_matrix_to_1x2_and_totals(P, tgt, p_over, method="tilt")
    return reconcile_matrix_to_1x2(P, tgt)


def arm_matrices(lams: np.ndarray, rho: float, targets: np.ndarray, anchors: Sequence[Optional[float]]) -> np.ndarray:
    m = len(lams)
    out = np.empty((m, GRID_SIDE, GRID_SIDE))
    for r in range(m):
        out[r] = served_matrix(lams[r, 0], lams[r, 1], rho, targets[r], anchors[r])
    return out


def anchor_list(values: np.ndarray, rows: Optional[np.ndarray] = None) -> List[Optional[float]]:
    """NaN -> None (no anchor), optionally restricted to `rows`."""
    vals = np.asarray(values, dtype=float)
    out: List[Optional[float]] = [None if not math.isfinite(v) else float(v) for v in vals]
    if rows is not None:
        keep = np.asarray(rows, dtype=bool)
        out = [v if k else None for v, k in zip(out, keep)]
    return out


def fit_constants(
    Y0: np.ndarray, V: np.ndarray, codes: np.ndarray, n_codes: int, w: np.ndarray, k: float = CONSTANT_SHRINKAGE
) -> Tuple[np.ndarray, np.ndarray]:
    """(global (K,), by_code (n_codes, K)): per-column weighted means over the
    rows where the column grades (V = 1; Y0 = the outcome with ungradable
    cells zeroed), each league's shrunk toward the global one with k effective
    matches: ft.fit_base_rates' (sum w y + k g) / (sum w + k), column by
    column. A partition's columns keep summing to 1."""
    from scipy import sparse

    Y0 = np.asarray(Y0)
    V = np.asarray(V)
    w = np.asarray(w, dtype=float)
    tot_v = w @ V
    if (tot_v <= 0).any():
        raise ValueError("a constant column has no weighted rows")
    g = (w @ Y0) / tot_v
    onehot = sparse.csr_matrix((w, (np.asarray(codes, dtype=int), np.arange(len(w)))), shape=(n_codes, len(w)))
    s_wy = np.asarray(onehot @ Y0, dtype=float)
    s_wv = np.asarray(onehot @ V, dtype=float)
    return g, (s_wy + k * g) / (s_wv + k)


def predict_constants(g: np.ndarray, by_code: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Each row's league rates, or the global rates for a league the fit never saw (code -1)."""
    codes = np.asarray(codes, dtype=int)
    out = np.tile(g, (len(codes), 1))
    known = codes >= 0
    out[known] = by_code[codes[known]]
    return out


def pair(challenger: np.ndarray, baseline: np.ndarray, folds: np.ndarray, mask: Optional[np.ndarray] = None):
    """ft.paired_delta plus both fold-clustered t(n_folds-1) bounds."""
    sel = np.ones(len(folds), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    out = ft.paired_delta(np.asarray(challenger)[sel], np.asarray(baseline)[sel], np.asarray(folds)[sel])
    out["upper_95_cluster_t"] = ht.t_upper(out["delta_cluster"], out["se_cluster"], out["n_folds"])
    out["lower_95_cluster_t"] = nhl.t_lower(out["delta_cluster"], out["se_cluster"], out["n_folds"])
    return out


def ship_strict(p: Dict[str, Any], baseline: str = "served") -> Dict[str, Any]:
    """ft.ship_verdict on a paired delta AND the t(n_folds-1) clustered upper
    bound below zero."""
    v = ft.ship_verdict(
        p.get("delta"),
        p.get("se"),
        baseline,
        delta_cluster=p.get("delta_cluster"),
        se_cluster=p.get("se_cluster"),
        n_folds=p.get("n_folds"),
    )
    v["upper_95_cluster_t"] = p.get("upper_95_cluster_t")
    v["ship_strict"] = bool(v["ship"] and p.get("upper_95_cluster_t") is not None and p["upper_95_cluster_t"] < 0)
    return v


def blocking_regressions(per_unit: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """ab_nhl_dixon_coles.regression (rule A2) on every BLOCKING unit."""
    out: Dict[str, str] = {}
    for unit in BLOCKING_UNITS:
        why = nhl.regression(per_unit.get(unit, {}))
        if why is not None:
            out[unit] = why
    return out


def decide_regime(per_unit: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Rules (a) and (b) for the deciding arm in one regime."""
    v = ship_strict(per_unit.get(PRIMARY, {}))
    regs = blocking_regressions(per_unit)
    ship = bool(v["ship_strict"]) and not regs
    if ship:
        reason = f"{PRIMARY} passes ship_strict vs served and no blocking unit is significantly worse"
    elif not v["ship_strict"]:
        reason = f"{PRIMARY} fails ship_strict vs served ({v['reason']})" + (
            f"; {len(regs)} blocking unit(s) also regress" if regs else ""
        )
    else:
        reason = f"{PRIMARY} passes, but {len(regs)} blocking unit(s) regress: {', '.join(regs)}"
    return {"ship": ship, "primary": v, "regressions": regs, "reason": reason}


def decide(per_regime: Dict[str, Dict[str, Any]], disqualified: Dict[str, Any], arm: str = DECIDING_ARM):
    """The pre-registered decision: the deciding arm passes (a) and (b) in
    every DECIDING_REGIMES regime and is not disqualified."""
    if arm in disqualified:
        return {"arm": arm, "ship": False, "reason": f"disqualified: {disqualified[arm].get('reason')}", "regimes": {}}
    missing = [r for r in DECIDING_REGIMES if r not in per_regime]
    if missing:
        return {"arm": arm, "ship": False, "reason": f"regime(s) not scored: {missing}", "regimes": {}}
    regimes = {r: decide_regime(per_regime[r]) for r in DECIDING_REGIMES}
    ship = all(d["ship"] for d in regimes.values())
    failing = [r for r, d in regimes.items() if not d["ship"]]
    reason = (
        "passes rules (a) and (b) in the live and the proxy regime"
        if ship
        else "; ".join(f"[{r}] {regimes[r]['reason']}" for r in failing)
    )
    return {"arm": arm, "ship": ship, "reason": reason, "regimes": regimes}


# ── Model plumbing (needs ml-models + api on the path) ───────────────


def dc_lambdas(model: Any, ev: pd.DataFrame) -> np.ndarray:
    """(m, 2) lambdas exactly as precompute_predictions asks for them."""
    out = np.empty((len(ev), 2))
    for r, (h, a, lg) in enumerate(zip(ev["home_team"], ev["away_team"], ev["league_id"])):
        h_lam, a_lam = model.lambdas_for_match(h, a, league=lg)
        out[r] = (float(h_lam), float(a_lam))
    return out


def derive_fidelity(
    lams: np.ndarray,
    rho: float,
    targets: np.ndarray,
    anchors: Sequence[Optional[float]],
    probs: Dict[str, np.ndarray],
    rows: Sequence[int],
) -> float:
    """Re-derive `rows` through derive_from_lambdas exactly as
    precompute_predictions calls it and compare every unit's served
    probabilities with the vectorised readout; returns the max |gap| or
    raises above DERIVE_ATOL."""
    from predictors.market_derivation import derive_from_lambdas  # type: ignore

    worst = 0.0
    for r in rows:
        mk = derive_from_lambdas(
            float(lams[r, 0]),
            float(lams[r, 1]),
            float(rho),
            target_1x2=(float(targets[r, 0]), float(targets[r, 1]), float(targets[r, 2])),
            target_over_25=anchors[r],
        )
        for unit in UNITS:
            if unit.kind == "grid":
                continue
            if unit.kind == "listed":
                cs = mk["correct_score"]
                keys = [
                    f"{divmod(int(ix), GRID_SIDE)[0]}-{divmod(int(ix), GRID_SIDE)[1]}"
                    for ix in probs["_listed_order"][r]
                ]
                if list(cs) != keys + ["other"]:
                    raise RuntimeError(f"correct_score keys differ on row {r}: {list(cs)} vs {keys + ['other']}")
                ref = np.array([cs[k] for k in keys + ["other"]])
            else:
                ref = np.array([mk[unit.market][k] for k in unit.keys])
            gap = float(np.max(np.abs(ref - probs[unit.name][r])))
            if not math.isfinite(gap) or gap > DERIVE_ATOL:
                raise RuntimeError(f"{unit.name} row {r}: derive_from_lambdas differs from the readout by {gap:.2e}")
            worst = max(worst, gap)
    return worst


def fit_refit(fit_df: pd.DataFrame, tmp: str, tag: str) -> Tuple[Any, Dict[str, Any]]:
    """DIXON_COLES_CONFIG on `fit_df`, saved and re-loaded through the
    side-artifact loader precompute_predictions uses (ht.fit_arm)."""
    return ht.fit_arm(fit_df, {}, tmp, tag)


def load_frame(database_url: str) -> pd.DataFrame:
    frame = members._read_sql(database_url, FRAME_SQL)
    if frame.empty:
        raise RuntimeError("no finished soccer matches — check --database-url")
    frame["match_date"] = pd.to_datetime(frame["match_date"], utc=True)
    dups = int(frame["match_id"].duplicated().sum())
    if dups:
        raise RuntimeError(f"the match frame carries {dups} duplicated match_id rows")
    frame["outcome"] = np.where(
        frame["home_score"] > frame["away_score"], 0, np.where(frame["home_score"] == frame["away_score"], 1, 2)
    )
    return frame.sort_values("match_date", kind="stable").reset_index(drop=True)


def settlement(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(h, a, ok) the full-time markets settle on: grading_outcomes.
    settlement_scores per row (the 90-minute pair when metadata carries one)."""
    h = np.zeros(len(frame), dtype=int)
    a = np.zeros(len(frame), dtype=int)
    ok = np.zeros(len(frame), dtype=bool)
    cols = zip(frame["home_score"], frame["away_score"], frame["regulation_home_score"], frame["regulation_away_score"])
    for r, (hs, as_, rh, ra) in enumerate(cols):
        meta = {}
        if rh is not None and not (isinstance(rh, float) and math.isnan(rh)):
            meta[go.REGULATION_HOME_KEY] = rh
        if ra is not None and not (isinstance(ra, float) and math.isnan(ra)):
            meta[go.REGULATION_AWAY_KEY] = ra
        s = go.settlement_scores(int(hs), int(as_), meta, sport="soccer", model_name=MODEL_NAME)
        if s is None or s[0] is None or s[1] is None:
            continue
        h[r], a[r], ok[r] = int(s[0]), int(s[1]), True
    return h, a, ok


def load_ou_rows(database_url: str, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, List[Dict[str, Any]]]:
    from utils.training_data import prematch_odds  # type: ignore

    rows = members._read_sql(
        database_url, prematch_odds(OU_ODDS_SQL), {"start": start.to_pydatetime(), "end": end.to_pydatetime()}
    )
    out: Dict[str, List[Dict[str, Any]]] = {}
    for rec in rows.to_dict("records"):
        out.setdefault(str(rec["match_id"]), []).append(rec)
    return out


def live_check(database_url: str, version: str, dc: Any) -> Dict[str, Any]:
    """Stored live predictions of `version` re-derived from the served
    Dixon-Coles with the STORED 1x2 as the target and the stored
    market_over_2.5 as the anchor (metadata.reconcile '1x2+ou2.5'): every
    stored probability of every derived market must come back to
    LIVE_CHECK_ATOL (LIVE_CHECK_ANCHORED_ATOL on an anchored row, whose
    anchor was stored rounded), or this raises."""
    from predictors.market_derivation import derive_from_lambdas  # type: ignore

    rows = members._read_sql(database_url, LIVE_CHECK_SQL, {"version": version})
    if rows.empty:
        raise RuntimeError(f"no stored predictions for model_version {version!r}")
    derived_types = set(u.market for u in UNITS)
    checked = anchored = skipped = 0
    worst = {"1x2": 0.0, "1x2+ou2.5": 0.0}
    worst_at: Dict[str, Optional[str]] = {"1x2": None, "1x2+ou2.5": None}
    for match_id, grp in rows.groupby("match_id"):
        by_type = {r.prediction_type: r for r in grp.itertuples()}
        mr, ou = by_type.get("match_result"), by_type.get("over_under")
        meta = (ou.metadata if ou is not None else None) or {}
        if mr is None or ou is None or meta.get("reconcile") not in ("1x2", "1x2+ou2.5"):
            skipped += 1
            continue
        tgt = tuple(float(mr.probabilities[k]) for k in ("home", "draw", "away"))
        p_over = float(meta["market_over_2.5"]) if meta.get("reconcile") == "1x2+ou2.5" else None
        first = grp.iloc[0]
        h_lam, a_lam = dc.lambdas_for_match(first["home_team"], first["away_team"], league=first["league_id"])
        mk = derive_from_lambdas(float(h_lam), float(a_lam), float(dc.rho), target_1x2=tgt, target_over_25=p_over)
        for ptype, rec in by_type.items():
            if ptype not in derived_types:
                continue
            stored = rec.probabilities or {}
            mine = mk[ptype]
            if set(stored) != set(mine):
                raise RuntimeError(f"live check {match_id} {ptype}: stored keys {sorted(stored)} vs {sorted(mine)}")
            gap = max(abs(float(stored[k]) - float(mine[k])) for k in stored)
            kind = meta["reconcile"]
            if gap > worst[kind]:
                worst[kind], worst_at[kind] = gap, f"{match_id} {ptype}"
        checked += 1
        anchored += int(p_over is not None)
    for kind, atol in (("1x2", LIVE_CHECK_ATOL), ("1x2+ou2.5", LIVE_CHECK_ANCHORED_ATOL)):
        if worst[kind] > atol:
            raise RuntimeError(
                f"live check: the served derivation differs from stored {kind} predictions by {worst[kind]:.2e} "
                f"at {worst_at[kind]} (tolerance {atol:.0e})"
            )
    return {
        "version": version,
        "matches": checked,
        "anchored": anchored,
        "skipped": skipped,
        "max_abs": worst["1x2"],
        "max_abs_anchored": worst["1x2+ou2.5"],
    }


# ── Fold scoring (pure given the models) ─────────────────────────────

TOTAL_GOALS_GRID = np.add.outer(np.arange(GRID_SIDE), np.arange(GRID_SIDE)).astype(float)


def regime_matrices(
    lams: np.ndarray, rho: float, targets: np.ndarray, a_live: np.ndarray, a_proxy: np.ndarray
) -> Dict[str, np.ndarray]:
    """{regime: (m, n, n)} one arm's served matrices in each anchor regime.
    a_live / a_proxy hold a VALID anchor or NaN; a proxy row that carries the
    live anchor reuses the live matrix (it is the same number)."""
    m = len(lams)
    none = arm_matrices(lams, rho, targets, [None] * m)
    live = none.copy()
    lv = np.isfinite(np.asarray(a_live, dtype=float))
    px = np.isfinite(np.asarray(a_proxy, dtype=float))
    for r in np.flatnonzero(lv):
        live[r] = served_matrix(lams[r, 0], lams[r, 1], rho, targets[r], float(a_live[r]))
    proxy = live.copy()
    for r in np.flatnonzero(px):
        if lv[r] and float(a_proxy[r]) == float(a_live[r]):
            continue
        proxy[r] = served_matrix(lams[r, 0], lams[r, 1], rho, targets[r], float(a_proxy[r]))
    return {"none": none, "live": live, "proxy": proxy}


def _sample_rows(n: int, preferred: np.ndarray, *seed_parts: str) -> List[int]:
    """Up to DERIVE_SAMPLE rows for the derive_from_lambdas check, anchored
    rows first so the tilt path is exercised whenever a fold has any.
    Deterministic (crc32 of the fold / arm / regime)."""
    import zlib

    rng = np.random.default_rng(zlib.crc32("|".join(seed_parts).encode()))
    pref = np.flatnonzero(np.asarray(preferred, dtype=bool))
    rest = np.setdiff1d(np.arange(n), pref)
    take = list(rng.permutation(pref)[: DERIVE_SAMPLE // 2]) + list(rng.permutation(rest))
    return sorted(int(x) for x in take[:DERIVE_SAMPLE])


def score_fold(
    label: str,
    ev: pd.DataFrame,
    targets: np.ndarray,
    a_live: np.ndarray,
    a_proxy: np.ndarray,
    truth: Dict[str, np.ndarray],
    models: Dict[str, Any],
    constants: Dict[str, Dict[str, np.ndarray]],
    weights: Dict[str, np.ndarray],
    served_lams: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, str], Dict[str, float]]:
    """Score one fold. Returns (per-row arrays, {arm: why disqualified},
    fidelity maxima). Every key exists for every arm of DC_ARMS and
    CONSTANT_ARMS — NaN for an arm that did not serve this fold — so folds
    always stack aligned.

    `a_live` / `a_proxy` hold a valid anchor or NaN; `truth` is vector_truth
    of the fold's settlement scores (h, a in ev['s_home'] / ev['s_away']);
    `constants` the per-league constants' unit probabilities. served must
    produce valid probabilities or this raises; a challenger that does not
    is reported as disqualified and left out."""
    m = len(ev)
    h = ev["s_home"].to_numpy(dtype=int)
    a = ev["s_away"].to_numpy(dtype=int)
    lv = np.isfinite(np.asarray(a_live, dtype=float))
    px = np.isfinite(np.asarray(a_proxy, dtype=float))
    anchors = {"none": [None] * m, "live": anchor_list(a_live), "proxy": anchor_list(a_proxy)}
    prefer = {"none": np.zeros(m, dtype=bool), "live": lv, "proxy": px}
    out: Dict[str, np.ndarray] = {}
    disq: Dict[str, str] = {}
    stats = {"derive": 0.0, "invariance": 0.0, "anchor": 0.0}
    probs: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for arm in DC_ARMS:
        mdl = models.get(arm)
        if mdl is None:
            continue
        lams = served_lams if (arm == "served" and served_lams is not None) else dc_lambdas(mdl, ev)
        rho = float(getattr(mdl, "rho", 0.0) or 0.0)
        mats = regime_matrices(lams, rho, targets, a_live, a_proxy)
        arm_probs: Dict[str, Dict[str, np.ndarray]] = {}
        bad: Optional[str] = None
        for regime in REGIMES:
            pr = unit_probabilities(mats[regime], weights)
            bad = invalid_probabilities(pr, UNIT_NAMES)
            if bad is not None:
                bad = f"[{regime}] {bad}"
                break
            arm_probs[regime] = pr
        if bad is not None:
            if arm == "served":
                raise RuntimeError(f"served fold {label}: {bad}")
            disq[arm] = bad
            continue
        for regime in REGIMES:
            rows = _sample_rows(m, prefer[regime], label, arm, regime)
            gap = derive_fidelity(lams, rho, targets, anchors[regime], arm_probs[regime], rows)
            stats["derive"] = max(stats["derive"], gap)
        probs[arm] = arm_probs
        out[f"lam_h::{arm}"] = lams[:, 0].astype(float)
        out[f"lam_a::{arm}"] = lams[:, 1].astype(float)
        out[f"rho::{arm}"] = np.full(m, rho)
        out[f"known::{arm}"] = members.known_teams(mdl, ev["home_team"], ev["away_team"])
        for regime in REGIMES:
            out[f"etot::{arm}::{regime}"] = np.einsum("rij,ij->r", mats[regime], TOTAL_GOALS_GRID)
    if "served" not in probs:
        raise RuntimeError(f"fold {label}: served did not score")
    for regime in REGIMES:
        stats["invariance"] = max(stats["invariance"], check_invariance({x: probs[x][regime] for x in probs}))
    for regime, ok, vals in (("live", lv, a_live), ("proxy", px, a_proxy)):
        if not ok.any():
            continue
        for arm in probs:
            gap = float(np.max(np.abs(probs[arm][regime][PRIMARY][ok, 0] - np.asarray(vals, dtype=float)[ok])))
            if not math.isfinite(gap) or gap > INVARIANCE_ATOL:
                raise RuntimeError(f"{arm} [{regime}] over_2.5 misses its anchor by {gap:.2e} in fold {label}")
            stats["anchor"] = max(stats["anchor"], gap)
    nan = np.full(m, np.nan)
    for arm in DC_ARMS:
        if arm not in probs:
            for key in ("lam_h", "lam_a", "rho", "known"):
                out[f"{key}::{arm}"] = nan.copy()
        for regime in REGIMES:
            if arm not in probs:
                out[f"etot::{arm}::{regime}"] = nan.copy()
                for u in UNIT_NAMES:
                    out[f"brier::{arm}::{regime}::{u}"] = nan.copy()
                out[f"p::{arm}::{regime}::{PRIMARY}"] = nan.copy()
                out[f"p::{arm}::{regime}::btts"] = nan.copy()
                continue
            pr = probs[arm][regime]
            for u in UNIT_NAMES:
                y = listed_truth(pr["_listed_order"], h, a) if u == LISTED else truth[u]
                out[f"brier::{arm}::{regime}::{u}"] = brier_rows(pr[u], y)
            out[f"p::{arm}::{regime}::{PRIMARY}"] = pr[PRIMARY][:, 0].copy()
            out[f"p::{arm}::{regime}::btts"] = pr["btts"][:, 0].copy()
    for cname in CONSTANT_ARMS:
        cp = constants[cname]
        bad = invalid_probabilities(cp, CONSTANT_UNITS)
        if bad is not None:
            raise RuntimeError(f"{cname} fold {label}: {bad}")
        for u in CONSTANT_UNITS:
            out[f"brier::{cname}::-::{u}"] = brier_rows(cp[u], truth[u])
        out[f"p::{cname}::-::{PRIMARY}"] = cp[PRIMARY][:, 0].copy()
        out[f"p::{cname}::-::btts"] = cp["btts"][:, 0].copy()
    return out, disq, stats


@dataclass
class ConstantFrame:
    """Every constant unit's realised outcome over the (date-sorted) frame,
    built once: Y0 = outcome with ungradable cells zeroed, V = 1 where the
    unit grades, codes = league index. A fold's fit rows are a prefix."""

    Y0: np.ndarray
    V: np.ndarray
    codes: np.ndarray
    n_codes: int
    layout: Dict[str, slice]
    league_codes: Dict[str, int]


def build_constant_frame(frame: pd.DataFrame) -> ConstantFrame:
    dates = frame["match_date"].to_numpy()
    if len(dates) > 1 and not (dates[1:] >= dates[:-1]).all():
        raise RuntimeError("the frame must be sorted by match_date (a fold's fit rows are a prefix)")
    truth = vector_truth(frame["s_home"].to_numpy(dtype=int), frame["s_away"].to_numpy(dtype=int))
    ok = frame["s_ok"].to_numpy(dtype=bool)
    layout: Dict[str, slice] = {}
    blocks = []
    at = 0
    for u in CONSTANT_UNITS:
        k = truth[u].shape[1]
        layout[u] = slice(at, at + k)
        at += k
        blocks.append(truth[u])
    Y = np.concatenate(blocks, axis=1)
    V = np.isfinite(Y) & ok[:, None]
    league_codes = {lg: k for k, lg in enumerate(sorted(frame["league_id"].astype(str).unique()))}
    codes = frame["league_id"].astype(str).map(league_codes).to_numpy(dtype=int)
    return ConstantFrame(
        Y0=np.where(V, Y, 0.0).astype(np.float32),
        V=V.astype(np.float32),
        codes=codes,
        n_codes=len(league_codes),
        layout=layout,
        league_codes=league_codes,
    )


def fold_constants(
    cf: ConstantFrame, n_fit: int, ev_leagues: Sequence[Any], w_dw: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """base_league (unweighted) and base_league_dw (recency-weighted) unit
    probabilities for a fold's eval rows, fitted on the frame's first n_fit rows."""
    ev_codes = np.array([cf.league_codes.get(str(lg), -1) for lg in ev_leagues], dtype=int)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for cname, w in (("base_league", np.ones(n_fit)), ("base_league_dw", np.asarray(w_dw, dtype=float))):
        g, by = fit_constants(cf.Y0[:n_fit], cf.V[:n_fit], cf.codes[:n_fit], cf.n_codes, w)
        pred = predict_constants(g, by, ev_codes)
        out[cname] = {u: pred[:, cf.layout[u]] for u in CONSTANT_UNITS}
    return out


# ── Run ──────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> Dict[str, Any]:
    members._add_src_paths()
    from predictors.market_derivation import MAX_GOALS_DERIVE, valid_ou_anchor  # type: ignore
    from predictors.model_config import DIXON_COLES_CONFIG  # type: ignore

    if MAX_GOALS_DERIVE + 1 != GRID_SIDE:
        raise RuntimeError(f"MAX_GOALS_DERIVE is {MAX_GOALS_DERIVE}; this harness reads an {GRID_SIDE}-wide grid")
    decay = float(DIXON_COLES_CONFIG.hyperparameters.get("time_decay", 0.0))
    frame = load_frame(args.database_url)
    ensemble, meta = members.load_ensemble(args.model_path)
    served = ensemble.models[SERVED_DC]
    fingerprint = members.member_fingerprint(args.model_path, SERVED_DC)
    if args.served_sha256 and fingerprint["sha256"] != args.served_sha256:
        raise RuntimeError(f"served artifact sha256 {fingerprint['sha256']} is not the pinned {args.served_sha256}")
    side_loaded = ht.load_serving(fingerprint["path"])
    fit_slice = members.served_fit_slice(frame, args.model_path, set(served.team_attack))
    if fit_slice.get("checked") and not fit_slice.get("identical"):
        logger.error("the served Dixon-Coles' teams are not the 70%% slice's teams: %s", fit_slice)
    seen_ids = members.load_seen_ids(args.model_path)

    merge_stats = None
    merge_map = None
    merge_new_ids: set = set()
    if args.simulate_merge_map:
        merge_map = pd.read_csv(args.simulate_merge_map, comment="#", dtype=str)
        merge_new_ids = set(merge_map["new_team_id"].astype(str))
    elif args.merge_map and Path(args.merge_map).exists():
        merge_new_ids = set(pd.read_csv(args.merge_map, comment="#", dtype=str)["new_team_id"].astype(str))
    frame["identity_split"] = frame["home_team_id"].isin(merge_new_ids) | frame["away_team_id"].isin(merge_new_ids)

    oos, seen_cutoff, n_late = members.out_of_sample_mask(frame, seen_ids)
    span_start = pd.Timestamp(frame.loc[oos, "match_date"].min())
    end_day = args.end or frame.loc[oos, "match_date"].max().strftime("%Y-%m-%d")
    start_day = args.start or span_start.strftime("%Y-%m-%d")
    folds = ft.build_folds(start_day, end_day)
    span_end = folds[-1].eval_end
    features = members.load_features(args.database_url, folds[0].eval_start, span_end)
    frame["has_features"] = frame["match_id"].isin(features.keys())
    if merge_map is not None:
        frame, merge_stats = members.simulate_team_merge(frame, merge_map)
        oos = (frame["match_date"] > seen_cutoff).to_numpy() & ~frame["match_id"].isin(seen_ids).to_numpy()
        logger.info("simulated merge: %s", merge_stats)
    frame["s_home"], frame["s_away"], frame["s_ok"] = settlement(frame)

    in_span = ((frame["match_date"] >= folds[0].eval_start) & (frame["match_date"] < span_end)).to_numpy()
    has_f = frame["has_features"].to_numpy(dtype=bool)
    s_ok = frame["s_ok"].to_numpy(dtype=bool)
    no_features = int((oos & in_span & ~has_f).sum())
    no_settlement = int((oos & in_span & has_f & ~s_ok).sum())
    eval_df = frame[oos & in_span & has_f & s_ok].copy()

    medians: Dict[str, float] = {}
    eval_feats = [features[mid] for mid in eval_df["match_id"]]
    if any(members.needs_medians(f) for f in eval_feats):
        medians = members.load_medians(args.database_url)
    rows = []
    n_filled = 0
    for f, (h, a, lg) in zip(eval_feats, eval_df[["home_team", "away_team", "league_id"]].itertuples(index=False)):
        row, k = members.serve_row(f, {"home_team": h, "away_team": a, "league_id": lg}, medians)
        rows.append(row)
        n_filled += k
    serve_x = pd.DataFrame(rows, index=eval_df.index)
    required = sorted({c for m in members.LEARNED_MEMBERS for c in ensemble.models[m].feature_names})
    unservable = members.rows_missing_features(serve_x, required)
    eval_df = eval_df[~unservable]
    serve_x = serve_x[~unservable]

    # The served 1x2, once: every arm reconciles to exactly these numbers.
    targets_all = np.asarray(ensemble.predict_proba(serve_x), dtype=float)
    members.require_finite("served 1x2 target", targets_all)
    fid_batch, fid_single, n_single = members.serve_path_fidelity(ensemble, serve_x, targets_all)
    pos = {idx: i for i, idx in enumerate(eval_df.index)}

    # Both loaders of the served artifact must agree to the bit.
    lam_member = dc_lambdas(served, eval_df)
    lam_side = dc_lambdas(side_loaded, eval_df)
    loader_gap = float(np.max(np.abs(lam_member - lam_side))) if len(eval_df) else 0.0
    if loader_gap > 0.0 or float(served.rho) != float(side_loaded.rho):
        raise RuntimeError(f"the ensemble member and the side loader disagree (lambda gap {loader_gap:.2e})")

    odds = load_ou_rows(args.database_url, folds[0].eval_start, span_end)
    a_live = np.full(len(eval_df), np.nan)
    a_proxy = np.full(len(eval_df), np.nan)
    for r, (mid, ko) in enumerate(zip(eval_df["match_id"], eval_df["match_date"])):
        recs = odds.get(str(mid), [])
        lv = live_anchor(recs, ko, str(mid))
        lv = lv if valid_ou_anchor(lv) else None
        px = lv if lv is not None else proxy_anchor(recs)
        px = px if valid_ou_anchor(px) else None
        a_live[r] = np.nan if lv is None else lv
        a_proxy[r] = np.nan if px is None else px

    eh = eval_df["s_home"].to_numpy(dtype=int)
    ea = eval_df["s_away"].to_numpy(dtype=int)
    truth_eval = vector_truth(eh, ea)
    for u, arr in graded_truth(eh, ea).items():
        mine = truth_eval[u]
        if not (
            np.array_equal(np.isnan(arr), np.isnan(mine)) and np.array_equal(np.nan_to_num(arr), np.nan_to_num(mine))
        ):
            raise RuntimeError(f"vectorised truth for {u} disagrees with grading_outcomes")
    cf = build_constant_frame(frame)
    frame_dates = frame["match_date"]

    weights = unit_weights()
    c = members.Collected()
    fits: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    disqualified: Dict[str, Dict[str, Any]] = {}
    fid = {"derive": 0.0, "invariance": 0.0, "anchor": 0.0}
    with tempfile.TemporaryDirectory() as tmp:
        frozen_fit = frame[frame_dates < folds[0].fit_cutoff]
        frozen, frozen_res = fit_refit(frozen_fit, tmp, "frozen")
        fits.append(_fit_record("frozen", "refit_frozen", frozen_fit, frozen_res))
        for fold in folds:
            fit_df, _ = ft.split_fold(frame, fold)
            ev = eval_df[(eval_df["match_date"] >= fold.eval_start) & (eval_df["match_date"] < fold.eval_end)]
            if args.limit:
                ev = ev.head(int(args.limit))
            if len(ev) < MIN_FOLD_EVAL:
                skipped.append({"fold": fold.label, "n_eval": int(len(ev))})
                logger.warning("fold %s skipped: %d eval rows < %d", fold.label, len(ev), MIN_FOLD_EVAL)
                continue
            members.check_fold_disjoint(fit_df, ev, fold)
            n_fit = len(fit_df)
            if n_fit and not (frame_dates.iloc[:n_fit] < fold.fit_cutoff).all():
                raise RuntimeError(f"fold {fold.label}: the fit rows are not the frame's first {n_fit} rows")
            idx = np.array([pos[i] for i in ev.index])
            fixtures = list(zip(ev["home_team"], ev["away_team"], ev["league_id"]))
            models: Dict[str, Any] = {"served": served}
            if "refit" not in disqualified:
                refit, res = fit_refit(fit_df, tmp, fold.label)
                fits.append(_fit_record(fold.label, "refit", fit_df, res, fold))
                why = htrainer.fit_refusal(res, refit, fixtures)
                if why is None:
                    models["refit"] = refit
                else:
                    disqualified["refit"] = {"fold": fold.label, "reason": f"the trainer would refuse this fit: {why}"}
                    logger.error("refit DISQUALIFIED at fold %s: %s", fold.label, why)
            if "refit_frozen" not in disqualified:
                why = htrainer.fit_refusal(frozen_res, frozen, fixtures)
                if why is None:
                    models["refit_frozen"] = frozen
                else:
                    disqualified["refit_frozen"] = {"fold": fold.label, "reason": f"the trainer would refuse it: {why}"}
                    logger.error("refit_frozen DISQUALIFIED at fold %s: %s", fold.label, why)
            constants = fold_constants(
                cf, n_fit, ev["league_id"].tolist(), ft.decay_weights(fit_df, fold.fit_cutoff, decay)
            )
            out, fold_disq, stats = score_fold(
                fold.label,
                ev,
                targets_all[idx],
                a_live[idx],
                a_proxy[idx],
                {u: v[idx] for u, v in truth_eval.items()},
                models,
                constants,
                weights,
                served_lams=lam_member[idx],
            )
            for arm, why in fold_disq.items():
                disqualified[arm] = {"fold": fold.label, "reason": why}
                logger.error("%s DISQUALIFIED at fold %s: %s", arm, fold.label, why)
            for key, val in stats.items():
                fid[key] = max(fid[key], val)
            for key, val in out.items():
                c.add(key, val)
            c.add("fold", np.full(len(ev), fold.label, dtype=object))
            c.add("match_id", ev["match_id"].to_numpy(dtype=object))
            c.add("match_date", ev["match_date"].astype(str).to_numpy(dtype=object))
            c.add("league_id", ev["league_id"].to_numpy(dtype=object))
            c.add("league", ev["league_name"].to_numpy(dtype=object))
            c.add("home_team", ev["home_team"].to_numpy(dtype=object))
            c.add("away_team", ev["away_team"].to_numpy(dtype=object))
            c.add("s_home", eh[idx].astype(float))
            c.add("s_away", ea[idx].astype(float))
            for k, cls in enumerate(("home", "draw", "away")):
                c.add(f"target::{cls}", targets_all[idx, k])
            c.add("anchor_live", a_live[idx])
            c.add("anchor_proxy", a_proxy[idx])
            c.add("identity_split", ev["identity_split"].to_numpy(dtype=float))
            c.add("espn_era", (ev["match_date"] >= ESPN_ERA_START).to_numpy(dtype=float))
            logger.info(
                "fold %s: fit=%d eval=%d live-anchored=%d proxy-anchored=%d",
                fold.label,
                n_fit,
                len(ev),
                int(np.isfinite(a_live[idx]).sum()),
                int(np.isfinite(a_proxy[idx]).sum()),
            )

    live = (
        live_check(args.database_url, args.live_check_model_version, served) if args.live_check_model_version else None
    )
    meta_out = {
        "model_path": args.model_path,
        "ensemble_artifact": str(meta),
        "served_fingerprint": fingerprint,
        "served_fit_slice": fit_slice,
        "seen_ids": len(seen_ids),
        "seen_cutoff": str(seen_cutoff),
        "unseen_rows_inside_seen_span": n_late,
        "eval_rows_without_features": no_features,
        "eval_rows_without_settlement": no_settlement,
        "eval_rows_unservable": int(unservable.sum()),
        "median_filled_values": n_filled,
        "fidelity_batch_max_abs": fid_batch,
        "fidelity_single_row_max_abs": fid_single,
        "fidelity_single_row_sample": n_single,
        "loader_lambda_gap": loader_gap,
        "derive_fidelity_max_abs": fid["derive"],
        "invariance_max_abs": fid["invariance"],
        "anchor_max_abs": fid["anchor"],
        "folds_requested": [f.label for f in folds],
        "skipped_folds": skipped,
        "disqualified": disqualified,
        "fits": fits,
        "simulated_merge": merge_stats,
        "corpus_rows": int(len(frame)),
        "live_check": live,
        "decay": decay,
    }
    return build_report(c.stack(), meta_out, out_dir=args.out_dir)


def _fit_record(label: str, arm: str, fit_df: pd.DataFrame, res: Dict[str, Any], fold: Any = None) -> Dict[str, Any]:
    rec = {
        "fold": label,
        "arm": arm,
        "n_fit": int(len(fit_df)),
        "fit_newest": str(fit_df["match_date"].max()),
        "teams": int(res.get("team_count") or 0),
        "leagues": int(res.get("league_count") or 0),
        "converged_at": res.get("iterations"),
        "rho": res.get("rho"),
        "fitted_total": res.get("fitted_mean_total_goals"),
        "observed_total": res.get("observed_mean_total_goals"),
    }
    if fold is not None:
        rec["newest_fit_gap_days"] = float((fold.fit_cutoff - fit_df["match_date"].max()).total_seconds() / 86400)
    return rec


# ── Report (pure given the stacked per-row arrays) ───────────────────


def matched_constant(arm: str) -> str:
    return "base_league_dw" if arm in DECAYED_ARMS else "base_league"


def build_report(s: Dict[str, np.ndarray], meta: Dict[str, Any], out_dir: Optional[str] = None) -> Dict[str, Any]:
    folds_arr = np.asarray(s["fold"])
    labels = list(dict.fromkeys(folds_arr.tolist()))
    disq = meta.get("disqualified") or {}

    def brier(arm: str, regime: str, unit: str) -> np.ndarray:
        if arm in CONSTANT_ARMS:
            return s[f"brier::{arm}::-::{unit}"]
        return s[f"brier::{arm}::{regime}::{unit}"]

    scored = [a for a in DC_ARMS if a not in disq]
    for regime in REGIMES:
        for u in UNIT_NAMES:
            if u != "draw_no_bet" and np.isnan(brier("served", regime, u)).any():
                raise RuntimeError(f"served carries a missing score for {u} [{regime}]")
    for arm in scored:
        if np.isnan(s[f"lam_h::{arm}"]).any():
            raise RuntimeError(f"{arm} is missing rows although it was never disqualified")

    vs_served = {
        regime: {
            arm: {u: pair(brier(arm, regime, u), brier("served", regime, u), folds_arr) for u in UNIT_NAMES}
            for arm in CHALLENGERS
            if arm in scored
        }
        for regime in REGIMES
    }
    decision = decide({r: vs_served[r].get(DECIDING_ARM, {}) for r in REGIMES if DECIDING_ARM in vs_served[r]}, disq)
    live_anch = np.isfinite(s["anchor_live"])
    proxy_anch = np.isfinite(s["anchor_proxy"])
    espn = s["espn_era"].astype(bool)
    known = s["known::served"]
    split = s["identity_split"].astype(bool)
    strata = {
        "live_anchored": live_anch,
        "not_live_anchored": ~live_anch,
        "proxy_anchored": proxy_anch,
        "football_data_era_to_2026_05": ~espn,
        "espn_era_2026_06_on": espn,
        "served_knows_both": known == 2,
        "served_knows_one": known == 1,
        "served_knows_neither": known == 0,
        "identity_split_fixtures": split,
    }
    y_over = (s["s_home"] + s["s_away"] > 2.5).astype(float)
    y_btts = ((s["s_home"] >= 1) & (s["s_away"] >= 1)).astype(float)
    report: Dict[str, Any] = {
        "meta": meta,
        "n_eval": int(len(folds_arr)),
        "folds": labels,
        "anchored": {
            "live": int(live_anch.sum()),
            "proxy": int(proxy_anch.sum()),
            "live_by_era": {"football_data": int((live_anch & ~espn).sum()), "espn": int((live_anch & espn).sum())},
            "espn_rows": int(espn.sum()),
        },
        "decision": decision,
        "brier": {
            regime: {
                arm: {
                    u: float(np.nanmean(brier(arm, regime, u)))
                    for u in (UNIT_NAMES if arm in DC_ARMS else CONSTANT_UNITS)
                }
                for arm in scored + list(CONSTANT_ARMS)
            }
            for regime in REGIMES
        },
        "vs_served": vs_served,
        "verdicts_vs_served": {
            regime: {arm: ship_strict(vs_served[regime][arm][PRIMARY]) for arm in vs_served[regime]}
            for regime in REGIMES
        },
        "regressions": {
            regime: {arm: blocking_regressions(vs_served[regime][arm]) for arm in vs_served[regime]}
            for regime in REGIMES
        },
        "vs_matched_constant": {
            regime: {
                arm: {
                    "constant": matched_constant(arm),
                    **{
                        u: pair(brier(arm, regime, u), brier(matched_constant(arm), regime, u), folds_arr)
                        for u in (PRIMARY, "btts", GRID)
                    },
                }
                for arm in scored
            }
            for regime in REGIMES
        },
        "head_to_head": {
            regime: (
                {
                    "refit_frozen - refit": pair(
                        brier("refit_frozen", regime, PRIMARY), brier("refit", regime, PRIMARY), folds_arr
                    )
                }
                if "refit" in scored and "refit_frozen" in scored
                else {}
            )
            for regime in REGIMES
        },
        "strata": {
            name: {
                "n": int(mask.sum()),
                **{
                    f"{arm} [{regime}]": pair(
                        brier(arm, regime, PRIMARY), brier("served", regime, PRIMARY), folds_arr, mask
                    )
                    for arm in CHALLENGERS
                    if arm in scored
                    for regime in ("live", "none")
                    if mask.sum() > 1
                },
            }
            for name, mask in strata.items()
        },
        "per_fold_vs_served": {
            arm: [
                {
                    "fold": lab,
                    "n": int((folds_arr == lab).sum()),
                    "delta_live": float(
                        np.nanmean(
                            brier(arm, "live", PRIMARY)[folds_arr == lab]
                            - brier("served", "live", PRIMARY)[folds_arr == lab]
                        )
                    ),
                    "anchored_live": int((live_anch & (folds_arr == lab)).sum()),
                }
                for lab in labels
            ]
            for arm in CHALLENGERS
            if arm in scored
        },
        "calibration_in_the_large": {
            "realised": {"over_2.5": float(y_over.mean()), "btts": float(y_btts.mean())},
            "mean_pred": {
                f"{arm} [{regime}]": {
                    "over_2.5": float(np.mean(s[f"p::{arm}::{regime}::{PRIMARY}"])),
                    "btts": float(np.mean(s[f"p::{arm}::{regime}::btts"])),
                    "e_total": float(np.mean(s[f"etot::{arm}::{regime}"])),
                }
                for arm in scored
                for regime in REGIMES
            },
            "realised_total": float((s["s_home"] + s["s_away"]).mean()),
        },
        "team_coverage": {
            arm: {
                "both": float((s[f"known::{arm}"] == 2).mean()),
                "one": float((s[f"known::{arm}"] == 1).mean()),
                "neither": float((s[f"known::{arm}"] == 0).mean()),
            }
            for arm in scored
        },
    }
    if out_dir:
        report["outputs"] = write_outputs(s, out_dir)
    return report


def write_outputs(s: Dict[str, np.ndarray], out_dir: str) -> Dict[str, str]:
    """per_match.csv.gz: one row per eval match with the served 1x2 target,
    both anchors, every Dixon-Coles arm's lambdas / rho (from which every
    served probability can be re-derived with derive_from_lambdas), the
    primary and BTTS probabilities per arm and regime, and the settlement
    score. scores.npz: every per-row unit Brier, keyed brier::<arm>::<regime>::<unit>."""
    os.makedirs(out_dir, exist_ok=True)
    cols = {k: v for k, v in s.items() if not k.startswith("brier::")}
    csv_path = os.path.join(out_dir, "per_match.csv.gz")
    pd.DataFrame(cols).to_csv(csv_path, index=False, compression="gzip")
    npz_path = os.path.join(out_dir, "scores.npz")
    np.savez_compressed(
        npz_path,
        fold=np.asarray(s["fold"]).astype(str),
        match_id=np.asarray(s["match_id"]).astype(str),
        **{k: v for k, v in s.items() if k.startswith("brier::")},
    )
    return {"per_match_csv": csv_path, "scores_npz": npz_path}


# ── Formatting ───────────────────────────────────────────────────────


def _f(v: Any, spec: str = "+.5f") -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, spec)


def _pair_line(name: str, p: Dict[str, Any]) -> str:
    return (
        f"  {name:<34} {_f(p.get('delta'))} ±{_f(p.get('se'), '.5f')}   cluster {_f(p.get('delta_cluster'))} "
        f"±{_f(p.get('se_cluster'), '.5f')}   t [{_f(p.get('lower_95_cluster_t'))}, {_f(p.get('upper_95_cluster_t'))}]"
        f"   n={p.get('n')}"
    )


def format_report(r: Dict[str, Any]) -> str:
    m = r["meta"]
    fp = m["served_fingerprint"]
    d = r["decision"]
    an = r["anchored"]
    L = [
        f"Soccer FULL-TIME Dixon-Coles (derived markets) A/B: {r['n_eval']} eval matches, folds "
        f"{r['folds'][0]}..{r['folds'][-1]} ({len(r['folds'])}; skipped {m['skipped_folds'] or 'none'})",
        f"served: {fp['path']} sha256 {fp['sha256'][:12]}, {fp['teams']} teams, {fp['league_baselines']} league "
        f"baselines, rho {fp['rho']}; fit slice {m['served_fit_slice']}",
        f"1x2 incumbent: {m['ensemble_artifact']}; seen ids {m['seen_ids']} (newest {m['seen_cutoff']})",
        f"excluded: {m['eval_rows_without_features']} without features, {m['eval_rows_unservable']} unservable, "
        f"{m['eval_rows_without_settlement']} without a settlement score, {m['unseen_rows_inside_seen_span']} unseen "
        f"inside the seen span; median-filled values {m['median_filled_values']}",
        f"fidelity: 1x2 single-row vs batch {m['fidelity_single_row_max_abs']:.1e} "
        f"(n={m['fidelity_single_row_sample']}); member vs side loader lambdas {m['loader_lambda_gap']:.1e}; "
        f"derive_from_lambdas vs readout {m['derive_fidelity_max_abs']:.1e}; "
        f"invariant units {m['invariance_max_abs']:.1e}; anchor hit {m['anchor_max_abs']:.1e}",
        f"anchored rows: live {an['live']} (football-data era {an['live_by_era']['football_data']}, ESPN era "
        f"{an['live_by_era']['espn']} of {an['espn_rows']}); proxy {an['proxy']}",
    ]
    if m.get("live_check"):
        lc = m["live_check"]
        L.append(
            f"live check ({lc['version']}): {lc['matches']} matches re-derived from the stored 1x2 + anchor: "
            f"1x2-only max |d| {lc['max_abs']:.1e}, {lc['anchored']} anchored max |d| {lc['max_abs_anchored']:.1e} "
            f"(anchor stored to 6 dp); skipped {lc['skipped']}"
        )
    if m.get("simulated_merge"):
        L.append(f"SIMULATED MERGE (corpus caveat run): {m['simulated_merge']}")
    for arm, dq in (m.get("disqualified") or {}).items():
        L.append(f"DISQUALIFIED {arm} at fold {dq['fold']}: {dq['reason']}")
    L += ["", "team coverage (both / one / neither known):"]
    for arm, cov in r["team_coverage"].items():
        L.append(f"  {arm:<15} {cov['both']:.1%} / {cov['one']:.1%} / {cov['neither']:.1%}")
    ci = r["calibration_in_the_large"]
    L += [
        "",
        f"calibration in the large: realised over_2.5 {ci['realised']['over_2.5']:.4f}, "
        f"btts {ci['realised']['btts']:.4f}, total goals {ci['realised_total']:.4f}",
    ]
    for k, v in ci["mean_pred"].items():
        L.append(f"  {k:<24} over_2.5 {v['over_2.5']:.4f}  btts {v['btts']:.4f}  E[total] {v['e_total']:.4f}")
    for regime in REGIMES:
        tag = "DECIDES" if regime in DECIDING_REGIMES else "context"
        L += [
            "",
            f"[{regime}] ({tag}) every unit vs served (pooled dBrier [t 95% lower bound]; P primary, I invariant,"
            " C context, R = regression on a blocking unit):",
        ]
        hdr = f"  {'unit':<30}{'served':>10}"
        for arm in r["vs_served"][regime]:
            hdr += f"{arm:>30}"
        L.append(hdr)
        for u in UNIT_NAMES:
            mark = "P" if u == PRIMARY else ("I" if u in INVARIANT_UNITS else ("C" if u in CONTEXT_UNITS else " "))
            line = f"  {mark} {u:<28}{r['brier'][regime]['served'][u]:>10.5f}"
            for arm, per in r["vs_served"][regime].items():
                p = per[u]
                rr = "R" if u in r["regressions"][regime].get(arm, {}) else " "
                line += f"{_f(p.get('delta')) + ' [' + _f(p.get('lower_95_cluster_t')) + ']' + rr:>30}"
            L.append(line)
        L += [f"  {PRIMARY} vs served:"]
        L += [_pair_line(arm, per[PRIMARY]) for arm, per in r["vs_served"][regime].items()]
        for name, p in r["head_to_head"][regime].items():
            L.append(_pair_line(name, p))
        L.append("  vs the matched per-league constant (refit arms vs base_league_dw, served vs base_league):")
        for arm, blk in r["vs_matched_constant"][regime].items():
            for u in (PRIMARY, "btts", GRID):
                L.append(_pair_line(f"{arm} - {blk['constant']} {u}", blk[u]))
    L += ["", f"strata ({PRIMARY}, challenger - served):"]
    for name, blk in r["strata"].items():
        L.append(f"  [{name}] n={blk['n']}")
        for k, v in blk.items():
            if isinstance(v, dict):
                L.append(_pair_line(f"    {k}", v))
    L += ["", f"per-fold {PRIMARY} delta vs served [live] (anchored rows per fold in the last line):"]
    L.append(f"  {'arm':<14}" + "".join(f"{lab[2:]:>8}" for lab in r["folds"]))
    for arm, rows in r["per_fold_vs_served"].items():
        L.append(f"  {arm:<14}" + "".join(f"{row['delta_live']:>+8.4f}" for row in rows))
    first = next(iter(r["per_fold_vs_served"].values()), [])
    L.append(f"  {'n':<14}" + "".join(f"{row['n']:>8}" for row in first))
    L.append(f"  {'anchored':<14}" + "".join(f"{row['anchored_live']:>8}" for row in first))
    unconv = [f["fold"] for f in m["fits"] if f.get("converged_at") is None]
    L += ["", f"fits: {len(m['fits'])}; not converged: {unconv or 'none'}"]
    L += ["", "VERDICTS (§4.3 ship_strict on the primary; A2 regressions on the blocking units):"]
    for regime in REGIMES:
        for arm, v in r["verdicts_vs_served"][regime].items():
            regs = r["regressions"][regime].get(arm, {})
            L.append(
                f"  [{regime}] {arm:<13} strict={'SHIP' if v['ship_strict'] else 'no'}  {v['line']}"
                + (f"  | regressions: {', '.join(regs)}" if regs else "")
            )
    L += [
        "",
        f"DECISION (pre-registered: {DECIDING_ARM}, live AND proxy): "
        f"{'SHIP' if d['ship'] else 'DO-NOT-SHIP'} — {d['reason']}",
    ]
    return "\n".join(L)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=None, help="first eval day (default: the 1x2 incumbent's first unseen match)")
    p.add_argument("--end", default=None, help="last eval day, inclusive (default: newest unseen match)")
    p.add_argument("--model-path", default=members.PRODUCTION_MODEL_PATH, help="registry holding the incumbent")
    p.add_argument("--served-sha256", default=None, help="refuse unless the served Dixon-Coles has this sha256")
    p.add_argument(
        "--merge-map",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "soccer_team_merge_map.csv"),
        help="merge map whose new_team_id rows define the identity-split stratum (read only)",
    )
    p.add_argument("--simulate-merge-map", default=None, help="apply this merge map in memory first (corpus caveat)")
    p.add_argument(
        "--live-check-model-version",
        default=None,
        help="also re-derive this model_version's stored live predictions from the served artifact (read only)",
    )
    p.add_argument("--limit", type=int, default=0, help="cap eval rows per fold (smoke run); 0 = all")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-dir", default=None, help="write per_match.csv.gz and scores.npz here")
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
        logger.error("ab_soccer_ft_dixon_coles failed: %s", exc, exc_info=True)
        return 1
    if args.out_json:  # before printing: a formatting bug must not lose the run
        Path(args.out_json).write_text(json.dumps(ht.clean_json(report), indent=2, allow_nan=False))
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
