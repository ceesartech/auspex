"""Soccer asian-handicap (AH) replay: does a monthly refit of the full-time
Dixon-Coles change the AH recommendations, and is there a Brier case for it?

THE QUESTION (operator, 2026-11-01)
  "For the derived-market soccer model, is there a way we can make this
  streamlined for the asian handicap alone?" The derived-market gate
  (scripts/ab_soccer_ft_dixon_coles.py, audit §4 "Soccer full-time
  Dixon-Coles that serves the derived markets") was DO-NOT-SHIP: a monthly
  refit improved the served over_2.5 by only -0.00375 (live regime), short of
  the -0.005 house gate. asian_handicap is the one rec-emitting soccer market
  that artifact shapes, so the question is whether an AH-only deployment (a
  side artifact used for the AH lines only) could be justified on its own.

  This is a READ-ONLY analysis of the gate's own window. Choosing the AH
  market after the gate's per-unit results were read is POST HOC: nothing
  here can ship a model. A forward, pre-registered gate is the only route
  (scripts/ab_soccer_ah_forward_gate.py, a PROPOSAL).

WHAT THE ARTIFACT CAN MOVE ON AH
  precompute_predictions reconciles the Dixon-Coles scoreline to the served
  1x2 (one IPF pass, or the 1x2 + O/U-2.5 tilt when >= 3 fresh books quote
  the 2.5 line), then stores asian_handicap at 17 home-perspective lines as
  effective home / away / push masses (a quarter line splits the stake). After
  the IPF, AH 0, +-0.25 and +-0.5 are functions of the 1x2 alone, so the two
  arms serve them identically; the shaped lines +-0.75..+-2 depend on how the
  artifact spreads mass INSIDE the home-win / draw / away-win regions (e.g.
  P(margin >= 2 | home win)).

INPUTS
  --ftdc-dir   the --out-dir of a run of scripts/ab_soccer_ft_dixon_coles.py
               (per_match.csv.gz: per eval match the served 1x2 target, both
               anchors, every arm's lambdas and rho; scores.npz: every unit's
               per-row Brier). Every arm's AH probabilities are re-derived
               from them through market_derivation.derive_from_lambdas,
               exactly as precompute_predictions calls it.
  the database (a replica of a read-only prod export, or prod itself, read
  only): finished soccer matches, their pre-match AH odds (prematch_odds), the
  odds_snapshots price history, the stored ensemble predictions, the
  production AH recommendations and the bankroll.

PART A — BRIER (context, post hoc; live regime; refit - served)
  * every AH line (12 shaped + 5 invariant) as the gate scored it: the 3-way
    Brier over the effective home / away / push masses;
  * the pooled shaped lines: per match, the MEAN of the 12 shaped-line
    Briers (a mean, so the scale stays a single 3-way Brier; a SUM over 12
    correlated lines would inflate the delta twelvefold without adding
    information);
  * a goal-difference ranked probability score (RPS) over the 7 ordered
    margin classes <=-3, -2, -1, 0, +1, +2, >=+3 — exactly the information
    every AH line in +-2 reads — normalised by K-1 = 6;
  * the scaled-shape bound: the in-sample best rescaling k of refit's
    correction (serve served + k (refit - served)) and its Brier gain, the
    most any shape along the refit's direction could gain on this window.
    [Amended after the run: this line first described a "shift ceiling",
    E[sum (p_refit - p_served)^2], as a bound. It is the gain refit would earn
    if it were the truth, not a ceiling: the measured gain exceeded it.]
  Re-derived matrices must reproduce scores.npz's AH Briers to 1e-12 (the run
  aborts otherwise).

PART A2 — THE FORWARD GATE'S OWN PAIR (amendment 6; planning, post hoc)
  scripts/ab_soccer_ah_forward_gate.py does not score served vs refit: it
  scores production's stored AH row (the newest created before kickoff)
  against the monthly refit reconciled to that row's stored 1x2 and anchor.
  This part scores exactly that pair on every gate eval match of this window
  with such a row, through the same function (score_ah_pair; an invariant-line
  mismatch aborts), and reports per line and pooled the per-match and
  fold-clustered deltas, the per-match SD and the between-fold variance tau2
  (between_fold_variance). Part D plans the forward gate's decision point on
  these numbers, not on the window pair's.

PART B — MONEY: the AH recommendation logic replayed per arm
  Population: every finished soccer match kicking off in [--start, --end)
  with at least one pre-match (prematch_odds) AH odds row. Recommendations
  only exist while the match is upcoming, and the generator deletes and
  re-writes a match's pending recs every 15-minute tick, so the set that
  stands is the one written at the LAST tick before kickoff. That tick is
  replayed: T = the last quarter-hour tick t with t + TICK_LAG_SECONDS before
  kickoff (production's AH recs since 2026-09-05 were written 12.8-14.1 min
  before kickoff on the hour, 3-4 min before kickoffs at :05).
  The odds state at T: a row exists when COALESCE(created_at, timestamp) <=
  T; its price at T is the last odds_snapshots price captured <= T (snapshots
  are written on every price change since 2026-07-13), else the row's own
  price when its timestamp <= T, else the row is dropped and counted; its age
  is T - COALESCE(last_seen_at, timestamp) when that is <= T, else 0 (it was
  still being quoted after T, counted). Then, per arm, exactly the soccer
  generator's path for market asian_handicap, through its own helpers
  (generate_recommendations.selection_value / kelly_fraction /
  display_selection / is_stale_odds, rec_gating.passes_gate / cap_stake):
    eligibility    both teams >= --min-team-history finished matches before T
    best price     per (selection, line): fresh first, then the highest price
                   (best_odds' DISTINCT ON order; ties broken by bookmaker)
    freshness      age <= rec_gating.MAX_ODDS_AGE_HOURS
    EV / floor     EV = win*O + push - 1 >= --ev-threshold, the no-push win
                   probability >= --prob-floor
    gate           ("soccer","asian_handicap"): odds < 4.0, EV <= 0.30,
                   model - market <= 0.10 with the market = the >= 3-book
                   de-vigged consensus at T (power de-vig; proportional before
                   POWER_DEVIG_SINCE in the reproduction check only)
    stake          quarter Kelly on the no-push conditional, capped at 2.5% of
                   the bankroll
  Settlement: grading_outcomes.grade_rec_selection (quarter lines and pushes
  included, 90-minute score) and rec_status_and_pl. Flat P/L stakes 1 unit.
  Price readouts: (i) same-book CLV: odds taken / the last pre-kickoff
  snapshot at the same book - 1 (vw_rec_clv's definition); (ii) market EV at
  bet time: odds taken x the >= 3-book power-de-vigged consensus at the bet's
  own tick - 1 (what the market implied about the bet when it was placed).
  NEITHER IS A CLOSING-LINE VALUE for a rec written at the last tick: no AH
  quote is captured after that tick (the odds fetch runs in the same tick,
  10-55 s before the generator reads; the run counts quotes captured after
  the tick and consensus values that differ at kickoff, and both are 0), so
  the "close" is the bet's own moment and (i) is 0 by construction.
  [Amendment 4: (ii) was first reported as "fair-close CLV" against a
  consensus "at kickoff"; that consensus is identical to the tick's.]
  Caveat on the window: until 2ec19b8 (2026-09-04) an odds row kept its first
  price and first-seen timestamp and had no last_seen_at, so today's
  freshness guard reads almost every pre-2026-09-05 quote as stale: ~93% of
  the window's replayed recs are from REPRO_START on.
  Arms:
    served           the gate's served arm: the 2026-11-01 artifact, the
                     current 1x2 incumbent's target, the live-replayed anchor
    refit            the gate's deciding refit (DIXON_COLES_CONFIG on every
                     finished match before the month), same target and anchor
    stored           production's stored asian_handicap probabilities: the
                     newest row (model_name 'ensemble') created before T
    refit_on_stored  refit reconciled to the stored 1x2 (the match_result row
                     of the stored row's model_version) and the stored anchor
                     (metadata.reconcile '1x2+ou2.5' -> market_over_2.5)
  Comparisons (PAIRS): served vs refit over the whole population (the
  window: today's generator logic applied to every match), and stored vs
  refit_on_stored where production stored predictions (the
  production-faithful pair; the forward gate's pair); both again from
  REPRO_START, where the stored arm is past the 2026-06-15..07-13
  constant-prior incident and production ran today's logic. served vs refit
  over the window is the headline readout. Nothing here decides a ship. Per
  arm: rec count, the overlap (same match and selection), the disagreement
  set (recs only one arm makes; the same side on a different line),
  stake-weighted and flat P/L, CLV. Uncertainty: match-clustered bootstrap
  (resampling matches, paired), and the per-match SE of the paired per-match
  difference.

PART B2 — THE EARLY BET: A CLOSING-LINE READOUT (amendment 5)
  A closing-line value needs the bet placed before the last observed quote.
  So each arm is also replayed at the last tick at least EARLY_HOURS = 6 h
  before kickoff, on matches kicking off from EARLY_SINCE = REPRO_START (the
  first full day with current prices, last_seen_at and snapshots), with the
  probabilities it had then (stored: the newest row created before that
  tick; refit_on_stored reconciled to it; served and refit unchanged), and
  each early rec is valued against the consensus at the LAST tick, the latest
  pre-kickoff state anyone observes:
    clv_fair  odds taken x p_last - 1  (the edge against the last observed
              fair price; CLV proper)
    ev_mkt    odds taken x p_early - 1 (what the market implied when placed)
    move      odds taken x (p_last - p_early) = clv_fair - ev_mkt (the market
              moving toward the bet)
    clv_same  odds taken / the same book's last pre-kickoff price - 1
  Per arm: recs, the four means, the share of recs the market moved toward
  and away from; per pair (served vs refit, stored vs refit_on_stored): the
  overlap and the match-clustered paired bootstrap of the clv_fair and move
  differences; and every fresh best quote at the early tick (a random side at
  the best price) as the baseline. 6 h is a single pre-declared horizon
  (prices move in it: ~23,000 AH snapshot changes fall in the last 6 h before
  kickoff of 2026-07..10 matches); no other horizon is read.

PART C — REPRODUCTION
  The stored arm, replayed with the logic era's de-vig, against production's
  actual AH recs (betting_recommendations, every status) on matches kicking
  off from REPRO_START (the 2026-09-04 gate + freshness-guard deploy): the
  share of production recs reproduced with the same match, selection and line,
  and of those the same price, book and stake. The served arm is compared the
  same way (its gap is model-input drift: each week's 1x2 and Dixon-Coles
  artifact differ from the 2026-11-01 ones). Production recs on matches that
  are not finished (stale 'scheduled' rows) are counted after the
  REPRO_START filter, never compared. Production's own settled P/L on the
  compared recs is reported with a match-clustered bootstrap SE and 95%
  interval, next to the stake-weighted market EV at bet time of the same
  bets (the loss a zero-edge stream would expect). [Amendment 7.]

PART D — POWER
  From this window's per-match variances: the n (and the months at the
  current monthly volume) a forward AH-only gate needs to detect the measured
  effect with 80% power at two-sided 5%, for each Brier primary and for the
  money / CLV primaries. [Amendment 8: the window pair's per-match z-test n
  is context only. The forward gate's Brier decision point is planned on its
  own pair (Part A2) under its own rule — the fold-clustered t(k-1) upper
  bound below 0 — with clustered_t_power: each fold mean is
  N(delta, sd^2 k / n + tau2), so power depends on the number of folds k,
  not only on n. For k = 4..18: the power at k x the pair's Aug-Oct monthly
  n, the n 80% power needs, and the smallest k reaching 80%. The money
  readouts are planned on the forward pair too (stake and flat P/L; early-bet
  clv_fair).]

THE RUN (fixed with this design, committed before the full run)
  --ftdc-dir = a re-run of the committed scripts/ab_soccer_ft_dixon_coles.py
  (4b8d768; --end 2026-10-31, --served-sha256 02141a6a..., --live-check-model-
  version served) on a local replica of the read-only REPEATABLE READ prod
  export of 2026-11-01 21:09:21Z (the soccer repair runbook had not started:
  no process, no state file), model files copied byte-identical (sha256
  checked); it reproduced the recorded gate exactly (n = 25,604, live
  -0.00375, every AH unit). This replay: --start 2026-05-01 --end
  2026-11-01 on the same replica. Before this commit the replay ran only as
  mechanics smokes (--mechanics, --limit 40 and 600): counts and fidelity
  lines, no P/L, CLV or Brier delta. (The 600-match smoke printed per-arm rec
  counts on its first 146 eligible matches, June-July 2026: served 8, refit
  9, stored 7, refit_on_stored 8.)

RESULT (2026-11-01; the run above, on the replica of the 21:09:21Z export;
updated after review with the amended readouts — amendments 4-10; every
Brier, P/L, reproduction and power number of the first run is unchanged)
  Inputs. The FT gate re-run reproduced the recorded gate exactly: n = 25,604,
  31 folds, live refit - served on over_2.5 -0.00375 (clustered -0.00358 +/-
  0.00044), every AH unit to the digit, live check 371/371 stored predictions
  of the current incumbent re-derived (1x2-only 2.8e-17, 69 anchored 5.0e-7).
  AH odds exist from 2026-05-30 06:00Z (first row created; 1,984 finished
  matches with pre-match AH rows to 2026-10-31, six US books); production's
  first AH rec was written 2026-06-11; odds_snapshots start 2026-07-13.
  1,113 of the 1,984 fail the eligibility gate at the tick (a team with < 10
  finished matches), leaving 871 (Jul 46, Aug 171, Sep 196, Oct 458 a month).
  PART A — Brier, refit - served (live regime, n = 25,604, 31 folds; the
  re-derived matrices reproduce scores.npz to 1.1e-15):
    best single line (ah_-2, chosen after reading)  -0.00212 +/- 0.00023
        (clustered -0.00211 +/- 0.00028, t upper bound -0.00154)
    pooled shaped lines (mean of 12)                -0.00112 +/- 0.00012
        (clustered -0.00110 +/- 0.00015, t upper bound -0.00079)
    goal-difference RPS (7 classes)                 -0.00047 +/- 0.00004
        (clustered -0.00047 +/- 0.00005; served RPS 0.1329)
    goal-difference 7-way Brier                     -0.00188 +/- 0.00023
  Every one fails the -0.005 house gate by a wide margin, clear of noise in
  the wrong direction for a ship: the effect is real but 2.4-10.6x too small.
  Why: after the IPF the 1x2 pins every region mass, so a Dixon-Coles shape
  can only move mass between margins inside a region; the refit moves the
  shaped lines by 0.007-0.020 per class (RMS) and the goal-difference classes
  by 0.015. Even the in-sample best rescaling of that correction (served +
  k (refit - served), k fitted on the scored rows) gains only 0.00230 on
  ah_-2 (k 1.39), 0.00118 pooled (k 1.28), 0.00051 on the RPS (k 1.38):
  under half the gate on the best line, about a quarter pooled. The pooled
  delta's between-fold variance tau2 is 2.1e-7 (31 folds).
  Choosing asian_handicap, a single line, a pooling or a score after the
  gate's per-unit results were read is post hoc; none of these numbers can
  ship anything.
  PART A2 — the forward gate's own pair, stored vs refit_on_stored (2,757
  matches with a stored AH row before kickoff, 2026-06..10; invariant lines
  agree to 7.8e-16, no abort): pooled shaped lines -0.00119 +/- 0.00043
  per match (SD 0.0228), clustered -0.00053 +/- 0.00071 over 5 folds of 66,
  225, 758, 773 and 935 matches. Its tau2 over all 5 folds is 1.5e-6, driven
  by the 66-match June fold of the constant-prior incident (+0.0020); from
  2026-07-14 on it is 0 (2,650 matches, -0.00123, SD 0.0231), and Aug-Oct
  alone 2.8e-7 (ad hoc) — the window pair's size.
  PART B — money at the last tick. served vs refit, every eligible match with
  pre-match AH odds (871 matches, 170 with a rec in either arm; today's logic
  throughout, so ~93% of the recs are from 2026-09-05 on — see the caveat):
    served  142 recs, stake 3,240.09, P/L -296.27 (ROI -9.1%, bootstrap SE
            7.2), flat -13.35 (-9.4%), market EV at bet -3.23% per rec
    refit   154 recs, stake 3,504.41, P/L -136.16 (ROI -3.9%, SE 7.2), flat
            -5.58 (-3.6%), market EV at bet -3.21% per rec
    overlap 120 of the 176 distinct recs are identical (108 at the same price
            and stake), 0 on a moved line; served-only 22 recs (-38% ROI),
            refit-only 34 (-4%): about 1 pick in 3 differs
    refit - served: ROI +5.3 pts (bootstrap SE 4.7, 95% [-3.9, +14.6]); flat
            ROI +5.8 pts (SE 4.5); total P/L +160 (SE 162); market EV at bet
            +0.0002 per rec (SE 0.0007)
  The production-faithful pair, stored vs refit_on_stored (871 matches):
    147 vs 155 recs, ROI -5.8% vs -5.3% (difference +0.4 pts, SE 4.9); 120 of
    182 distinct recs identical.
  From 2026-09-05 (647 matches): served vs refit ROI -12.0% vs -6.4% (+5.5
  pts, SE 5.1); stored vs refit_on_stored -7.0% (SE 7.2) vs -6.8% (+0.15 pts,
  SE 5.1).
  These recs are written at the last tick, and no AH quote is captured after
  it (0 quotes between the tick and kickoff; 0 consensus values differ at
  kickoff), so no closing-line value exists for them: same-book CLV is 0 by
  construction, and the "market EV at bet" — -3.2% per rec in every arm,
  -3.59% for every fresh best quote (a random side at the best price) — only
  says the best price sits ~3% below the market's fair price when the bet is
  placed, while the model claims ~+11.5%.
  PART B2 — the early bet (6 h before kickoff, kickoffs 2026-09-05..10-31,
  647 matches), valued against the last tick's consensus:
    served           134 recs, CLV -2.34% per rec (bootstrap SE 0.44),
                     market EV when placed -3.65%, move +1.38% (SE 0.43;
                     59% of recs moved toward, 36% away)
    refit            131 recs, CLV -2.37% (SE 0.46), move +1.31%
    stored           145 recs, CLV -2.29% (SE 0.43), move +1.38% (SE 0.42)
    refit_on_stored  146 recs, CLV -2.45% (SE 0.45), move +1.26%
    every fresh best quote at the early tick (random side): CLV -3.91%,
    market EV -3.89%, move +0.05% (SE 0.11). Ad hoc: a market-only pick (per
    match the best quote that beats its own consensus most) moves -0.43%
    (SE 0.15).
    refit - served CLV -0.03 pts (SE 0.25), stored-pair -0.15 pts (SE 0.26):
    none. Same-book CLV of the early bets +1.3% to +1.5% per rec.
  So the market does move toward the model's AH picks (+1.3 pts, where a
  random side moves 0 and a market-only pick moves against itself), but not
  by enough: an early bet still finishes 2.3% below the last observed fair
  price, and the refit changes none of it.
  PART C — reproduction, 2026-09-05..2026-10-31: production wrote 142 AH recs
  on matches kicking off then; 2 (away_2 and home_-1.75, 25.00 each, still
  pending) sit on two matches still marked 'scheduled', which the replay
  never loads. The stored arm reproduces the other 140, on 134 matches, and
  nothing else (recall 1.000, precision 1.000): 140/140 the same price and
  the same stake; the book differs on 54, every one a book tied at the same
  price at the tick (best_odds' SQL picks among ties arbitrarily).
  Production's settled P/L on them is -249.81 on 3,239.65 staked: ROI -7.7%
  (match bootstrap SE 7.3, 95% [-21.9%, +6.5%]); the market EV at bet of the
  same bets, stake-weighted, is -3.3%. The window cannot tell this stream
  from break-even or from a zero-edge stream paying the margin. The served
  arm (today's artifact and 1x2 incumbent) reproduces 107 of the 140 (recall
  0.76, precision 0.81), every one at the same price, 70 at the same stake:
  that gap is input drift, not the replay (the stored AH probabilities
  differ from the served arm's by a median 0.021, p90 0.061, on the shaped
  lines, because each week's 1x2 ensemble and artifact differ from the
  2026-11-01 ones).
  PART D — power (80%, two-sided 5%):
    window pair, per-match z (context): pooled shaped lines 2,241 matches,
        ah_-2 2,394, GD RPS 1,687; the -0.005 gate is unreachable at these
        effect sizes at ANY n
    forward pair, the forward gate's clustered t(k-1) rule, pooled shaped
        lines (delta -0.00119, SD 0.0228, planning tau2 2.1e-7, 822 matches a
        month Aug-Oct, folds projected with the 2024-26 calendar shape: June
        ~0.18 and July ~0.49 of an Aug-Oct month): 6 folds (2026-11..2027-04,
        4,832 matches) 70% power; 7 folds 81%; adding the small June fold
        drops 8 folds to 74%; >= 80% from 10 folds on; 12 folds
        (2026-11..2027-10, 8,780 matches) 94% (66% under the sensitivity
        tau2 1.5e-6), the 12-fold n floor for 80% 5,224
    money, forward pair: stake P/L +0.006 a match (SD 5.77): 7.6 million
        matches; flat P/L 645,000; early-bet CLV -0.0002 a match (SD 0.0146):
        36,500 matches (~106 months at ~345 early-bet matches a month) — no
        money metric can decide this pair in practice (the window pair's
        +0.18 a match, 6,934 matches, is the noise the forward pair removes)
  ANSWER. A streamlined AH-only deployment is buildable (design in scripts/
  ab_soccer_ah_forward_gate.py) but this window gives no case for it: the
  Brier gain is real and too small for the house gate; the replayed
  recommendations agree on about 2 picks in 3, and the P/L gap in the window
  pair is noise that vanishes in the production-faithful pair; and the one
  closing-line readout the data allows, the early bet, is the same for both
  models (-2.3% against the last observed fair price).

AMENDMENTS AFTER THE RUN (1-3 change no money, reproduction or Brier delta;
the full run was re-run after them and every one of those numbers is
identical, compared field by field; 4-10 are the review round below)
  1. Part A's "shift ceiling" readout was mis-described as a bound (the
     measured gain exceeded it). It is replaced by the scaled-shape bound
     (shape_moments / scaled_shape).
  2. Part B also reports the fair-close CLV of every fresh best quote at the
     tick (the random-side baseline); recs record the books tied at their
     price, and Part C counts the book mismatches that are such ties.
  3. The monthly volume query truncates match_date in UTC explicitly (it had
     used the session time zone; the recorded run's session was UTC).
  After review (2026-11-01, committed before any of these readouts ran on
  the full window; before this commit they ran only as --mechanics smokes):
  4. "fair-close CLV" is renamed what it is — the market-implied EV at bet
     time (ev_mkt) — because no AH quote is captured after the last tick; the
     run counts quotes captured after the tick and consensus values that
     differ at kickoff, so the identity is checked, not assumed. The
     same-book CLV's 0 is explained by the same fact (it was attributed to
     "no AH price moved in the final 14 minutes", which nothing shows).
  5. Part B2, the early bet: the only closing-line readout this data allows.
  6. Part A2, the forward gate's own pair.
  7. Part C counts production recs on unfinished matches after the
     REPRO_START filter (they were counted over the whole window) and
     reports production's realised ROI with a bootstrap SE and the market EV
     of the same bets.
  8. Part D plans the forward gate on its own pair and its own clustered rule.
  9. The window caveat above (pre-2026-09-05 quotes read as stale).
  After the first full run of 4-9 (planning and context only; every Brier,
  money, early-bet and reproduction number of that run is unchanged,
  compared field by field, and recs.csv is byte-identical):
  10. Part A2 also reports the forward pair without the constant-prior
      incident, and Part D plans the forward gate with projected seasonal
      fold sizes (clustered_t_power_sizes, n_floor_for_sizes,
      seasonal_fold_sizes; the rule's unweighted fold mean gives a June fold
      of ~150 matches full weight), a planning tau2 that excludes the
      incident and a sensitivity tau2 that keeps it; the early-bet random
      side also reports its movement on the quotes where both consensus
      values exist.

Read-only: SELECTs only. Usage (local replica):
    python scripts/ab_soccer_ah_recs_replay.py --ftdc-dir <ft gate --out-dir> \\
        --database-url postgresql://localhost/<replica> --end 2026-11-01 --out-json ah.json
Mechanics smoke (no deltas, no money numbers): add --mechanics --limit 40.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_soccer_ensemble_members as members  # noqa: E402
import ab_soccer_ft_dixon_coles as ftdc  # noqa: E402
import generate_recommendations as gr  # noqa: E402
import grading_outcomes as go  # noqa: E402
import rec_gating  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("ab_soccer_ah_recs_replay")

SPORT = "soccer"
MARKET = "asian_handicap"
TICK_MINUTES = 15  # auspex_pipeline runs */15
TICK_LAG_SECONDS = 60  # the soccer chain reaches generate_recommendations about a minute after the tick
# Commit 1c6e307 (2026-10-25T19:53:33-06:00): the market consensus switched to
# the power de-vig. Before it the gap cap read a proportional de-vig.
POWER_DEVIG_SINCE = pd.Timestamp("2026-10-26T01:53:33Z")
# Commits dc93233 / 2ec19b8 (2026-09-04): the per-stream gate, the AH caps
# and the freshness guard on the live quote. The reproduction check starts
# with the first full day after them.
REPRO_START = pd.Timestamp("2026-09-05T00:00:00Z")
SHAPED_LINES: Tuple[float, ...] = tuple(x for x in ftdc.AH_LINES if x not in ftdc.AH_INVARIANT_LINES)
SHAPED_UNITS: Tuple[str, ...] = tuple(f"ah_{ftdc.fmt_line(x)}" for x in SHAPED_LINES)
ALL_AH_UNITS: Tuple[str, ...] = tuple(f"ah_{ftdc.fmt_line(x)}" for x in ftdc.AH_LINES)
GD_CLASSES = ("<=-3", "-2", "-1", "0", "+1", "+2", ">=+3")
REGIME = "live"
ARMS = ("served", "refit", "stored", "refit_on_stored")
# (a, b, first kickoff or None): the window pair, the production-faithful
# pair, and both again over the current-logic era (from REPRO_START), where
# the stored arm is no longer the 2026-06-15..07-13 constant-prior incident.
PAIRS: Tuple[Tuple[str, str, Optional[pd.Timestamp]], ...] = (
    ("served", "refit", None),
    ("stored", "refit_on_stored", None),
    ("served", "refit", REPRO_START),
    ("stored", "refit_on_stored", REPRO_START),
)
REPRO_ARMS = ("stored", "served")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261101
RECONSTRUCT_ATOL = 1e-12
Z_ALPHA = 1.959963984540054  # two-sided 5%
Z_BETA = 0.8416212335729143  # 80% power
POWER_TARGET = 0.80
# Amendment 4 (after the run). No AH quote is captured after the last
# generator tick: the odds fetch runs in the same tick, 10-55 s before the
# generator reads. So the consensus "at kickoff" IS the consensus at the tick,
# and odds taken at the tick x that consensus - 1 is the market-implied EV of
# the bet when it is placed, not a closing-line value. A closing-line readout
# needs a bet placed before the last observed quote: the EARLY bet, replayed
# at the last tick at least EARLY_HOURS before kickoff, against the consensus
# at the last tick (the latest pre-kickoff state anyone observes).
EARLY_HOURS = 6
# The early bet needs a quote's price AND age at a tick hours before kickoff.
# Prices: odds_snapshots, from 2026-07-13. Ages: until 2ec19b8 (2026-09-04)
# an odds row kept its FIRST price and first-seen timestamp forever and had
# no last_seen_at (stamped from 2026-09-04 19:42Z), so before then every quote
# reads as days old and today's freshness guard suppresses it. The early bet
# therefore starts with REPRO_START, the first full day of the current schema.
EARLY_SINCE = REPRO_START
EARLY_PAIRS: Tuple[Tuple[str, str, Optional[pd.Timestamp]], ...] = (
    ("served", "refit", None),
    ("stored", "refit_on_stored", None),
)
INVARIANT_ATOL = 1e-9
POOLED = "pooled_shaped_mean"
# Amendment 10. Stored soccer predictions of 2026-06-15..07-13 were the
# constant-prior incident; the forward pair's between-fold variance is also
# reported without them (the forward window has no such rows).
INCIDENT_END = pd.Timestamp("2026-07-14T00:00:00Z")
FORWARD_FIRST_FOLD = pd.Timestamp("2026-11-01T00:00:00Z")
FORWARD_FIRST_FOLD_SHARE = 29.0 / 30.0  # the forward window opens 2026-11-02

AH_ODDS_SQL = """
    SELECT
        o.match_id::text AS match_id,
        o.bookmaker,
        lower(o.selection) AS selection,
        o.line::float AS line,
        o.odds_decimal::float AS odds_decimal,
        o."timestamp" AS ts,
        o.last_seen_at,
        COALESCE(o.created_at, o."timestamp") AS created,
        o.is_live
    FROM odds o
    JOIN matches m ON m.id = o.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
    WHERE m.status = 'finished'
      AND m.match_date >= %(start)s
      AND m.match_date <  %(end)s
      AND o.market_type = 'asian_handicap'
"""

SNAPSHOTS_SQL = """
    SELECT s.match_id::text AS match_id, s.bookmaker, lower(s.selection) AS selection, s.line::float AS line,
           s.odds_decimal::float AS odds_decimal, s.captured_at
    FROM odds_snapshots s
    JOIN matches m ON m.id = s.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
    WHERE m.status = 'finished'
      AND m.match_date >= %(start)s
      AND m.match_date <  %(end)s
      AND s.market_type = 'asian_handicap'
"""

MATCHES_SQL = """
    SELECT m.id::text AS match_id, m.match_date, m.home_team_id::text AS home_team_id,
           m.away_team_id::text AS away_team_id, m.home_score, m.away_score, m.metadata,
           l.name AS league_name
    FROM matches m
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
    WHERE m.status = 'finished'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
      AND m.match_date >= %(start)s
      AND m.match_date <  %(end)s
"""

# generate_recommendations.list_upcoming counts every finished match of a team.
HISTORY_SQL = """
    SELECT m.home_team_id::text AS home_team_id, m.away_team_id::text AS away_team_id, m.match_date
    FROM matches m
    WHERE m.status = 'finished'
"""

STORED_SQL = """
    SELECT p.match_id::text AS match_id, p.model_version, p.prediction_type, p.probabilities, p.metadata,
           p.created_at
    FROM predictions p
    JOIN matches m ON m.id = p.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
    WHERE p.model_name = 'ensemble'
      AND p.prediction_type IN ('asian_handicap', 'match_result')
      AND m.status = 'finished'
      AND m.match_date >= %(start)s
      AND m.match_date <  %(end)s
"""

# The forward Brier gate's population: finished matches with a stored AH row.
STORED_AH_PER_MONTH_SQL = """
    SELECT to_char(date_trunc('month', m.match_date AT TIME ZONE 'UTC'), 'YYYY-MM') AS month,
           count(DISTINCT p.match_id) AS n
    FROM predictions p
    JOIN matches m ON m.id = p.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
    WHERE p.model_name = 'ensemble'
      AND p.prediction_type = 'asian_handicap'
      AND m.status = 'finished'
      AND p.created_at < m.match_date
      AND m.match_date >= %(start)s
      AND m.match_date <  %(end)s
    GROUP BY 1
"""

RECS_SQL = """
    SELECT br.id::text AS rec_id, br.match_id::text AS match_id, br.selection,
           br.odds_at_recommendation::float AS odds_decimal, br.bookmaker,
           br.recommended_stake::float AS stake, br.expected_value::float AS ev, br.status, br.actual_result,
           br.profit_loss::float AS profit_loss, br.created_at, m.match_date AS kickoff, m.status AS match_status
    FROM betting_recommendations br
    JOIN matches m ON m.id = br.match_id
    JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
    WHERE br.bet_type = 'asian_handicap'
      AND m.match_date >= %(start)s
      AND m.match_date <  %(end)s
"""


# ── Pure: the tick and the odds state ────────────────────────────────


def _ts(value: Any) -> Optional[pd.Timestamp]:
    return ftdc._ts(value)


def last_tick_before(
    kickoff: Any, tick_minutes: int = TICK_MINUTES, lag_seconds: int = TICK_LAG_SECONDS
) -> pd.Timestamp:
    """The moment the last generator run before kickoff executed: the latest
    quarter-hour tick t with t + lag strictly before kickoff, plus the lag
    (list_upcoming keeps a match only while match_date >= NOW())."""
    ko = _ts(kickoff)
    if ko is None:
        raise ValueError(f"kickoff {kickoff!r} is not a timestamp")
    step = pd.Timedelta(minutes=tick_minutes)
    lag = pd.Timedelta(seconds=lag_seconds)
    t = (ko - lag).floor(f"{tick_minutes}min")
    if t + lag >= ko:
        t -= step
    return t + lag


@dataclass(frozen=True)
class Quote:
    """One book's AH quote as it stood at a moment."""

    bookmaker: str
    selection: str  # 'home' / 'away'
    line: float  # the side's own signed line
    odds: float
    age_hours: Optional[float]
    moved_at: Optional[pd.Timestamp]


STATE_DROP_REASONS = ("live", "not_yet_listed", "price_unknown_at_tick", "seen_after_tick_kept_fresh")


def snapshot_index(
    snaps: Iterable[Dict[str, Any]],
) -> Dict[Tuple[str, str, str, float], List[Tuple[pd.Timestamp, float]]]:
    """{(match, book, selection, line): [(captured_at, price), ...] sorted by time}."""
    idx: Dict[Tuple[str, str, str, float], List[Tuple[pd.Timestamp, float]]] = {}
    for s in snaps:
        at = _ts(s["captured_at"])
        if at is None:
            raise ValueError(f"snapshot without captured_at: {s}")
        key = (str(s["match_id"]), str(s["bookmaker"]), str(s["selection"]).lower(), round(float(s["line"]), 4))
        idx.setdefault(key, []).append((at, float(s["odds_decimal"])))
    for v in idx.values():
        v.sort(key=lambda x: x[0])
    return idx


def quote_state(
    rows: Sequence[Dict[str, Any]],
    snaps: Dict[Tuple[str, str, str, float], List[Tuple[pd.Timestamp, float]]],
    at: Any,
    drops: Optional[Dict[str, int]] = None,
) -> List[Quote]:
    """Every AH quote of one match as it stood at `at` (see the docstring:
    exists when created <= at; price = the last snapshot <= at, else the row's
    own price if it last moved <= at; age from COALESCE(last_seen_at,
    timestamp), 0 when that is after `at`)."""
    t = _ts(at)
    if t is None:
        raise ValueError(f"{at!r} is not a timestamp")

    def drop(reason: str) -> None:
        if drops is not None:
            drops[reason] = drops.get(reason, 0) + 1

    out: List[Quote] = []
    for r in rows:
        if bool(r.get("is_live")):
            drop("live")
            continue
        created = _ts(r.get("created"))
        if created is None or created > t:
            drop("not_yet_listed")
            continue
        key = (str(r["match_id"]), str(r["bookmaker"]), str(r["selection"]).lower(), round(float(r["line"]), 4))
        hist = [h for h in snaps.get(key, []) if h[0] <= t]
        moved = _ts(r.get("ts"))
        if hist:
            moved_at, price = hist[-1]
        elif moved is not None and moved <= t:
            moved_at, price = moved, float(r["odds_decimal"])
        else:
            drop("price_unknown_at_tick")
            continue
        seen = _ts(r.get("last_seen_at")) or moved
        if seen is None:
            age = None
        elif seen > t:
            drop("seen_after_tick_kept_fresh")
            age = 0.0
        else:
            age = (t - seen).total_seconds() / 3600.0
        out.append(Quote(str(r["bookmaker"]), str(r["selection"]).lower(), float(r["line"]), price, age, moved_at))
    return out


def is_fresh(q: Quote, max_age_hours: float = rec_gating.MAX_ODDS_AGE_HOURS) -> bool:
    return q.age_hours is not None and q.age_hours <= max_age_hours


def best_quotes(state: Sequence[Quote], max_age_hours: float = rec_gating.MAX_ODDS_AGE_HOURS) -> List[Quote]:
    """generate_recommendations.best_odds on the state: per (selection, line)
    the fresh quotes first (unknown age last), then the highest price; ties go
    to the bookmaker that sorts first. Ordered like the SQL (selection, line)."""
    by_key: Dict[Tuple[str, float], List[Quote]] = {}
    for q in state:
        by_key.setdefault((q.selection, round(q.line, 4)), []).append(q)
    out = []
    for key in sorted(by_key):
        qs = sorted(by_key[key], key=lambda q: (0 if is_fresh(q, max_age_hours) else 1, -q.odds, q.bookmaker))
        out.append(qs[0])
    return out


def price_ties(state: Sequence[Quote], max_age_hours: float = rec_gating.MAX_ODDS_AGE_HOURS) -> int:
    """How many (selection, line) best prices are shared by two or more fresh books."""
    by_key: Dict[Tuple[str, float], List[float]] = {}
    for q in state:
        if is_fresh(q, max_age_hours):
            by_key.setdefault((q.selection, round(q.line, 4)), []).append(q.odds)
    return sum(1 for v in by_key.values() if len(v) > 1 and sorted(v)[-1] == sorted(v)[-2])


def tied_books(state: Sequence[Quote], best: Quote, max_age_hours: float = rec_gating.MAX_ODDS_AGE_HOURS) -> List[str]:
    """Every fresh book quoting `best`'s selection and line at exactly its
    price: best_odds' SQL picks among them arbitrarily."""
    return sorted(
        q.bookmaker
        for q in state
        if q.selection == best.selection
        and abs(q.line - best.line) < 1e-9
        and q.odds == best.odds
        and is_fresh(q, max_age_hours)
    )


def consensus_prob(
    state: Sequence[Quote],
    selection: str,
    line: float,
    devig: Callable[[Dict[str, float], float], Optional[Dict[str, float]]] = rec_gating.devig_power,
    max_age_hours: float = rec_gating.MAX_ODDS_AGE_HOURS,
    min_books: int = rec_gating.MIN_CONSENSUS_BOOKS,
) -> Optional[float]:
    """rec_gating.market_consensus_prob for one AH selection on the state:
    fresh quotes of the selection at `line` and of the other side at -line,
    per book de-vigged (`devig`), averaged over >= min_books books. A test
    pins it to market_consensus_prob itself for the power de-vig."""
    sel = selection.strip().lower()
    by_book: Dict[str, Dict[str, float]] = {}
    for q in state:
        if not is_fresh(q, max_age_hours):
            continue
        wanted = line if q.selection == sel else -line
        if abs(q.line - wanted) > 1e-9:
            continue
        by_book.setdefault(q.bookmaker.strip().lower(), {})[q.selection] = q.odds
    vals = []
    for prices in by_book.values():
        if len(prices) < 2 or sel not in prices or any(p <= 0.0 for p in prices.values()):
            continue
        fair = devig(prices, 1.0)
        if fair is not None:
            vals.append(fair[sel])
    if len(vals) < min_books:
        return None
    return sum(vals) / len(vals)


# ── Pure: the generator's pricing path ───────────────────────────────


@dataclass
class Rec:
    match_id: str
    selection: str  # display selection, e.g. 'home_-0.75'
    side: str
    line: float
    odds: float
    bookmaker: str
    ev: float
    p_raw: float
    p_kelly: float
    market_prob: Optional[float]
    kelly_stake: float
    stake: float
    age_hours: Optional[float]
    extra: Dict[str, Any] = field(default_factory=dict)


def replay_match(
    match_id: str,
    probs: Dict[str, float],
    state: Sequence[Quote],
    bankroll: float,
    ev_threshold: float,
    prob_floor: float,
    devig: Callable[[Dict[str, float], float], Optional[Dict[str, float]]] = rec_gating.devig_power,
    max_age_hours: float = rec_gating.MAX_ODDS_AGE_HOURS,
) -> Tuple[List[Rec], Dict[str, int]]:
    """generate_recommendations.recommend_for_match for the asian_handicap
    rows of one match, on the replayed odds state, with the arm's stored-form
    AH probabilities. Returns (recs, suppressed by reason)."""
    suppressed: Dict[str, int] = {}
    recs: List[Rec] = []
    if not probs:
        return recs, suppressed
    for q in best_quotes(state, max_age_hours):
        if q.odds <= 1.0:
            continue
        if gr.is_stale_odds(q.age_hours, max_age_hours):
            suppressed[gr.STALE_ODDS_REASON] = suppressed.get(gr.STALE_ODDS_REASON, 0) + 1
            continue
        valued = gr.selection_value(MARKET, probs, q.selection, q.line, q.odds)
        if valued is None:
            continue
        p_kelly, ev, p_raw = valued
        if ev < ev_threshold or p_kelly < prob_floor:
            continue
        gate = rec_gating.gate_for(SPORT, MARKET)
        market_prob = None
        if gate.enabled and gate.max_gap is not None:
            market_prob = consensus_prob(state, q.selection, q.line, devig, max_age_hours)
        allowed, reason = rec_gating.passes_gate(
            SPORT, MARKET, odds=q.odds, ev=ev, model_prob=p_raw, market_prob=market_prob
        )
        if not allowed:
            key = str(reason) if str(reason).startswith("stream_disabled") else f"{MARKET}:{reason}"
            suppressed[key] = suppressed.get(key, 0) + 1
            continue
        f = gr.kelly_fraction(p_kelly, q.odds)
        if f <= 0:
            continue
        recs.append(
            Rec(
                match_id=match_id,
                selection=gr.display_selection(MARKET, q.selection, q.line),
                side=q.selection,
                line=q.line,
                odds=q.odds,
                bookmaker=q.bookmaker,
                ev=round(ev, 4),
                p_raw=p_raw,
                p_kelly=p_kelly,
                market_prob=market_prob,
                kelly_stake=round(bankroll * f, 2),
                stake=rec_gating.cap_stake(bankroll * f * 0.25, bankroll),
                age_hours=q.age_hours,
                extra={"tie_books": tied_books(state, q, max_age_hours)},
            )
        )
    return recs, suppressed


def settle(selection: str, odds: float, stake: float, home: int, away: int, metadata: Optional[dict]):
    """(verdict, stake P/L, flat P/L) through grading_outcomes, or Nones when ungradable."""
    verdict = go.grade_rec_selection(
        MARKET, selection, int(home), int(away), sport=SPORT, model_name="ensemble", metadata=metadata or {}
    )
    if verdict is None:
        return None, None, None
    _, pl = go.rec_status_and_pl(verdict, stake, odds)
    _, flat = go.rec_status_and_pl(verdict, 1.0, odds)
    return verdict, float(pl), float(flat)


def closing_price(
    snaps: Dict[Tuple[str, str, str, float], List[Tuple[pd.Timestamp, float]]],
    rows: Sequence[Dict[str, Any]],
    match_id: str,
    bookmaker: str,
    selection: str,
    line: float,
    kickoff: Any,
) -> Optional[float]:
    """vw_rec_clv's closing price: the last snapshot strictly before kickoff at
    the same book; for a key with no snapshot, the row's own price when it
    last moved before kickoff."""
    ko = _ts(kickoff)
    key = (str(match_id), str(bookmaker), selection.lower(), round(float(line), 4))
    hist = [h for h in snaps.get(key, []) if ko is not None and h[0] < ko]
    if hist:
        return hist[-1][1]
    for r in rows:
        if (str(r["bookmaker"]), str(r["selection"]).lower(), round(float(r["line"]), 4)) == key[1:]:
            moved = _ts(r.get("ts"))
            if moved is not None and ko is not None and moved < ko and not snaps.get(key):
                return float(r["odds_decimal"])
    return None


def captured_after(
    rows: Sequence[Dict[str, Any]],
    snaps: Dict[Tuple[str, str, str, float], List[Tuple[pd.Timestamp, float]]],
    after: Any,
    kickoff: Any,
) -> int:
    """How many observations of this match's AH quotes fall strictly between
    `after` (the last tick) and kickoff: snapshots, and rows whose price moved
    or which were last seen in that gap. 0 means the market is not observed
    after the tick, so nothing measured at kickoff can differ from the tick."""
    t, ko = _ts(after), _ts(kickoff)
    if t is None or ko is None:
        raise ValueError(f"captured_after needs timestamps, got {after!r} / {kickoff!r}")
    n = 0
    for r in rows:
        key = (str(r["match_id"]), str(r["bookmaker"]), str(r["selection"]).lower(), round(float(r["line"]), 4))
        n += sum(1 for at, _ in snaps.get(key, []) if t < at < ko)
        n += sum(1 for v in (_ts(r.get("ts")), _ts(r.get("last_seen_at"))) if v is not None and t < v < ko)
    return n


def handicap_vector(probs: Dict[str, float], line: float) -> np.ndarray:
    """(home, away, push) of one line from a stored-form asian_handicap dict."""
    lbl = ftdc.fmt_line(line)
    try:
        return np.array([float(probs[f"{lbl}_{s}"]) for s in ("home", "away", "push")])
    except KeyError as exc:
        raise RuntimeError(f"asian_handicap dict lacks line {lbl}: {exc}") from exc


def score_ah_pair(
    served: Dict[str, float], refit: Dict[str, float], home: int, away: int
) -> Dict[str, Tuple[float, float]]:
    """{unit: (served Brier, refit Brier)} for the 12 shaped lines and their
    mean, each a 3-way Brier over the effective home / away / push masses
    against grading_outcomes' handicap settlement on the 90-minute score.
    Raises when an invariant line (AH 0, +-0.25, +-0.5: a function of the 1x2
    after the reconcile) differs between the arms beyond INVARIANT_ATOL — the
    arms were then not reconciled to the same 1x2, which is a bug."""
    for line in ftdc.AH_INVARIANT_LINES:
        gap = float(np.max(np.abs(handicap_vector(served, line) - handicap_vector(refit, line))))
        if not math.isfinite(gap) or gap > INVARIANT_ATOL:
            raise RuntimeError(f"invariant AH line {line} differs between served and refit by {gap:.2e}")
    out: Dict[str, Tuple[float, float]] = {}
    for unit, line in zip(SHAPED_UNITS, SHAPED_LINES):
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


# ── Pure: comparisons and statistics ─────────────────────────────────


def compare_recs(a: Sequence[Rec], b: Sequence[Rec]) -> Dict[str, Any]:
    """Overlap of two arms' rec sets on the same matches. same: identical
    (match, selection incl. line); moved_line: the same match and side on a
    different line (paired one-to-one per match and side); a_only / b_only:
    the rest."""
    ka = {(r.match_id, r.selection): r for r in a}
    kb = {(r.match_id, r.selection): r for r in b}
    same = sorted(set(ka) & set(kb))
    rest_a = [r for k, r in ka.items() if k not in kb]
    rest_b = [r for k, r in kb.items() if k not in ka]
    moved: List[Tuple[Rec, Rec]] = []
    pool_b: Dict[Tuple[str, str], List[Rec]] = {}
    for r in rest_b:
        pool_b.setdefault((r.match_id, r.side), []).append(r)
    a_only = []
    for r in rest_a:
        cand = pool_b.get((r.match_id, r.side))
        if cand:
            moved.append((r, cand.pop(0)))
        else:
            a_only.append(r)
    b_only = [r for v in pool_b.values() for r in v]
    return {"same": same, "moved_line": moved, "a_only": a_only, "b_only": b_only}


def money_summary(recs: Sequence[Rec]) -> Dict[str, Any]:
    """Counts, stake, stake-weighted and flat P/L and ROI, verdicts, the
    same-book CLV and the market-implied EV at bet time (odds x the consensus
    at the tick - 1) of settled recs (ungradable ones are counted, never
    scored)."""
    settled = [r for r in recs if r.extra.get("verdict") is not None]
    stake = sum(r.stake for r in settled)
    pl = sum(r.extra["pl"] for r in settled)
    flat = sum(r.extra["flat"] for r in settled)
    clv_s = [r.extra["clv_same"] for r in recs if r.extra.get("clv_same") is not None]
    ev_m = [r.extra["ev_mkt"] for r in recs if r.extra.get("ev_mkt") is not None]
    ev_m_w = [(r.extra["ev_mkt"], r.stake) for r in recs if r.extra.get("ev_mkt") is not None]
    verdicts: Dict[str, int] = {}
    for r in settled:
        verdicts[r.extra["verdict"]] = verdicts.get(r.extra["verdict"], 0) + 1
    return {
        "recs": len(recs),
        "settled": len(settled),
        "ungradable": len(recs) - len(settled),
        "matches": len({r.match_id for r in recs}),
        "stake": stake,
        "pl": pl,
        "roi_stake": pl / stake if stake > 0 else None,
        "flat_pl": flat,
        "roi_flat": flat / len(settled) if settled else None,
        "mean_odds": float(np.mean([r.odds for r in recs])) if recs else None,
        "mean_ev_claimed": float(np.mean([r.ev for r in recs])) if recs else None,
        "verdicts": verdicts,
        "clv_same_mean": float(np.mean(clv_s)) if clv_s else None,
        "clv_same_n": len(clv_s),
        "ev_mkt_mean": float(np.mean(ev_m)) if ev_m else None,
        "ev_mkt_stake_weighted": (
            float(sum(c * s for c, s in ev_m_w) / sum(s for _, s in ev_m_w))
            if ev_m_w and sum(s for _, s in ev_m_w) > 0
            else None
        ),
        "ev_mkt_n": len(ev_m),
    }


def per_match_totals(recs: Sequence[Rec], match_ids: Sequence[str]) -> Dict[str, np.ndarray]:
    """Per match (aligned to match_ids): stake, P/L, flat count and P/L of settled
    recs, and the sum / count of the market-implied EV at bet time."""
    pos = {m: i for i, m in enumerate(match_ids)}
    n = len(match_ids)
    out = {k: np.zeros(n) for k in ("stake", "pl", "flat_n", "flat_pl", "ev_sum", "ev_n")}
    for r in recs:
        i = pos.get(r.match_id)
        if i is None:
            raise KeyError(f"rec on match {r.match_id} outside the compared population")
        if r.extra.get("verdict") is not None:
            out["stake"][i] += r.stake
            out["pl"][i] += r.extra["pl"]
            out["flat_n"][i] += 1
            out["flat_pl"][i] += r.extra["flat"]
        if r.extra.get("ev_mkt") is not None:
            out["ev_sum"][i] += r.extra["ev_mkt"]
            out["ev_n"][i] += 1
    return out


def per_match_sum(recs: Sequence[Rec], match_ids: Sequence[str], key: str) -> Tuple[np.ndarray, np.ndarray]:
    """(sum, count) per match (aligned to match_ids) of rec.extra[key], skipping Nones."""
    pos = {m: i for i, m in enumerate(match_ids)}
    s, c = np.zeros(len(match_ids)), np.zeros(len(match_ids))
    for r in recs:
        i = pos.get(r.match_id)
        if i is None:
            raise KeyError(f"rec on match {r.match_id} outside the compared population")
        v = r.extra.get(key)
        if v is not None:
            s[i] += float(v)
            c[i] += 1
    return s, c


def _ratio(num: np.ndarray, den: np.ndarray) -> float:
    d = float(den.sum())
    return float(num.sum()) / d if d > 0 else float("nan")


def paired_bootstrap(
    ta: Dict[str, np.ndarray], tb: Dict[str, np.ndarray], draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> Dict[str, Any]:
    """Match-clustered paired bootstrap of each arm's ROI (stake, flat), mean
    market-implied EV at bet time, and the differences b - a (ROI, total P/L
    per match). Returns the point estimate, the bootstrap SE and the
    2.5 / 97.5 percentiles."""
    n = len(ta["pl"])
    if n == 0:
        return {}
    rng = np.random.default_rng(seed)
    stats: Dict[str, Callable[[np.ndarray], float]] = {
        "a_roi_stake": lambda ix: _ratio(ta["pl"][ix], ta["stake"][ix]),
        "b_roi_stake": lambda ix: _ratio(tb["pl"][ix], tb["stake"][ix]),
        "a_roi_flat": lambda ix: _ratio(ta["flat_pl"][ix], ta["flat_n"][ix]),
        "b_roi_flat": lambda ix: _ratio(tb["flat_pl"][ix], tb["flat_n"][ix]),
        "a_ev_mkt": lambda ix: _ratio(ta["ev_sum"][ix], ta["ev_n"][ix]),
        "b_ev_mkt": lambda ix: _ratio(tb["ev_sum"][ix], tb["ev_n"][ix]),
        "diff_roi_stake": lambda ix: _ratio(tb["pl"][ix], tb["stake"][ix]) - _ratio(ta["pl"][ix], ta["stake"][ix]),
        "diff_roi_flat": lambda ix: _ratio(tb["flat_pl"][ix], tb["flat_n"][ix])
        - _ratio(ta["flat_pl"][ix], ta["flat_n"][ix]),
        "diff_pl_total": lambda ix: float(tb["pl"][ix].sum() - ta["pl"][ix].sum()),
        "diff_flat_pl_total": lambda ix: float(tb["flat_pl"][ix].sum() - ta["flat_pl"][ix].sum()),
        "diff_ev_mkt": lambda ix: _ratio(tb["ev_sum"][ix], tb["ev_n"][ix]) - _ratio(ta["ev_sum"][ix], ta["ev_n"][ix]),
    }
    full = np.arange(n)
    point = {k: f(full) for k, f in stats.items()}
    boot = {k: np.empty(draws) for k in stats}
    for b in range(draws):
        ix = rng.integers(0, n, n)
        for k, f in stats.items():
            boot[k][b] = f(ix)
    out = {}
    for k in stats:
        v = boot[k][np.isfinite(boot[k])]
        out[k] = {
            "estimate": point[k],
            "se": float(v.std(ddof=1)) if v.size > 1 else None,
            "lo95": float(np.percentile(v, 2.5)) if v.size else None,
            "hi95": float(np.percentile(v, 97.5)) if v.size else None,
        }
    out["matches"] = n
    out["draws"] = draws
    return out


def per_match_diff(x: np.ndarray) -> Dict[str, Any]:
    """Mean, SD and per-match SE of a per-match paired difference."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else None, "sd": None, "se": None}
    sd = float(x.std(ddof=1))
    return {"n": n, "mean": float(x.mean()), "sd": sd, "se": sd / math.sqrt(n)}


def n_for_power(sd: Optional[float], delta: Optional[float], z_alpha: float = Z_ALPHA, z_beta: float = Z_BETA):
    """Units needed for a two-sided 5% test of a paired mean to have 80% power
    against `delta`, given the per-unit SD; None when undefined (delta 0)."""
    if sd is None or delta is None or not math.isfinite(sd) or not math.isfinite(delta) or delta == 0:
        return None
    return int(math.ceil(((z_alpha + z_beta) * sd / abs(delta)) ** 2))


def ratio_bootstrap(
    num_a: np.ndarray,
    den_a: np.ndarray,
    num_b: np.ndarray,
    den_b: np.ndarray,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Match-clustered paired bootstrap of two per-match ratio means
    (sum num / sum den, e.g. the mean CLV per rec) and their difference b - a."""
    n = len(num_a)
    if n == 0:
        return {}
    rng = np.random.default_rng(seed)
    stats: Dict[str, Callable[[np.ndarray], float]] = {
        "a": lambda ix: _ratio(num_a[ix], den_a[ix]),
        "b": lambda ix: _ratio(num_b[ix], den_b[ix]),
        "diff": lambda ix: _ratio(num_b[ix], den_b[ix]) - _ratio(num_a[ix], den_a[ix]),
    }
    full = np.arange(n)
    boot = {k: np.empty(draws) for k in stats}
    for b in range(draws):
        ix = rng.integers(0, n, n)
        for k, f in stats.items():
            boot[k][b] = f(ix)
    out: Dict[str, Any] = {}
    for k, f in stats.items():
        v = boot[k][np.isfinite(boot[k])]
        est = f(full)
        out[k] = {
            "estimate": est if math.isfinite(est) else None,
            "se": float(v.std(ddof=1)) if v.size > 1 else None,
            "lo95": float(np.percentile(v, 2.5)) if v.size else None,
            "hi95": float(np.percentile(v, 97.5)) if v.size else None,
        }
    out["matches"] = n
    return out


def between_fold_variance(d: np.ndarray, folds: np.ndarray) -> Dict[str, Any]:
    """Method-of-moments split of a per-match paired difference into its
    within-fold noise and tau2, the variance of the TRUE fold means (a monthly
    refit shares one fit per fold, so its effect drifts month to month):
    var(fold means) - mean(within variance / fold size), floored at 0. tau2
    does not shrink with more matches per fold, only with more folds — which
    is why the clustered rule's power is set by the fold count."""
    d = np.asarray(d, dtype=float)
    f = np.asarray(folds).astype(str)
    ok = np.isfinite(d)
    d, f = d[ok], f[ok]
    labels = sorted(set(f.tolist()))
    sizes = np.array([(f == x).sum() for x in labels])
    keep = [x for x, s in zip(labels, sizes) if s >= 2]
    if len(keep) < 3:
        return {"n_folds": len(keep), "tau2": None, "fold_sizes": dict(zip(labels, sizes.tolist()))}
    means = np.array([d[f == x].mean() for x in keep])
    within = np.array([d[f == x].var(ddof=1) / (f == x).sum() for x in keep])
    tau2 = float(means.var(ddof=1) - within.mean())
    return {
        "n_folds": len(keep),
        "var_fold_means": float(means.var(ddof=1)),
        "mean_within_var_over_n": float(within.mean()),
        "tau2": max(tau2, 0.0),
        "tau2_raw": tau2,
        "fold_sizes": dict(zip(labels, sizes.tolist())),
    }


def clustered_t_power(delta: float, sd: float, tau2: float, k: int, n: int) -> float:
    """Power of the fold-clustered rule 'the t(k-1) upper bound of dBrier < 0'
    with k equal folds and n matches in all, against a true effect `delta`
    (< 0) with per-match SD `sd` and between-fold variance tau2: each fold
    mean is N(delta, sd^2 k / n + tau2), the rule's statistic is their
    one-sample t with k - 1 degrees of freedom, so the power is
    P(noncentral t(k-1, |delta| / sqrt(sd^2/n + tau2/k)) > t_0.975(k-1))."""
    from scipy.stats import nct
    from scipy.stats import t as student_t

    if k < 2 or n < k or delta == 0:
        return 0.0
    var = sd * sd / n + tau2 / k
    if var <= 0:
        return 1.0
    ncp = abs(delta) / math.sqrt(var)
    return float(nct.sf(student_t.ppf(0.975, k - 1), k - 1, ncp))


def clustered_t_power_sizes(delta: float, sd: float, tau2: float, sizes: Sequence[float]) -> float:
    """clustered_t_power for UNEQUAL folds: the rule's estimate is the
    unweighted mean of the k fold means, so its variance is
    sum_i (sd^2 / n_i + tau2) / k^2 and a small fold weighs as much as a big
    one; the power is read from a noncentral t(k-1) at that variance (equal
    sizes give clustered_t_power exactly)."""
    from scipy.stats import nct
    from scipy.stats import t as student_t

    k = len(sizes)
    if k < 2 or delta == 0 or any(n <= 0 for n in sizes):
        return 0.0
    var = sum(sd * sd / n + tau2 for n in sizes) / (k * k)
    if var <= 0:
        return 1.0
    return float(nct.sf(student_t.ppf(0.975, k - 1), k - 1, abs(delta) / math.sqrt(var)))


def n_floor_for_sizes(
    delta: float, sd: float, tau2: float, sizes: Sequence[float], target: float = POWER_TARGET
) -> Optional[int]:
    """The total n below which the projected fold sizes, scaled down
    uniformly, fall under `target` power (None when even 100x the projection
    does not reach it): the floor a pinned fold set should refuse to decide
    below."""
    total = float(sum(sizes))
    if total <= 0 or clustered_t_power_sizes(delta, sd, tau2, [n * 100 for n in sizes]) < target:
        return None
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid > 0 and clustered_t_power_sizes(delta, sd, tau2, [n * mid for n in sizes]) >= target:
            hi = mid
        else:
            lo = mid
    return int(math.ceil(hi * total))


def seasonal_fold_sizes(window_sizes: Dict[str, int], rate: float, k: int) -> List[float]:
    """Projected matches in the forward folds 2026-11.. (k of them): `rate`
    (matches a month, Aug-Oct 2026) times each calendar month's share of an
    Aug-Oct month in the window pair's eval set over its last 24 complete
    months (summer and winter breaks shrink folds; 2024-25: June ~0.2, July
    ~0.5, December/January ~0.7 of an Aug-Oct month), the first fold clipped to
    the window's 2026-11-02 start."""
    months = sorted(window_sizes)[-24:]
    by_cal: Dict[int, List[float]] = {}
    for m in months:
        by_cal.setdefault(int(m[5:7]), []).append(float(window_sizes[m]))
    base = float(np.mean([v for c in (8, 9, 10) for v in by_cal.get(c, [])]))
    out = []
    for i in range(k):
        cal = (FORWARD_FIRST_FOLD.month - 1 + i) % 12 + 1
        share = float(np.mean(by_cal[cal])) / base if by_cal.get(cal) and base > 0 else 1.0
        out.append(rate * share * (FORWARD_FIRST_FOLD_SHARE if i == 0 else 1.0))
    return out


def n_for_clustered_power(delta: float, sd: float, tau2: float, k: int, target: float = POWER_TARGET):
    """The smallest n with clustered_t_power >= target at k folds, or None when
    tau2 alone keeps it below target at any n."""
    hi = 10**8
    if clustered_t_power(delta, sd, tau2, k, hi) < target:
        return None
    lo = k
    if clustered_t_power(delta, sd, tau2, k, lo) >= target:
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if clustered_t_power(delta, sd, tau2, k, mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


# ── Pure: Brier readouts ─────────────────────────────────────────────


def gd_probabilities(P: np.ndarray) -> np.ndarray:
    """(m, 7) P(margin class) from served matrices (m, n, n), classes GD_CLASSES."""
    P = np.asarray(P, dtype=float)
    n = P.shape[1]
    i, j = np.indices((n, n))
    mar = i - j
    masks = [mar <= -3, mar == -2, mar == -1, mar == 0, mar == 1, mar == 2, mar >= 3]
    return np.stack([np.einsum("rij,ij->r", P, m.astype(float)) for m in masks], axis=1)


def gd_class(home: np.ndarray, away: np.ndarray) -> np.ndarray:
    mar = np.asarray(home, dtype=int) - np.asarray(away, dtype=int)
    return np.clip(mar, -3, 3) + 3


def rps_rows(p: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """Ranked probability score per row over ordered classes, / (K - 1)."""
    p = np.asarray(p, dtype=float)
    k = p.shape[1]
    y = np.zeros_like(p)
    y[np.arange(len(p)), np.asarray(cls, dtype=int)] = 1.0
    return np.sum((np.cumsum(p, axis=1)[:, :-1] - np.cumsum(y, axis=1)[:, :-1]) ** 2, axis=1) / (k - 1)


def handicap_probs(P: np.ndarray, line: float, weights: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """(m, 3) effective home / away / push masses of one AH line (the gate's readout)."""
    w = (weights or ftdc.unit_weights())[f"ah_{ftdc.fmt_line(line)}"]
    return np.einsum("rij,kij->rk", np.asarray(P, dtype=float), w)


def shape_moments(served: np.ndarray, refit: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """(E[sum_k d^2], E[sum_k d (served - y)]) with d = refit - served, over
    the rows (m, k) where y is finite."""
    ok = np.isfinite(y).all(axis=1)
    d = np.asarray(refit, dtype=float)[ok] - np.asarray(served, dtype=float)[ok]
    r = np.asarray(served, dtype=float)[ok] - np.asarray(y, dtype=float)[ok]
    return float(np.mean(np.sum(d * d, axis=1))), float(np.mean(np.sum(d * r, axis=1)))


def scaled_shape(parts: Sequence[Tuple[float, float]]) -> Dict[str, Optional[float]]:
    """The in-sample best rescaling of refit's correction: serve
    served + k (refit - served) for one k. Brier(k) - Brier(0) = 2k E[d r] +
    k^2 E[d^2] (a mean over `parts` for a pooled score), so the best k is
    -E[d r] / E[d^2] and its gain E[d r]^2 / E[d^2]. That is the most any
    shape along the refit's direction could gain on this window, fitted on
    the scored rows themselves (an optimistic bound, never a model)."""
    e_dd = float(np.mean([p[0] for p in parts]))
    e_dr = float(np.mean([p[1] for p in parts]))
    if e_dd <= 0:
        return {"e_d2": e_dd, "gain_at_refit": None, "best_k": None, "gain_at_best_k": None}
    return {
        "e_d2": e_dd,
        "gain_at_refit": -(2.0 * e_dr + e_dd),
        "best_k": -e_dr / e_dd,
        "gain_at_best_k": e_dr * e_dr / e_dd,
    }


# ── Data ─────────────────────────────────────────────────────────────


def load_ftdc(ftdc_dir: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    pm = pd.read_csv(os.path.join(ftdc_dir, "per_match.csv.gz"), low_memory=False)
    with np.load(os.path.join(ftdc_dir, "scores.npz"), allow_pickle=False) as z:
        scores = {k: z[k] for k in z.files}
    if list(scores["match_id"].astype(str)) != list(pm["match_id"].astype(str)):
        raise RuntimeError("per_match.csv.gz and scores.npz are not row-aligned")
    pm["match_date"] = pd.to_datetime(pm["match_date"], utc=True)
    return pm, scores


def load_db(database_url: str, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Any]:
    from utils.training_data import prematch_odds  # type: ignore

    params = {"start": start.to_pydatetime(), "end": end.to_pydatetime()}
    odds = members._read_sql(database_url, prematch_odds(AH_ODDS_SQL), params)
    snaps = members._read_sql(database_url, SNAPSHOTS_SQL, params)
    matches = members._read_sql(database_url, MATCHES_SQL, params)
    history = members._read_sql(database_url, HISTORY_SQL)
    stored = members._read_sql(database_url, STORED_SQL, params)
    recs = members._read_sql(database_url, RECS_SQL, params)
    per_month = members._read_sql(database_url, STORED_AH_PER_MONTH_SQL, params)
    import psycopg2
    from psycopg2.extras import RealDictCursor

    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            bankroll = gr.get_bankroll(cur)
    for df, col in ((matches, "match_date"), (history, "match_date"), (stored, "created_at"), (recs, "created_at")):
        df[col] = pd.to_datetime(df[col], utc=True)
    return {
        "odds": odds,
        "snaps": snaps,
        "matches": matches,
        "history": history,
        "stored": stored,
        "recs": recs,
        "bankroll": float(bankroll),
        "stored_ah_per_month": {str(k): int(v) for k, v in zip(per_month["month"], per_month["n"])},
    }


class TeamHistory:
    """Finished-match counts per team strictly before a moment (list_upcoming's
    team_history at the tick)."""

    def __init__(self, history: pd.DataFrame):
        dates: Dict[str, List[pd.Timestamp]] = {}
        for h, a, d in zip(history["home_team_id"], history["away_team_id"], history["match_date"]):
            dates.setdefault(str(h), []).append(d)
            dates.setdefault(str(a), []).append(d)
        self._dates = {k: np.sort(np.asarray([x.value for x in v], dtype=np.int64)) for k, v in dates.items()}

    def count_before(self, team_id: str, at: pd.Timestamp) -> int:
        arr = self._dates.get(str(team_id))
        if arr is None:
            return 0
        return int(np.searchsorted(arr, at.value, side="left"))


def stored_inputs(stored: Optional[pd.DataFrame], match_id: str, before: pd.Timestamp) -> Optional[Dict[str, Any]]:
    """The newest stored asian_handicap row of `match_id` created before
    `before`, with the match_result row of the same model_version and the
    stored anchor; None when there is no such AH row. Raises on an AH row with
    no 1x2 sibling (precompute writes both in one transaction)."""
    if stored is None or stored.empty:
        return None
    rows = stored[(stored["match_id"] == match_id) & (stored["created_at"] < before)]
    ah = rows[rows["prediction_type"] == MARKET]
    if ah.empty:
        return None
    row = ah.sort_values("created_at").iloc[-1]
    mr = rows[(rows["prediction_type"] == "match_result") & (rows["model_version"] == row["model_version"])]
    if mr.empty:
        raise RuntimeError(f"stored AH row of {match_id} ({row['model_version']}) has no match_result sibling")
    meta = row["metadata"] or {}
    probs1x2 = mr.iloc[-1]["probabilities"] or {}
    anchor = float(meta["market_over_2.5"]) if meta.get("reconcile") == "1x2+ou2.5" else None
    return {
        "model_version": row["model_version"],
        "probabilities": {str(k): float(v) for k, v in (row["probabilities"] or {}).items()},
        "target": tuple(float(probs1x2[k]) for k in ("home", "draw", "away")),
        "anchor": anchor,
        "reconcile": meta.get("reconcile"),
    }


def derive_ah(
    h_lam: float, a_lam: float, rho: float, target: Sequence[float], anchor: Optional[float]
) -> Dict[str, float]:
    """The stored-form asian_handicap dict precompute_predictions would write."""
    from predictors.market_derivation import derive_from_lambdas  # type: ignore

    mk = derive_from_lambdas(
        float(h_lam),
        float(a_lam),
        float(rho),
        target_1x2=tuple(float(x) for x in target),
        target_over_25=anchor,
    )
    return {str(k): float(v) for k, v in mk[MARKET].items()}


def _anchor(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ── Part A: Brier ────────────────────────────────────────────────────


def brier_part(pm: pd.DataFrame, scores: Dict[str, np.ndarray], mechanics: bool, limit: int = 0) -> Dict[str, Any]:
    """AH Brier readouts (refit - served, live regime) from the gate's outputs,
    plus the GD RPS and the scaled-shape bound from re-derived matrices."""
    folds = np.asarray(scores["fold"]).astype(str)
    idx = np.arange(len(pm)) if not limit else np.arange(min(limit, len(pm)))
    weights = ftdc.unit_weights()
    mats: Dict[str, np.ndarray] = {}
    for arm in ("served", "refit"):
        lams = pm[[f"lam_h::{arm}", f"lam_a::{arm}"]].to_numpy(dtype=float)[idx]
        rho = pm[f"rho::{arm}"].to_numpy(dtype=float)[idx]
        tgt = pm[["target::home", "target::draw", "target::away"]].to_numpy(dtype=float)[idx]
        anc = pm["anchor_live"].to_numpy(dtype=float)[idx]
        mats[arm] = np.stack(
            [
                ftdc.served_matrix(
                    lams[r, 0], lams[r, 1], rho[r], tgt[r], None if not math.isfinite(anc[r]) else anc[r]
                )
                for r in range(len(idx))
            ]
        )
    h = pm["s_home"].to_numpy(dtype=int)[idx]
    a = pm["s_away"].to_numpy(dtype=int)[idx]
    truth = ftdc.vector_truth(h, a)
    worst = 0.0
    for unit in ALL_AH_UNITS:
        for arm in ("served", "refit"):
            p = np.einsum("rij,kij->rk", mats[arm], weights[unit])
            mine = ftdc.brier_rows(p, truth[unit])
            ref = scores[f"brier::{arm}::{REGIME}::{unit}"][idx]
            gap = float(np.max(np.abs(mine - ref))) if len(idx) else 0.0
            if not math.isfinite(gap) or gap > RECONSTRUCT_ATOL:
                raise RuntimeError(f"re-derived {arm} {unit} Brier differs from scores.npz by {gap:.2e}")
            worst = max(worst, gap)
    out: Dict[str, Any] = {"n": int(len(idx)), "reconstruct_max_abs": worst}
    if mechanics:
        return out
    f = folds[idx]

    def b(arm: str, unit: str) -> np.ndarray:
        return scores[f"brier::{arm}::{REGIME}::{unit}"][idx]

    out["units"] = {u: ftdc.pair(b("refit", u), b("served", u), f) for u in ALL_AH_UNITS}
    pooled_s = np.mean(np.stack([b("served", u) for u in SHAPED_UNITS]), axis=0)
    pooled_r = np.mean(np.stack([b("refit", u) for u in SHAPED_UNITS]), axis=0)
    out["pooled_shaped_mean"] = ftdc.pair(pooled_r, pooled_s, f)
    out["pooled_shaped_mean"]["sd_per_match"] = per_match_diff(pooled_r - pooled_s)["sd"]
    out["pooled_shaped_mean"]["between_folds"] = between_fold_variance(pooled_r - pooled_s, f)
    cls = gd_class(h, a)
    gd = {arm: gd_probabilities(mats[arm]) for arm in mats}
    rps = {arm: rps_rows(gd[arm], cls) for arm in gd}
    out["gd_rps"] = ftdc.pair(rps["refit"], rps["served"], f)
    out["gd_rps"]["sd_per_match"] = per_match_diff(rps["refit"] - rps["served"])["sd"]
    out["gd_rps"]["served"] = float(rps["served"].mean())
    out["gd_brier_7way"] = ftdc.pair(
        ftdc.brier_rows(gd["refit"], np.eye(7)[cls]), ftdc.brier_rows(gd["served"], np.eye(7)[cls]), f
    )
    parts: Dict[str, Tuple[float, float]] = {}
    for u in ALL_AH_UNITS:
        ps = np.einsum("rij,kij->rk", mats["served"], weights[u])
        pr = np.einsum("rij,kij->rk", mats["refit"], weights[u])
        parts[u] = shape_moments(ps, pr, truth[u])
    cum = {arm: np.cumsum(gd[arm], axis=1)[:, :-1] for arm in gd}
    cum_y = np.cumsum(np.eye(7)[cls], axis=1)[:, :-1]
    e_dd, e_dr = shape_moments(cum["served"], cum["refit"], cum_y)
    best = min(SHAPED_UNITS, key=lambda u: out["units"][u]["delta"])
    out["scaled_shape_bound"] = {
        "best_single_line": scaled_shape([parts[best]]),
        "pooled_shaped_mean": scaled_shape([parts[u] for u in SHAPED_UNITS]),
        "gd_rps": scaled_shape([(e_dd / 6.0, e_dr / 6.0)]),
        **{u: scaled_shape([parts[u]]) for u in SHAPED_UNITS},
    }
    out["best_single_line"] = {"unit": best, **out["units"][best]}
    out["best_single_line"]["sd_per_match"] = per_match_diff(b("refit", best) - b("served", best))["sd"]
    out["verdicts"] = {
        name: ftdc.ship_strict(p)
        for name, p in (
            ("best_single_line", out["best_single_line"]),
            ("pooled_shaped_mean", out["pooled_shaped_mean"]),
            ("gd_rps", out["gd_rps"]),
        )
    }
    return out


def forward_pair_brier(
    pm: pd.DataFrame,
    scores: Dict[str, np.ndarray],
    stored_by: Dict[str, pd.DataFrame],
    mechanics: bool,
    limit: int = 0,
) -> Dict[str, Any]:
    """PART A2 — the forward gate's OWN pair on this window: on every gate
    eval match with a stored asian_handicap row created before kickoff,
    production's stored AH probabilities (served) against the gate's monthly
    refit reconciled to that row's stored 1x2 and anchor (refit_on_stored),
    scored exactly as scripts/ab_soccer_ah_forward_gate.py scores them
    (score_ah_pair; an invariant-line mismatch aborts). Its per-match SD and
    between-fold variance are what the forward gate's decision point is
    planned on (Part D)."""
    folds_all = np.asarray(scores["fold"]).astype(str)
    rows: List[Dict[str, Any]] = []
    no_stored = 0
    worst = 0.0
    for i, g in enumerate(pm.itertuples(index=False)):
        if limit and len(rows) >= limit:
            break
        mid = str(g.match_id)
        if mid not in stored_by:
            continue
        st = stored_inputs(stored_by[mid], mid, pd.Timestamp(g.match_date))
        if st is None:
            no_stored += 1
            continue
        g_ = pm.iloc[i]
        ros = derive_ah(g_["lam_h::refit"], g_["lam_a::refit"], g_["rho::refit"], st["target"], st["anchor"])
        for line in ftdc.AH_INVARIANT_LINES:
            worst = max(worst, float(np.max(np.abs(handicap_vector(st["probabilities"], line)
                                                   - handicap_vector(ros, line)))))  # fmt: skip
        sc = score_ah_pair(st["probabilities"], ros, int(g_["s_home"]), int(g_["s_away"]))
        row: Dict[str, Any] = {"match_id": mid, "fold": folds_all[i], "kickoff": g.match_date}
        for k, (a, b) in sc.items():
            row[f"{k}::served"], row[f"{k}::refit"] = a, b
        rows.append(row)
    out: Dict[str, Any] = {"n": len(rows), "no_stored_before_kickoff": no_stored, "invariant_max_abs": worst}
    if mechanics or not rows:
        return out
    df = pd.DataFrame(rows)
    f = df["fold"].to_numpy()
    out["fold_sizes"] = {str(k): int(v) for k, v in df["fold"].value_counts().sort_index().items()}
    out["units"] = {
        u: ftdc.pair(df[f"{u}::refit"].to_numpy(), df[f"{u}::served"].to_numpy(), f) for u in (*SHAPED_UNITS, POOLED)
    }
    d = df[f"{POOLED}::refit"].to_numpy() - df[f"{POOLED}::served"].to_numpy()
    out["units"][POOLED]["sd_per_match"] = per_match_diff(d)["sd"]
    out["units"][POOLED]["between_folds"] = between_fold_variance(d, f)
    post = (pd.to_datetime(df["kickoff"], utc=True) >= INCIDENT_END).to_numpy()
    out["units"][POOLED]["post_incident"] = {
        "n": int(post.sum()),
        "delta": float(d[post].mean()) if post.any() else None,
        "sd_per_match": per_match_diff(d[post])["sd"],
        "between_folds": between_fold_variance(d[post], f[post]),
    }
    return out


# ── Part B / C: the replay ───────────────────────────────────────────


def run(args: argparse.Namespace) -> Dict[str, Any]:
    members._add_src_paths()
    pm, scores = load_ftdc(args.ftdc_dir)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    db = load_db(args.database_url, start, end)
    report: Dict[str, Any] = {
        "window": {"start": str(start), "end": str(end)},
        "mechanics": bool(args.mechanics),
        "bankroll": db["bankroll"],
        "generator": {
            "ev_threshold": args.ev_threshold,
            "prob_floor": args.prob_floor,
            "min_team_history": args.min_team_history,
            "max_odds_age_hours": rec_gating.MAX_ODDS_AGE_HOURS,
            "gate": dict(rec_gating.gate_for(SPORT, MARKET).__dict__),
            "max_stake_fraction": rec_gating.MAX_STAKE_FRACTION,
            "min_consensus_books": rec_gating.MIN_CONSENSUS_BOOKS,
        },
        "early_bet": {"hours_before_kickoff": EARLY_HOURS, "kickoffs_from": str(EARLY_SINCE)},
        "stored_ah_matches_per_month": db["stored_ah_per_month"],
    }
    report["brier"] = brier_part(pm, scores, args.mechanics, args.limit)
    stored_by = {str(k): g for k, g in db["stored"].groupby("match_id")}
    report["brier_forward_pair"] = forward_pair_brier(pm, scores, stored_by, args.mechanics, args.limit)

    odds = db["odds"]
    report["ah_odds"] = {
        "rows": int(len(odds)),
        "matches": int(odds["match_id"].nunique()) if len(odds) else 0,
        "first_created": str(pd.to_datetime(odds["created"], utc=True).min()) if len(odds) else None,
        "books": sorted(odds["bookmaker"].unique().tolist()) if len(odds) else [],
        "snapshots": int(len(db["snaps"])),
    }
    rows_by_match: Dict[str, List[Dict[str, Any]]] = {}
    for rec in odds.to_dict("records"):
        rows_by_match.setdefault(str(rec["match_id"]), []).append(rec)
    snaps = snapshot_index(db["snaps"].to_dict("records"))
    matches = db["matches"].set_index("match_id")
    pm_by = pm.set_index("match_id")
    hist = TeamHistory(db["history"])
    cand = sorted((m for m in rows_by_match if m in matches.index), key=lambda m: (matches.loc[m, "match_date"], m))
    if args.limit:
        cand = cand[: int(args.limit)]
    bankroll = db["bankroll"]

    state_drops: Dict[str, int] = {k: 0 for k in STATE_DROP_REASONS}
    early_drops: Dict[str, int] = {k: 0 for k in STATE_DROP_REASONS}
    recs: Dict[str, List[Rec]] = {arm: [] for arm in ARMS}
    recs_early: Dict[str, List[Rec]] = {arm: [] for arm in ARMS}
    repro_recs: List[Rec] = []
    suppressed: Dict[str, Dict[str, int]] = {arm: {} for arm in ARMS}
    defined: Dict[str, List[str]] = {arm: [] for arm in ARMS}
    defined_early: Dict[str, List[str]] = {arm: [] for arm in ARMS}
    status: Dict[str, str] = {}
    counts: Dict[str, Any] = {
        "candidates": len(cand),
        "ineligible": 0,
        "not_in_gate_eval": 0,
        "no_stored": 0,
        "price_ties": 0,
        "quotes_captured_after_tick": 0,
        "consensus_differs_at_kickoff": 0,
        "early_candidates": 0,
        "early_ineligible": 0,
        "early_no_stored": 0,
    }
    stored_vs_served_ah: List[float] = []
    baseline: List[Tuple[bool, float]] = []
    baseline_early: List[Tuple[bool, Optional[float], Optional[float]]] = []
    shaped_keys = [f"{ftdc.fmt_line(x)}_{s}" for x in SHAPED_LINES for s in ("home", "away", "push")]
    for mid in cand:
        mrow = matches.loc[mid]
        ko = mrow["match_date"]
        tick = last_tick_before(ko)
        if (
            hist.count_before(mrow["home_team_id"], tick) < args.min_team_history
            or hist.count_before(mrow["away_team_id"], tick) < args.min_team_history
        ):
            counts["ineligible"] += 1
            status[mid] = "ineligible"
            continue
        status[mid] = "eligible"
        rows = rows_by_match[mid]
        state = quote_state(rows, snaps, tick, state_drops)
        # The consensus at kickoff equals the tick's when no quote is captured
        # in between (amendment 4): both are counted, never assumed.
        counts["quotes_captured_after_tick"] += captured_after(rows, snaps, tick, ko)
        close_state = quote_state(rows, snaps, ko - pd.Timedelta(microseconds=1))
        counts["price_ties"] += price_ties(state)
        for q in best_quotes(state):
            p_q = consensus_prob(state, q.selection, q.line) if is_fresh(q) else None
            if p_q is not None:
                baseline.append((ko >= REPRO_START, q.odds * p_q - 1.0))
                if consensus_prob(close_state, q.selection, q.line) != p_q:
                    counts["consensus_differs_at_kickoff"] += 1
        meta = mrow["metadata"] if isinstance(mrow["metadata"], dict) else {}
        arm_probs: Dict[str, Dict[str, float]] = {}
        g = pm_by.loc[mid] if mid in pm_by.index else None
        if g is None:
            counts["not_in_gate_eval"] += 1
        else:
            tgt = (float(g["target::home"]), float(g["target::draw"]), float(g["target::away"]))
            for arm in ("served", "refit"):
                arm_probs[arm] = derive_ah(
                    g[f"lam_h::{arm}"], g[f"lam_a::{arm}"], g[f"rho::{arm}"], tgt, _anchor(g["anchor_live"])
                )
        st = stored_inputs(stored_by.get(mid), mid, tick)
        if st is None:
            counts["no_stored"] += 1
        else:
            arm_probs["stored"] = st["probabilities"]
            if g is not None:
                arm_probs["refit_on_stored"] = derive_ah(
                    g["lam_h::refit"], g["lam_a::refit"], g["rho::refit"], st["target"], st["anchor"]
                )
                gaps = [abs(st["probabilities"][k] - arm_probs["served"][k]) for k in shaped_keys]
                stored_vs_served_ah.append(max(gaps))
            if ko >= REPRO_START:
                devig = rec_gating.devig_power if tick >= POWER_DEVIG_SINCE else rec_gating.devig_proportional
                rr, _ = replay_match(
                    mid, st["probabilities"], state, bankroll, args.ev_threshold, args.prob_floor, devig
                )
                for r in rr:
                    p_m = consensus_prob(state, r.side, r.line)
                    r.extra["ev_mkt"] = (r.odds * p_m - 1.0) if p_m is not None else None
                repro_recs.extend(rr)
        for arm, probs in arm_probs.items():
            defined[arm].append(mid)
            rr, sup = replay_match(mid, probs, state, bankroll, args.ev_threshold, args.prob_floor)
            for k, v in sup.items():
                suppressed[arm][k] = suppressed[arm].get(k, 0) + v
            for r in rr:
                verdict, pl, flat = settle(r.selection, r.odds, r.stake, mrow["home_score"], mrow["away_score"], meta)
                close = closing_price(snaps, rows, mid, r.bookmaker, r.side, r.line, ko)
                p_m = consensus_prob(state, r.side, r.line)
                r.extra.update(
                    {
                        "kickoff": str(ko),
                        "tick": str(tick),
                        "verdict": verdict,
                        "pl": pl,
                        "flat": flat,
                        "clv_same": (r.odds / close - 1.0) if close else None,
                        "ev_mkt": (r.odds * p_m - 1.0) if p_m is not None else None,
                        "p_mkt": p_m,
                    }
                )
            recs[arm].extend(rr)
        if ko >= EARLY_SINCE:
            counts["early_candidates"] += 1
            early_bets(
                mid, mrow, g, arm_probs, rows, snaps, state, stored_by, hist, bankroll, args, early_drops,
                recs_early, defined_early, baseline_early, counts,
            )  # fmt: skip
    counts["state_drops"] = state_drops
    counts["early_state_drops"] = early_drops
    counts["defined"] = {arm: len(v) for arm, v in defined.items()}
    counts["defined_early"] = {arm: len(v) for arm, v in defined_early.items()}
    counts["stored_vs_served_max_abs_shaped"] = (
        {
            "n": len(stored_vs_served_ah),
            "median": float(np.median(stored_vs_served_ah)),
            "p90": float(np.percentile(stored_vs_served_ah, 90)),
        }
        if stored_vs_served_ah
        else None
    )
    report["replay"] = counts
    report["suppressed"] = suppressed
    report["ev_mkt_baseline"] = {
        name: per_match_diff(np.array([v for since, v in baseline if keep(since)]))
        for name, keep in (("all", lambda x: True), ("from_repro_start", lambda x: x))
    }
    report["early_baseline"] = {
        name: {
            "clv_fair": per_match_diff(
                np.array([c for since, c, _ in baseline_early if keep(since) and c is not None])
            ),
            "ev_mkt": per_match_diff(np.array([e for since, _, e in baseline_early if keep(since) and e is not None])),
            "move_both_known": per_match_diff(
                np.array([c - e for since, c, e in baseline_early if keep(since) and c is not None and e is not None])
            ),
        }
        for name, keep in (("all", lambda x: True), ("from_repro_start", lambda x: x))
    }
    by_month: Dict[str, int] = {}
    for mid, st_ in status.items():
        if st_ == "eligible":
            key = matches.loc[mid, "match_date"].strftime("%Y-%m")
            by_month[key] = by_month.get(key, 0) + 1
    report["eligible_ah_matches_by_month"] = dict(sorted(by_month.items()))
    recent = [v for k, v in by_month.items() if "2026-08" <= k <= "2026-10"]
    report["ah_odds_matches_per_month"] = float(np.mean(recent)) if recent else None
    if args.mechanics:
        report["rec_counts"] = {arm: len(v) for arm, v in recs.items()}
        report["early_rec_counts"] = {arm: len(v) for arm, v in recs_early.items()}
        return report
    if args.out_csv:
        write_recs_csv(args.out_csv, {**recs, **{f"{a}@early": v for a, v in recs_early.items()}})

    report["money"] = {arm: money_summary(v) for arm, v in recs.items()}
    report["pairs"] = {
        pair_name(a, b, since): compare_pair(recs, defined, matches, a, b, since, args.bootstrap_draws)
        for a, b, since in PAIRS
    }
    report["early"] = {arm: early_summary(v) for arm, v in recs_early.items()}
    report["early_pairs"] = {
        pair_name(a, b, since): compare_early_pair(
            recs_early, defined_early, matches, a, b, since, args.bootstrap_draws
        )
        for a, b, since in EARLY_PAIRS
    }
    report["reproduction"] = reproduction(
        db["recs"], repro_recs, recs["served"], defined, status, matches, args.bootstrap_draws
    )
    report["power"] = power(report)
    return report


def early_bets(
    mid: str,
    mrow: Any,
    g: Any,
    arm_probs: Dict[str, Dict[str, float]],
    rows: Sequence[Dict[str, Any]],
    snaps: Dict[Tuple[str, str, str, float], List[Tuple[pd.Timestamp, float]]],
    state_last: Sequence[Quote],
    stored_by: Dict[str, pd.DataFrame],
    hist: TeamHistory,
    bankroll: float,
    args: argparse.Namespace,
    drops: Dict[str, int],
    recs_early: Dict[str, List[Rec]],
    defined_early: Dict[str, List[str]],
    baseline_early: List[Tuple[bool, Optional[float], Optional[float]]],
    counts: Dict[str, Any],
) -> None:
    """The early bet of one match (amendment 4): the generator replayed at the
    last tick at least EARLY_HOURS before kickoff, each arm with the
    probabilities it had then (stored: the newest row created before that
    tick; refit_on_stored reconciled to it), and each rec valued against the
    consensus at the LAST tick (state_last), the latest pre-kickoff quote
    state anyone observes: clv_fair = odds x p_last - 1 (the edge against the
    last observed fair price), ev_mkt = odds x p_early - 1 (what the market
    implied when the bet was placed), move = odds x (p_last - p_early) (the
    market moving toward the bet: CLV proper), clv_same = odds / the same
    book's last pre-kickoff price - 1 (vw_rec_clv's definition)."""
    ko = mrow["match_date"]
    t_e = last_tick_before(ko - pd.Timedelta(hours=EARLY_HOURS))
    if (
        hist.count_before(mrow["home_team_id"], t_e) < args.min_team_history
        or hist.count_before(mrow["away_team_id"], t_e) < args.min_team_history
    ):
        counts["early_ineligible"] += 1
        return
    state_e = quote_state(rows, snaps, t_e, drops)
    for q in best_quotes(state_e):
        if not is_fresh(q):
            continue
        p_l, p_e = consensus_prob(state_last, q.selection, q.line), consensus_prob(state_e, q.selection, q.line)
        baseline_early.append(
            (
                ko >= REPRO_START,
                (q.odds * p_l - 1.0) if p_l is not None else None,
                (q.odds * p_e - 1.0) if p_e is not None else None,
            )
        )
    probs_e: Dict[str, Dict[str, float]] = {a: arm_probs[a] for a in ("served", "refit") if a in arm_probs}
    st_e = stored_inputs(stored_by.get(mid), mid, t_e)
    if st_e is None:
        counts["early_no_stored"] += 1
    else:
        probs_e["stored"] = st_e["probabilities"]
        if g is not None:
            probs_e["refit_on_stored"] = derive_ah(
                g["lam_h::refit"], g["lam_a::refit"], g["rho::refit"], st_e["target"], st_e["anchor"]
            )
    for arm, probs in probs_e.items():
        defined_early[arm].append(mid)
        rr, _ = replay_match(mid, probs, state_e, bankroll, args.ev_threshold, args.prob_floor)
        for r in rr:
            p_e = consensus_prob(state_e, r.side, r.line)
            p_l = consensus_prob(state_last, r.side, r.line)
            close = closing_price(snaps, rows, mid, r.bookmaker, r.side, r.line, ko)
            r.extra.update(
                {
                    "kickoff": str(ko),
                    "tick": str(t_e),
                    "p_mkt": p_e,
                    "p_last": p_l,
                    "ev_mkt": (r.odds * p_e - 1.0) if p_e is not None else None,
                    "clv_fair": (r.odds * p_l - 1.0) if p_l is not None else None,
                    "move": (r.odds * (p_l - p_e)) if p_l is not None and p_e is not None else None,
                    "clv_same": (r.odds / close - 1.0) if close else None,
                }
            )
        recs_early[arm].extend(rr)


def early_summary(recs: Sequence[Rec]) -> Dict[str, Any]:
    """Per-arm readout of the early bets: count, mean edge against the last
    observed consensus, mean market-implied EV when placed, mean movement, and
    the same-book CLV."""

    def mean(key: str) -> Tuple[Optional[float], int]:
        v = [float(r.extra[key]) for r in recs if r.extra.get(key) is not None]
        return (float(np.mean(v)) if v else None), len(v)

    out: Dict[str, Any] = {"recs": len(recs), "matches": len({r.match_id for r in recs})}
    for key in ("clv_fair", "ev_mkt", "move", "clv_same"):
        out[f"{key}_mean"], out[f"{key}_n"] = mean(key)
    mv = [float(r.extra["move"]) for r in recs if r.extra.get("move") is not None]
    out["share_moved_toward"] = (sum(1 for x in mv if x > 0) / len(mv)) if mv else None
    out["share_moved_away"] = (sum(1 for x in mv if x < 0) / len(mv)) if mv else None
    return out


def compare_early_pair(
    recs: Dict[str, List[Rec]],
    defined: Dict[str, List[str]],
    matches: pd.DataFrame,
    a: str,
    b: str,
    since: Optional[pd.Timestamp],
    draws: int,
) -> Dict[str, Any]:
    """Arms a and b's early bets on the matches both are defined for: overlap,
    each arm's readout, and the match-clustered paired bootstrap of the mean
    clv_fair and mean move per rec (b - a)."""
    common = sorted(set(defined[a]) & set(defined[b]))
    if since is not None:
        common = [m for m in common if matches.loc[m, "match_date"] >= since]
    keep = set(common)
    ra = [r for r in recs[a] if r.match_id in keep]
    rb = [r for r in recs[b] if r.match_id in keep]
    cmp_ = compare_recs(ra, rb)
    out: Dict[str, Any] = {
        "matches": len(common),
        "a": early_summary(ra),
        "b": early_summary(rb),
        "same": len(cmp_["same"]),
        "moved_line": len(cmp_["moved_line"]),
        "a_only": len(cmp_["a_only"]),
        "b_only": len(cmp_["b_only"]),
    }
    for key in ("clv_fair", "move"):
        sa, ca = per_match_sum(ra, common, key)
        sb, cb = per_match_sum(rb, common, key)
        out[f"bootstrap_{key}"] = ratio_bootstrap(sa, ca, sb, cb, draws)
        out[f"per_match_{key}_sum_diff"] = per_match_diff(sb - sa)
    return out


def pair_name(a: str, b: str, since: Optional[pd.Timestamp]) -> str:
    return f"{a} vs {b}" + (f" from {since.date()}" if since is not None else "")


def compare_pair(
    recs: Dict[str, List[Rec]],
    defined: Dict[str, List[str]],
    matches: pd.DataFrame,
    a: str,
    b: str,
    since: Optional[pd.Timestamp],
    draws: int,
) -> Dict[str, Any]:
    """Arms a and b on the matches both are defined for (kicking off from
    `since`): overlap, disagreement set, money, paired bootstrap."""
    common = sorted(set(defined[a]) & set(defined[b]))
    if since is not None:
        common = [m for m in common if matches.loc[m, "match_date"] >= since]
    keep = set(common)
    ra = [r for r in recs[a] if r.match_id in keep]
    rb = [r for r in recs[b] if r.match_id in keep]
    cmp_ = compare_recs(ra, rb)
    ka = {(r.match_id, r.selection): r for r in ra}
    kb = {(r.match_id, r.selection): r for r in rb}
    ta, tb = per_match_totals(ra, common), per_match_totals(rb, common)
    return {
        "matches": len(common),
        "matches_with_any_rec": int(((ta["flat_n"] + tb["flat_n"]) > 0).sum()),
        "a": money_summary(ra),
        "b": money_summary(rb),
        "same": len(cmp_["same"]),
        "same_price_and_stake": sum(
            1 for k in cmp_["same"] if ka[k].odds == kb[k].odds and abs(ka[k].stake - kb[k].stake) < 0.005
        ),
        "moved_line": len(cmp_["moved_line"]),
        "a_only": money_summary(cmp_["a_only"]),
        "b_only": money_summary(cmp_["b_only"]),
        "moved_line_a": money_summary([x for x, _ in cmp_["moved_line"]]),
        "moved_line_b": money_summary([y for _, y in cmp_["moved_line"]]),
        "bootstrap": paired_bootstrap(ta, tb, draws),
        "per_match_pl_diff": per_match_diff(tb["pl"] - ta["pl"]),
        "per_match_flat_diff": per_match_diff(tb["flat_pl"] - ta["flat_pl"]),
        "per_match_ev_sum_diff": per_match_diff(tb["ev_sum"] - ta["ev_sum"]),
        "a_shaped_share": _shaped_share(ra),
        "b_shaped_share": _shaped_share(rb),
    }


def _shaped_share(recs: Sequence[Rec]) -> Optional[float]:
    if not recs:
        return None
    return sum(1 for r in recs if abs(r.line) >= 0.75 - 1e-9) / len(recs)


def reproduction(
    prod: pd.DataFrame,
    stored_recs: Sequence[Rec],
    served_recs: Sequence[Rec],
    defined: Dict[str, List[str]],
    status: Dict[str, str],
    matches: pd.DataFrame,
    draws: int = BOOTSTRAP_DRAWS,
) -> Dict[str, Any]:
    """Production's AH recs on matches kicking off from REPRO_START against
    the stored-arm replay (logic-era de-vig) and the served-arm replay. Every
    production rec is accounted for: on a match not finished (counted, never
    compared: the replay loads finished matches only), compared, on a match
    the replay found ineligible, on a match the arm is not defined for, or
    written after kickoff (excluded). Also production's own realised P/L on
    the compared recs with a match-clustered bootstrap, next to the
    market-implied EV of the same bets (what a zero-edge stream loses)."""
    prod = prod.copy()
    prod["kickoff"] = pd.to_datetime(prod["kickoff"], utc=True)
    prod = prod[prod["kickoff"] >= REPRO_START]
    unfinished = ~prod["match_id"].isin(matches.index)
    left = prod[unfinished]
    prod = prod[~unfinished]
    after = prod["created_at"] >= prod["kickoff"]
    pre = prod[~after]
    out: Dict[str, Any] = {
        "production_recs_from_repro_start": int(len(prod) + len(left)),
        "production_recs_on_unfinished_matches": int(len(left)),
        "production_unfinished_detail": {
            "matches": int(left["match_id"].nunique()),
            "match_status": {str(k): int(v) for k, v in left["match_status"].value_counts().items()},
            "rec_status": {str(k): int(v) for k, v in left["status"].value_counts().items()},
            "stake": float(left["stake"].sum()),
        },
        "production_recs": int(len(prod)),
        "production_written_after_kickoff": int(after.sum()),
        "production_matches": int(pre["match_id"].nunique()),
        "production_by_status": {str(k): int(v) for k, v in pre["status"].value_counts().items()},
    }
    settled = pre[pre["profit_loss"].notna()]
    mids = sorted(settled["match_id"].astype(str).unique())
    pos = {m: i for i, m in enumerate(mids)}
    pl, stake = np.zeros(len(mids)), np.zeros(len(mids))
    for r in settled.itertuples():
        pl[pos[str(r.match_id)]] += float(r.profit_loss)
        stake[pos[str(r.match_id)]] += float(r.stake)
    bs = ratio_bootstrap(pl, stake, pl, stake, draws) if mids else {}
    out["production_settled"] = {
        "recs": int(len(settled)),
        "matches": len(mids),
        "stake": float(stake.sum()),
        "pl": float(pl.sum()),
        "roi_stake": bs.get("a"),
    }
    pk = {(str(r.match_id), str(r.selection)): r for r in pre.itertuples()}
    for name, rr in (("stored", stored_recs), ("served", served_recs)):
        pool = {m for m in defined[name] if matches.loc[m, "match_date"] >= REPRO_START}
        mine = {(r.match_id, r.selection): r for r in rr if r.match_id in pool}
        prod_in = {k: v for k, v in pk.items() if k[0] in pool}
        outside: Dict[str, int] = {}
        for k in pk:
            if k[0] in pool:
                continue
            why = status.get(k[0], "no_prematch_ah_odds_in_export")
            why = "arm_undefined" if why == "eligible" else why
            outside[why] = outside.get(why, 0) + 1
        both = sorted(set(mine) & set(prod_in))
        po = {k: float(prod_in[k].odds_decimal) for k in both}
        ev_w = [(mine[k].extra.get("ev_mkt"), float(prod_in[k].stake)) for k in both]
        ev_w = [(e, w) for e, w in ev_w if e is not None]
        out[name] = {
            "matches_in_pool": len(pool),
            "production_recs_in_pool": len(prod_in),
            "production_recs_outside_pool": outside,
            "replay_recs": len(mine),
            "both": len(both),
            "production_only": len(set(prod_in) - set(mine)),
            "replay_only": len(set(mine) - set(prod_in)),
            "both_same_price": sum(1 for k in both if abs(po[k] - mine[k].odds) < 1e-9),
            "both_price_within_0.05": sum(1 for k in both if abs(po[k] - mine[k].odds) <= 0.05 + 1e-12),
            "both_same_book": sum(1 for k in both if str(prod_in[k].bookmaker) == mine[k].bookmaker),
            "both_other_book_tied_at_the_price": sum(
                1
                for k in both
                if str(prod_in[k].bookmaker) != mine[k].bookmaker
                and str(prod_in[k].bookmaker) in mine[k].extra.get("tie_books", [])
            ),
            "both_same_stake": sum(1 for k in both if abs(float(prod_in[k].stake) - mine[k].stake) < 0.005),
            "recall": len(both) / len(prod_in) if prod_in else None,
            "precision": len(both) / len(mine) if mine else None,
            "both_ev_mkt_stake_weighted": (
                sum(e * w for e, w in ev_w) / sum(w for _, w in ev_w) if ev_w and sum(w for _, w in ev_w) > 0 else None
            ),
        }
    return out


def power(report: Dict[str, Any]) -> Dict[str, Any]:
    """n (and months at the current volume) to detect each measured effect with
    80% power at two-sided 5%: per-match z tests for the window pair (context)
    and the money readouts, and the fold-clustered t rule the forward gate
    decides on for its own pair (stored vs refit_on_stored)."""
    br = report["brier"]
    per_month = report.get("stored_ah_matches_per_month") or {}
    recent = [v for k, v in per_month.items() if k >= "2026-08"]
    stored_rate = float(np.mean(recent)) if recent else None
    pr = report["pairs"][pair_name("served", "refit", None)]
    pf = report["pairs"][pair_name("stored", "refit_on_stored", None)]
    out: Dict[str, Any] = {"stored_ah_matches_per_month_since_2026_08": stored_rate}
    for name in ("best_single_line", "pooled_shaped_mean", "gd_rps"):
        p = br[name]
        n = n_for_power(p.get("sd_per_match"), p.get("delta"))
        out[f"window_brier_{name}"] = {
            "delta": p.get("delta"),
            "sd_per_match": p.get("sd_per_match"),
            "n_matches_80pct_z": n,
            "months_at_current_volume": (n / stored_rate) if n and stored_rate else None,
            "house_gate": (
                "unreachable at this effect size (|delta| < 0.005)"
                if p.get("delta") is None or p["delta"] > -0.005
                else "the point estimate clears -0.005"
            ),
        }
    out["forward_brier_pooled"] = forward_brier_plan(report)
    ah_rate = report.get("ah_odds_matches_per_month")
    for pair, tag in ((pr, "window"), (pf, "forward")):
        for key, label in (("per_match_pl_diff", "money_stake_pl"), ("per_match_flat_diff", "money_flat_pl")):
            d = pair[key]
            n = n_for_power(d.get("sd"), d.get("mean"))
            out[f"{tag}_{label}"] = {
                "mean_per_match": d.get("mean"),
                "sd_per_match": d.get("sd"),
                "n_matches_80pct_z": n,
                "months_at_current_volume": (n / ah_rate) if n and ah_rate else None,
            }
    early_rate = _early_rate(report)
    for name, tag in (
        (pair_name("served", "refit", None), "window"),
        (pair_name("stored", "refit_on_stored", None), "forward"),
    ):
        d = report["early_pairs"][name]["per_match_clv_fair_sum_diff"]
        n = n_for_power(d.get("sd"), d.get("mean"))
        out[f"{tag}_early_clv_fair"] = {
            "mean_per_match": d.get("mean"),
            "sd_per_match": d.get("sd"),
            "n_matches_80pct_z": n,
            "months_at_current_volume": (n / early_rate) if n and early_rate else None,
        }
    out["ah_odds_matches_per_month_since_2026_08"] = ah_rate
    out["early_matches_per_month_since_2026_08"] = early_rate
    return out


def _early_rate(report: Dict[str, Any]) -> Optional[float]:
    """Early-bet matches per month: the matches the early pair is defined for,
    over the months from EARLY_SINCE to the window end."""
    n = (report.get("replay") or {}).get("defined_early", {}).get("stored")
    end = pd.Timestamp(report["window"]["end"])
    months = (end - EARLY_SINCE).total_seconds() / 86400 / 30.4375
    return (float(n) / months) if n and months > 0 else None


def forward_brier_plan(report: Dict[str, Any]) -> Dict[str, Any]:
    """The forward gate's Brier decision point, planned on its OWN pair
    (Part A2) under its OWN rule (the fold-clustered t(k-1) upper bound < 0).
    For each fold count k from 2026-11: the projected matches (seasonal fold
    sizes at the pair's Aug-Oct 2026 monthly n), the power at them, the power
    with equal folds of the same mean, the n floor (the projection scaled
    down to 80% power), and the power under the sensitivity tau2. Planning
    tau2 = the larger of the window pair's (31 folds) and the forward pair's
    without the constant-prior incident; sensitivity tau2 = the forward pair's
    over all its folds (inflated by the incident's 66-match June fold)."""
    fp = (report.get("brier_forward_pair") or {}).get("units", {}).get(POOLED)
    if not fp:
        return {}
    sizes = report["brier_forward_pair"].get("fold_sizes", {})
    recent = [v for k, v in sizes.items() if "2026-08" <= k <= "2026-10"]
    rate = float(np.mean(recent)) if recent else None
    win_bf = report["brier"].get("pooled_shaped_mean", {}).get("between_folds") or {}
    tau_w = win_bf.get("tau2")
    tau_post = ((fp.get("post_incident") or {}).get("between_folds") or {}).get("tau2")
    tau_all = (fp.get("between_folds") or {}).get("tau2")
    tau2 = max(x for x in (tau_w, tau_post, 0.0) if x is not None)
    delta, sd = fp["delta"], fp["sd_per_match"]
    plan: Dict[str, Any] = {
        "delta": delta,
        "sd_per_match": sd,
        "tau2_window_pair": tau_w,
        "tau2_forward_post_incident": tau_post,
        "tau2_forward_all_folds": tau_all,
        "tau2_used": tau2,
        "matches_per_month": rate,
        "n_80pct_z_per_match": n_for_power(sd, delta),
        "by_folds": {},
    }
    if delta is None or sd is None or not rate or delta >= 0 or not win_bf.get("fold_sizes"):
        return plan
    for k in range(4, 19):
        proj = seasonal_fold_sizes(win_bf["fold_sizes"], rate, k)
        plan["by_folds"][k] = {
            "projected_sizes": [int(round(x)) for x in proj],
            "projected_n": int(round(sum(proj))),
            "power_projected": clustered_t_power_sizes(delta, sd, tau2, proj),
            "power_equal_folds": clustered_t_power(delta, sd, tau2, k, int(round(sum(proj)))),
            "n_floor_80pct": n_floor_for_sizes(delta, sd, tau2, proj),
            "power_projected_sensitivity_tau2": (
                clustered_t_power_sizes(delta, sd, tau_all, proj) if tau_all is not None else None
            ),
        }
    ok = [k for k, v in plan["by_folds"].items() if v["power_projected"] >= POWER_TARGET]
    plan["smallest_k_with_80pct"] = min(ok) if ok else None
    ks = sorted(plan["by_folds"])
    stable = [k for k in ks if all(plan["by_folds"][j]["power_projected"] >= POWER_TARGET for j in ks if j >= k)]
    plan["smallest_k_staying_at_80pct"] = min(stable) if stable else None
    return plan


def write_recs_csv(path: str, recs: Dict[str, List[Rec]]) -> None:
    rows = []
    for arm, rr in recs.items():
        for r in rr:
            rows.append(
                {
                    "arm": arm,
                    "match_id": r.match_id,
                    "selection": r.selection,
                    "odds": r.odds,
                    "bookmaker": r.bookmaker,
                    "ev": r.ev,
                    "p_raw": r.p_raw,
                    "p_kelly": r.p_kelly,
                    "market_prob": r.market_prob,
                    "stake": r.stake,
                    "age_hours": r.age_hours,
                    **r.extra,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


# ── Formatting ───────────────────────────────────────────────────────


def _f(v: Any, spec: str = "+.5f") -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, spec)


def _money_line(name: str, m: Dict[str, Any]) -> str:
    return (
        f"  {name:<22} recs {m['recs']:>4} (settled {m['settled']}, {m['matches']} matches)  stake {m['stake']:>9.2f}"
        f"  P/L {m['pl']:>+9.2f}  ROI {_f(m['roi_stake'], '+.3f')}  flat {m['flat_pl']:>+7.2f}"
        f" ROI {_f(m['roi_flat'], '+.3f')}  odds {_f(m['mean_odds'], '.2f')}  CLV same {_f(m['clv_same_mean'], '+.4f')}"
        f"  market EV at bet {_f(m['ev_mkt_mean'], '+.4f')} (n={m['ev_mkt_n']})"
    )


def _early_line(name: str, m: Dict[str, Any]) -> str:
    return (
        f"  {name:<22} recs {m['recs']:>4} ({m['matches']} matches)  CLV vs last consensus "
        f"{_f(m['clv_fair_mean'], '+.4f')} (n={m['clv_fair_n']})  market EV when placed {_f(m['ev_mkt_mean'], '+.4f')}"
        f"  move {_f(m['move_mean'], '+.4f')} (toward {_f(m['share_moved_toward'], '.2f')}, away "
        f"{_f(m['share_moved_away'], '.2f')})  CLV same book {_f(m['clv_same_mean'], '+.4f')} (n={m['clv_same_n']})"
    )


def _boot_line(k: str, v: Dict[str, Any]) -> str:
    v = v or {}
    return (
        f"    {k:<20} {_f(v.get('estimate'), '+.4f')}  bootstrap SE {_f(v.get('se'), '.4f')}  95% "
        f"[{_f(v.get('lo95'), '+.4f')}, {_f(v.get('hi95'), '+.4f')}]"
    )


def format_report(r: Dict[str, Any]) -> str:
    rp = r["replay"]
    br = r["brier"]
    fw = r["brier_forward_pair"]
    L = [
        f"Soccer AH replay: window {r['window']['start']}..{r['window']['end']}; bankroll {r['bankroll']}",
        f"AH odds: {r['ah_odds']['rows']} pre-match rows on {r['ah_odds']['matches']} finished matches, first created "
        f"{r['ah_odds']['first_created']}; books {r['ah_odds']['books']}; {r['ah_odds']['snapshots']} snapshots",
        f"candidates {rp['candidates']}: ineligible {rp['ineligible']}, outside the gate's eval set "
        f"{rp['not_in_gate_eval']}, no stored prediction {rp['no_stored']}; arms defined {rp['defined']}",
        f"odds state at the tick: {rp['state_drops']}; best prices shared by 2+ books {rp['price_ties']}",
        f"quotes captured after the last tick, before kickoff: {rp['quotes_captured_after_tick']}; best-quote "
        f"consensus different at kickoff: {rp['consensus_differs_at_kickoff']}",
        f"early bet ({EARLY_HOURS} h, kickoffs from {EARLY_SINCE.date()}): candidates {rp['early_candidates']}, "
        f"ineligible {rp['early_ineligible']}, no stored row {rp['early_no_stored']}; arms defined "
        f"{rp['defined_early']}; odds state {rp['early_state_drops']}",
        f"stored vs served shaped AH max |d| per match: {rp['stored_vs_served_max_abs_shaped']}",
        f"Brier re-derivation vs scores.npz: max |d| {br['reconstruct_max_abs']:.1e} over {br['n']} rows",
        f"forward pair (stored vs refit_on_stored): {fw['n']} matches, no stored row before kickoff "
        f"{fw['no_stored_before_kickoff']}, invariant lines max |d| {fw['invariant_max_abs']:.1e}",
    ]
    if r["mechanics"]:
        L.append(f"MECHANICS ONLY (no deltas, no money): rec counts {r['rec_counts']}; early {r['early_rec_counts']}")
        return "\n".join(L)
    L += ["", "PART A — Brier, refit - served, live regime (post hoc; context only):"]
    for u in ALL_AH_UNITS:
        L.append(ftdc._pair_line(u, br["units"][u]))
    for name in ("best_single_line", "pooled_shaped_mean", "gd_rps", "gd_brier_7way"):
        L.append(ftdc._pair_line(name, br[name]))
    for name, v in br["verdicts"].items():
        L.append(f"  house gate on {name}: {'PASS' if v['ship_strict'] else 'FAIL'} — {v['line']}")
    L.append(f"  pooled between folds: {br['pooled_shaped_mean'].get('between_folds')}")
    L.append(
        "  scaled-shape bound (in-sample best k for served + k*(refit - served); gain = Brier reduction vs served):"
    )
    for name, v in br["scaled_shape_bound"].items():
        L.append(
            f"    {name:<20} E[sum d^2] {_f(v['e_d2'], '.5f')}  gain at refit {_f(v['gain_at_refit'])}  best k "
            f"{_f(v['best_k'], '.2f')}  gain at best k {_f(v['gain_at_best_k'])}"
        )
    L += ["", "PART A2 — the forward gate's own pair (stored vs refit_on_stored; refit - stored):"]
    for u, p in (fw.get("units") or {}).items():
        L.append(ftdc._pair_line(u, p))
    L.append(f"  fold sizes {fw.get('fold_sizes')}")
    L.append(f"  pooled between folds: {(fw.get('units') or {}).get(POOLED, {}).get('between_folds')}")
    L.append(f"  pooled without the incident: {(fw.get('units') or {}).get(POOLED, {}).get('post_incident')}")
    L += ["", "PART B — money (replayed at the last tick; settled by grading_outcomes):"]
    L.append(
        "  'market EV at bet' = odds x the >= 3-book power-de-vigged consensus at the bet's own tick - 1. It is NOT "
        "closing-line value: no AH quote is captured after the last tick, so the 'close' is the bet's own moment."
    )
    for name, v in r["ev_mkt_baseline"].items():
        L.append(
            f"  market EV at bet of EVERY fresh best quote (a random side at the best price) [{name}]: n={v['n']} "
            f"mean {_f(v['mean'], '+.4f')} sd {_f(v['sd'], '.4f')}"
        )
    for arm, m in r["money"].items():
        L.append(_money_line(arm, m))
    for name, p in r["pairs"].items():
        bs = p["bootstrap"]
        L += [
            "",
            f"  [{name}] {p['matches']} matches ({p['matches_with_any_rec']} with a rec in either arm); same rec "
            f"{p['same']} (same price and stake {p['same_price_and_stake']}), same side moved line {p['moved_line']}",
            _money_line("a", p["a"]),
            _money_line("b", p["b"]),
            _money_line("a only", p["a_only"]),
            _money_line("b only", p["b_only"]),
            _money_line("moved line, a", p["moved_line_a"]),
            _money_line("moved line, b", p["moved_line_b"]),
            f"    shaped-line share a {_f(p['a_shaped_share'], '.3f')} b {_f(p['b_shaped_share'], '.3f')}",
        ]
        for k in (
            "a_roi_stake",
            "b_roi_stake",
            "diff_roi_stake",
            "a_roi_flat",
            "b_roi_flat",
            "diff_roi_flat",
            "diff_pl_total",
            "diff_flat_pl_total",
            "a_ev_mkt",
            "b_ev_mkt",
            "diff_ev_mkt",
        ):
            L.append(_boot_line(k, bs.get(k)))
        for k in ("per_match_pl_diff", "per_match_flat_diff", "per_match_ev_sum_diff"):
            d = p[k]
            L.append(f"    {k:<24} mean {_f(d['mean'], '+.4f')}  sd {_f(d['sd'], '.4f')}  SE {_f(d['se'], '.4f')}")
    L += [
        "",
        f"PART B2 — the early bet: replayed {EARLY_HOURS} h before kickoff, valued against the consensus at the last "
        "tick (the latest observed pre-kickoff state; the closing-line readout):",
    ]
    for name, v in r["early_baseline"].items():
        L.append(
            f"  EVERY fresh best quote at the early tick [{name}]: CLV vs last consensus n={v['clv_fair']['n']} mean "
            f"{_f(v['clv_fair']['mean'], '+.4f')}; market EV when placed mean {_f(v['ev_mkt']['mean'], '+.4f')}; move "
            f"(both known, n={v['move_both_known']['n']}) {_f(v['move_both_known']['mean'], '+.4f')} SE "
            f"{_f(v['move_both_known']['se'], '.4f')}"
        )
    for arm, m in r["early"].items():
        L.append(_early_line(arm, m))
    for name, p in r["early_pairs"].items():
        L += [
            "",
            f"  [{name}] {p['matches']} matches; same rec {p['same']}, moved line {p['moved_line']}, a only "
            f"{p['a_only']}, b only {p['b_only']}",
            _early_line("a", p["a"]),
            _early_line("b", p["b"]),
        ]
        for key in ("clv_fair", "move"):
            bs = p[f"bootstrap_{key}"]
            for k in ("a", "b", "diff"):
                L.append(_boot_line(f"{key} {k}", bs.get(k)))
            d = p[f"per_match_{key}_sum_diff"]
            L.append(f"    per-match {key} sum diff mean {_f(d['mean'], '+.4f')}  sd {_f(d['sd'], '.4f')}")
    rep = r["reproduction"]
    ps = rep["production_settled"]
    roi = ps.get("roi_stake") or {}
    L += [
        "",
        f"PART C — reproduction vs production's AH recs from {REPRO_START.date()}: "
        f"{rep['production_recs_from_repro_start']} recs, of which {rep['production_recs_on_unfinished_matches']} on "
        f"matches not finished ({rep['production_unfinished_detail']}); {rep['production_recs']} on finished matches "
        f"({rep['production_written_after_kickoff']} written after kickoff, excluded) on {rep['production_matches']} "
        "matches",
        f"  production's settled recs: {ps['recs']} on {ps['matches']} matches, stake {ps['stake']:.2f}, P/L "
        f"{ps['pl']:+.2f}, ROI {_f(roi.get('estimate'), '+.4f')} (match bootstrap SE {_f(roi.get('se'), '.4f')}, 95% "
        f"[{_f(roi.get('lo95'), '+.4f')}, {_f(roi.get('hi95'), '+.4f')}])",
    ]
    for name in REPRO_ARMS:
        v = rep[name]
        L.append(
            f"  {name:<7} production recs in pool {v['production_recs_in_pool']}, replay {v['replay_recs']}, both "
            f"{v['both']} (recall {_f(v['recall'], '.3f')}, precision {_f(v['precision'], '.3f')}); of both: same "
            f"price {v['both_same_price']}, within 0.05 {v['both_price_within_0.05']}, same book "
            f"{v['both_same_book']} (other book tied at the price {v['both_other_book_tied_at_the_price']}), "
            f"same stake {v['both_same_stake']}, stake-weighted market EV at bet "
            f"{_f(v['both_ev_mkt_stake_weighted'], '+.4f')}; production recs outside the pool "
            f"{v['production_recs_outside_pool']}"
        )
    pw = r["power"]
    L += [
        "",
        f"PART D — power (80%, two-sided 5%); AH-odds matches per month since 2026-08: "
        f"{_f(pw['ah_odds_matches_per_month_since_2026_08'], '.0f')}",
    ]
    for k, v in pw.items():
        if k == "forward_brier_pooled":
            continue
        if isinstance(v, dict):
            L.append(f"  {k:<32} {v}")
    fp = pw.get("forward_brier_pooled") or {}
    L.append(
        f"  forward gate, pooled shaped lines, clustered t rule: delta {_f(fp.get('delta'))} sd "
        f"{_f(fp.get('sd_per_match'), '.4f')} tau2 used {_f(fp.get('tau2_used'), '.3e')} (window "
        f"{_f(fp.get('tau2_window_pair'), '.3e')}, forward without the incident "
        f"{_f(fp.get('tau2_forward_post_incident'), '.3e')}; sensitivity, forward all folds "
        f"{_f(fp.get('tau2_forward_all_folds'), '.3e')}); {_f(fp.get('matches_per_month'), '.0f')} matches a month; "
        f"per-match z n {fp.get('n_80pct_z_per_match')}"
    )
    for k, v in (fp.get("by_folds") or {}).items():
        L.append(
            f"    k={k:<3} projected n {v['projected_n']:>6}  power {v['power_projected']:.3f} (equal folds "
            f"{v['power_equal_folds']:.3f}; sensitivity {_f(v['power_projected_sensitivity_tau2'], '.3f')})  n floor "
            f"{v['n_floor_80pct']}  sizes {v['projected_sizes']}"
        )
    L.append(
        f"    smallest k with 80% at the projected n: {fp.get('smallest_k_with_80pct')}; smallest k from which it "
        f"stays >= 80%: {fp.get('smallest_k_staying_at_80pct')}"
    )
    return "\n".join(L)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    defaults = gr.parse_args([])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ftdc-dir", required=True, help="the --out-dir of scripts/ab_soccer_ft_dixon_coles.py")
    p.add_argument("--start", default="2026-05-01", help="first kickoff day (AH odds exist from 2026-05-30)")
    p.add_argument("--end", default="2026-11-01", help="kickoffs before this day (the gate's window ends 2026-10-31)")
    p.add_argument("--ev-threshold", type=float, default=defaults.ev_threshold)
    p.add_argument("--prob-floor", type=float, default=defaults.prob_floor)
    p.add_argument("--min-team-history", type=int, default=defaults.min_team_history)
    p.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    p.add_argument("--limit", type=int, default=0, help="cap the candidate matches and Brier rows (smoke)")
    p.add_argument("--mechanics", action="store_true", help="print only counts and fidelity lines, never a delta")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None, help="every replayed rec, per arm")
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
        logger.error("ab_soccer_ah_recs_replay failed: %s", exc, exc_info=True)
        return 1
    if args.out_json and not args.mechanics:
        Path(args.out_json).write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False))
    print(format_report(report))
    return 0


def _jsonable(obj: Any) -> Any:
    from ab_halftime_dixon_coles import clean_json

    return clean_json(obj)


if __name__ == "__main__":
    sys.exit(main())
