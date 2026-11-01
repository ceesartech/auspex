"""A/B gate for the NHL regulation-tie structure (dixon_coles_nhl).

THE FINDING BEING ADDRESSED
---------------------------
The 2026-10-31 NHL Dixon-Coles gate (scripts/ab_nhl_dixon_coles.py, audit §4
"NHL Dixon-Coles refit — MEASURED — NOT SHIPPED") found every Dixon-Coles arm
(rho forced to 0, fitted on FINAL scores) under-predicting REGULATION TIES —
served 15.4% vs 24.6% realised over 1,603 games — so a per-season constant
beat the served artifact on regulation 1x2 by -0.023 Brier, double chance by
-0.007..-0.008, total goals by -0.020 and the correct-score grid by -0.009.
Those markets are display-only (NHL recs use the ensembles' moneyline / puck
line / total), but the numbers users see are wrong.

STEP 1 — THE MECHANISM (read-only, 2026-11-01, one REPEATABLE READ export of
prod at 02:37Z; frame 8,169 games 2021-01-13..2026-10-31)
  * "Regulation" is metadata.regulation_winner: 'home' / 'away', or 'tie'
    ('draw' on 38 ESPN rows written before 2026-10) when the game went to
    overtime or a shootout. All 1,829 such games end with a final margin of
    exactly one goal (the OT / SO winner is credited one goal) and no game
    decided in regulation has a tied final, so the regulation score is the
    final score, or (k, k) with k the loser's final goals. The loader's
    metadata.regulation_home/away_score (8,169 of 8,234 finished rows) agree
    with that on every row.
  * Settlement (grading_outcomes): regulation_1x2 and double_chance grade on
    regulation_winner; puck_line, over_under, total_goals, clean_sheet,
    win_to_nil and correct_score grade on the FINAL score (OT / SO goal
    included). All eight are stored by precompute_predictions_nhl and shown
    on the match page (GET /predictions/match/{id} returns every stored row).
  * Pre-window rows (2021-01..2025-09, n=6,551 — the only rows the arm
    design below used), realised regulation scoreline vs an in-sample
    Poisson fitted on regulation scores: tie 21.8% vs 16.1%; |margin| 1
    17.9% vs 29.6%; 2 19.8% vs 22.9%; 3+ 40.5% vs 31.4%. Per margin,
    realised/predicted: 0 1.35, +/-1 0.60, +/-2 0.83..0.90, +/-3 1.52..1.56
    — the late goalie pull (a side down one equalises or concedes into the
    empty net; a side down two concedes a third). Diagonal cells realised /
    served: 0-0 0.43%/0.18%, 1-1 2.96/1.72, 2-2 7.05/4.23, 3-3 6.84/4.68,
    4-4+ 4.55/4.58. Final scores: 67.4% odd totals realised vs 50.0% served
    (an OT game adds one goal to an even regulation total) and no tie ever,
    where served puts 15.4% on the diagonal. Served's level is inflated
    (E[final total] 6.39 vs 6.18).
  * Disclosed: the same step-1 run also printed decision-window descriptives
    (realised regulation tie 24.7%, |margin| 1 18.4%, 2 17.7%, 3+ 39.3%).
    Nothing below was chosen from them.

PRE-REGISTERED DESIGN (fixed 2026-11-01, committed before the first run of this gate)
  Model   predictors.hockey_regulation: the REGULATION scoreline
              P_R(i, j) ∝ Pois(i; λh) · Pois(j; λa) · exp(θ_b(|i-j|))
          on the served (MAX_GOALS_DERIVE+1)^2 grid; team strengths and θ are
          fitted JOINTLY by weighted maximum likelihood on each fit frame's
          regulation scorelines (fit_structure: block coordinate ascent,
          strengths by the tilted moment equations with the model's own
          prior, θ by Newton). The final scoreline moves each regulation tie
          (k, k) to (k+1, k) w.p. q, else (k, k+1) — the NHL scoring rule;
          q = the weighted ML home share of the fit frame's OT / SO games.
          regulation_1x2 / double_chance read P_R, every other market the
          final matrix (market_derivation.derive_hockey_markets(P, final=)).
          NOT A CLOSED LEVER: calibration (audit §4, closed) is a post-hoc map
          of emitted probabilities. θ and the strengths are parameters of the
          regulation likelihood, fitted inside it on each fold's training
          data; q is an ML estimate of the overtime winner's side. No
          emitted probability is re-mapped. Nor is it one of the 2026-10-31
          refit arms, which all fitted FINAL scores with rho 0.
  Arms    every fitted arm is round-tripped through save() and the serve
          loader (precompute_predictions_nhl._load_hockey_dixon_coles_artifact)
          and read through the serve path's own hockey_dc_markets /
          hockey_dc_matrices — no league passed. Strengths for every
          candidate: DixonColesPredictor(DIXON_COLES_CONFIG named
          dixon_coles_nhl), rho 0, via train_hockey_dixon_coles.
          fit_regulation_model (the code a ship would run).
    served             the production file (sha256 ac4e7e10...), incumbent.
    refit_frozen       reference from the 2026-10-31 gate: DIXON_COLES_CONFIG
                       on FINAL scores, rho 0, fitted once on every row before
                       2025-10-01.
    reg_poisson_frozen margin_buckets=(): Poisson on regulation scores + the
                       overtime resolution — the reframing alone (ablation).
    reg_diag_frozen    margin_buckets=(0,): the diagonal-inflated model, one
                       tie log-factor (ablation).
    reg_margin_frozen  margin_buckets=(0, 1, 2, 3, 4) — THE DECIDING ARM. The
                       buckets were chosen on the pre-window rows only by a
                       rule fixed before its result was read: add bucket K
                       while the joint fit's in-sample log-likelihood gain
                       exceeds the BIC penalty 0.5*ln(6551) = 4.39. Gains:
                       0 +75.0, 1 +182.5, 2 +82.5, 3 +37.9, 4 +27.4, then 5
                       +0.4 -> (0..4).
                       The three *_frozen candidates are fitted ONCE on every
                       row before 2025-10-01 and scored on every fold: the
                       artifact is fitted by hand and served for months, so
                       this staleness is what would ship.
    reg_margin         the deciding model refitted monthly (walk-forward):
                       context — what a monthly retrain would serve.
    base_global / base_season / base_season_dw
                       ab_nhl_dixon_coles's walk-forward constants (context);
                       base_season is the per-season constant of the finding.
  Window  monthly walk-forward folds (ft.build_folds / split_fold, fit strictly
          before the fold's month start; thin folds pooled or skipped as
          ab_nhl_dixon_coles.pool_thin_folds) from 2025-10-01 to the newest
          match: the 2026-10-31 gate's decision window plus the games since.
          Every row is out of sample for served (fit ends 2025-04-04) and for
          every frozen arm (fit < 2025-10-01); the pre-window rows the arm
          design used are never scored.
  Units   ab_nhl_dixon_coles.UNITS, graded by grading_outcomes itself
          (regulation markets on regulation_winner, the rest on the final
          score); correct_score_grid is each arm's FINAL matrix as a 121-way
          Brier (served / refit_frozen: their one matrix).
  Primary regulation_1x2 (3-way Brier, home/draw/away).

DECISION RULE (pre-registered)
  SHIP only if reg_margin_frozen, over the window,
    (a) passes ship_strict vs served on regulation_1x2: ft.ship_verdict
        (dBrier <= -0.005, per-match AND fold-clustered z=2 95% upper bounds
        < 0) AND the fold-clustered t(n_folds-1) upper bound < 0; and
    (b) is significantly worse than served on NO blocking unit — every unit
        but regulation_1x2 and correct_score_listed: the fold-clustered
        t(n_folds-1) 95% LOWER bound of dBrier (challenger - served) is not
        > 0 (per-match delta - 2*SE when no clustered SE exists); a unit with
        no gradable row or no bound blocks (ab_nhl_dixon_coles.regression).
  reg_margin_frozen disqualified (the trainer would refuse a fit —
  train_hockey_dixon_coles.regulation_fit_refusal — or an invalid
  probability) -> DO-NOT-SHIP. Every other arm, every comparison with
  refit_frozen or the constants, correct_score_listed, the log score and
  the freshness blocks are context and never decide.
  SHIP would mean: train_hockey_dixon_coles.py's main() fits
  fit_regulation_model (DIXON_COLES_CONFIG, margin buckets 0..4), the
  artifact carries regulation_structure, scripts/promote_dc_model.py gets an
  "nhl" entry and an operator-run train + promote wrapper. DO-NOT-SHIP
  would mean: no model change; the NHL Dixon-Coles markets that lose to the
  per-season constant stop being displayed.

FRESHNESS (confirmation, never decides)
  The 2026-10-31 finding was read off that gate's pooled decision window.
  Reported for every arm vs served on every unit — per-match dBrier +/- SE
  (one fold: no clustered SE), sign and magnitude:
    (i)  the 2026-27 season (match_date >= 2026-08-01). 209 of these games
         were scored by the 2026-10-31 gate (its fold 2026-10, for over_5.5);
    (ii) the games after that gate's newest match (2026-10-31T00:00Z),
         scored by no earlier NHL analysis.
  The primary is also split into the folds 2025-10..2026-02 and the rest.

RESULT (2026-11-01, the committed harness run once, unamended, on a local
replica of the 02:37Z export; 8 folds 2025-10..2026-10, n=1,611)
  DECISION: DO-NOT-SHIP. reg_margin_frozen passes the primary by a wide
  margin — regulation_1x2 -0.0172 vs served (fold-clustered -0.0168 +/-
  0.0029, t-bound -0.0101; tie mass 21.5% vs served 15.4%, realised 24.6%)
  — but is significantly worse on two blocking units: clean_sheet_home
  +0.00043 (95% lower bound +0.00006) and win_to_nil_home +0.00047 (+0.00006).
  It improves total_goals -0.0205, correct_score_grid -0.0093, over_4.5
  -0.0014, double_chance -0.004..-0.007; over_5.5 -0.0012 and over_6.5 +0.0000
  sit inside the noise.
  Context only (never decides, so nothing ships on it): reg_diag_frozen (one
  tie factor) reads -0.0171 (t-bound -0.0099) with NO blocking regression
  (clean_sheet_home +0.00009, lower bound -0.00008); the walk-forward
  reg_margin -0.0231 (t-bound -0.0166), no regression; reg_poisson_frozen
  (the reframing alone) only -0.0076 — the tie factor, not the reframing,
  carries the regulation gain (reg_diag - reg_poisson -0.0094, t-bound
  -0.0059). Every structured arm still trails the per-season constant on
  regulation_1x2 (frozen +0.0063, walk-forward +0.0003; inside the noise):
  past the season's tie rate, team strength adds little to the 3-way.
  Served vs base_season, the units where the constant is better beyond the
  fold-clustered t bound: regulation_1x2 (-0.0235, upper -0.0081),
  double_chance_12 (-0.0084, -0.0038), double_chance_X2 (-0.0080, -0.0006),
  total_goals (-0.0197, -0.0123), over_4.5 (-0.0029, -0.0009),
  correct_score_grid (-0.0093, -0.0070); puck line, clean sheet and win to
  nil are inside the noise.
  Freshness: 2026-27 (n=217) reg_margin_frozen regulation_1x2 -0.0060 +/-
  0.0068 (same sign, inside the noise), total_goals -0.0144 +/- 0.0074, grid
  -0.0093 +/- 0.0012, clean_sheet_home +0.0007 +/- 0.0004; the 8 games after
  the previous gate: regulation_1x2 -0.031 +/- 0.042 — no information.
  Per-game rows / grids / JSON / log: the closeout scratch dir
  closeout/nhl_regtie/gate/ (ab_nhl_regtie_rows.csv recomputes every delta).
  Post-review refactor (no amendment): rule (a)'s ship_strict moved out of
  run() into the pure strict_verdict(), logic unchanged, so a test pins its
  t(n_folds-1) half; the serve loader also refuses a structured artifact
  with rho != 0 (none of these arms is one). Re-run on the same export:
  JSON, per-game rows and grids identical to the recorded run.

RE-GATE — PRE-REGISTERED 2026-11-01, committed before any re-gate run (--regate)
  Why. The gate above was DO-NOT-SHIP on its single deciding arm, but its
  context arm reg_diag_frozen (one tie log-factor) passed the primary with no
  blocking regression. Switching arms after reading that would be choosing
  the winner after the fact, so the audit §4 row's revisit condition asks for
  a fresh-window re-gate with the tie-factor model PRE-REGISTERED as the
  deciding arm. This is that pre-registration. It changes nothing in the
  gate above (its mode, GATE_2026_11_01, is the default and reproduces the
  recorded run).
  Deciding arm  reg_diag: margin_buckets (0,) — the regulation scoreline
                Pois x Pois x exp(theta_0 [i == j]) fitted jointly by
                train_hockey_dixon_coles.fit_regulation_model with
                DIXON_COLES_CONFIG (dixon_coles_nhl), rho 0, REFITTED MONTHLY:
                each fold's fit uses every game strictly before the fold's
                month start (ft.split_fold). The monthly refit is the arm, not
                an option: it is the object a monthly NHL job would serve, and
                the soccer gates showed a one-off fit decays.
  Context       reg_diag_frozen (the same model fitted once before
                2026-11-01), refit_frozen, served, and the constants. Never
                decide.
  Served        the production dixon_coles_nhl file, sha256 ac4e7e10...
                (SERVED_SHA256). If production's file changes before the
                decision, this run refuses it and the re-gate must be
                re-registered against the new incumbent.
  Window        folds from 2026-11 (REGATE_FOLD_START), pooled / skipped as
                above. A game is scored only if it kicks off at or after
                2026-11-02T00:00Z (REGATE_FIRST_KICKOFF: games on or before
                2026-11-01 UTC are never scored, and a guard refuses the run
                if one reaches scoring) AND is a regular-season game
                (metadata.season_type 'regular', or game_type 'regular' when
                season_type is absent; preseason, postseason and untagged
                games are excluded).
  When          the decision is read once >= 1,000 such games exist
                (REGATE_MIN_GAMES; about early April 2027 at ~6.6 games a day).
                Below that the run reports counts only — no delta, no
                verdict — so nobody can peek and stop early. If the 2026-27
                regular season ends short of 1,000 scored games, the decision
                waits for the next regular season's games; the rule does not
                change.
  Units, primary, statistics: as the gate above (regulation_1x2 primary,
                ab_nhl_dixon_coles.UNITS graded by grading_outcomes, paired
                per-game deltas with fold-clustered SE and t(n_folds-1)
                bounds).
  DECISION RULE SHIP only if reg_diag (not disqualified) (a) passes
                ship_strict vs served on regulation_1x2 (dBrier <= -0.005,
                per-match AND fold-clustered z=2 upper bounds < 0, AND the
                t(n_folds-1) clustered upper bound < 0) and (b) is
                significantly worse on NO blocking unit under rule A2 (a unit
                regresses when its fold-clustered t(n_folds-1) 95% LOWER bound
                of dBrier is > 0; per-match delta - 2 SE without a clustered
                SE; an unassessable unit blocks).
  SHIP would mean: train_hockey_dixon_coles.py fits fit_regulation_model with
  margin buckets (0,) on a monthly schedule outside retrain_models,
  promote_dc_model.py gets an "nhl" entry, an operator-run train + promote
  wrapper, and a markets un-hide only where the gated artifact beats the
  per-season constant. This commit runs nothing of that.
  Mechanics. Before this commit the re-gate ran only as a mechanics smoke
  (--regate --mechanics --limit 60) on a local replica of a read-only
  REPEATABLE READ export of prod's NHL matches at 2026-11-01 21:36:55Z (8,339
  rows; the served file copied, sha256 ac4e7e10 checked): 0 scorable games
  (none after 2026-11-01 had been played), counts only. The 2026-11-01
  gate's own path ran the same way (--mechanics --limit 60: 480 games, 8
  folds, 12 fits, counts only). No delta of either was computed for output.
  Usage once the n is reached (api container):
    python /app/scripts/ab_nhl_regulation_ties.py --regate --out-json /tmp/ab_nhl_regate.json

Per-game output for an independent recomputation: --out-rows (CSV: per arm
the regulation and over-line probabilities and every unit's per-game Brier),
--out-grids (npz: per arm the 121-cell final grid per game).

Read-only: SELECTs, plus temp files for the round-trip. Usage (api container,
or locally against a replica with --served-artifact):
    python /app/scripts/ab_nhl_regulation_ties.py --out-json /tmp/ab_nhl_regtie.json
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
import ab_nhl_dixon_coles as nhl  # noqa: E402

ht = nhl.ht
ft = nhl.ft
trainer = nhl.trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("ab_nhl_regulation_ties")

SERVED_ARTIFACT = nhl.SERVED_ARTIFACT
SERVED_SHA256 = "ac4e7e10410f9035e33ecc933059152c177ecb90da75997711168c73a260b8a9"
DEFAULT_START = "2025-10-01"  # the window, and the frozen arms' fit cutoff
PREVIOUS_GATE_NEWEST = "2026-10-31T00:00:00+00:00"  # the 2026-10-31 gate's newest match
FRESH_SEASON = 2026  # 2026-27
DISCOVERY_FOLDS = ("2025-10", "2025-11", "2025-12", "2026-01", "2026-02")

PRIMARY = "regulation_1x2"
UNIT_NAMES = nhl.UNIT_NAMES
CONTEXT_UNITS = (nhl.CORRECT_SCORE_LISTED,)
BLOCKING_UNITS = tuple(u for u in UNIT_NAMES if u != PRIMARY and u not in CONTEXT_UNITS)

# Candidate arms: name -> (margin buckets, frozen?)
CANDIDATES: Dict[str, Tuple[Tuple[int, ...], bool]] = {
    "reg_poisson_frozen": ((), True),
    "reg_diag_frozen": ((0,), True),
    "reg_margin_frozen": (trainer.REGULATION_MARGIN_BUCKETS, True),
    "reg_margin": (trainer.REGULATION_MARGIN_BUCKETS, False),
}
DECIDING_ARM = "reg_margin_frozen"
REFERENCE_FROZEN = "refit_frozen"
CHALLENGERS = (REFERENCE_FROZEN, *CANDIDATES)
CONSTANT_ARMS = nhl.CONSTANT_ARMS
ARMS = ("served", *CHALLENGERS, *CONSTANT_ARMS)

# ── The re-gate (pre-registered 2026-11-01; docstring "RE-GATE") ─────
REGATE_FOLD_START = "2026-11-01"  # first fold month; reg_diag_frozen (context) is fitted before it
REGATE_FIRST_KICKOFF = "2026-11-02T00:00:00+00:00"  # a game on or before 2026-11-01 (UTC) is never scored
REGATE_MIN_GAMES = 1000  # no decision, and no delta, below this many scored games
REGATE_DECIDING_ARM = "reg_diag"
REGATE_CANDIDATES: Dict[str, Tuple[Tuple[int, ...], bool]] = {
    "reg_diag": ((0,), False),  # THE DECIDING ARM: one tie log-factor, refitted monthly
    "reg_diag_frozen": ((0,), True),  # context: the same model fitted once before the first fold
}
SEASON_TAGS_SQL = """
    SELECT m.id::text AS match_id, m.metadata->>'season_type' AS season_type,
           m.metadata->>'game_type' AS game_type
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    WHERE l.sport = 'nhl' AND m.status = 'finished'
"""


@dataclass(frozen=True)
class GateMode:
    """What one run of this harness gates: its window start, its candidate
    arms, the deciding arm, and the re-gate's scoring rules."""

    name: str
    start: str
    candidates: Dict[str, Tuple[Tuple[int, ...], bool]]
    deciding_arm: str
    first_kickoff: Optional[str] = None  # never score a game before this
    regular_season_only: bool = False
    min_games: int = 0  # refuse to decide (and to report a delta) below this


GATE_2026_11_01 = GateMode("2026-11-01 gate", DEFAULT_START, CANDIDATES, DECIDING_ARM)
REGATE = GateMode(
    "re-gate (pre-registered 2026-11-01)",
    REGATE_FOLD_START,
    REGATE_CANDIDATES,
    REGATE_DECIDING_ARM,
    first_kickoff=REGATE_FIRST_KICKOFF,
    regular_season_only=True,
    min_games=REGATE_MIN_GAMES,
)


# ── Pure ─────────────────────────────────────────────────────────────


def decide(
    verdicts_vs_served: Dict[str, Dict[str, Any]],
    regressions: Dict[str, Dict[str, str]],
    disqualified: Dict[str, Any],
    arm: str = DECIDING_ARM,
) -> Dict[str, Any]:
    """The pre-registered decision for ``arm``: ship_strict on the primary
    AND no significant regression on a blocking unit."""
    if arm in disqualified or arm not in verdicts_vs_served:
        why = disqualified.get(arm, {}).get("reason", "not scored")
        return {"arm": arm, "ship": False, "primary_pass": False, "regressions": {}, "reason": f"disqualified: {why}"}
    primary_pass = bool(verdicts_vs_served[arm]["ship_strict"])
    regs = dict(regressions.get(arm, {}))
    ship = primary_pass and not regs
    if ship:
        reason = f"{PRIMARY} passes ship_strict vs served and no blocking unit is significantly worse"
    elif not primary_pass:
        reason = f"{PRIMARY} fails ship_strict vs served" + (f"; {len(regs)} unit(s) also regress" if regs else "")
    else:
        reason = f"{PRIMARY} passes, but {len(regs)} unit(s) regress: {', '.join(regs)}"
    return {"arm": arm, "ship": ship, "primary_pass": primary_pass, "regressions": regs, "reason": reason}


def strict_verdict(p: Dict[str, Any], baseline: str) -> Dict[str, Any]:
    """ft.ship_verdict on one paired comparison (run()'s pair() output) plus
    rule (a)'s ship_strict: that verdict's SHIP (dBrier <= -0.005, per-match
    AND fold-clustered z=2 upper bounds < 0) AND the fold-clustered
    t(n_folds-1) 95% upper bound < 0. The t bound is the stricter one at few
    folds (multiplier 2.36 at 8 folds against z=2), so a SHIP on the z=2
    bounds alone is not a pass; no t bound (fewer than 2 folds) never ships."""
    v = ft.ship_verdict(
        p["delta"],
        p["se"],
        baseline,
        delta_cluster=p["delta_cluster"],
        se_cluster=p["se_cluster"],
        n_folds=p["n_folds"],
    )
    t_upper = p["upper_95_cluster_t"]
    v["upper_95_cluster_t"] = t_upper
    v["ship_strict"] = bool(v["ship"] and t_upper is not None and t_upper < 0)
    return v


def blocking_regressions(per_unit: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """ab_nhl_dixon_coles.regression on every BLOCKING unit of this gate."""
    out: Dict[str, str] = {}
    for unit in BLOCKING_UNITS:
        why = nhl.regression(per_unit.get(unit, {}))
        if why is not None:
            out[unit] = why
    return out


def per_match_pair(c: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Per-match paired dBrier (c - b) and its SE over the finite rows — the
    freshness blocks, where one fold allows no clustered SE."""
    c, b = np.asarray(c, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(c) & np.isfinite(b)
    n = int(ok.sum())
    if n == 0:
        return {"n": 0, "delta": None, "se": None}
    d = c[ok] - b[ok]
    return {"n": n, "delta": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else None}


def _present(v: Any) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def is_regular_season(season_type: Any, game_type: Any) -> bool:
    """A regular-season NHL game: metadata.season_type 'regular', or (no
    season_type) the legacy loader's game_type 'regular'. Preseason,
    postseason / playoff and untagged games are not."""
    if _present(season_type):
        return str(season_type).strip().lower() == "regular"
    return _present(game_type) and str(game_type).strip().lower() == "regular"


def scorable(ev: pd.DataFrame, mode: GateMode) -> np.ndarray:
    """Which of a fold's games this mode may score: kicking off at or after
    mode.first_kickoff and, for the re-gate, in the regular season."""
    keep = np.ones(len(ev), dtype=bool)
    if mode.first_kickoff is not None:
        keep &= (pd.to_datetime(ev["match_date"], utc=True) >= pd.Timestamp(mode.first_kickoff)).to_numpy()
    if mode.regular_season_only:
        keep &= np.array([is_regular_season(s, g) for s, g in zip(ev["season_type"], ev["game_type"])], dtype=bool)
    return keep


def check_scored(ev: pd.DataFrame, mode: GateMode) -> None:
    """Refuse loudly if a game the mode must never score reached scoring."""
    if not len(ev):
        return
    if mode.first_kickoff is not None:
        early = pd.to_datetime(ev["match_date"], utc=True) < pd.Timestamp(mode.first_kickoff)
        if early.any():
            raise RuntimeError(
                f"{mode.name}: {int(early.sum())} game(s) before {mode.first_kickoff} reached scoring — refused"
            )
    if mode.regular_season_only:
        bad = [not is_regular_season(s, g) for s, g in zip(ev["season_type"], ev["game_type"])]
        if any(bad):
            raise RuntimeError(f"{mode.name}: {sum(bad)} non-regular-season game(s) reached scoring — refused")


def decidable(n_scored: int, mode: GateMode) -> Tuple[bool, str]:
    """(True, '') when the mode may decide on n_scored games, else (False, why)."""
    if n_scored < mode.min_games:
        return False, f"{n_scored} scored game(s) < the pre-registered {mode.min_games}: not decidable, no delta read"
    return True, ""


def mode_for(args: argparse.Namespace) -> Tuple[GateMode, str]:
    """(mode, window start). The re-gate's window is pinned: any other
    --start is refused."""
    if getattr(args, "regate", False):
        if args.start is not None and args.start != REGATE.start:
            raise RuntimeError(f"the re-gate window is pinned to {REGATE.start}; --start {args.start} refused")
        return REGATE, REGATE.start
    return GATE_2026_11_01, args.start or GATE_2026_11_01.start


def load_season_tags(database_url: str) -> pd.DataFrame:
    from sqlalchemy import create_engine

    engine = create_engine(database_url)
    try:
        return pd.read_sql(SEASON_TAGS_SQL, engine)
    finally:
        engine.dispose()


# ── Model plumbing (needs ml-models on the path) ─────────────────────


def serve_module():
    """precompute_predictions_nhl — the serve path every arm is read through."""
    import precompute_predictions_nhl as ppn  # type: ignore

    return ppn


def model_arm(
    model: Any, ev: pd.DataFrame
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[np.ndarray], Optional[str]]:
    """(per-unit probabilities, E[final total], final matrices, invalid) for
    one served model, every market from hockey_dc_markets and the grid from
    hockey_dc_matrices' final matrix."""
    ppn = serve_module()
    markets, finals = [], []
    for home, away in zip(ev["home_team"], ev["away_team"]):
        R, F, _ = ppn.hockey_dc_matrices(model, home, away)
        finals.append(F)
        markets.append(ppn.hockey_dc_markets(model, home, away))
    bad = nhl.invalid_matrices(finals)
    if bad:
        return {}, np.array([]), [], bad
    for m in markets:
        if set(m) != set(nhl.SERVED_MARKETS):
            raise RuntimeError(
                f"hockey_dc_markets yields {sorted(m)}, this harness scores {sorted(nhl.SERVED_MARKETS)}"
            )
    return nhl.readout(markets, ev, finals), nhl.expected_totals(finals), finals, None


def fit_candidate(
    fit_df: pd.DataFrame, buckets: Tuple[int, ...], cfg: Any, tmpdir: str, tag: str
) -> Tuple[Any, Dict[str, Any], str, Optional[str]]:
    """Fit a candidate exactly as the trainer would (fit_regulation_model),
    then round-trip it through save() + the serve loader."""
    ppn = serve_module()
    cols = [*nhl.FIT_COLS, "regulation_winner"]
    model, result, refusal = trainer.fit_regulation_model(fit_df[cols].copy(), cfg, buckets)
    path = os.path.join(tmpdir, f"{tag}.bin")
    model.save(path)
    served = ppn._load_hockey_dixon_coles_artifact(path, f"candidate {tag}")
    if served is None:
        raise RuntimeError(f"candidate {tag}: the serve loader refused the round-tripped artifact")
    return served, result, path, refusal


# ── Run ──────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> Dict[str, Any]:
    sys.path.insert(0, ht.ML_SRC)
    mode, start = mode_for(args)
    candidates = mode.candidates
    challengers_all = (REFERENCE_FROZEN, *candidates)
    arms_all = ("served", *challengers_all, *CONSTANT_ARMS)
    frame, frame_audit = nhl.load_frame(args.database_url)
    if mode.regular_season_only:
        frame = frame.merge(load_season_tags(args.database_url), on="match_id", how="left", validate="many_to_one")
    else:
        frame["season_type"], frame["game_type"] = None, None
    frame["match_date"] = pd.to_datetime(frame["match_date"], utc=True)
    frame = frame.sort_values("match_date", kind="stable").reset_index(drop=True)
    dups = int(frame["match_id"].duplicated().sum())
    if dups:
        logger.error("NHL frame carries %d duplicated match_id rows — dropped for scoring", dups)
        frame = frame.drop_duplicates("match_id", keep="first").reset_index(drop=True)
    frame["season"] = [nhl.season_of(t) for t in frame["match_date"]]
    end = args.end or frame["match_date"].max().strftime("%Y-%m-%d")
    limit = int(getattr(args, "limit", 0) or 0)

    def fold_rows(fold: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
        fit_df, ev = ft.split_fold(frame, fold)
        ev = ev[scorable(ev, mode)]
        if limit:
            ev = ev.head(limit)
        check_scored(ev, mode)
        return fit_df, ev

    raw_folds = ft.build_folds(start, end) if pd.Timestamp(end, tz="UTC") >= pd.Timestamp(start, tz="UTC") else []
    counts = [len(fold_rows(f)[1]) for f in raw_folds]
    folds, pooled, skipped = nhl.pool_thin_folds(raw_folds, counts)
    if not folds:
        if mode.min_games:
            _, why = decidable(0, mode)
            logger.warning("%s: no fold has >= %d scorable games — %s", mode.name, nhl.MIN_FOLD_EVAL, why)
            return counts_only_report(mode, start, end, 0, [], pooled, skipped, [], {}, why)
        raise RuntimeError(f"no fold has >= {nhl.MIN_FOLD_EVAL} eval rows between {start} and {end}")
    for p in pooled:
        logger.info("fold %s (%d eval rows) pooled into %s", p["fold"], p["n_eval"], p["into"])
    for s in skipped:
        logger.warning("fold %s skipped: %d eval rows < %d", s["fold"], s["n_eval"], nhl.MIN_FOLD_EVAL)
    frozen_cutoff = folds[0].fit_cutoff
    logger.info(
        "frame: %d NHL matches %s..%s; folds %s; frozen arms fitted on match_date < %s",
        len(frame),
        frame["match_date"].min(),
        frame["match_date"].max(),
        [f.label for f in folds],
        frozen_cutoff,
    )

    served_fp = ht.fingerprint(args.served_artifact)
    if served_fp["sha256"] != args.served_sha256:
        raise RuntimeError(
            f"served artifact {args.served_artifact} has sha256 {served_fp['sha256'][:12]}, not the incumbent "
            f"{args.served_sha256[:12]} — point --served-artifact at the incumbent (or its backup)"
        )
    ppn = serve_module()
    served = ppn._load_hockey_dixon_coles_artifact(args.served_artifact, "served")
    if served is None:
        raise RuntimeError(f"the serve loader refused the served artifact {args.served_artifact}")

    cfgs = nhl.arm_configs()
    cfg = cfgs["refit"]
    decay = float(cfg.hyperparameters.get("time_decay", 0.0))
    arms: List[str] = list(arms_all)
    scores: Dict[str, Dict[str, List[np.ndarray]]] = {a: {u: [] for u in UNIT_NAMES} for a in arms}
    log_scores: Dict[str, List[np.ndarray]] = {a: [] for a in arms}
    p_reg: Dict[str, List[np.ndarray]] = {a: [] for a in arms}
    p_over: Dict[str, Dict[str, List[np.ndarray]]] = {a: {k: [] for k in ("4.5", "5.5", "6.5")} for a in arms}
    e_total: Dict[str, List[np.ndarray]] = {a: [] for a in arms}
    grids: Dict[str, List[np.ndarray]] = {a: [] for a in arms}
    ev_frames: List[pd.DataFrame] = []
    fold_of: List[np.ndarray] = []
    y_reg: List[np.ndarray] = []
    fits: List[Dict[str, Any]] = []
    disqualified: Dict[str, Dict[str, Any]] = {}
    frozen: Dict[str, Any] = {}

    def disqualify(arm: str, label: str, reason: str) -> None:
        disqualified[arm] = {"fold": label, "reason": reason}
        logger.error("%s DISQUALIFIED at fold %s: %s", arm, label, reason)
        if arm in arms:
            arms.remove(arm)

    def record(label: str, arm: str, fit_df: pd.DataFrame, cutoff: pd.Timestamp, result, path, refusal) -> None:
        fits.append(
            {
                "fold": label,
                "arm": arm,
                "n_fit": int(len(fit_df)),
                "teams": int(result.get("team_count") or 0),
                "poisson_iterations": result.get("poisson_iterations", result.get("iterations")),
                "structure_iterations": result.get("iterations") if "margin_buckets" in result else None,
                "margin_buckets": result.get("margin_buckets"),
                "margin_log_factors": result.get("margin_log_factors"),
                "ot_home_share": result.get("ot_home_share"),
                "fitted_tie_rate": result.get("fitted_tie_rate"),
                "observed_tie_rate": result.get("observed_tie_rate"),
                "fitted_total": result.get("fitted_mean_total_goals"),
                "observed_total": result.get("observed_mean_total_goals"),
                "loglik_gain": result.get("loglik_gain"),
                "refusal": refusal,
                "newest_fit_gap_days": float((cutoff - fit_df["match_date"].max()).total_seconds() / 86400),
                "sha256": ht.fingerprint(path)["sha256"][:12],
            }
        )

    with tempfile.TemporaryDirectory() as tmp:
        first_fit = frame[frame["match_date"] < frozen_cutoff]
        label0 = f"frozen@{frozen_cutoff.date()}"
        model, result, path, refusal = nhl.fit_arm(first_fit, cfg, tmp, "refit_frozen")
        record(label0, REFERENCE_FROZEN, first_fit, frozen_cutoff, result, path, refusal)
        frozen[REFERENCE_FROZEN] = model
        if refusal is not None:
            disqualify(REFERENCE_FROZEN, label0, f"the NHL trainer would refuse this fit: {refusal}")
        for arm, (buckets, is_frozen) in candidates.items():
            if not is_frozen:
                continue
            try:
                model, result, path, refusal = fit_candidate(first_fit, buckets, cfg, tmp, arm)
            except ValueError as e:  # the trainer refuses the frame itself (regulation_frame)
                disqualify(arm, label0, f"the NHL trainer would refuse this frame: {e}")
                continue
            record(label0, arm, first_fit, frozen_cutoff, result, path, refusal)
            frozen[arm] = model
            if refusal is not None:
                disqualify(arm, label0, f"the NHL trainer would refuse this fit: {refusal}")

        for fold in folds:
            fit_df, ev = fold_rows(fold)
            truth = nhl.fixed_truth(ev)
            arm_probs: Dict[str, Dict[str, np.ndarray]] = {}
            arm_totals: Dict[str, np.ndarray] = {}
            arm_grids: Dict[str, List[np.ndarray]] = {}

            probs, tot, fin, bad = model_arm(served, ev)
            if bad:
                raise RuntimeError(f"served fold {fold.label}: {bad}")
            arm_probs["served"], arm_totals["served"], arm_grids["served"] = probs, tot, fin
            for arm in [REFERENCE_FROZEN, *(a for a, (_, f) in candidates.items() if f)]:
                if arm not in arms:
                    continue
                probs, tot, fin, bad = model_arm(frozen[arm], ev)
                if bad:
                    disqualify(arm, fold.label, bad)
                    continue
                arm_probs[arm], arm_totals[arm], arm_grids[arm] = probs, tot, fin
            for arm, (buckets, is_frozen) in candidates.items():
                if is_frozen or arm not in arms:
                    continue
                try:
                    model, result, path, refusal = fit_candidate(
                        fit_df, buckets, cfg, tmp, f"{arm}_{fold.fit_cutoff:%Y%m%d}"
                    )
                except ValueError as e:
                    disqualify(arm, fold.label, f"the NHL trainer would refuse this frame: {e}")
                    continue
                record(fold.label, arm, fit_df, fold.fit_cutoff, result, path, refusal)
                if refusal is not None:
                    disqualify(arm, fold.label, f"the NHL trainer would refuse this fit: {refusal}")
                    continue
                probs, tot, fin, bad = model_arm(model, ev)
                if bad:
                    disqualify(arm, fold.label, bad)
                    continue
                arm_probs[arm], arm_totals[arm], arm_grids[arm] = probs, tot, fin
            w = ft.decay_weights(fit_df, fold.fit_cutoff, decay)
            for arm, weights, by_season in (
                ("base_global", None, False),
                ("base_season", None, True),
                ("base_season_dw", w, True),
            ):
                markets, mats = nhl.constant_markets(fit_df, ev, weights, by_season)
                bad = nhl.invalid_matrices(mats)
                if bad:
                    raise RuntimeError(f"{arm} fold {fold.label}: {bad}")
                arm_probs[arm], arm_totals[arm], arm_grids[arm] = (
                    nhl.readout(markets, ev, mats),
                    nhl.expected_totals(mats),
                    mats,
                )
            for arm in list(arm_probs):
                bad = nhl.invalid_probabilities(arm_probs[arm])
                if bad is None:
                    continue
                if arm in challengers_all:
                    disqualify(arm, fold.label, bad)
                    del arm_probs[arm]
                else:
                    raise RuntimeError(f"{arm} fold {fold.label}: {bad}")
            if set(arm_probs) != set(arms):
                raise RuntimeError(f"fold {fold.label}: scored arms {sorted(arm_probs)} != expected {sorted(arms)}")
            for arm in arms:
                for unit, s in nhl.score_arm(arm_probs[arm], truth).items():
                    scores[arm][unit].append(s)
                log_scores[arm].append(nhl.grid_log_score(arm_probs[arm], truth))
                p_reg[arm].append(arm_probs[arm]["regulation_1x2"])
                for k in p_over[arm]:
                    p_over[arm][k].append(arm_probs[arm][f"over_{k}"][:, 0])
                e_total[arm].append(arm_totals[arm])
                grids[arm].append(np.array([np.asarray(m, dtype=float).ravel() for m in arm_grids[arm]]))
            y_reg.append(truth["regulation_1x2"])
            ev_frames.append(ev)
            fold_of.append(np.full(len(ev), fold.label, dtype=object))
            logger.info("fold %s: fit=%d eval=%d arms=%s", fold.label, len(fit_df), len(ev), ",".join(arms))

    # A disqualified arm's earlier folds are dropped with it.
    live = [a for a in arms_all if a in arms]
    folds_arr = np.concatenate(fold_of)
    ok, why = decidable(int(len(folds_arr)), mode)
    if getattr(args, "mechanics", False) or not ok:
        return counts_only_report(
            mode,
            start,
            end,
            int(len(folds_arr)),
            [f.label for f in folds],
            pooled,
            skipped,
            fits,
            disqualified,
            "mechanics run: no delta read" if getattr(args, "mechanics", False) else why,
        )
    cat = {a: {u: np.concatenate(scores[a][u]) for u in UNIT_NAMES} for a in live}
    cat_log = {a: np.concatenate(log_scores[a]) for a in live}
    ev_all = pd.concat(ev_frames, ignore_index=True)
    yr = np.vstack(y_reg)
    realised_total = (ev_all["home_score"] + ev_all["away_score"]).to_numpy(dtype=float)
    seasons = ev_all["season"].to_numpy(dtype=int)
    fold_labels = list(dict.fromkeys(folds_arr.tolist()))
    challengers = [a for a in live if a in challengers_all]
    every = np.ones(len(folds_arr), dtype=bool)

    def pair(a: str, b: str, unit: str = PRIMARY, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        sel = every if mask is None else np.asarray(mask, dtype=bool)
        out = ft.paired_delta(cat[a][unit][sel], cat[b][unit][sel], folds_arr[sel])
        out["upper_95_cluster_t"] = ht.t_upper(out["delta_cluster"], out["se_cluster"], out["n_folds"])
        out["lower_95_cluster_t"] = nhl.t_lower(out["delta_cluster"], out["se_cluster"], out["n_folds"])
        return out

    def verdict(a: str, b: str, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        return strict_verdict(pair(a, b, mask=mask), b)

    def calibration(a: str, mask: np.ndarray) -> Dict[str, Any]:
        pr = np.vstack(p_reg[a])
        po = np.concatenate(p_over[a]["5.5"])
        ok = mask & np.isfinite(yr).all(axis=1)
        return {
            "n": int(mask.sum()),
            "regulation_pred": {c: float(pr[ok, k].mean()) for k, c in enumerate(nhl.REGULATION_CLASSES)},
            "regulation_real": {c: float(yr[ok, k].mean()) for k, c in enumerate(nhl.REGULATION_CLASSES)},
            "over_5.5_pred": float(po[mask].mean()),
            "over_5.5_real": float(np.mean(realised_total[mask] > 5.5)),
            "expected_total": float(np.concatenate(e_total[a])[mask].mean()),
            "realised_total": float(realised_total[mask].mean()),
        }

    def freshness(mask: np.ndarray) -> Dict[str, Any]:
        return {
            "n": int(mask.sum()),
            "vs_served": {
                a: {u: per_match_pair(cat[a][u][mask], cat["served"][u][mask]) for u in UNIT_NAMES}
                for a in live
                if a != "served"
            },
            "brier": {a: {u: _mean(cat[a][u][mask]) for u in UNIT_NAMES} for a in live},
        }

    vs_served = {a: {u: pair(a, "served", u) for u in UNIT_NAMES} for a in live if a != "served"}
    regressions = {a: blocking_regressions(vs_served[a]) for a in challengers}
    verdicts_vs_served = {a: verdict(a, "served") for a in challengers}
    fresh_season = seasons == FRESH_SEASON
    newer = (ev_all["match_date"] > pd.Timestamp(PREVIOUS_GATE_NEWEST)).to_numpy()
    discovery = np.isin(folds_arr, DISCOVERY_FOLDS)
    report: Dict[str, Any] = {
        "n_eval": int(len(folds_arr)),
        "mode": mode.name,
        "window": {"start": start, "end": end, "frozen_fit_cutoff": frozen_cutoff.isoformat()},
        "folds": fold_labels,
        "pooled_folds": pooled,
        "skipped_folds": skipped,
        "duplicated_match_ids": dups,
        "frame": frame_audit,
        "served_artifact": served_fp,
        "disqualified": disqualified,
        "fits": fits,
        "n_gradable": {u: int(np.isfinite(cat["served"][u]).sum()) for u in UNIT_NAMES},
        "brier": {a: {u: _mean(cat[a][u]) for u in UNIT_NAMES} for a in live},
        "grid_log_score": {
            a: {
                "mean": float(cat_log[a].mean()) if np.isfinite(cat_log[a]).all() else None,
                "zero_probability_rows": int(np.isinf(cat_log[a]).sum()),
            }
            for a in live
        },
        "vs_served": vs_served,
        "regressions": regressions,
        "verdicts_vs_served": verdicts_vs_served,
        "verdicts_vs_refit_frozen": {
            a: verdict(a, REFERENCE_FROZEN) for a in challengers if a != REFERENCE_FROZEN and REFERENCE_FROZEN in live
        },
        "verdicts_vs_base_season": {a: verdict(a, "base_season") for a in ["served", *challengers]},
        "head_to_head": {
            f"{a} - {b}": pair(a, b)
            for a, b in (
                ("reg_margin_frozen", "reg_diag_frozen"),
                ("reg_diag_frozen", "reg_poisson_frozen"),
                ("reg_poisson_frozen", "refit_frozen"),
                ("reg_margin", "reg_margin_frozen"),
                ("reg_diag", "reg_diag_frozen"),
            )
            if a in live and b in live
        },
        "per_fold_vs_served": {
            a: [
                {
                    "fold": lab,
                    "n": int((folds_arr == lab).sum()),
                    "delta": _mean(cat[a][PRIMARY][folds_arr == lab] - cat["served"][PRIMARY][folds_arr == lab]),
                }
                for lab in fold_labels
            ]
            for a in live
            if a != "served"
        },
        "primary_discovery_split": {
            name: {a: pair(a, "served", mask=m) for a in challengers} | {"n": int(m.sum())}
            for name, m in (("folds 2025-10..2026-02", discovery), ("folds 2026-03..", ~discovery))
            if m.sum() > 1
        },
        "by_season": {
            nhl.season_label(int(s)): {
                "n": int((seasons == s).sum()),
                **{a: pair(a, "served", mask=seasons == s) for a in challengers},
            }
            for s in sorted(set(seasons.tolist()))
            if (seasons == s).sum() > 1
        },
        "calibration": {a: calibration(a, every) for a in live},
        "freshness": {
            "season_2026_27": freshness(fresh_season),
            "after_previous_gate": freshness(newer),
        },
        "decision": decide(verdicts_vs_served, regressions, disqualified, arm=mode.deciding_arm),
    }
    if args.out_rows:
        write_rows(args.out_rows, ev_all, folds_arr, live, cat, p_reg, p_over)
    if args.out_grids:
        np.savez_compressed(
            args.out_grids,
            match_id=ev_all["match_id"].to_numpy(dtype=str),
            **{a: np.vstack(grids[a]) for a in live},
        )
    return report


def counts_only_report(
    mode: GateMode,
    start: str,
    end: str,
    n_scored: int,
    folds: List[str],
    pooled: List[Dict[str, Any]],
    skipped: List[Dict[str, Any]],
    fits: List[Dict[str, Any]],
    disqualified: Dict[str, Any],
    why: str,
) -> Dict[str, Any]:
    """The report of a run that may not read a delta (a mechanics smoke, or
    the re-gate below its n): counts and fit bookkeeping only."""
    return {
        "counts_only": True,
        "mode": mode.name,
        "window": {"start": start, "end": end, "first_kickoff": mode.first_kickoff},
        "n_eval": n_scored,
        "min_games": mode.min_games,
        "folds": folds,
        "pooled_folds": pooled,
        "skipped_folds": skipped,
        "fits": [{k: f.get(k) for k in ("fold", "arm", "n_fit", "teams", "refusal")} for f in fits],
        "disqualified": disqualified,
        "decision": {"arm": mode.deciding_arm, "ship": False, "decidable": False, "reason": why},
    }


def _mean(x: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else None


def write_rows(
    path: str,
    ev: pd.DataFrame,
    folds: np.ndarray,
    arms: Sequence[str],
    cat: Dict[str, Dict[str, np.ndarray]],
    p_reg: Dict[str, List[np.ndarray]],
    p_over: Dict[str, Dict[str, List[np.ndarray]]],
) -> None:
    """One row per game: identity, scores, regulation label, then per arm the
    regulation probabilities, the over-line probabilities and every unit's
    per-game Brier — enough to recompute every delta and SE in the report."""
    out = pd.DataFrame(
        {
            "match_id": ev["match_id"].to_numpy(),
            "match_date": ev["match_date"].astype(str).to_numpy(),
            "fold": folds,
            "season": ev["season"].to_numpy(),
            "home_team": ev["home_team"].to_numpy(),
            "away_team": ev["away_team"].to_numpy(),
            "home_score": ev["home_score"].to_numpy(),
            "away_score": ev["away_score"].to_numpy(),
            "regulation_winner": ev["regulation_winner"].to_numpy(),
        }
    )
    cols: Dict[str, np.ndarray] = {}
    for a in arms:
        pr = np.vstack(p_reg[a])
        for k, c in enumerate(nhl.REGULATION_CLASSES):
            cols[f"{a}__p_reg_{c}"] = pr[:, k]
        for line, chunks in p_over[a].items():
            cols[f"{a}__p_over_{line}"] = np.concatenate(chunks)
        for u in UNIT_NAMES:
            cols[f"{a}__brier__{u}"] = cat[a][u]
    pd.concat([out, pd.DataFrame(cols)], axis=1).to_csv(path, index=False)


def _f(v: Any, spec: str = "+.5f") -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, spec)


def format_report(r: Dict[str, Any]) -> str:
    if r.get("counts_only"):
        d = r["decision"]
        refused = [f["arm"] for f in r["fits"] if f["refusal"]]
        return "\n".join(
            [
                f"NHL regulation-tie A/B [{r['mode']}] — COUNTS ONLY: {r['n_eval']} scorable games (minimum "
                f"{r['min_games']}), folds {r['folds'] or 'none'}, window {r['window']}",
                f"fits: {len(r['fits'])}; refused: {refused or 'none'}; "
                f"disqualified: {sorted(r['disqualified']) or 'none'}",
                f"NO DECISION ({d['arm']}): {d['reason']}",
            ]
        )
    w = r["window"]
    L = [
        f"NHL regulation-tie A/B — {r['n_eval']} games, folds {r['folds'][0]}..{r['folds'][-1]} ({len(r['folds'])});"
        f" pooled {[p['fold'] + '->' + p['into'] for p in r['pooled_folds']] or 'none'};"
        f" skipped {[s['fold'] for s in r['skipped_folds']] or 'none'}; frozen arms fitted on match_date <"
        f" {w['frozen_fit_cutoff']}",
        f"frame: {r['frame']['rows']} rows, {r['frame']['without_regulation_winner']} without regulation_winner;"
        f" served sha256 {r['served_artifact']['sha256'][:12]}",
    ]
    for arm, d in r["disqualified"].items():
        L.append(f"DISQUALIFIED {arm} at fold {d['fold']}: {d['reason']}")
    L += ["", "fits (frozen arms once, walk-forward per fold):"]
    for f in r["fits"]:
        if f["fold"].startswith("frozen") or f["arm"] == "reg_margin":
            theta = f.get("margin_log_factors")
            L.append(
                f"  {f['fold']:<18} {f['arm']:<19} n={f['n_fit']} teams={f['teams']}"
                + (
                    f" theta={[round(t, 3) for t in theta]} q={_f(f['ot_home_share'], '.3f')}"
                    f" tie {_f(f['fitted_tie_rate'], '.3f')}/{_f(f['observed_tie_rate'], '.3f')}"
                    f" it={f['structure_iterations']}"
                    if theta is not None
                    else ""
                )
                + f" level {_f(f['fitted_total'], '.3f')}/{_f(f['observed_total'], '.3f')}"
                + (f" REFUSED: {f['refusal']}" if f["refusal"] else "")
            )
    arms = list(r["brier"])
    L += ["", "Brier by arm and unit (lower is better; P = primary, C = context):"]
    L.append(f"  {'unit':<24}" + "".join(f"{a[:18]:>19}" for a in arms))
    for u in UNIT_NAMES:
        tag = " P" if u == PRIMARY else (" C" if u in CONTEXT_UNITS else "")
        L.append(f"  {u + tag:<24}" + "".join(f"{_f(r['brier'][a][u], '.5f'):>19}" for a in arms))
    g = r["grid_log_score"]
    L.append(
        f"  {'grid log score C':<24}"
        + "".join(f"{(_f(g[a]['mean'], '.5f') if g[a]['mean'] is not None else 'inf'):>19}" for a in arms)
    )
    L += ["", f"{PRIMARY} vs served:"]
    L += [ht._pair_line(a, d[PRIMARY]) for a, d in r["vs_served"].items()]
    L += [
        "",
        "every unit vs served (pooled dBrier [fold-clustered t 95% lower bound]; R = significantly worse on a"
        " blocking unit):",
    ]
    others = list(r["vs_served"])
    L.append(f"  {'unit':<22}" + "".join(f"{a[:22]:>26}" for a in others))
    for u in UNIT_NAMES:
        cells = []
        for a in others:
            p = r["vs_served"][a][u]
            mark = " R" if u in r["regressions"].get(a, {}) else (" P" if u == PRIMARY else "  ")
            cells.append(f"{_f(p.get('delta')) + ' [' + _f(p.get('lower_95_cluster_t')) + ']' + mark:>26}")
        L.append(f"  {u:<22}" + "".join(cells))
    L += ["", f"{PRIMARY} — context comparisons:"]
    for title, key in (("vs refit_frozen", "verdicts_vs_refit_frozen"), ("vs base_season", "verdicts_vs_base_season")):
        for a, v in r[key].items():
            L.append(
                f"  {a + ' ' + title:<40} {'PASS' if v['ship_strict'] else 'FAIL'}  {v['line'].split(': ', 1)[-1]}"
                f"  (t-bound {_f(v.get('upper_95_cluster_t'))})"
            )
    L += ["", "head to head (primary):"]
    L += [ht._pair_line(k, v) for k, v in r["head_to_head"].items()]
    L += ["", f"{PRIMARY} vs served, discovery folds vs the rest (context):"]
    for name, block in r["primary_discovery_split"].items():
        L.append(f"  [{name}] n={block['n']}")
        L += [ht._pair_line(f"{a}", v) for a, v in block.items() if isinstance(v, dict)]
    L += ["", f"{PRIMARY} by season (context):"]
    for s, block in r["by_season"].items():
        L.append(f"  [{s}] n={block['n']}")
        L += [ht._pair_line(f"{a}", v) for a, v in block.items() if isinstance(v, dict)]
    first = next(iter(r["per_fold_vs_served"].values()))
    L += ["", f"{PRIMARY} per-fold delta vs served:"]
    L.append(f"  {'arm':<19}" + "".join(f"{row['fold'][2:]:>9}" for row in first))
    for a, rows in r["per_fold_vs_served"].items():
        L.append(f"  {a[:19]:<19}" + "".join(f"{_f(row['delta'], '+.4f'):>9}" for row in rows))
    L.append(f"  {'n':<19}" + "".join(f"{row['n']:>9}" for row in first))
    L += ["", "calibration (regulation H/D/A predicted vs realised; over 5.5; E[final total]):"]
    for a, c in r["calibration"].items():
        rp, rr = c["regulation_pred"], c["regulation_real"]
        L.append(
            f"  {a:<19} reg {rp['home']:.3f}/{rp['draw']:.3f}/{rp['away']:.3f} real {rr['home']:.3f}/{rr['draw']:.3f}"
            f"/{rr['away']:.3f}  over5.5 {c['over_5.5_pred']:.3f}/{c['over_5.5_real']:.3f}"
            f"  E[total] {c['expected_total']:.3f}/{c['realised_total']:.3f}"
        )
    for name, title in (
        ("season_2026_27", "the 2026-27 season"),
        ("after_previous_gate", f"games after {PREVIOUS_GATE_NEWEST} (no earlier analysis)"),
    ):
        blk = r["freshness"][name]
        L += [
            "",
            f"FRESHNESS (confirmation, never decides) — {title}: n={blk['n']}; per-match dBrier vs served +/- SE:",
        ]
        if not blk["n"]:
            continue
        others = list(blk["vs_served"])
        L.append(f"  {'unit':<22}" + "".join(f"{a[:22]:>26}" for a in others))
        for u in UNIT_NAMES:
            cells = []
            for a in others:
                p = blk["vs_served"][a][u]
                cells.append(f"{_f(p['delta']) + ' +/- ' + _f(p['se'], '.5f'):>26}")
            L.append(f"  {u:<22}" + "".join(cells))
    d = r["decision"]
    L += [
        "",
        f"DECISION (pre-registered: {d['arm']} must pass ship_strict vs served on {PRIMARY} AND be significantly"
        f" worse on no blocking unit): {'SHIP' if d['ship'] else 'DO-NOT-SHIP'} — {d['reason']}",
    ]
    for u, why in d["regressions"].items():
        L.append(f"  regression {u}: {why}")
    for a, v in r["verdicts_vs_served"].items():
        regs = r["regressions"].get(a, {})
        L.append(
            f"  {a:<19} {'PASS' if v['ship_strict'] else 'FAIL'}  {v['line'].split(': ', 1)[-1]}"
            f"  (t-bound {_f(v.get('upper_95_cluster_t'))}); regressions: {', '.join(regs) or 'none'}"
        )
    return "\n".join(L)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--start",
        default=None,
        help=f"first eval day (default {DEFAULT_START}); the frozen arms are fitted before its month. "
        "Refused with --regate (its window is pinned)",
    )
    p.add_argument(
        "--regate",
        action="store_true",
        help="the pre-registered re-gate: reg_diag decides, regular-season games kicking off from "
        f"{REGATE_FIRST_KICKOFF} only, no decision and no delta below {REGATE_MIN_GAMES} games",
    )
    p.add_argument("--limit", type=int, default=0, help="cap scored games per fold (mechanics smoke); 0 = all")
    p.add_argument("--mechanics", action="store_true", help="counts and fit bookkeeping only: never read a delta")
    p.add_argument("--end", default=None, help="last eval day (default: newest NHL match)")
    p.add_argument("--served-artifact", default=SERVED_ARTIFACT)
    p.add_argument("--served-sha256", default=SERVED_SHA256, help="refuse any other incumbent")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-rows", default=None, help="per-game CSV (probabilities and per-unit Brier per arm)")
    p.add_argument("--out-grids", default=None, help="per-game final grids per arm (.npz)")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    report = run(args)
    if args.out_json:  # before printing: a formatting bug must not lose the run
        Path(args.out_json).write_text(json.dumps(ht.clean_json(report), indent=2, allow_nan=False))
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
