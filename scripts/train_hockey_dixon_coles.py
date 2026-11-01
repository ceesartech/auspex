"""Train a Dixon-Coles model for NHL — into STAGING only.

NOT PROMOTABLE. The served NHL artifact
(/app/models/production/dixon_coles_nhl/1.0.0/model.bin) has never been gated,
and neither has anything this script writes: its gate is
scripts/ab_nhl_dixon_coles.py (pre-registered there, amended 2026-10-31 after a
smoke run, not yet run over its full window). scripts/promote_dc_model.py has no
"nhl" entry, and nothing staged here may be copied into production by hand.
Until the gate passes, the config below stays the pinned legacy one and
production keeps serving its current file.

Hockey scorelines are low-discrete just like soccer (modal ~3-2, median total
~5.5), so the same Dixon-Coles + Poisson machinery works directly. This trainer
pulls the NHL goals (matches.home_score / away_score, FINAL scores including
OT/SO) instead of soccer FT/HT scores.

What it feeds once promoted: precompute_predictions_nhl.py loads the production
file (_load_hockey_dixon_coles), takes lambdas_for_match(home, away) — NO
league argument — builds build_dc_matrix(..., rho, MAX_GOALS_DERIVE) and serves
8 markets from derive_hockey_markets (regulation_1x2, puck_line, over_under,
total_goals, double_chance, clean_sheet, win_to_nil, correct_score; its
moneyline is dropped).

Architecture (the half-time trainer's guards, reused — not copied):
  * Fit on ALL rows of NHL_DC_TRAINING_QUERY (preseason excluded) with
    nhl_config() — the pinned legacy config, which converges on the NHL frame
    (7 iterations). --holdout-fraction (default 0) only holds out the newest
    slice as a diagnostic, and the diagnostic is a hockey one
    (holdout_diagnostic: over-5.5 Brier and level read from the matrices
    production would serve). DixonColesPredictor.train's own validation
    metrics are never computed here: they score a soccer 3-way 1x2 (class 1 =
    draw) against this query's binary match_outcome (1 = AWAY win), before rho
    is forced — numbers that mean nothing for NHL.
  * force_rho_zero: final scores are never 0-0 or 1-1, so the rho NLL runs away
    (~1e27 observed) instead of fitting; rho is forced to 0 (pure Poisson) after
    the fit, as before. The optimiser's value is logged.
  * train_halftime_dixon_coles.fit_refusal, AFTER forcing rho, refuses an
    unconverged fit, a fitted vs observed mean total goals gap past +/-2%, and
    any training fixture whose served matrix (lambdas_for_match WITHOUT a
    league, as precompute_predictions_nhl calls it) is not finite and
    normalised. Its rho bound is trivially met at 0 — the runaway rho is
    handled by the forcing, not refused. Nothing is written on a refusal.
  * write_artifact writes model.bin + manifest.json atomically (run id, git sha,
    rows, newest match, convergence, level, rho, sha256 of model.bin).
  * output_dir_refusal refuses /app/models/staging (the weekly retrain_models
    swap_production promotes everything in it, with no checks) and
    /app/models/production, symlinks and `..` resolved. The default is
    /app/models/staging-nhl/dixon_coles_nhl/1.0.0, which nothing promotes.

Until 2026-10-31 this script fitted the OLDEST 80% only (so no 2025-26 match
reached the fit: 'Utah Mammoth' is unknown to the served file), never checked
convergence or level, and wrote straight over the production file. A
2026-10-31 dry run of this version (to a /tmp dir) on all 8,161 rows: converged
after 7 iterations, fitted vs observed mean total goals 6.189 / 6.189, the
optimiser's rho 8.2e27 forced to 0, 34 teams.

Run on prod via (stages only; read the log):
    docker compose exec api python /app/scripts/train_hockey_dixon_coles.py
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

# The preseason-exclusion predicate is defined ONCE, in
# services/ml-models/src/utils/training_data.py. Same sys.path dance as
# scripts/compute_features_nfl.py: the api container already has that tree
# on PYTHONPATH, and the repo-relative path keeps local dev + the unit
# tests working. Position 0 also guarantees `utils` resolves to
# ml-models' package rather than the same-named one under
# services/data-ingestion/src.
_ML_MODELS_SRC = str(Path(__file__).resolve().parent.parent / "services" / "ml-models" / "src")
for _p in ("/app/services/ml-models/src", _ML_MODELS_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# The half-time trainer's guards, manifest and atomic write live next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_halftime_dixon_coles as halftime  # noqa: E402
from utils.training_data import preseason_exclusion_sql  # noqa: E402

LOGGER = logging.getLogger("train_hockey_dixon_coles")

MODEL_NAME = "dixon_coles_nhl"
DEFAULT_OUTPUT_DIR = Path("/app/models/staging-nhl/dixon_coles_nhl/1.0.0")


# Preseason is excluded here for the same reason it is excluded from every
# other training frame: preseason lineups and scoring rates are structurally
# different, and this artifact feeds 8 live NHL markets via
# derive_hockey_markets. Unlike the NBA/NFL spread+total frames there is no
# incidental odds join to keep it out — nothing but this predicate. NHL rows
# loaded by load_nhl_historical.py carry the legacy metadata.game_type marker
# instead of season_type, which preseason_exclusion_sql also honours.
NHL_DC_TRAINING_QUERY = f"""
    SELECT
        m.id::text AS match_id,
        m.match_date,
        m.league_id::text AS league_id,
        m.home_team_id::text AS home_team_id,
        m.away_team_id::text AS away_team_id,
        ht.name AS home_team,
        at.name AS away_team,
        -- Use FINAL home/away scores (including OT/SO) so the
        -- Dixon-Coles model captures the actual scoring rate per
        -- team. The NHL_REGULATION market gets its own model
        -- elsewhere; this one is for total goals + final-result
        -- markets that include OT.
        m.home_score AS home_score,
        m.away_score AS away_score,
        CASE
            WHEN m.home_score > m.away_score THEN 0
            ELSE 1
        END AS match_outcome,
        NULL::numeric AS odds_home,
        NULL::numeric AS odds_draw,
        NULL::numeric AS odds_away,
        NULL::numeric AS odds_over25,
        NULL::numeric AS odds_under25,
        NULL::jsonb AS features
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    WHERE l.sport = 'nhl'
      AND m.status = 'finished'
      AND m.home_score IS NOT NULL
      AND m.away_score IS NOT NULL
      AND {preseason_exclusion_sql('m')}
    ORDER BY m.match_date ASC
"""


def nhl_config() -> Any:
    """The config this trainer fits — scripts/ab_nhl_dixon_coles.py's pinned arm
    imports it, so the gate measures exactly this.

    NOT DIXON_COLES_CONFIG. That object carries the 2026-09 SOCCER refit knobs
    (max_goals 10, a live time_decay, regularization as a 20-effective-match
    prior, per-league baselines), and not one hockey number was measured for
    any of them; ab_nhl_dixon_coles.py's refit arms measure them now. Pinned to
    the pre-refit values until that gate passes. The NHL defects (as corrected
    2026-10-31 in model_config.py, next to
    dixon_coles_config_pinned_to_legacy_fit), each needing its own measurement:
      * level inflation: the served artifact predates the scale-gauge
        correction in PoissonMatchPredictor.train, so its lambdas carry that
        positive level bias (a fit here does not; fit_refusal checks it);
      * a stale fit: the served file has no 2025-26 data, so 'Utah Mammoth' is
        an unknown team there and serves attack = defense = 1.0 (a fit here
        uses every row);
      * rho forced to 0 (force_rho_zero);
      * a regulation-tie deficit: fitted on FINAL scores (OT/SO included), its
        draw cells — what regulation_1x2 and double_chance price — held 15.5%
        on average vs a 23.4% observed regulation-tie rate over the 286 graded
        matches to 2026-10-31. Nothing here changes that.
    (max_goals=6 is not one: precompute_predictions_nhl builds every served
    matrix at market_derivation.MAX_GOALS_DERIVE=10, and home_advantage=0.25
    roughly matches NHL's observed home edge.)
    """
    from predictors.model_config import dixon_coles_config_pinned_to_legacy_fit  # type: ignore

    return dixon_coles_config_pinned_to_legacy_fit(MODEL_NAME)


def force_rho_zero(model: Any, result: Dict[str, Any]) -> Optional[float]:
    """Force rho to 0 on `model` and in `result`; return the optimiser's value.

    NHL goal scoring has no soccer-style low-score dependence (the
    0-0/1-1/0-1/1-0 cluster the DC tau corrects), and final scores are never
    tied, so the rho NLL runs off (~1e27 observed during initial training).
    rho = 0 is pure Poisson — the right baseline for hockey. The team strengths
    are fitted before rho and do not depend on it. result['rho_fitted'] keeps
    the optimiser's value (None when non-finite) for the log.
    """
    fitted = getattr(model, "rho", None)
    model.rho = 0.0
    result["rho_fitted"] = float(fitted) if halftime._finite(fitted) else None
    result["rho"] = 0.0
    return result["rho_fitted"]


def served_fixtures(frame: Any) -> Iterable[Tuple[str, str, None]]:
    """(home, away, None) per distinct fixture: precompute_predictions_nhl calls
    lambdas_for_match(home, away) with no league, so fit_refusal checks the
    matrices production would actually build."""
    pairs = frame[["home_team", "away_team"]].drop_duplicates()
    return ((home, away, None) for home, away in pairs.itertuples(index=False))


# ── The regulation-scoreline model (predictors.hockey_regulation) ────
# Gated by scripts/ab_nhl_regulation_ties.py, which fits its candidate arms
# through fit_regulation_model below so the gate measures exactly this code.

# The frame above plus metadata.regulation_winner ('home' / 'away', or 'tie' —
# 'draw' on ESPN rows written before 2026-10 — when the game went past
# regulation), which the regulation scoreline is derived from.
NHL_DC_REGULATION_QUERY = NHL_DC_TRAINING_QUERY.replace(
    "NULL::jsonb AS features",
    "NULL::jsonb AS features,\n        m.metadata->>'regulation_winner' AS regulation_winner",
)
REGULATION_LABELS = {"home": False, "away": False, "tie": True, "draw": True}  # label -> went past regulation
# The margin buckets of the deciding arm, pre-registered in
# ab_nhl_regulation_ties.py (the BIC rule on the 2021-01..2025-09 rows).
REGULATION_MARGIN_BUCKETS: Tuple[int, ...] = (0, 1, 2, 3, 4)
# fit_refusal's level tolerance, applied to the regulation fit's served level.
MAX_REGULATION_LEVEL_BIAS = 0.02


def regulation_frame(frame: Any) -> Any:
    """``frame`` (FIT columns + regulation_winner, FINAL scores) with
    home_score / away_score replaced by the REGULATION scoreline and an
    ``overtime_home_win`` column (1.0 / 0.0 for a game that went past
    regulation, NaN otherwise). Raises on a row without a known
    regulation_winner or whose final score contradicts it
    (hockey_regulation.regulation_scores) — guessing either would fit on it
    silently."""
    import numpy as np
    from predictors.hockey_regulation import regulation_scores  # type: ignore

    labels = frame["regulation_winner"].map(lambda v: REGULATION_LABELS.get(v) if isinstance(v, str) else None)
    unknown = labels.isna()
    if unknown.any():
        raise ValueError(
            f"{int(unknown.sum())} NHL row(s) without a known regulation_winner "
            f"(e.g. {frame.loc[unknown, 'regulation_winner'].head(3).tolist()}) — no regulation score to fit"
        )
    past = labels.astype(bool).to_numpy()
    final_h = frame["home_score"].to_numpy(dtype=int)
    final_a = frame["away_score"].to_numpy(dtype=int)
    reg_h, reg_a = regulation_scores(final_h, final_a, past)
    out = frame.copy()
    out["home_score"], out["away_score"] = reg_h, reg_a
    out["overtime_home_win"] = np.where(past, (final_h > final_a).astype(float), np.nan)
    return out


def regulation_fit_refusal(
    result: Dict[str, Any], model: Any, fixtures: Iterable[Tuple[str, str, Any]]
) -> Optional[str]:
    """Why a regulation-model fit must NOT be written, or None.

    The Poisson start converged (result['poisson_iterations']) and the joint
    structure fit converged (result['iterations']); the served level — mean
    REGULATION total goals under the fitted regulation matrices vs observed
    — is within +/-2%; the structure is valid; rho is 0; and every fixture's
    served regulation AND final matrices (precompute_predictions_nhl's
    lambdas_for_match without a league) are finite and normalised, the final
    one with no tie mass."""
    import numpy as np
    from predictors.hockey_regulation import structure_of  # type: ignore
    from predictors.market_derivation import MAX_GOALS_DERIVE, build_dc_matrix  # type: ignore

    if result.get("poisson_iterations") is None:
        return "the Poisson start did not converge"
    if result.get("iterations") is None:
        return "the regulation-structure fit did not converge"
    fitted, observed = result.get("fitted_mean_total_goals"), result.get("observed_mean_total_goals")
    if not (halftime._finite(fitted) and halftime._finite(observed) and observed > 0):
        return f"no usable level check (fitted {fitted!r}, observed {observed!r})"
    bias = fitted / observed - 1.0
    if abs(bias) > MAX_REGULATION_LEVEL_BIAS:
        return (
            f"fitted mean regulation total goals {fitted:.4f} vs observed {observed:.4f} ({100 * bias:+.2f}%) is "
            f"outside +/-{100 * MAX_REGULATION_LEVEL_BIAS:.0f}%"
        )
    if float(getattr(model, "rho", 0.0) or 0.0) != 0.0:
        return f"rho is {model.rho!r}, not 0"
    try:
        structure = structure_of(model)
    except ValueError as e:
        return f"invalid regulation structure: {e}"
    if structure is None:
        return "the model carries no regulation structure"
    for home, away, _ in fixtures:
        P = build_dc_matrix(*model.lambdas_for_match(home, away), 0.0, max_goals=MAX_GOALS_DERIVE)
        R, F = structure.matrices(P)
        for name, M in (("regulation", R), ("final", F)):
            if not (np.isfinite(M).all() and (M >= 0).all() and abs(float(M.sum()) - 1.0) < 1e-6):
                return f"invalid served {name} matrix for {home} v {away}"
        if float(np.trace(F)) != 0.0:
            return f"the served final matrix for {home} v {away} holds tie mass"
    return None


def fit_regulation_model(
    frame: Any, cfg: Any, margin_buckets: Tuple[int, ...] = REGULATION_MARGIN_BUCKETS
) -> Tuple[Any, Dict[str, Any], Optional[str]]:
    """Fit the regulation-scoreline model on ``frame`` (FIT columns +
    regulation_winner, FINAL scores): DixonColesPredictor(cfg) on the
    regulation scoreline, rho forced to 0, then the team strengths and the
    margin log-factors jointly by maximum likelihood with the overtime home
    share (hockey_regulation.fit_structure); the structure rides in the
    artifact as model.regulation_structure. Returns (model, result,
    refusal) — refusal from regulation_fit_refusal, None when the fit may be
    written. ``margin_buckets`` () is the plain Poisson on regulation scores
    (the overtime resolution still applies)."""
    from predictors.hockey_regulation import fit_structure  # type: ignore
    from predictors.market_derivation import MAX_GOALS_DERIVE  # type: ignore
    from predictors.poisson_models import DixonColesPredictor  # type: ignore

    reg = regulation_frame(frame)
    fit_cols = ["home_team", "away_team", "home_score", "away_score", "match_date", "league_id"]
    reg_df = reg[fit_cols].copy()
    model = DixonColesPredictor(cfg)
    result = model.train(reg_df)
    force_rho_zero(model, result)
    poisson = {k: result.get(k) for k in ("iterations", "fitted_mean_total_goals", "observed_mean_total_goals")}
    structure, sresult = fit_structure(
        model, reg_df, reg["overtime_home_win"].to_numpy(), margin_buckets, int(MAX_GOALS_DERIVE)
    )
    model.regulation_structure = structure.to_payload()
    result.update(sresult)
    result["poisson_iterations"] = poisson["iterations"]
    result["poisson_fitted_mean_total_goals"] = poisson["fitted_mean_total_goals"]
    refusal = regulation_fit_refusal(result, model, list(served_fixtures(reg_df)))
    return model, result, refusal


def holdout_diagnostic(model: Any, train_df: Any, holdout_df: Any) -> Dict[str, Any]:
    """The --holdout-fraction diagnostic, in hockey terms: each held-out match
    read exactly as precompute_predictions_nhl serves it —
    lambdas_for_match(home, away) with NO league, build_dc_matrix(..., rho,
    MAX_GOALS_DERIVE), derive_markets("nhl", P)["over_under"]["over_5.5"] —
    and graded on the FINAL score as grading_outcomes grades over_under.

    Returns the over-5.5 Brier, the same Brier for a constant at the train
    slice's over-5.5 rate (the floor a fit has to beat), mean predicted vs
    realised P(over 5.5), mean E[total goals] vs realised, and how many held-out
    matches involve a team the fit never saw (served at attack = defense = 1.0).
    A diagnostic only: one slice, one fit, no SE — scripts/ab_nhl_dixon_coles.py
    is the gate.
    """
    import numpy as np
    from predictors.market_derivation import MAX_GOALS_DERIVE, build_dc_matrix, derive_markets  # type: ignore

    rho = float(getattr(model, "rho", 0.0) or 0.0)
    known = set(getattr(model, "team_attack", {}) or {})
    p_over, e_total = [], []
    for home, away in zip(holdout_df["home_team"], holdout_df["away_team"]):
        h_lam, a_lam = model.lambdas_for_match(home, away)
        P = build_dc_matrix(float(h_lam), float(a_lam), rho, max_goals=MAX_GOALS_DERIVE)
        p_over.append(float(derive_markets("nhl", P)["over_under"]["over_5.5"]))
        i, j = np.indices(P.shape)
        e_total.append(float((P * (i + j)).sum()))
    p = np.asarray(p_over, dtype=float)
    total = (holdout_df["home_score"].astype(int) + holdout_df["away_score"].astype(int)).to_numpy(dtype=float)
    y = (total > 5.5).astype(float)
    train_total = train_df["home_score"].astype(int) + train_df["away_score"].astype(int)
    base_rate = float((train_total > 5.5).mean())
    unknown = sum(
        1
        for home, away in zip(holdout_df["home_team"], holdout_df["away_team"])
        if home not in known or away not in known
    )
    return {
        "n": int(len(y)),
        "over_5.5_brier": float(np.mean((p - y) ** 2)),
        "constant_over_5.5_rate": base_rate,
        "constant_over_5.5_brier": float(np.mean((base_rate - y) ** 2)),
        "mean_p_over_5.5": float(p.mean()),
        "realised_over_5.5": float(y.mean()),
        "mean_expected_total": float(np.mean(e_total)),
        "realised_mean_total": float(total.mean()),
        "matches_with_an_unknown_team": int(unknown),
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write model.bin + manifest.json (default: staging-nhl — nothing promotes it). Refused under "
        "/app/models/staging (the weekly retrain auto-promotes it) or /app/models/production, after resolving "
        "symlinks.",
    )
    parser.add_argument("--run-id", default=None, help="Recorded in manifest.json. Default: a UTC stamp.")
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help="Diagnostic only: hold out the newest fraction, fit the rest, and log the held-out over-5.5 Brier "
        "(vs a constant at the train slice's rate), P(over 5.5) and E[total goals] vs realised, read from the "
        "matrices production would serve. No 1x2 validation metrics: the predictor's would score a soccer 3-way "
        "result against NHL's binary final result. 0 (default) fits everything. The manifest records it, and "
        "promote_dc_model.py only ever promotes a 0.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s - %(message)s",
    )
    if not args.database_url:
        LOGGER.error("DATABASE_URL not set.")
        return 1
    if not 0.0 <= args.holdout_fraction < 1.0:
        LOGGER.error("--holdout-fraction must be in [0, 1)")
        return 1
    refusal = halftime.output_dir_refusal(args.output_dir)
    if refusal:
        LOGGER.error("Refusing --output-dir: %s", refusal)
        return 1
    run_id = args.run_id or halftime.default_run_id()

    from predictors.poisson_models import DixonColesPredictor
    from utils.training_data import load_training_frame

    LOGGER.info("Run %s: loading NHL Dixon-Coles training frame...", run_id)
    frame = load_training_frame(database_url=args.database_url, query=NHL_DC_TRAINING_QUERY)
    if frame.empty:
        LOGGER.error("The NHL training frame is empty — refusing to fit")
        return 1
    LOGGER.info(
        "Loaded %d NHL matches (%s..%s).",
        len(frame),
        frame["match_date"].min(),
        frame["match_date"].max(),
    )

    # Fit everything by default. The query's ORDER BY keeps match-date order,
    # so a holdout (diagnostic only) is the temporally latest slice.
    n = len(frame)
    cutoff = n - int(n * args.holdout_fraction)
    train_df = frame.iloc[:cutoff].copy()
    holdout_df = frame.iloc[cutoff:].copy() if cutoff < n else None
    LOGGER.info("Train=%d Holdout=%d", len(train_df), 0 if holdout_df is None else len(holdout_df))

    model = DixonColesPredictor(nhl_config())
    # No val_df, ever: DixonColesPredictor.train would evaluate its 3-way 1x2
    # (class 1 = draw) against NHL_DC_TRAINING_QUERY's binary match_outcome
    # (1 = AWAY win) with the optimiser's runaway rho — meaningless for NHL.
    # The holdout gets holdout_diagnostic below instead.
    result = model.train(train_df)
    rho_fitted = force_rho_zero(model, result)
    LOGGER.info(
        "Forced rho=0 for hockey (the optimiser produced %s; final scores carry no soccer-style low-score "
        "dependence).",
        "a non-finite value" if rho_fitted is None else f"{rho_fitted:.6g}",
    )
    LOGGER.info("Train result: %s", {k: v for k, v in result.items() if k not in ("team_attack", "team_defense")})

    # Converged, level-true, and every training fixture serves a valid matrix
    # as precompute_predictions_nhl builds it — or nothing is written.
    fixtures = list(served_fixtures(train_df))
    refusal = halftime.fit_refusal(result, model, fixtures)
    if refusal:
        LOGGER.error("Run %s: %s — refusing to write an artifact", run_id, refusal)
        return 1
    LOGGER.info(
        "Fit accepted: converged after %s iterations, fitted/observed mean total goals %.4f/%.4f, rho 0 (forced); "
        "%d distinct fixtures all serve a finite, normalised matrix",
        result["iterations"],
        result["fitted_mean_total_goals"],
        result["observed_mean_total_goals"],
        len(fixtures),
    )
    if holdout_df is not None:
        diag = holdout_diagnostic(model, train_df, holdout_df)
        LOGGER.info(
            "Holdout diagnostic (newest %d matches, %s..%s; read as precompute_predictions_nhl serves, graded on the "
            "final score): over 5.5 Brier %.5f vs %.5f for a constant at the train rate %.3f; mean P(over 5.5) %.3f "
            "vs realised %.3f; mean E[total goals] %.3f vs realised %.3f; %d match(es) with a team the fit never saw. "
            "One slice, no SE: ab_nhl_dixon_coles.py is the gate.",
            diag["n"],
            holdout_df["match_date"].min(),
            holdout_df["match_date"].max(),
            diag["over_5.5_brier"],
            diag["constant_over_5.5_brier"],
            diag["constant_over_5.5_rate"],
            diag["mean_p_over_5.5"],
            diag["realised_over_5.5"],
            diag["mean_expected_total"],
            diag["realised_mean_total"],
            diag["matches_with_an_unknown_team"],
        )

    manifest = halftime.write_artifact(
        model,
        args.output_dir,
        dict(
            result=result,
            train_df=train_df,
            run_id=run_id,
            git_sha=os.environ.get("GIT_SHA") or "unknown",
            holdout_fraction=args.holdout_fraction,
        ),
        model_name=MODEL_NAME,
    )
    LOGGER.info(
        "Staged NHL Dixon-Coles at %s (sha256 %s, run %s, %d teams, newest match %s). NOT promotable: its gate "
        "(scripts/ab_nhl_dixon_coles.py) has not passed; production keeps the served artifact.",
        args.output_dir / halftime.MODEL_FILE,
        manifest["model_sha256"],
        run_id,
        manifest["teams"],
        manifest["newest_match_date"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
