"""Promote a staged Dixon-Coles side artifact into production — today the
soccer half-time model (--model ht); REGISTRY takes the second-half and NHL
artifacts once their trainers write a manifest.

These artifacts are deliberately NOT in the weekly retrain_models swap: that
DAG copies everything in /app/models/staging into production with no checks.
Their trainers write to their own staging dirs instead, and this script is the
only way from there into production:

  1. check the staged artifact (evaluate_promotion — pure):
       * manifest.json present and its model_sha256 is the staged model.bin's;
       * with --expect-run-id, the manifest is that run's (the
         retrain_halftime_dc DAG passes its own run id, so a stale or foreign
         staging file cannot ride along on a successful train task);
       * staged sha256 == production sha256 -> "nothing to promote", exit 0:
         a stale staging file is never re-promoted;
       * the expected config name, a fit on every row, per-league baselines,
         a finite rho;
       * no coverage regression: teams >= 98% of production's and leagues >=
         production's - 1 (skipped against a legacy production artifact, which
         has no per-league baselines), data no older than production's;
     then loads the staged file through the serve path's own loader helper and
     derives the HT 1x2 for one fixture;
  2. backs up production (model.bin + manifest.json if any) to
     /app/models/backup-<name>-<UTC stamp>/, sha256 verified;
  3. promotes atomically: copy to *.new in the production dir, sha256 checked,
     then os.replace — model.bin and manifest.json;
  4. verifies: production sha256 == staged, and the serve loader
     (precompute_predictions._load_halftime_dixon_coles) loads it and a fixture
     yields a valid HT 1x2. Prints an atomic rollback command.

precompute_predictions.py reads the file on every run, so the next 15-min
pipeline tick serves it — no restart.

Run in the api container (scripts/ops/promote_ht_model.sh wraps this on the host):
    docker compose exec -T api python /app/scripts/promote_dc_model.py --model ht --check-only
    docker compose exec -T api python /app/scripts/promote_dc_model.py --model ht
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("promote_dc_model")

MODELS_ROOT = Path("/app/models")
ML_SRC = "/app/services/ml-models/src"
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = "model.bin"
MANIFEST_FILE = "manifest.json"

# Coverage-regression bounds against a production artifact that has
# per-league baselines. A monthly refit only ever ADDS matches, so its team
# count should not shrink; 2% of slack absorbs a merge of duplicate teams.
# One league may drop out (a competition whose last match aged out of a
# source), not more.
MIN_TEAM_RATIO = 0.98
MAX_LEAGUE_DROP = 1


@dataclass(frozen=True)
class DCModelSpec:
    name: str  # artifact name: the directory under staging-*/ and production/
    staging_dir: Path
    production_dir: Path
    expected_config_name: str  # payload["config"]["name"] and manifest["model"]
    serve_loader: str  # precompute_predictions function that loads the production file
    smoke_market_group: Optional[str] = None  # market_derivation.derive_markets sport key
    smoke_market: Optional[str] = None  # its home/draw/away market, checked on one fixture


REGISTRY: Dict[str, DCModelSpec] = {
    # Soccer half-time. Trainer: scripts/train_halftime_dixon_coles.py
    # (writes the manifest); gate: scripts/ab_halftime_dixon_coles.py.
    "ht": DCModelSpec(
        name="dixon_coles_ht_soccer",
        staging_dir=MODELS_ROOT / "staging-ht" / "dixon_coles_ht_soccer" / "1.0.0",
        production_dir=MODELS_ROOT / "production" / "dixon_coles_ht_soccer" / "1.0.0",
        expected_config_name="dixon_coles_ht_soccer",
        serve_loader="_load_halftime_dixon_coles",
        smoke_market_group="soccer_halftime",
        smoke_market="match_result_ht",
    ),
    # "h2" (dixon_coles_h2_soccer, served config name "dixon_coles") and "nhl"
    # (dixon_coles_nhl) belong here once their trainers write a manifest into
    # their own staging-* dir — both are still ungated (audit §4 registry).
}


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class Decision:
    action: str  # "promote" | "noop" | "refuse"
    checks: Tuple[Check, ...]

    @property
    def refusals(self) -> List[Check]:
        return [c for c in self.checks if not c.ok]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _short(sha: Optional[str]) -> str:
    return (sha or "none")[:12]


def _parse_ts(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def evaluate_promotion(
    spec: DCModelSpec,
    *,
    staged_payload: Dict[str, Any],
    staged_sha: str,
    staged_manifest: Optional[Dict[str, Any]],
    production_payload: Optional[Dict[str, Any]],
    production_sha: Optional[str],
    production_manifest: Optional[Dict[str, Any]] = None,
    expect_run_id: Optional[str] = None,
) -> Decision:
    """Decide promote / noop / refuse from the two artifacts' contents.

    Pure: no I/O. `*_payload` is the parsed model.bin JSON, `*_sha` its
    sha256; production_* are None when there is no production artifact (or
    no manifest — every artifact written before 2026-10-31 is manifest-less).
    """
    checks: List[Check] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append(Check(name, bool(ok), detail))
        return bool(ok)

    # 1. The manifest describes THIS file, from the expected run.
    if not check("staged manifest.json present", staged_manifest is not None):
        return Decision("refuse", tuple(checks))
    assert staged_manifest is not None
    check(
        "manifest model_sha256 matches the staged model.bin",
        staged_manifest.get("model_sha256") == staged_sha,
        f"manifest {_short(staged_manifest.get('model_sha256'))} vs file {_short(staged_sha)}",
    )
    if expect_run_id is not None:
        check(
            "staged manifest is from the expected run",
            staged_manifest.get("run_id") == expect_run_id,
            f"manifest run_id {staged_manifest.get('run_id')!r} vs expected {expect_run_id!r}",
        )
    if any(not c.ok for c in checks):
        return Decision("refuse", tuple(checks))

    # 2. Never re-promote what production already serves.
    if production_sha is not None and production_sha == staged_sha:
        return Decision("noop", tuple(checks))

    # 3. The artifact is the gated model, fitted the gated way.
    config_name = (staged_payload.get("config") or {}).get("name")
    check(
        f"config name {spec.expected_config_name}",
        config_name == spec.expected_config_name and staged_manifest.get("model") == spec.expected_config_name,
        f"artifact {config_name!r}, manifest {staged_manifest.get('model')!r}",
    )
    holdout = staged_manifest.get("holdout_fraction")
    check("fitted on every row (holdout_fraction 0)", holdout == 0, f"holdout_fraction {holdout!r}")
    check(
        "per-league baselines fitted",
        bool(staged_payload.get("league_baselines")) and staged_payload.get("global_home_baseline") is not None,
    )
    rho = staged_payload.get("rho")
    check("finite rho", _finite(rho), f"rho {rho!r}")

    # 4. No coverage regression against a production artifact of the same kind.
    teams = len(staged_payload.get("team_attack") or {})
    leagues = len(staged_payload.get("league_baselines") or {})
    if production_payload and production_payload.get("league_baselines"):
        prod_teams = len(production_payload.get("team_attack") or {})
        prod_leagues = len(production_payload["league_baselines"])
        check(
            f"teams >= {MIN_TEAM_RATIO:.0%} of production",
            teams >= MIN_TEAM_RATIO * prod_teams,
            f"{teams} vs {prod_teams}",
        )
        check(
            f"leagues >= production - {MAX_LEAGUE_DROP}",
            leagues >= prod_leagues - MAX_LEAGUE_DROP,
            f"{leagues} vs {prod_leagues}",
        )
    else:
        why = (
            "no readable production artifact"
            if production_payload is None
            else "legacy production (no per-league baselines)"
        )
        check("teams / leagues vs production", True, f"skipped: {why}; staged {teams} teams, {leagues} leagues")
    staged_newest = staged_manifest.get("newest_match_date")
    prod_newest = (production_manifest or {}).get("newest_match_date")
    if prod_newest is not None:
        s_ts, p_ts = _parse_ts(staged_newest), _parse_ts(prod_newest)
        check(
            "trained on data no older than production's",
            s_ts is not None and p_ts is not None and s_ts >= p_ts,
            f"newest match {staged_newest!r} vs production {prod_newest!r}",
        )
    return Decision("promote" if all(c.ok for c in checks) else "refuse", tuple(checks))


def serve_smoke(model: Any, spec: DCModelSpec) -> Tuple[Optional[str], str]:
    """(error or None, description) after serving one fixture from `model`
    the way precompute_predictions does: lambdas -> build_dc_matrix ->
    derive_markets."""
    if model is None:
        return "the loader returned no model (see its log line above)", ""
    teams = list(getattr(model, "team_attack", None) or {})
    if len(teams) < 2:
        return f"the model knows {len(teams)} team(s)", ""
    league = next(iter(getattr(model, "league_baselines", None) or {}), None)
    home, away = teams[0], teams[1]

    import numpy as np
    from predictors.market_derivation import MAX_GOALS_DERIVE, build_dc_matrix, derive_markets  # type: ignore

    h_lam, a_lam = model.lambdas_for_match(home, away, league=league)
    P = build_dc_matrix(float(h_lam), float(a_lam), float(getattr(model, "rho", 0.0)), max_goals=MAX_GOALS_DERIVE)
    if not (np.isfinite(P).all() and abs(float(P.sum()) - 1.0) < 1e-6):
        return f"invalid scoreline matrix for {home} v {away}", ""
    desc = f"{home} v {away}: lambdas {float(h_lam):.3f}/{float(a_lam):.3f}"
    if spec.smoke_market_group and spec.smoke_market:
        p = derive_markets(spec.smoke_market_group, P).get(spec.smoke_market) or {}
        vals = [p.get(k) for k in ("home", "draw", "away")]
        if not all(_finite(v) and 0.0 < v < 1.0 for v in vals) or abs(sum(vals) - 1.0) > 1e-6:
            return f"invalid {spec.smoke_market} for {home} v {away}: {p}", desc
        desc += f"; {spec.smoke_market} home={vals[0]:.3f} draw={vals[1]:.3f} away={vals[2]:.3f}"
    return None, desc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Parsed JSON, or None when the file does not exist. A file that exists
    but does not parse raises — the caller decides how loud to be."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _copy_durable(src: Path, dst: Path) -> None:
    shutil.copyfile(src, dst)
    with open(dst, "rb") as f:
        os.fsync(f.fileno())


def rollback_command(spec: DCModelSpec, backup_dir: Optional[Path], backup_has_manifest: bool) -> str:
    """The host command that atomically restores what production held before
    this promotion (copy beside, then rename — never a cp over the live file)."""
    prod = spec.production_dir
    if backup_dir is None:
        # First promotion: production had nothing, so rolling back removes it.
        inner = f"rm -f {prod}/{MODEL_FILE} {prod}/{MANIFEST_FILE}"
    else:
        steps = [
            f"cp -a {backup_dir}/{MODEL_FILE} {prod}/{MODEL_FILE}.new",
            f"mv -f {prod}/{MODEL_FILE}.new {prod}/{MODEL_FILE}",
        ]
        if backup_has_manifest:
            steps += [
                f"cp -a {backup_dir}/{MANIFEST_FILE} {prod}/{MANIFEST_FILE}.new",
                f"mv -f {prod}/{MANIFEST_FILE}.new {prod}/{MANIFEST_FILE}",
            ]
        else:
            # A legacy production file had no manifest; leaving the promoted
            # one would describe a model that is no longer served.
            steps.append(f"rm -f {prod}/{MANIFEST_FILE}")
        inner = " && ".join(steps)
    return f"cd /opt/auspex && docker compose exec -T api sh -c '{inner}'"


def _log_checks(decision: Decision) -> None:
    for c in decision.checks:
        line = f"  {'ok  ' if c.ok else 'FAIL'}  {c.name}" + (f" ({c.detail})" if c.detail else "")
        (LOGGER.info if c.ok else LOGGER.error)(line)


def promote(
    spec: DCModelSpec,
    *,
    check_only: bool,
    expect_run_id: Optional[str],
    load_staged: Callable[[Path], Any],
    load_production: Callable[[], Any],
    models_root: Path = MODELS_ROOT,
    stamp: Optional[str] = None,
) -> int:
    """Check, back up, promote, verify. Returns the process exit code.

    load_staged(path) / load_production() are the serve path's loaders
    (injected so the flow is testable without /app); everything else is
    files under spec's dirs and models_root."""
    staged_model = spec.staging_dir / MODEL_FILE
    production_model = spec.production_dir / MODEL_FILE
    LOGGER.info("1/4 check the staged %s artifact at %s", spec.name, staged_model)
    if not staged_model.exists():
        LOGGER.error("No staged artifact at %s — nothing to check, refusing", staged_model)
        return 1
    staged_sha = sha256_file(staged_model)
    try:
        staged_payload = read_json(staged_model) or {}
        staged_manifest = read_json(spec.staging_dir / MANIFEST_FILE)
    except (OSError, ValueError) as e:
        LOGGER.error("Staged artifact or manifest is unreadable (%s) — refusing", e)
        return 1

    production_sha = sha256_file(production_model) if production_model.exists() else None
    production_payload: Optional[Dict[str, Any]] = None
    production_manifest: Optional[Dict[str, Any]] = None
    try:
        production_payload = read_json(production_model)
    except (OSError, ValueError) as e:
        LOGGER.error(
            "Production %s is unreadable (%s) — the serve path cannot load it either; "
            "teams/leagues comparisons skipped",
            production_model,
            e,
        )
    try:
        production_manifest = read_json(spec.production_dir / MANIFEST_FILE)
    except (OSError, ValueError) as e:
        LOGGER.error("Production manifest is unreadable (%s) — ignoring it", e)

    decision = evaluate_promotion(
        spec,
        staged_payload=staged_payload,
        staged_sha=staged_sha,
        staged_manifest=staged_manifest,
        production_payload=production_payload,
        production_sha=production_sha,
        production_manifest=production_manifest,
        expect_run_id=expect_run_id,
    )
    _log_checks(decision)
    if decision.action == "refuse":
        LOGGER.error(
            "Refusing to promote %s (sha256 %s): %s",
            spec.name,
            _short(staged_sha),
            "; ".join(c.name for c in decision.refusals),
        )
        return 1
    if decision.action == "noop":
        LOGGER.info(
            "Nothing to promote: the staged %s is already production (sha256 %s)", spec.name, _short(staged_sha)
        )
        return 0
    assert staged_manifest is not None  # evaluate_promotion refuses without one
    LOGGER.info(
        "  run %s, git %s, %s rows to %s, %s teams, rho %s",
        staged_manifest.get("run_id"),
        staged_manifest.get("git_sha"),
        staged_manifest.get("n_rows"),
        staged_manifest.get("newest_match_date"),
        staged_manifest.get("teams"),
        staged_payload.get("rho"),
    )

    try:
        err, desc = serve_smoke(load_staged(staged_model), spec)
    except Exception as e:  # noqa: BLE001 — any failure here is a refusal, logged with context
        err, desc = f"{type(e).__name__}: {e}", ""
    if err:
        LOGGER.error("The staged artifact does not serve through the loader helper: %s — refusing", err)
        return 1
    LOGGER.info("  staged artifact serves: %s", desc)
    if check_only:
        LOGGER.info(
            "--check-only: every check passed; would promote %s over production %s",
            _short(staged_sha),
            _short(production_sha),
        )
        return 0

    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir: Optional[Path] = None
    backup_has_manifest = False
    if production_sha is not None:
        backup_dir = models_root / f"backup-{spec.name}-{stamp}"
        LOGGER.info("2/4 back up production -> %s", backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(production_model, backup_dir / MODEL_FILE)
        if (spec.production_dir / MANIFEST_FILE).exists():
            shutil.copy2(spec.production_dir / MANIFEST_FILE, backup_dir / MANIFEST_FILE)
            backup_has_manifest = True
        backup_sha = sha256_file(backup_dir / MODEL_FILE)
        LOGGER.info("  %s  %s", production_sha, production_model)
        LOGGER.info("  %s  %s", backup_sha, backup_dir / MODEL_FILE)
        LOGGER.info("  %s  %s", staged_sha, staged_model)
        if backup_sha != production_sha:
            LOGGER.error("The backup does not match production — production untouched, refusing")
            return 1
    else:
        LOGGER.warning("2/4 no production artifact at %s — first promotion, nothing to back up", production_model)

    LOGGER.info("3/4 promote")
    spec.production_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        (spec.staging_dir / MODEL_FILE, spec.production_dir / (MODEL_FILE + ".new"), spec.production_dir / MODEL_FILE),
        (
            spec.staging_dir / MANIFEST_FILE,
            spec.production_dir / (MANIFEST_FILE + ".new"),
            spec.production_dir / MANIFEST_FILE,
        ),
    ]
    for src, tmp, _ in pending:
        _copy_durable(src, tmp)
    # The staging file is re-hashed AS COPIED: a trainer run that rewrote
    # staging after the checks above must not slip an unchecked file through.
    copied_sha = sha256_file(pending[0][1])
    copied_manifest = read_json(pending[1][1]) or {}
    if copied_sha != staged_sha or copied_manifest.get("model_sha256") != staged_sha:
        for _, tmp, _ in pending:
            tmp.unlink(missing_ok=True)
        LOGGER.error(
            "Staging changed while promoting (checked %s, copied %s) — production untouched, refusing",
            _short(staged_sha),
            _short(copied_sha),
        )
        return 1
    for _, tmp, dst in pending:
        os.replace(tmp, dst)

    rollback = rollback_command(spec, backup_dir, backup_has_manifest)
    LOGGER.info("4/4 verify")
    promoted_sha = sha256_file(production_model)
    if promoted_sha != staged_sha:
        LOGGER.error("Production sha256 %s != staged %s. Roll back with: %s", promoted_sha, staged_sha, rollback)
        return 1
    try:
        err, desc = serve_smoke(load_production(), spec)
    except Exception as e:  # noqa: BLE001 — verification failure, logged with the rollback
        err, desc = f"{type(e).__name__}: {e}", ""
    if err:
        LOGGER.error("The serve loader cannot serve the promoted artifact: %s. Roll back with: %s", err, rollback)
        return 1
    LOGGER.info("  serve loader ok: %s", desc)
    LOGGER.info("DONE — promoted %s %s (run %s)", spec.name, _short(staged_sha), staged_manifest.get("run_id"))
    LOGGER.info("Rollback: %s", rollback)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(REGISTRY))
    parser.add_argument("--check-only", action="store_true", help="Run every check, change nothing.")
    parser.add_argument(
        "--expect-run-id",
        default=None,
        help="Refuse unless the staged manifest's run_id is this (the DAG passes its own run id).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

    spec = REGISTRY[args.model]
    for path in (ML_SRC, SCRIPTS_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)
    # The serve path's own loaders — imported before anything is touched, so
    # an import failure cannot leave a half-done promotion.
    import precompute_predictions as pp  # type: ignore

    return promote(
        spec,
        check_only=args.check_only,
        expect_run_id=args.expect_run_id,
        load_staged=lambda path: pp._load_dixon_coles_artifact(str(path), f"staged {args.model}"),
        load_production=getattr(pp, spec.serve_loader),
    )


if __name__ == "__main__":
    sys.exit(main())
