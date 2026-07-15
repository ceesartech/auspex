"""Pull a Modal training run's artifacts from B2, apply the PROMOTE-GATE, and
stage only the winners for the existing swap_production step.

Runs INSIDE the api container (which already carries the B2 + Telegram env).
For a given --run-id it:

  1. downloads modal-train/<run_id>/** into /app/models/modal-incoming/<run_id>/,
  2. per bundle, reads the served held-out Brier from that bundle's gate.json and
     compares it to the INCUMBENT (production held_out_metrics.json sidecar),
  3. copies ONLY bundles that PASS the gate into /app/models/staging/ (so the
     unchanged swap_production merges just the winners; a rejected bundle never
     reaches staging → its current production model is left untouched),
  4. writes the promoted bundle's held_out_metrics.json sidecar into staging (so
     it merges to production and becomes next run's incumbent),
  5. writes promote_decisions.json + Telegram-pages a one-line digest.

GATE (per bundle, keyed by ensemble_name; lower Brier = better):
  - challenger served Brier <= incumbent + NOISE_FLOOR (0.009)  → PROMOTE
  - no incumbent sidecar yet (first run / new bundle)           → PROMOTE (seed)
  - training ok but no held-out test (e.g. tennis/mma, n=None)  → PROMOTE (can't
        gate; the model is valid — mirrors the calibration gate's own default)
  - gate.json missing/malformed, or bundle status != ok         → REJECT + alert
        (silent-failure guard: a broken bundle must NOT slip in)

--shadow gates + reports but writes NOTHING into staging (the bootstrap run: pull,
gate, eyeball promote_decisions.json vs live Brier, then let a real run swap).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import b2_io  # noqa: E402
import model_metrics_store as mstore  # noqa: E402
from modal_bundles import ARTIFACT_PREFIX, BUNDLE_TO_ENSEMBLE, NOISE_FLOOR, served_brier  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("pull_modal_artifacts")


def _telegram(text: str) -> None:
    """Best-effort one-line page; never raises. Gated on ENABLE_TELEGRAM_NOTIFICATIONS."""
    if os.environ.get("ENABLE_TELEGRAM_NOTIFICATIONS", "false").lower() != "true":
        return
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    try:
        import requests

        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telegram page failed: %s", exc)


def _read_gate(bundle_dir: Path) -> dict:
    """The bundle's served-Brier decision, from the gate.json the Modal function
    uploaded (falls back to training_report.json['holdout_test'])."""
    gj = bundle_dir / "gate.json"
    if gj.exists():
        try:
            return json.loads(gj.read_text())
        except (ValueError, OSError, json.JSONDecodeError):
            pass
    tr = bundle_dir / "training_report.json"
    if tr.exists():
        try:
            return {"status": "ok", "gate": json.loads(tr.read_text()).get("holdout_test") or {}}
        except (ValueError, OSError, json.JSONDecodeError):
            pass
    return {}


def _merge_bundle_into_staging(bundle_dir: Path, staging: Path) -> None:
    """Copy each top-level item of a bundle's artifact tree into staging, rm-then-
    cp per item — exactly how the on-VM retrain builds staging (unique model dirs
    accumulate; shared registry_index.json/training_report.json overwrite, which
    is benign since the serve path loads model.bin by mtime, not the index)."""
    staging.mkdir(parents=True, exist_ok=True)
    for item in bundle_dir.iterdir():
        if item.name == "gate.json":
            continue  # gate.json is our sidecar, not a model artifact
        dest = staging / item.name
        if dest.is_dir():
            shutil.rmtree(dest)
        elif dest.exists():
            dest.unlink()
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)


def gate_and_stage(run_id: str, models_dir: str, shadow: bool, database_url: str | None) -> dict:
    production = str(Path(models_dir) / "production")
    staging = Path(models_dir) / "staging"
    incoming = Path(models_dir) / "modal-incoming" / run_id

    n_files = b2_io.download_prefix(f"{ARTIFACT_PREFIX}{run_id}", incoming)
    logger.info("Pulled %d artifact files for run %s", n_files, run_id)

    decisions = []
    promoted = 0
    for bundle, ensemble_name in BUNDLE_TO_ENSEMBLE.items():
        bdir = incoming / bundle
        if not bdir.is_dir():
            logger.info("No artifacts for %s in this run — skipping", bundle)
            continue

        gate = _read_gate(bdir)
        status = gate.get("status", "ok")
        brier, kept, n = served_brier(gate.get("gate") or gate)
        incumbent = mstore.read_incumbent_brier(production, ensemble_name)

        if status != "ok" or not gate:
            decision, reason = "reject", f"training status={status!r} / no gate.json — kept incumbent"
        elif brier is None:
            decision, reason = "promote", "no held-out test to gate on — promoted (model valid)"
        elif incumbent is None:
            decision, reason = "promote", f"no incumbent yet — promoted to seed (brier={brier:.4f})"
        elif brier <= incumbent + NOISE_FLOOR:
            decision, reason = "promote", f"brier {brier:.4f} <= incumbent {incumbent:.4f} + {NOISE_FLOOR}"
        else:
            decision, reason = (
                "reject",
                f"brier {brier:.4f} > incumbent {incumbent:.4f} + {NOISE_FLOOR} — kept incumbent",
            )

        row = {
            "bundle": bundle,
            "ensemble_name": ensemble_name,
            "challenger_brier": brier,
            "incumbent_brier": incumbent,
            "kept_calibration": kept,
            "n": n,
            "decision": decision,
            "reason": reason,
        }
        decisions.append(row)
        logger.info("[gate] %-22s %s (%s)", bundle, decision.upper(), reason)

        if decision == "promote" and not shadow:
            _merge_bundle_into_staging(bdir, staging)
            # The sidecar rides along to production and becomes next run's
            # incumbent. Only write it when we HAVE a number.
            if brier is not None:
                payload = mstore.build_payload(ensemble_name, served_brier=brier, kept=kept, n=n, run_id=run_id)
                mstore.write_sidecar(str(staging), payload)
                mstore.mirror_to_db(database_url, payload)
            promoted += 1

    summary = {"run_id": run_id, "shadow": shadow, "promoted": promoted, "decisions": decisions}
    # Drop the decision log where an operator can find it: into staging on a real
    # run (rides to production), into incoming on a shadow run.
    out_dir = incoming if shadow else staging
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "promote_decisions.json").write_text(json.dumps(summary, indent=2))

    rejects = [d["bundle"] for d in decisions if d["decision"] == "reject"]
    head = f"Modal retrain {run_id}{' (SHADOW)' if shadow else ''}: promoted {promoted}/{len(decisions)}"
    if rejects:
        head += f"; kept incumbent for {', '.join(rejects)}"
    logger.info(head)
    _telegram("🤖 " + head)
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True, help="B2 prefix modal-train/<run_id>/ to pull + gate.")
    p.add_argument("--models-dir", default="/app/models")
    p.add_argument("--shadow", action="store_true", help="Gate + report only; write nothing into staging.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = gate_and_stage(args.run_id, args.models_dir, args.shadow, args.database_url)
    # Non-zero exit if EVERY bundle was rejected — a real problem worth failing
    # the task on (a single reject is normal partial-retrain behavior).
    if summary["decisions"] and summary["promoted"] == 0 and not args.shadow:
        logger.error("Every bundle was rejected — leaving production untouched and failing loudly.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
