"""Pull a Modal training run's artifacts from B2, apply the PROMOTE-GATE, and
stage only the winners for the existing swap_production step.

Runs INSIDE the api container (which already carries the B2 + Telegram env).
For a given --run-id it:

  1. downloads modal-train/<run_id>/** into /app/models/modal-incoming/<run_id>/,
  2. per bundle, reads the served held-out Brier from that bundle's gate.json and
     compares it to the INCUMBENT (production held_out_metrics.json sidecar) —
     but ONLY when both numbers were scored on comparable frames (see below),
  3. copies ONLY bundles that PASS the gate into /app/models/staging/ (so the
     unchanged swap_production merges just the winners; a rejected bundle never
     reaches staging → its current production model is left untouched),
  4. writes the promoted bundle's held_out_metrics.json sidecar into staging (so
     it merges to production and becomes next run's incumbent),
  5. writes promote_decisions.json + Telegram-pages a one-line digest.

GATE (per bundle, keyed by ensemble_name; lower Brier = better):
  - gate.json missing/malformed, or bundle status != ok          → REJECT + alert
        (silent-failure guard: a broken bundle must NOT slip in)
  - training ok but no held-out test (e.g. tennis/mma, n=None)   → PROMOTE (can't
        gate; the model is valid — mirrors the calibration gate's own default)
  - no incumbent sidecar yet (first run / new bundle)            → PROMOTE (seed)
  - COMPARABLE frames (matching fingerprints) → SCALAR path: challenger Brier
        <= the BEST comparable served Brier on record (the last K runs AND the
        all-time champion, which never ages out of that window) + gate_tolerance(n).
        Not last-week's, because gating against only the previous run lets a
        bundle ratchet upward one tolerance-sized regression at a time (nfl_tot
        drifted 0.5052 → 0.5430 across four "passing" runs at n=128, where tol
        is 0.047) — and not a sliding window alone, which only slows that
        ratchet to one tolerance per K runs.
  - DIFFERENT frames → PAIRED path: the two stored scalars are computed on
        different test sets and mean nothing against each other (when the soccer
        frame grew ~7x on 2026-08-06 the incumbent's 0.59345 on n=3,562 rows was
        being compared to challengers scored on a wholly different population;
        the better model was rejected four weeks running). Both models are
        RE-SCORED on one shared set of rows — rows NEITHER model has seen —
        and the decision is made on the paired ΔBrier and its standard error.
        If that shared set cannot be built — no DB, missing artifacts, no way
        to bound what the incumbent already saw, a frame that drifted under us
        since the Modal run, fewer than MIN_PAIRED_N rows left — the bundle is
        REJECTED loudly and flagged for a manual decision. It is never quietly
        decided by the broken scalar comparison, and never decided on rows one
        of the two models trained on.

--shadow gates + reports but writes NOTHING into staging (the bootstrap run: pull,
gate, eyeball promote_decisions.json vs live Brier, then let a real run swap).

Force-promote (OPERATIONS.md) is unchanged: delete the incumbent sidecar
(rm /app/models/production/<ensemble_name>/held_out_metrics.json) and the next
run treats the bundle as un-gated (promote-to-seed), or copy the bundle tree from
modal-incoming into staging by hand and re-run swap.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

import b2_io  # noqa: E402
import model_metrics_store as mstore  # noqa: E402
from modal_bundles import (  # noqa: E402
    ARTIFACT_PREFIX,
    BUNDLE_TO_ENSEMBLE,
    HISTORY_LIMIT,
    MIN_PAIRED_N,
    fingerprint_from_report,
    fingerprints_comparable,
    gate_tolerance,
    paired_decision,
    served_brier,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("pull_modal_artifacts")

# train_all_models' split defaults; the paired path rebuilds the challenger's
# held-out tail with the same ratios the Modal run trained with.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


class PairedEvalError(RuntimeError):
    """The shared evaluation set for a paired re-scoring could not be built.
    Raised loudly — the caller REJECTS the bundle rather than falling back to the
    scalar comparison the fingerprints just told us is invalid."""


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


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        logger.warning("Unreadable %s: %s", path, exc)
        return None
    return doc if isinstance(doc, dict) else None


def _read_gate(bundle_dir: Path) -> dict:
    """The bundle's served-Brier decision, from the gate.json the Modal function
    uploaded (falls back to training_report.json['holdout_test'])."""
    gj = _read_json(bundle_dir / "gate.json")
    if gj is not None:
        return gj
    tr = _read_json(bundle_dir / "training_report.json")
    if tr is not None:
        return {"status": "ok", "gate": tr.get("holdout_test") or {}}
    return {}


def _challenger_fingerprint(bundle_dir: Path, gate: dict) -> Optional[dict]:
    """The shape of the frame this challenger was scored on. gate.json does not
    carry it today, so it is derived from the sibling training_report.json
    (data_quality.rows/date_min/date_max + holdout_test.n) — the fields
    train_all_models already writes. If a future train_modal starts embedding a
    'fingerprint' in gate.json, that wins."""
    embedded = gate.get("fingerprint")
    if isinstance(embedded, dict) and embedded:
        return embedded
    report = _read_json(bundle_dir / "training_report.json")
    if report is None:
        logger.warning("No readable training_report.json in %s — challenger has no frame fingerprint", bundle_dir)
        return None
    return fingerprint_from_report(report)


def recover_incumbent_fingerprint(incoming_root: Optional[Path], bundle: str, incumbent_doc: Optional[dict]):
    """Frame fingerprint for a LEGACY incumbent sidecar, recovered from the
    training_report.json of the run that produced it.

    Sidecars written before the fingerprint existed carry only a run_id, so the
    paired path had nothing to bound leakage with and fell back to the
    sidecar's `trained_at` — a WALL-CLOCK timestamp, not a data cut-off. For an
    incumbent built on 2026-08-30 that leaves almost no rows after the cutoff,
    so every legacy bundle failed with "shared evaluation set too small" and
    the gate deadlocked: it could not promote until sidecars carried
    fingerprints, and sidecars only gain one by being promoted.

    The escape is that each Modal run's per-bundle training_report.json is
    retained under models/modal-incoming/<run_id>/<bundle>/, and the sidecar
    records exactly which run_id it came from. That report carries the real
    data_quality.date_max, which is what the cutoff should have been all along.

    Returns (fingerprint | None, human description of where it came from).
    """
    run_id = (incumbent_doc or {}).get("run_id")
    if not run_id or incoming_root is None:
        return None, "no run_id on the incumbent sidecar"
    report_path = Path(incoming_root) / str(run_id) / bundle / "training_report.json"
    report = _read_json(report_path)
    if report is None:
        return None, f"no retained training_report.json for incumbent run {run_id}"
    fp = fingerprint_from_report(report)
    if not fp:
        return None, f"incumbent run {run_id} report carries no row count"
    return fp, f"recovered from the incumbent's own run {run_id}"


def _fp_summary(fp: Optional[dict]) -> str:
    if not fp:
        return "unknown frame"
    return f"rows={fp.get('rows')} holdout_n={fp.get('holdout_n')} {fp.get('date_min')}..{fp.get('date_max')}"


def _per_row_brier(proba, y):
    """Per-row multiclass Brier — the row-wise decomposition of
    training.calibration.brier_multiclass (whose value is this vector's mean).
    Needed because a PAIRED test needs per-row deltas, not just two means."""
    import numpy as np

    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y)), np.asarray(y).astype(int)] = 1.0
    return np.sum((proba - onehot) ** 2, axis=1)


def _frame_cutoff(incumbent_fp: Optional[dict], incumbent_doc: Optional[dict]):
    """The instant after which a row is certainly UNSEEN by the incumbent, as
    (pandas Timestamp, human description).

    Preference order:
      1. the incumbent's own frame end (fingerprint date_max) — exact, and we
         push it to the END of that day because date_max is date-precision;
      2. failing that, the incumbent sidecar's `trained_at`. Every sidecar
         carries it, including the legacy ones seeded at the 2026-08 cutover
         (which have no fingerprint at all), and a model cannot have trained on
         a match that had not been played when the model was written.

    Returns (None, reason) when neither is available — the caller must then
    refuse to decide rather than score the incumbent on rows it memorised.
    """
    import pandas as pd

    date_max = (incumbent_fp or {}).get("date_max")
    if date_max:
        ts = pd.to_datetime(date_max, utc=True, errors="coerce")
        if not pd.isna(ts):
            # date-precision → exclude the whole of that final day
            return ts.normalize() + pd.Timedelta(days=1), f"the incumbent frame end {date_max}"

    trained_at = (incumbent_doc or {}).get("trained_at")
    if trained_at:
        ts = pd.to_datetime(trained_at, utc=True, errors="coerce")
        if not pd.isna(ts):
            return ts, f"the incumbent's trained_at {trained_at} (legacy sidecar: no frame fingerprint)"

    return None, "the incumbent sidecar carries neither a frame fingerprint nor a trained_at"


def paired_rescore(
    bundle: str,
    production: str,
    challenger_dir: Path,
    database_url: Optional[str],
    incumbent_fp: Optional[dict],
    incumbent_doc: Optional[dict],
    challenger_fp: Optional[dict],
) -> dict:
    """Re-score the INCUMBENT and the CHALLENGER on one shared set of rows and
    return {n, delta, se, basis, incumbent_brier, challenger_brier}.

    Only runs where both artifacts and the DB actually exist — the VM. Reuses the
    existing serve-path loader (walk_forward_predictions.load_snapshot_ensemble →
    services.prediction_service._build_ensemble_for_task) so both models are
    reconstituted exactly as production reconstitutes them, calibrator included,
    and the bundle's own training loader for the frame. No new inference path.

    The shared set is rows NEITHER model has seen: the challenger's held-out
    tail (never trained or validated on), narrowed to what falls after the
    incumbent's own cutoff (_frame_cutoff). That narrowing is MANDATORY. It
    used to be best-effort — kept only when a fingerprint gave a cutoff and
    enough rows sat after it, otherwise the whole tail was scored and the
    basis string said so — but every incumbent sidecar in production today is a
    legacy one with no fingerprint, so in practice that fell through to scoring
    the incumbent on its own training and calibration rows. "Biases against
    promotion" is not a defence: a contaminated comparison is exactly what this
    path exists to replace, and it would reject the better soccer model a fifth
    week running. If a clean set cannot be built, we say so and the bundle goes
    to a human.

    The frame is re-loaded from the LIVE database at pull time, while the
    challenger trained on Modal against a restored nightly dump. If rows landed
    mid-history in between, the 85%-of-rows split boundary moves EARLIER and the
    "held-out" tail starts including rows the challenger trained on — which
    biases the delta FOR promotion, the unsafe direction. So the reloaded frame
    is fingerprinted and must still be comparable to the challenger's own.

    Raises PairedEvalError (loudly) if the set can't be built or is too small."""
    if not database_url:
        raise PairedEvalError("no DATABASE_URL — cannot rebuild a shared evaluation set")

    for path in ("/app/services/ml-models/src", "/app/services/api/src"):
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        import pandas as pd
        from training.calibration import brier_multiclass
        from training.train_all_models import SPORT_BUNDLES, _split_temporally
        from utils.training_data import get_feature_columns
        from walk_forward_predictions import load_snapshot_ensemble
    except ImportError as exc:
        raise PairedEvalError(f"paired re-scoring needs ml-models + api on sys.path: {exc}") from exc

    spec = SPORT_BUNDLES.get(bundle)
    if spec is None:
        raise PairedEvalError(f"no SPORT_BUNDLES entry for {bundle!r}")

    target = spec.target_column
    try:
        frame = spec.load_frame(database_url=database_url)
    except Exception as exc:  # noqa: BLE001 — surfaced as a loud reject, not swallowed
        raise PairedEvalError(f"could not load the {bundle} frame: {exc}") from exc
    if frame is None or frame.empty or target not in frame.columns:
        raise PairedEvalError(f"{bundle} frame is empty or missing target {target!r}")

    try:
        _, _, test_df = _split_temporally(frame, TRAIN_RATIO, VAL_RATIO)
    except ValueError as exc:
        raise PairedEvalError(f"could not split the {bundle} frame: {exc}") from exc

    # Encode against the classes of the WHOLE frame (sorted, i.e. exactly what
    # LabelEncoder gives the trainer) so column i of predict_proba is class i
    # even when the shared slice happens to miss a class.
    classes = sorted(frame[target].dropna().unique().tolist())

    # Did the frame drift under us since the Modal run trained on it?
    frame_dates = pd.to_datetime(frame["match_date"], errors="coerce", utc=True)
    frame_fp = fingerprint_from_report(
        {
            "data_quality": {
                "rows": len(frame),
                "date_min": frame_dates.min().isoformat() if len(frame) else None,
                "date_max": frame_dates.max().isoformat() if len(frame) else None,
                # get_feature_columns, NOT spec.feature_columns: the report's
                # data_quality.feature_count comes from validate_training_frame,
                # which uses this generic selector for every bundle. Deriving it
                # any other way would make the two fingerprints never match and
                # send every paired decision to a human.
                "feature_count": len(get_feature_columns(frame, target)),
                "target_classes": len(classes),
            },
            "holdout_test": {"n": len(test_df)},
        }
    )
    if not fingerprints_comparable(frame_fp, challenger_fp):
        raise PairedEvalError(
            "the frame drifted since the Modal run — re-splitting it would score the challenger on "
            f"rows it trained on (now {_fp_summary(frame_fp)} vs trained on {_fp_summary(challenger_fp)})"
        )

    cutoff, cutoff_desc = _frame_cutoff(incumbent_fp, incumbent_doc)
    if cutoff is None:
        raise PairedEvalError(
            f"cannot establish rows the incumbent has never seen: {cutoff_desc} — refusing to score it "
            "on rows it may have trained or calibrated on"
        )

    dates = pd.to_datetime(test_df["match_date"], errors="coerce", utc=True)
    eval_df = test_df[dates > cutoff]
    basis = f"rows after {cutoff_desc} (unseen by BOTH models)"

    n = int(len(eval_df))
    if n < MIN_PAIRED_N:
        raise PairedEvalError(
            f"shared evaluation set too small (n={n} < {MIN_PAIRED_N}) after excluding rows before " f"{cutoff_desc}"
        )

    index_of = {c: i for i, c in enumerate(classes)}
    y = [index_of.get(v) for v in eval_df[target].tolist()]
    if any(v is None for v in y):
        raise PairedEvalError(f"{bundle} shared set has target values outside the frame's classes")

    scored = {}
    for label, root in (("incumbent", Path(production)), ("challenger", Path(challenger_dir))):
        try:
            ensemble, _task, _meta = load_snapshot_ensemble(bundle, root)
        except Exception as exc:  # noqa: BLE001 — a missing/broken artifact must reject, not pass
            raise PairedEvalError(f"could not load the {label} ensemble from {root}: {exc}") from exc
        try:
            proba = ensemble.predict_proba(eval_df)
        except Exception as exc:  # noqa: BLE001
            raise PairedEvalError(f"{label} predict_proba failed on the shared set: {exc}") from exc
        if getattr(proba, "ndim", 0) != 2 or proba.shape[0] != n or proba.shape[1] != len(classes):
            raise PairedEvalError(
                f"{label} produced {getattr(proba, 'shape', '?')} for n={n}, {len(classes)} classes — not comparable"
            )
        scored[label] = proba

    rows_inc = _per_row_brier(scored["incumbent"], y)
    rows_chal = _per_row_brier(scored["challenger"], y)
    delta_rows = rows_chal - rows_inc
    se = float(delta_rows.std(ddof=1) / math.sqrt(n)) if n > 1 else float("inf")
    return {
        "n": n,
        "delta": float(delta_rows.mean()),
        "se": se,
        "basis": basis,
        "incumbent_brier": float(brier_multiclass(scored["incumbent"], y)),
        "challenger_brier": float(brier_multiclass(scored["challenger"], y)),
    }


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


def decide_bundle(
    bundle: str,
    bundle_dir: Path,
    production: str,
    database_url: Optional[str],
    paired_fn=paired_rescore,
    incoming_root: Optional[Path] = None,
) -> dict:
    """The whole promote decision for one bundle, as a promote_decisions.json row.

    Keys challenger_brier / incumbent_brier / kept_calibration / n / decision /
    reason are the original shape and keep their meaning; everything else is
    additive."""
    ensemble_name = BUNDLE_TO_ENSEMBLE[bundle]
    gate = _read_gate(bundle_dir)
    status = gate.get("status", "ok")
    brier, kept, n = served_brier(gate.get("gate") or gate)
    challenger_fp = _challenger_fingerprint(bundle_dir, gate)

    incumbent_doc = mstore.read_incumbent(production, ensemble_name)
    incumbent = mstore.read_incumbent_brier(production, ensemble_name)
    incumbent_fp = mstore.incumbent_fingerprint(incumbent_doc)
    fp_source = "sidecar"
    if not incumbent_fp:
        # Legacy sidecar: recover the real frame end from the incumbent's own
        # retained run report, so the leakage cutoff is a DATA boundary rather
        # than the wall-clock moment the model happened to be built.
        incumbent_fp, fp_source = recover_incumbent_fingerprint(incoming_root, bundle, incumbent_doc)
        if incumbent_fp:
            logger.info("[gate] %s: incumbent fingerprint %s", bundle, fp_source)

    tol = gate_tolerance(n)
    comparison = "none"
    baseline: Optional[float] = None
    baseline_run: Optional[str] = None
    paired: Optional[dict] = None
    needs_manual = False

    if status != "ok" or not gate:
        decision, reason = "reject", f"training status={status!r} / no gate.json — kept incumbent"
    elif brier is None:
        comparison = "ungated"
        decision, reason = "promote", "no held-out test to gate on — promoted (model valid)"
    elif incumbent is None:
        comparison = "seed"
        decision, reason = "promote", f"no incumbent yet — promoted to seed (brier={brier:.4f})"
    elif fingerprints_comparable(incumbent_fp, challenger_fp):
        comparison = "scalar"
        baseline, baseline_run = mstore.best_comparable_brier(incumbent_doc, challenger_fp)
        if baseline is None:  # comparable fingerprints but no usable history row
            baseline = incumbent
            baseline_run = incumbent_doc.get("run_id") if incumbent_doc else None
        against = f"best-of-{HISTORY_LIMIT} baseline {baseline:.4f}"
        if baseline_run:
            against += f" (run {baseline_run})"
        if brier <= baseline + tol:
            decision, reason = "promote", f"brier {brier:.4f} <= {against} + tol {tol:.4f} (n={n})"
        else:
            decision, reason = "reject", f"brier {brier:.4f} > {against} + tol {tol:.4f} (n={n}) — kept incumbent"
    else:
        comparison = "paired"
        frames = f"frames differ (incumbent {_fp_summary(incumbent_fp)} vs challenger {_fp_summary(challenger_fp)})"
        try:
            paired = paired_fn(
                bundle=bundle,
                production=production,
                challenger_dir=bundle_dir,
                database_url=database_url,
                incumbent_fp=incumbent_fp,
                incumbent_doc=incumbent_doc,
                challenger_fp=challenger_fp,
            )
        except PairedEvalError as exc:
            needs_manual = True
            decision = "reject"
            reason = f"{frames}; stored Briers are NOT comparable and paired re-scoring failed: {exc} — MANUAL DECISION NEEDED"  # noqa: E501
            logger.error("[gate] %s: %s", bundle, reason)
        except Exception as exc:  # noqa: BLE001 — never let an unexpected error decide silently
            needs_manual = True
            decision = "reject"
            reason = f"{frames}; paired re-scoring raised {type(exc).__name__}: {exc} — MANUAL DECISION NEEDED"
            logger.error("[gate] %s: %s", bundle, reason, exc_info=True)
        else:
            ok, verdict = paired_decision(paired["delta"], paired["se"], paired["n"])
            decision = "promote" if ok else "reject"
            reason = f"{frames}; re-scored both on {paired['basis']}: {verdict}"
            if not ok:
                reason += " — kept incumbent"

    return {
        "bundle": bundle,
        "ensemble_name": ensemble_name,
        "challenger_brier": brier,
        "incumbent_brier": incumbent,
        "kept_calibration": kept,
        "n": n,
        "decision": decision,
        "reason": reason,
        # additive fields (promote_decisions.json readers keyed on the above are unaffected)
        "comparison": comparison,
        "baseline_brier": baseline,
        "baseline_run_id": baseline_run,
        "challenger_fingerprint": challenger_fp,
        "incumbent_fingerprint": incumbent_fp,
        "paired": paired,
        "needs_manual": needs_manual,
    }


def gate_and_stage(
    run_id: str,
    models_dir: str,
    shadow: bool,
    database_url: Optional[str],
    paired_fn=paired_rescore,
) -> dict:
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

        row = decide_bundle(bundle, bdir, production, database_url, paired_fn=paired_fn, incoming_root=incoming.parent)
        decisions.append(row)
        logger.info("[gate] %-22s %s (%s)", bundle, row["decision"].upper(), row["reason"])

        if row["decision"] == "promote":
            # Count the DECISION in both modes so a shadow run honestly reports
            # "would promote N" (not always 0). Only STAGE + persist on a real run.
            promoted += 1
            if not shadow:
                _merge_bundle_into_staging(bdir, staging)
                # The sidecar rides along to production and becomes next run's
                # incumbent. Only write it when we HAVE a number.
                if row["challenger_brier"] is not None:
                    payload = mstore.build_payload(
                        ensemble_name,
                        served_brier=row["challenger_brier"],
                        kept=row["kept_calibration"],
                        n=row["n"],
                        run_id=run_id,
                        fingerprint=row["challenger_fingerprint"],
                        previous=mstore.read_incumbent(production, ensemble_name),
                    )
                    mstore.write_sidecar(str(staging), payload)
                    mstore.mirror_to_db(database_url, payload)

    manual = [d["bundle"] for d in decisions if d.get("needs_manual")]
    summary = {
        "run_id": run_id,
        "shadow": shadow,
        "promoted": promoted,
        "decisions": decisions,
        "needs_manual": manual,
    }
    # Drop the decision log where an operator can find it: into staging on a real
    # run (rides to production), into incoming on a shadow run.
    out_dir = incoming if shadow else staging
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "promote_decisions.json").write_text(json.dumps(summary, indent=2))

    rejects = [d["bundle"] for d in decisions if d["decision"] == "reject"]
    verb = "would promote" if shadow else "promoted"
    head = f"Modal retrain {run_id}{' (SHADOW)' if shadow else ''}: {verb} {promoted}/{len(decisions)}"
    if rejects:
        head += f"; kept incumbent for {', '.join(rejects)}"
    if manual:
        head += f"; ⚠️ MANUAL DECISION NEEDED for {', '.join(manual)} (frames not comparable, paired re-score failed)"
        logger.error(
            "%d bundle(s) could not be gated at all — decide by hand: %s",
            len(manual),
            ", ".join(manual),
        )
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
