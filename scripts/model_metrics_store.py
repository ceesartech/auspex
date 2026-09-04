"""Persist + read each promoted ensemble's held-out (served) Brier.

The promote-gate (scripts/pull_modal_artifacts.py) compares a challenger model
against the INCUMBENT's held-out Brier — a number NOT persisted before this
migration (the file registry stores only validation metrics;
model_performance_logs is free-form post-hoc). So for every bundle we PROMOTE we
drop a sidecar beside its ensemble artifact:

    /app/models/production/<ensemble_name>/held_out_metrics.json
      = {ensemble_name, served_brier, kept, n, run_id, trained_at,
         fingerprint, history, champion}

and best-effort mirror it into Postgres model_performance_logs (queryable +
survives a models-dir wipe). Next week's gate reads the sidecar as the incumbent.

Three fields beyond the original shape (all ADDED, nothing renamed — old
sidecars still read fine):

  fingerprint  the shape of the frame this Brier was scored on
               ({rows, date_min, date_max, holdout_n}, see modal_bundles). Two
               Briers are only comparable when their fingerprints are; a legacy
               sidecar has none and therefore compares against nothing.
  history      the last HISTORY_LIMIT promoted runs, each
               {run_id, served_brier, kept, n, trained_at, fingerprint}. The gate
               scores a challenger against the BEST comparable Brier in this
               history, not just last week's — otherwise a bundle ratchets
               upward one tolerance-sized regression at a time (nfl_tot went
               0.5052 → 0.5430 over four "passing" runs).
  champion     the best comparable run EVER recorded, carried forward run after
               run and never aged out. `history` is a sliding window of K, so on
               its own it only slows the ratchet by a factor of K: once the best
               entry falls off the end, the drift resumes one tolerance per K
               runs (simulated: nfl_total reaches 0.741 — worse than a coin
               flip — after 39 "PROMOTE" runs). The champion is what makes the
               baseline monotone. It resets only when the fingerprint family
               changes, because a Brier from another frame is not a bar this
               one can be held to.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(__file__))

from modal_bundles import HISTORY_LIMIT, fingerprints_comparable  # noqa: E402

logger = logging.getLogger("model_metrics_store")

SIDECAR_NAME = "held_out_metrics.json"


def sidecar_path(models_dir: str, ensemble_name: str) -> Path:
    return Path(models_dir) / ensemble_name / SIDECAR_NAME


def read_incumbent(models_dir: str, ensemble_name: str) -> Optional[dict]:
    """The whole incumbent sidecar doc, or None when there is no readable one
    (first Modal run, a brand-new bundle, or a corrupt file) → the gate then
    promotes to seed it."""
    p = sidecar_path(models_dir, ensemble_name)
    if not p.exists():
        return None
    try:
        doc = json.loads(p.read_text())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        logger.warning("Unreadable incumbent sidecar %s: %s — treating as no incumbent", p, exc)
        return None
    if not isinstance(doc, dict):
        logger.warning("Incumbent sidecar %s is not an object — treating as no incumbent", p)
        return None
    return doc


def read_incumbent_brier(models_dir: str, ensemble_name: str) -> Optional[float]:
    """The incumbent's served held-out Brier, or None if no sidecar exists yet."""
    doc = read_incumbent(models_dir, ensemble_name)
    if doc is None:
        return None
    v = doc.get("served_brier")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        logger.warning("Non-numeric served_brier in sidecar for %s — treating as no incumbent", ensemble_name)
        return None


def incumbent_fingerprint(doc: Optional[dict]) -> Optional[dict]:
    """The frame fingerprint the incumbent's Brier was scored on. None for a
    legacy sidecar — which fails CLOSED (never 'comparable')."""
    if not isinstance(doc, dict):
        return None
    fp = doc.get("fingerprint")
    return fp if isinstance(fp, dict) and fp else None


def history_entries(doc: Optional[dict]) -> list[dict]:
    """The bundle's promoted-run history, newest last. A legacy sidecar (no
    history key) is projected to a single entry from its top-level fields so the
    best-of-K baseline still has something to chew on."""
    if not isinstance(doc, dict):
        return []
    hist = doc.get("history")
    if isinstance(hist, list) and hist:
        return [e for e in hist if isinstance(e, dict)]
    if doc.get("served_brier") is None:
        return []
    return [
        {
            "run_id": doc.get("run_id"),
            "served_brier": doc.get("served_brier"),
            "kept": doc.get("kept"),
            "n": doc.get("n"),
            "trained_at": doc.get("trained_at"),
            "fingerprint": incumbent_fingerprint(doc),
        }
    ]


def champion_entry(doc: Optional[dict]) -> Optional[dict]:
    """The bundle's monotone best-ever run, or None when there is not one."""
    if not isinstance(doc, dict):
        return None
    champ = doc.get("champion")
    return champ if isinstance(champ, dict) and champ.get("served_brier") is not None else None


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_entry(entries, fingerprint: Optional[dict]) -> Optional[dict]:
    """The lowest-Brier entry whose fingerprint is comparable to `fingerprint`,
    or None when none of them is."""
    best: Optional[dict] = None
    best_brier: Optional[float] = None
    for entry in entries:
        brier = _as_float(entry.get("served_brier"))
        if brier is None:
            continue
        if not fingerprints_comparable(entry.get("fingerprint"), fingerprint):
            continue
        if best_brier is None or brier < best_brier:
            best, best_brier = entry, brier
    return best


def best_comparable_brier(doc: Optional[dict], challenger_fp: Optional[dict]) -> tuple[Optional[float], Optional[str]]:
    """(best served Brier, its run_id) a challenger must beat: the best of the
    last HISTORY_LIMIT promoted runs AND the all-time champion, counting only
    runs whose fingerprint is comparable to the challenger's.

    This — not merely last week's number — is what a challenger must beat.
    Gating against only the immediately-previous run lets a bundle drift upward
    forever in tolerance-sized steps, each one individually "within noise" of
    the step before it. The window alone is not enough either: the best entry
    ages out of it, so the drift merely slows to one tolerance per K runs. The
    champion never ages out, which makes the bar monotone.

    Returns (None, None) when nothing comparable is on record; the caller must
    then take the paired path, not guess."""
    candidates = list(history_entries(doc)[-HISTORY_LIMIT:])
    champ = champion_entry(doc)
    if champ is not None:
        candidates.append(champ)
    best = _best_entry(candidates, challenger_fp)
    if best is None:
        return None, None
    return _as_float(best.get("served_brier")), best.get("run_id")


def build_payload(
    ensemble_name: str,
    *,
    served_brier,
    kept,
    n,
    run_id: str,
    fingerprint: Optional[dict] = None,
    previous: Optional[dict] = None,
) -> dict:
    """The sidecar doc for a bundle we are promoting. `previous` is the incumbent
    doc being replaced — its history is carried forward (capped at HISTORY_LIMIT)
    with this run appended, and its champion (the best comparable run ever, which
    never ages out of the window) is carried forward or beaten, so next week's
    gate scores against a monotone bar instead of just last week's number."""
    entry: dict[str, Any] = {
        "ensemble_name": ensemble_name,
        "served_brier": served_brier,
        "kept": kept,
        "n": n,
        "run_id": run_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
    }
    run_summary = {k: entry[k] for k in ("run_id", "served_brier", "kept", "n", "trained_at", "fingerprint")}

    history = list(history_entries(previous))
    history.append(run_summary)
    entry["history"] = history[-HISTORY_LIMIT:]

    # Monotone champion. Replaced only by a strictly better run scored on a
    # COMPARABLE frame; reset to this run when the frame family changed, since
    # a Brier from another population is not a bar this one can be held to.
    # A sidecar written before champions existed has none; bootstrap it from
    # the best comparable run its history still remembers, so the bar does not
    # restart at whatever this run happens to score.
    champ = champion_entry(previous) or _best_entry(history_entries(previous), fingerprint)
    champ_brier = _as_float((champ or {}).get("served_brier"))
    mine = _as_float(served_brier)
    if champ is None or not fingerprints_comparable(champ.get("fingerprint"), fingerprint):
        entry["champion"] = run_summary
    elif mine is not None and champ_brier is not None and mine < champ_brier:
        entry["champion"] = run_summary
    else:
        entry["champion"] = champ
    return entry


def write_sidecar(models_dir: str, payload: dict) -> Path:
    """Write held_out_metrics.json into the ensemble's dir under models_dir. When
    written into a STAGING tree, it rides along with model.bin through
    swap_production's per-item merge into production."""
    p = sidecar_path(models_dir, payload["ensemble_name"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote incumbent sidecar %s (served_brier=%s)", p, payload.get("served_brier"))
    return p


def mirror_to_db(database_url: Optional[str], payload: dict) -> None:
    """Best-effort insert into model_performance_logs; NEVER raises (the sidecar
    is authoritative). sport is derived from the ensemble name
    (ensemble_<sport>_… → <sport>); model_version pinned to the training version."""
    if not database_url or payload.get("served_brier") is None:
        return
    try:
        import psycopg2
        from psycopg2.extras import Json

        parts = payload["ensemble_name"].split("_")
        sport = parts[1] if len(parts) > 1 else "unknown"
        with psycopg2.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_performance_logs
                        (model_name, model_version, sport, evaluation_date, metrics, sample_size, notes)
                    VALUES (%s, %s, %s, CURRENT_DATE, %s, %s, %s)
                    """,
                    (
                        payload["ensemble_name"],
                        "1.0.0",
                        sport,
                        Json(
                            {
                                "held_out_brier": payload["served_brier"],
                                "kept": payload["kept"],
                                "run_id": payload["run_id"],
                                "fingerprint": payload.get("fingerprint"),
                            }
                        ),
                        payload.get("n"),
                        f"promote-gate incumbent (run {payload['run_id']})",
                    ),
                )
                conn.commit()
    except Exception as exc:  # noqa: BLE001 — sidecar already written; DB mirror is optional
        logger.warning("Could not mirror held-out Brier to model_performance_logs: %s", exc)
