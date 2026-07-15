"""Persist + read each promoted ensemble's held-out (served) Brier.

The promote-gate (scripts/pull_modal_artifacts.py) compares a challenger model
against the INCUMBENT's held-out Brier — a number NOT persisted before this
migration (the file registry stores only validation metrics;
model_performance_logs is free-form post-hoc). So for every bundle we PROMOTE we
drop a sidecar beside its ensemble artifact:

    /app/models/production/<ensemble_name>/held_out_metrics.json
      = {ensemble_name, served_brier, kept, n, run_id, trained_at}

and best-effort mirror it into Postgres model_performance_logs (queryable +
survives a models-dir wipe). Next week's gate reads the sidecar as the incumbent.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("model_metrics_store")

SIDECAR_NAME = "held_out_metrics.json"


def sidecar_path(models_dir: str, ensemble_name: str) -> Path:
    return Path(models_dir) / ensemble_name / SIDECAR_NAME


def read_incumbent_brier(models_dir: str, ensemble_name: str) -> Optional[float]:
    """The incumbent's served held-out Brier, or None if no sidecar exists yet
    (first Modal run, or a brand-new bundle) → the gate then promotes to seed it."""
    p = sidecar_path(models_dir, ensemble_name)
    if not p.exists():
        return None
    try:
        v = json.loads(p.read_text()).get("served_brier")
        return float(v) if v is not None else None
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        logger.warning("Unreadable incumbent sidecar %s: %s — treating as no incumbent", p, exc)
        return None


def build_payload(ensemble_name: str, *, served_brier, kept, n, run_id: str) -> dict:
    return {
        "ensemble_name": ensemble_name,
        "served_brier": served_brier,
        "kept": kept,
        "n": n,
        "run_id": run_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


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
                            }
                        ),
                        payload.get("n"),
                        f"promote-gate incumbent (run {payload['run_id']})",
                    ),
                )
                conn.commit()
    except Exception as exc:  # noqa: BLE001 — sidecar already written; DB mirror is optional
        logger.warning("Could not mirror held-out Brier to model_performance_logs: %s", exc)
