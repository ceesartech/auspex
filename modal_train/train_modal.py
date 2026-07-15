"""Production Modal training app — one NAMED function per bundle, run in parallel.

Promoted from modal_trial/train_trial.py. Differences from the trial:
  - 13 DISTINCT NAMED Modal functions (soccer_match_result_training,
    nhl_moneyline_training, … mma_moneyline_training) generated DRY from one
    body, so each is individually visible in the Modal dashboard / logs / billing.
  - Data-in is the real B2 dump pulled at runtime (no `modal volume put`); the VM
    Postgres is never exposed.
  - Artifacts + a gate.json (served held-out Brier) are pushed to
    modal-train/<run_id>/<bundle>/ in B2; the VM's scripts/pull_modal_artifacts.py
    pulls, gates, and stages them for the existing swap_production.
  - Structured per-bundle logging + a Telegram page on any bundle failure.
  - Secrets (auspex-b2, auspex-telegram) injected per function; NO DATABASE_URL.

Trigger (the retrain_models DAG, or by hand):
    modal run modal_train/train_modal.py --run-id <id>
    modal run modal_train/train_modal.py --run-id smoke1 --bundles soccer_match_result
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

# Bundle list + prefixes are the shared source of truth (scripts/). Needed at
# DEFINITION time to mint the named functions, so put scripts/ on the path here.
# BUNDLES is a list and the prefixes are strings — all serialize by value into
# the cloudpickled functions. `served_brier` (a function from modal_bundles) is
# imported at RUNTIME inside _train_one instead, so it's not a cross-module
# reference cloudpickle would try (and fail) to re-import in the container.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from modal_bundles import ARTIFACT_PREFIX, BUNDLES, DUMP_PREFIX  # noqa: E402

app = modal.App("auspex-train")

# Same image as the trial (PG17 pin is load-bearing — the api's pg_dump 17 writes
# archive v1.16 that only pg_restore>=17 reads). Adds scripts/ so the container
# can import b2_io at runtime.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "gnupg", "libgomp1", "build-essential", "libpq-dev")
    .run_commands(
        "install -d /usr/share/postgresql-common/pgdg",
        "curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc "
        "https://www.postgresql.org/media/keys/ACCC4CF8.asc",
        'echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] '
        'https://apt.postgresql.org/pub/repos/apt $(. /etc/os-release && echo $VERSION_CODENAME)-pgdg main" '
        "> /etc/apt/sources.list.d/pgdg.list",
        "apt-get update",
        "apt-get install -y --no-install-recommends postgresql-17",
    )
    .pip_install_from_requirements("requirements.txt")
    .run_commands("pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu")
    .pip_install("onnxmltools==1.13.0", "onnxruntime==1.20.1")
    .add_local_dir("services/ml-models/src", "/app/ml-src")
    .add_local_dir("scripts", "/app/scripts")
    # So runtime imports of b2_io / modal_bundles resolve in the container
    # (also makes them importable at deserialize time for serialized functions).
    .env({"PYTHONPATH": "/app/scripts"})
)

# B2 creds (AWS_ACCESS_KEY_ID/SECRET + BACKUP_S3_*) and Telegram creds. No
# DATABASE_URL — data-in is exclusively the B2 dump.
SECRETS = [modal.Secret.from_name("auspex-b2"), modal.Secret.from_name("auspex-telegram")]

PGDATA = "/tmp/pgdata"
DB_URL = "postgresql://postgres@localhost:5432/trial"


# ── Postgres restore (identical mechanics to the trial) ──────────────


def _pgbin() -> str:
    import glob

    hits = sorted(glob.glob("/usr/lib/postgresql/*/bin"))
    if not hits:
        raise RuntimeError("No postgresql server binaries under /usr/lib/postgresql/*/bin")
    return hits[-1]


def _run(cmd, **kw):
    return subprocess.run(cmd, check=kw.pop("check", True), text=True, capture_output=True, **kw)


def _restore_dump(dump_path: str) -> None:
    pgbin = _pgbin()

    def pg(*args, check=True):
        return _run(["runuser", "-u", "postgres", "--", f"{pgbin}/{args[0]}", *args[1:]], check=check)

    os.makedirs(PGDATA, exist_ok=True)
    _run(["chown", "-R", "postgres:postgres", PGDATA])
    pg("initdb", "-D", PGDATA, "--auth=trust", "--username=postgres")
    pg("pg_ctl", "-D", PGDATA, "-o", "-p 5432", "-l", "/tmp/pg.log", "-w", "start")
    pg("createdb", "-p", "5432", "trial")
    res = pg("pg_restore", "-p", "5432", "--no-owner", "--no-acl", "-d", "trial", dump_path, check=False)
    if res.returncode != 0:
        print(f"[pg_restore rc={res.returncode}] {res.stderr[-800:]}")
    check = pg("psql", "-p", "5432", "-d", "trial", "-tAc", "SELECT count(*) FROM matches", check=False)
    n = check.stdout.strip() if check.returncode == 0 else "ERROR"
    if check.returncode != 0 or n in ("", "0", "ERROR"):
        raise RuntimeError(f"Restore looks empty/broken (matches={n!r}); aborting bundle.")
    print(f"[restore] matches rows = {n}")


def _telegram(text: str) -> None:
    """Best-effort page from inside the bundle's own container; never raises."""
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
    except Exception:  # noqa: BLE001
        pass


# ── The shared training body (one bundle) ────────────────────────────


def _train_one(bundle: str, run_id: str) -> dict:
    log = logging.getLogger(f"train.{bundle}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    def phase(name, **kw):
        log.info(json.dumps({"run_id": run_id, "bundle": bundle, "phase": name, **kw}))

    sys.path.insert(0, "/app/scripts")
    import b2_io  # noqa: E402 — runtime import (needs boto3 + B2 env from the secret)
    from modal_bundles import served_brier  # noqa: E402 — runtime, not a serialized cross-module ref

    t0 = time.time()
    out_dir = f"/tmp/out/{bundle}"
    os.makedirs(out_dir, exist_ok=True)
    try:
        phase("download_dump")
        meta = b2_io.download_latest(DUMP_PREFIX, Path("/tmp/dump.dump"))
        phase("restore", dump=meta["key"])
        _restore_dump("/tmp/dump.dump")

        phase("train")
        env = {**os.environ, "PYTHONPATH": "/app/ml-src", "DATABASE_URL": DB_URL}
        proc = subprocess.run(
            [
                "python",
                "-m",
                "training.train_all_models",
                "--sport",
                bundle,
                "--model-type",
                "all",
                "--database-url",
                DB_URL,
                "--output-dir",
                out_dir,
                "--export-onnx",
            ],
            cwd="/app",
            env=env,
            text=True,
            capture_output=True,
        )
        ok = proc.returncode == 0
        if not ok:
            print(f"[{bundle}] TRAIN FAILED rc={proc.returncode}\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")

        # Served held-out Brier from the training report's calibration gate.
        gate_metrics = {}
        report = Path(out_dir) / "training_report.json"
        if report.exists():
            try:
                gate_metrics = json.loads(report.read_text()).get("holdout_test") or {}
            except (ValueError, OSError):
                pass
        brier, kept, n = served_brier(gate_metrics)

        # gate.json is the compact decision the VM gate reads without re-parsing.
        gate_doc = {"status": "ok" if ok else "error", "bundle": bundle, "run_id": run_id, "gate": gate_metrics}
        (Path(out_dir) / "gate.json").write_text(json.dumps(gate_doc, indent=2))

        artifact_prefix = f"{ARTIFACT_PREFIX}{run_id}/{bundle}"
        phase("upload", prefix=artifact_prefix)
        n_files = b2_io.upload_tree(Path(out_dir), artifact_prefix)

        elapsed = round(time.time() - t0, 1)
        if not ok:
            _telegram(f"🤖❌ Modal train {bundle} FAILED (run {run_id}, rc={proc.returncode})")
        phase("done", status="ok" if ok else "error", served_brier=brier, seconds=elapsed, files=n_files)
        return {
            "bundle": bundle,
            "status": "ok" if ok else "error",
            "returncode": proc.returncode,
            "served_brier": brier,
            "kept": kept,
            "n": n,
            "seconds": elapsed,
            "artifact_prefix": artifact_prefix,
        }
    except Exception as exc:  # noqa: BLE001 — isolate this bundle
        _telegram(f"🤖❌ Modal train {bundle} ERROR (run {run_id}): {exc}")
        phase("error", error=str(exc))
        return {"bundle": bundle, "status": "error", "error": str(exc), "seconds": round(time.time() - t0, 1)}


# ── 13 NAMED functions, generated DRY ────────────────────────────────


def _register(bundle: str):
    # serialized=True is REQUIRED: these functions are generated in a factory
    # (not at module/global scope), so Modal can't reference them by import path
    # and raises "must apply to functions in global scope, unless serialized=True".
    # Serialized functions are cloudpickled by value; the entrypoint runs as
    # __main__ so _train_one + its helpers serialize by value too. name= still
    # registers each as a distinct <bundle>_training function in the dashboard.
    @app.function(
        name=f"{bundle}_training",
        image=image,
        secrets=SECRETS,
        cpu=4.0,
        memory=8192,
        timeout=3600,
        retries=1,
        serialized=True,
    )
    def _fn(run_id: str, _bundle: str = bundle) -> dict:
        return _train_one(_bundle, run_id)

    return _fn


TRAINERS = {b: _register(b) for b in BUNDLES}


@app.local_entrypoint()
def main(run_id: str = "", bundles: str = ""):
    """Fan out training across the NAMED per-bundle functions, in parallel.

    modal run modal_train/train_modal.py --run-id <airflow_run_id>
    modal run modal_train/train_modal.py --run-id smoke1 --bundles soccer_match_result
    """
    import uuid
    from datetime import datetime, timezone

    run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    selected = [b.strip() for b in bundles.split(",") if b.strip()] or BUNDLES
    unknown = [b for b in selected if b not in BUNDLES]
    if unknown:
        raise SystemExit(f"Unknown bundle(s): {unknown}\nValid: {BUNDLES}")

    print(f"run_id={run_id}\nSpawning {len(selected)} named training functions in parallel: {selected}\n")
    handles = [TRAINERS[b].spawn(run_id) for b in selected]
    results = [h.get() for h in handles]

    print("\n" + "=" * 92)
    print(f"{'bundle':24} {'status':6} {'secs':>6}  {'served Brier':>12}   artifact prefix")
    print("-" * 92)
    for r in sorted(results, key=lambda x: x["bundle"]):
        b = r.get("served_brier")
        bs = f"{b:.4f}" if isinstance(b, (int, float)) else "—"
        print(f"{r['bundle']:24} {r['status']:6} {r.get('seconds', 0):>6}  {bs:>12}   {r.get('artifact_prefix', '')}")
    print("=" * 92)
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"run_id={run_id}  ok={ok}  errored={len(results) - ok}")
    print(f"Artifacts in B2 under {ARTIFACT_PREFIX}{run_id}/ — pull + gate on the VM with:")
    print(f"  docker compose exec -T api python /app/scripts/pull_modal_artifacts.py --run-id {run_id}")
