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

# Bundle list + helpers are the shared source of truth (scripts/). Put scripts/
# on the path both here (definition time) and, in the container, via the image's
# PYTHONPATH=/app/scripts so this same top-level import resolves when Modal
# imports the module to run a (non-serialized) function.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from modal_bundles import ARTIFACT_PREFIX, BUNDLES, DUMP_PREFIX, served_brier  # noqa: E402

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
    # PYTHONPATH so runtime imports of b2_io / modal_bundles resolve in the
    # container. MUST come before add_local_* — Modal requires add_local_* to be
    # the LAST build steps (they're layered on at container startup, not baked).
    .env({"PYTHONPATH": "/app/scripts"})
    .add_local_dir("services/ml-models/src", "/app/ml-src")
    .add_local_dir("scripts", "/app/scripts")
)

# B2 creds (AWS_ACCESS_KEY_ID/SECRET + BACKUP_S3_*) and Telegram creds. No
# DATABASE_URL — data-in is exclusively the B2 dump.
SECRETS = [modal.Secret.from_name("auspex-b2"), modal.Secret.from_name("auspex-telegram")]

# Shared dump cache. prep_dump downloads the nightly B2 dump ONCE into this
# volume; the 13 trainers read it from here instead of each pulling the ~148 MB
# dump from B2 (13× the download blew B2's daily download cap). ~1.9 GB/run →
# ~148 MB/run, and it's faster.
dump_cache = modal.Volume.from_name("auspex-dump-cache", create_if_missing=True)
CACHED_DUMP = "/cache/current.dump"

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


# ── Dump prep: download the nightly B2 dump ONCE into the shared volume ──


@app.function(image=image, secrets=SECRETS, volumes={"/cache": dump_cache}, timeout=1800)
def prep_dump() -> dict:
    """Download the latest B2 dump into the cache volume (once per run), so the
    13 trainers read it from the volume instead of each pulling it from B2.
    Skips the download when the cached copy already matches the latest dump."""
    import sys
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    sys.path.insert(0, "/app/scripts")
    import b2_io

    s3 = b2_io.s3_client()
    objs = b2_io._list(s3, DUMP_PREFIX)
    if not objs:
        raise RuntimeError(f"No dump under s3://{b2_io.bucket()}/{DUMP_PREFIX}")
    latest = max(objs, key=lambda o: o["LastModified"])
    age = datetime.now(timezone.utc) - latest["LastModified"]
    if age > timedelta(hours=30):
        raise RuntimeError(f"Latest dump {latest['Key']} is {age} old (>30h) — refusing to train on a stale dump.")

    dest = Path(CACHED_DUMP)
    if dest.exists() and dest.stat().st_size == latest["Size"]:
        print(f"[prep] cache hit: {latest['Key']} ({latest['Size']} bytes)")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[prep] downloading {latest['Key']} ({latest['Size']} bytes) → {CACHED_DUMP}")
        s3.download_file(b2_io.bucket(), latest["Key"], str(dest))
        if dest.stat().st_size != latest["Size"]:
            raise RuntimeError(f"Cached dump size mismatch: {dest.stat().st_size} vs {latest['Size']}")
        dump_cache.commit()
    return {"key": latest["Key"], "size": latest["Size"]}


# ── The shared training body (one bundle) ────────────────────────────


def _train_one(bundle: str, run_id: str) -> dict:
    log = logging.getLogger(f"train.{bundle}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    def phase(name, **kw):
        log.info(json.dumps({"run_id": run_id, "bundle": bundle, "phase": name, **kw}))

    sys.path.insert(0, "/app/scripts")
    import b2_io  # noqa: E402 — runtime import (needs boto3 + B2 env from the secret)

    t0 = time.time()
    out_dir = f"/tmp/out/{bundle}"
    os.makedirs(out_dir, exist_ok=True)
    try:
        # Read the dump prep_dump already cached in the shared volume — no B2
        # download here (that's what blew the daily cap at 13× the dump size).
        dump_cache.reload()
        phase("restore", dump=CACHED_DUMP)
        _restore_dump(CACHED_DUMP)

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


# ── 13 NAMED functions (explicit global-scope defs) ──────────────────
# Deliberately NOT a factory + NOT serialized=True. Modal requires @app.function
# on a global-scope function; a NON-serialized function is *imported* in the
# container (which runs its own Python 3.11), so the caller's Python version is
# irrelevant — this avoids the serialized=True constraint that the definer's
# Python must match the image's. Verbose, but bulletproof. Each just dispatches
# to the shared _train_one body, and each registers as its own <bundle>_training
# function in the Modal dashboard/logs/billing.
_FN_KW = dict(
    image=image, secrets=SECRETS, volumes={"/cache": dump_cache}, cpu=4.0, memory=8192, timeout=3600, retries=1
)


@app.function(name="soccer_match_result_training", **_FN_KW)
def soccer_match_result_training(run_id: str) -> dict:
    return _train_one("soccer_match_result", run_id)


@app.function(name="nhl_moneyline_training", **_FN_KW)
def nhl_moneyline_training(run_id: str) -> dict:
    return _train_one("nhl_moneyline", run_id)


@app.function(name="nhl_regulation_training", **_FN_KW)
def nhl_regulation_training(run_id: str) -> dict:
    return _train_one("nhl_regulation", run_id)


@app.function(name="nhl_puck_line_training", **_FN_KW)
def nhl_puck_line_training(run_id: str) -> dict:
    return _train_one("nhl_puck_line", run_id)


@app.function(name="nhl_total_training", **_FN_KW)
def nhl_total_training(run_id: str) -> dict:
    return _train_one("nhl_total", run_id)


@app.function(name="nba_moneyline_training", **_FN_KW)
def nba_moneyline_training(run_id: str) -> dict:
    return _train_one("nba_moneyline", run_id)


@app.function(name="nba_spread_training", **_FN_KW)
def nba_spread_training(run_id: str) -> dict:
    return _train_one("nba_spread", run_id)


@app.function(name="nba_total_training", **_FN_KW)
def nba_total_training(run_id: str) -> dict:
    return _train_one("nba_total", run_id)


@app.function(name="nfl_moneyline_training", **_FN_KW)
def nfl_moneyline_training(run_id: str) -> dict:
    return _train_one("nfl_moneyline", run_id)


@app.function(name="nfl_spread_training", **_FN_KW)
def nfl_spread_training(run_id: str) -> dict:
    return _train_one("nfl_spread", run_id)


@app.function(name="nfl_total_training", **_FN_KW)
def nfl_total_training(run_id: str) -> dict:
    return _train_one("nfl_total", run_id)


@app.function(name="tennis_moneyline_training", **_FN_KW)
def tennis_moneyline_training(run_id: str) -> dict:
    return _train_one("tennis_moneyline", run_id)


@app.function(name="mma_moneyline_training", **_FN_KW)
def mma_moneyline_training(run_id: str) -> dict:
    return _train_one("mma_moneyline", run_id)


# bundle key → its named function, for the entrypoint to spawn in parallel.
TRAINERS = {b: globals()[f"{b}_training"] for b in BUNDLES}


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

    print(f"run_id={run_id}\n[prep] downloading the dump once into the shared cache volume …")
    dm = prep_dump.remote()
    print(f"[prep] dump ready: {dm['key']} ({dm['size']} bytes)\n")
    print(f"Spawning {len(selected)} named training functions in parallel: {selected}\n")
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
