"""Modal trial — parallel per-bundle model training, faithful to the on-VM path.

STANDALONE EXPERIMENT. This is wired into nothing: no DAG, no CI, no change to
train_all_models.py or retrain_models.py. Its only job is to answer, with real
numbers, three questions before we consider a consolidated Modal-backed retrain
DAG:

  1. Does `train_all_models` run correctly on Modal?
  2. Does training all 13 bundles IN PARALLEL (one container each) work, with no
     resource contention — the reason the on-VM DAG trains sequentially?
  3. How long does it take and what does it cost?

WHY THE POSTGRES RESTORE (not a CSV snapshot). Each bundle has its own SQL query
(soccer / NHL / NBA / NFL / tennis / MMA differ), and correctness is the
priority. So every container restores the real nightly pg_dump into a local
Postgres and trains via `--database-url`, i.e. the *identical* pd.read_sql path
production uses — zero CSV round-trip risk across any bundle. It also mirrors
exactly what a future DAG→Modal design would do (pull the B2 dump, restore,
train, ship artifacts back).

CORRECTNESS CHECK. XGBoost/LightGBM are seeded but the torch NN is not, so the
artifacts won't be byte-identical to a VM run. We compare the held-out TEST
Brier — the `CALIBRATION GATE (...)` line each bundle logs — against a VM
baseline, judged within the ΔBrier ≈ 0.009 noise floor the audit uses. Expected
soccer_match_result ensemble Brier ≈ 0.594–0.596 (per the audit's held-out runs).

Usage: see modal_trial/README.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import modal

# The 13 trainable bundles = the keys of SPORT_BUNDLES in
# services/ml-models/src/training/train_all_models.py. Horse racing is excluded
# — it trains via its own precompute path, not train_all_models.
BUNDLES = [
    "soccer_match_result",
    "nhl_moneyline",
    "nhl_regulation",
    "nhl_puck_line",
    "nhl_total",
    "nba_moneyline",
    "nba_spread",
    "nba_total",
    "nfl_moneyline",
    "nfl_spread",
    "nfl_total",
    "tennis_moneyline",
    "mma_moneyline",
]

app = modal.App("auspex-train-trial")

# Faithful-enough image: the app's own requirements.txt (numpy/pandas/sklearn/
# xgboost/lightgbm/sqlalchemy/psycopg2-binary — the exact production pins), CPU
# torch (the VM has no GPU, and the NN auto-selects cpu), the optional ONNX
# exporters (so --export-onnx exercises the full path), and a Postgres server to
# restore the dump into. libgomp1 is xgboost/lightgbm's OpenMP runtime.
#
# Postgres 17 specifically: the VM's server is PG15, but the api container's
# pg_dump is 17.10 (its Debian base ships client 17), so the nightly dump is a
# custom-format archive version 1.16 that ONLY pg_restore >= 17 can read
# (pg_restore 15 → "unsupported version (1.16) in file header"). Modal's
# debian_slim is bookworm (PG15), so pull PG17 from the PGDG apt repo. _pgbin()
# then globs /usr/lib/postgresql/*/bin and picks 17. Restoring a PG15-server
# dump into a PG17 server is forward-compatible (plain tables + rows).
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
    # The training package (services/ml-models/src → /app/ml-src). Run with
    # PYTHONPATH=/app/ml-src so `python -m training.train_all_models` resolves,
    # mirroring the VM's `cd services/ml-models; PYTHONPATH=src ...`.
    .add_local_dir("services/ml-models/src", "/app/ml-src")
)

# Staged once by the operator: the latest nightly pg_dump lands at /data/dump.dump
# (see README). Artifacts + the run summary collect in /models.
data_vol = modal.Volume.from_name("auspex-trial-data", create_if_missing=True)
models_vol = modal.Volume.from_name("auspex-trial-models", create_if_missing=True)

PGDATA = "/tmp/pgdata"
DB_URL = "postgresql://postgres@localhost:5432/trial"
DUMP_PATH = "/data/dump.dump"


def _pgbin() -> str:
    """Locate the Postgres binary dir at runtime rather than hardcoding the
    major version — Modal's debian_slim base is bookworm (PG15) today, but a
    -Fc dump restores fine into a newer server, so glob whatever got installed."""
    import glob

    hits = sorted(glob.glob("/usr/lib/postgresql/*/bin"))
    if not hits:
        raise RuntimeError("No postgresql server binaries found under /usr/lib/postgresql/*/bin")
    return hits[-1]  # highest version present


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command, streaming nothing, raising on failure unless check=False."""
    return subprocess.run(cmd, check=kw.pop("check", True), text=True, capture_output=True, **kw)


def _restore_dump() -> None:
    """Bring up an ephemeral local Postgres and restore the -Fc dump into it.

    Runs as the `postgres` system user (the server refuses to run as root).
    initdb --auth=trust so the (root) training process can connect over TCP
    without a password. --no-owner/--no-acl drop the betting_user ownership +
    grants the dump carries; a fresh DB gets every table/index/extension."""
    pgbin = _pgbin()

    def pg(*args: str, check: bool = True) -> subprocess.CompletedProcess:
        return _run(["runuser", "-u", "postgres", "--", f"{pgbin}/{args[0]}", *args[1:]], check=check)

    os.makedirs(PGDATA, exist_ok=True)
    _run(["chown", "-R", "postgres:postgres", PGDATA])
    pg("initdb", "-D", PGDATA, "--auth=trust", "--username=postgres")
    pg("pg_ctl", "-D", PGDATA, "-o", "-p 5432", "-l", "/tmp/pg.log", "-w", "start")
    pg("createdb", "-p", "5432", "trial")
    # pg_restore returns non-zero on benign warnings (skipped GRANTs, missing
    # roles); don't fail on that — verify by row count below instead.
    res = pg("pg_restore", "-p", "5432", "--no-owner", "--no-acl", "-d", "trial", DUMP_PATH, check=False)
    if res.returncode != 0:
        # Surface the tail so a REAL restore failure is debuggable, but continue.
        print(f"[pg_restore rc={res.returncode}] {res.stderr[-800:]}")
    # Sanity gate: the training queries need a populated `matches` table.
    check = pg("psql", "-p", "5432", "-d", "trial", "-tAc", "SELECT count(*) FROM matches", check=False)
    n = check.stdout.strip() if check.returncode == 0 else "ERROR"
    print(f"[restore] matches rows = {n}")
    if check.returncode != 0 or n in ("", "0", "ERROR"):
        raise RuntimeError(f"Restore looks empty/broken (matches={n!r}); aborting bundle.")


def _parse_gate(stdout: str) -> dict:
    """Pull the held-out metrics out of the `CALIBRATION GATE (name): {json}`
    line train_all_models logs for the ensemble. Returns {} if absent."""
    for line in stdout.splitlines():
        if "CALIBRATION GATE" in line and "{" in line:
            try:
                return json.loads(line[line.index("{") :])
            except (ValueError, json.JSONDecodeError):
                continue
    return {}


@app.function(
    image=image,
    volumes={"/data": data_vol, "/models": models_vol},
    cpu=4.0,
    memory=8192,
    timeout=3600,
)
def train_bundle(bundle: str) -> dict:
    """Restore the dump + train one bundle. Returns a result row. Never raises
    on a training failure — a dead bundle reports status='error' so .map keeps
    the others (mirrors the DAG's trigger_rule=ALL_DONE isolation)."""
    t0 = time.time()
    out_dir = f"/models/{bundle}"
    os.makedirs(out_dir, exist_ok=True)
    try:
        _restore_dump()
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
        elapsed = round(time.time() - t0, 1)
        gate = _parse_gate(proc.stdout + "\n" + proc.stderr)
        models_vol.commit()
        ok = proc.returncode == 0
        if not ok:
            print(f"[{bundle}] FAILED rc={proc.returncode}\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
        return {
            "bundle": bundle,
            "status": "ok" if ok else "error",
            "returncode": proc.returncode,
            "seconds": elapsed,
            "gate": gate,
        }
    except Exception as exc:  # noqa: BLE001 — isolate this bundle's failure
        return {
            "bundle": bundle,
            "status": "error",
            "returncode": None,
            "seconds": round(time.time() - t0, 1),
            "error": str(exc),
            "gate": {},
        }


@app.local_entrypoint()
def main(bundles: str = ""):
    """Fan out training across bundles IN PARALLEL (one container each).

    modal run modal_trial/train_trial.py                    # all 13
    modal run modal_trial/train_trial.py --bundles soccer_match_result
    modal run modal_trial/train_trial.py --bundles soccer_match_result,nhl_total
    """
    selected = [b.strip() for b in bundles.split(",") if b.strip()] or BUNDLES
    unknown = [b for b in selected if b not in BUNDLES]
    if unknown:
        raise SystemExit(f"Unknown bundle(s): {unknown}\nValid: {BUNDLES}")

    print(f"Training {len(selected)} bundle(s) in parallel on Modal: {selected}\n")
    wall0 = time.time()
    results = list(train_bundle.map(selected))  # parallel fan-out
    wall = round(time.time() - wall0, 1)

    ok = [r for r in results if r["status"] == "ok"]
    bad = [r for r in results if r["status"] != "ok"]

    print("\n" + "=" * 78)
    print(f"{'bundle':26} {'status':7} {'secs':>7}  held-out Brier / ECE (ensemble)")
    print("-" * 78)
    for r in sorted(results, key=lambda x: x["bundle"]):
        g = r.get("gate") or {}
        brier = g.get("brier") if isinstance(g, dict) else None
        ece = g.get("ece") if isinstance(g, dict) else None
        # _holdout_metrics may nest raw/calibrated; show whatever is present.
        metric = ""
        if brier is not None:
            metric = f"Brier={brier:.4f}" + (f"  ECE={ece:.4f}" if ece is not None else "")
        elif g:
            metric = json.dumps(g)[:40]
        print(f"{r['bundle']:26} {r['status']:7} {r['seconds']:>7}  {metric}")
    print("=" * 78)
    print(f"parallel wall-clock: {wall}s   ok={len(ok)}  errored={len(bad)}")
    if bad:
        print(f"failed bundles: {[r['bundle'] for r in bad]} (see per-container logs above)")

    # Persist the summary alongside the artifacts for later comparison.
    summary = {"wall_seconds": wall, "results": results}
    with open("/tmp/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nArtifacts + this summary are in the 'auspex-trial-models' volume.")
    print("Compare held-out Brier vs the VM baseline within the ~0.009 noise floor.")
