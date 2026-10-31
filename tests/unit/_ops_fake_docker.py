"""A scripted stand-in for `docker compose` on the VM, for running the
scripts/ops wrappers (and the runbook) for real in unit tests: no database, no
containers, no Airflow.

`install(tmp_path, rules, airflow=...)` writes `bin/docker` (this file, run as
a script), a no-op `sleep`, and the rules; `run(...)` runs a wrapper with that
bin first on PATH. Every docker call is logged to calls.log, one per line.

* Airflow (`exec -T airflow-scheduler ...`) is a small Airflow 3.3 model kept
  in airflow.json: per DAG `paused` and `runs` (each {run_id, state, ticks,
  start_date}). Every `dags list-runs` call is one scheduler tick for that DAG,
  and — as in 3.3 — a PAUSED DAG's runs never advance: unpaused, queued ->
  running and a running run counts its ticks down to success. `slip_in`
  {dag: n} queues a new run on that DAG's n-th pause (the run that slips in
  between a drain and a pause). `trigger db_backup_daily` lands a fake dump
  (see b2) unless `backup_fails` is set.
* `compose cp <file> postgres:/tmp/<name>` copies the file to cp/<name> and
  cp/<n>_<name> (so a test can read every SQL a wrapper sent, in order).
* Everything else is matched against the rules in order: {"match": regex
  over the joined argv, "out": stdout, "rc": exit code, "times": n (optional:
  the rule then retires)}. A call no rule matches exits 99 and is logged as
  UNMATCHED.
* The B2 probe (`api python -c ... b2_io ...`) prints the last landed dump.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.000000+00:00", time.gmtime())


def _airflow(state_path: Path, cmd: str, fake: Path) -> tuple[str, int]:
    st = json.loads(state_path.read_text())
    dags = st.setdefault("dags", {})

    def dag(name: str) -> dict:
        return dags.setdefault(name, {"paused": False, "runs": []})

    out, rc = "", 0
    m = re.search(r"airflow dags (details|list-runs|pause|unpause|trigger)\s+(?:-y\s+)?(\S+)(.*)", cmd)
    if not m:
        return "", 2
    verb, name, rest = m.group(1), m.group(2), m.group(3)
    d = dag(name)
    if verb == "details":
        out = json.dumps([{"dag_id": name, "is_paused": "True" if d["paused"] else "False"}])
    elif verb == "list-runs":
        if not d["paused"]:
            for r in d["runs"]:
                if r["state"] == "queued":
                    r["state"], r["start_date"] = "running", _now_iso()
                elif r["state"] == "running":
                    r["ticks"] -= 1
                    if r["ticks"] <= 0:
                        r["state"], r["end_date"] = "success", _now_iso()
        sm = re.search(r"--state (\S+)", rest)
        want = sm.group(1) if sm else None
        out = json.dumps(
            [
                {"dag_id": name, "run_id": r["run_id"], "state": r["state"], "start_date": r.get("start_date")}
                for r in d["runs"]
                if want is None or r["state"] == want
            ]
        )
    elif verb == "pause":
        d["paused"] = True
        d["pauses"] = d.get("pauses", 0) + 1
        if st.get("slip_in", {}).get(name) == d["pauses"]:
            d["runs"].append({"run_id": f"slipped_{d['pauses']}", "state": "queued", "ticks": 1})
    elif verb == "unpause":
        d["paused"] = False
        for r in st.get("on_unpause", {}).get(name, []):
            d["runs"].append(dict(r, start_date=_now_iso()))
        st.get("on_unpause", {}).pop(name, None)
    elif verb == "trigger":
        st["triggers"] = st.get("triggers", 0) + 1
        if name == "db_backup_daily" and not st.get("backup_fails"):
            st["dump"] = f"{int(time.time()) + 5} postgres/fake_{st['triggers']}.dump"
    state_path.write_text(json.dumps(st))
    return out, rc


def main(argv: list[str]) -> int:
    fake = Path(os.environ["FAKE_DIR"])
    joined = " ".join(argv)
    with (fake / "calls.log").open("a") as fh:
        fh.write(joined.replace("\n", "\\n") + "\n")
    if len(argv) >= 4 and argv[1] == "cp" and argv[3].startswith("postgres:"):
        (fake / "cp").mkdir(exist_ok=True)
        name = Path(argv[3].split(":", 1)[1]).name
        shutil.copy(argv[2], fake / "cp" / name)
        shutil.copy(argv[2], fake / "cp" / f"{len(list((fake / 'cp').glob('[0-9]*')))}_{name}")
        return 0
    if "airflow-scheduler" in argv:
        out, rc = _airflow(fake / "airflow.json", joined, fake)
        if out:
            print(out)
        return rc
    if " api python -c " in f" {joined} " and "b2_io" in joined:
        st = json.loads((fake / "airflow.json").read_text())
        print(st.get("dump", "0 none"))
        return 0
    rules_path = fake / "rules.json"
    rules = json.loads(rules_path.read_text())
    for rule in rules:
        if rule.get("times", 1) <= 0:
            continue
        if re.search(rule["match"], joined, re.S):
            if "times" in rule:
                rule["times"] -= 1
                rules_path.write_text(json.dumps(rules))
            if rule.get("out"):
                print(rule["out"])
            return int(rule.get("rc", 0))
    with (fake / "calls.log").open("a") as fh:
        fh.write("UNMATCHED " + joined.replace("\n", "\\n") + "\n")
    return 99


def install(tmp_path: Path, rules: list[dict], airflow: dict | None = None) -> dict:
    """bin/ (docker, sleep), fake/ (rules.json, airflow.json); returns the env to run with."""
    bin_dir, fake = tmp_path / "bin", tmp_path / "fake"
    bin_dir.mkdir(exist_ok=True)
    fake.mkdir(exist_ok=True)
    (bin_dir / "docker").write_text(
        f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "import _ops_fake_docker\nsys.exit(_ops_fake_docker.main(sys.argv[1:]))\n"
    )
    (bin_dir / "sleep").write_text("#!/bin/sh\nexit 0\n")
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    (fake / "rules.json").write_text(json.dumps(rules))
    (fake / "airflow.json").write_text(json.dumps(airflow or {}))
    (tmp_path / "auspex").mkdir(exist_ok=True)
    (tmp_path / "tmp").mkdir(exist_ok=True)
    return {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "FAKE_DIR": str(fake),
        "AUSPEX_DIR": str(tmp_path / "auspex"),
        "TMPDIR": str(tmp_path / "tmp"),
        "AF_POLL_SECONDS": "0",
    }


def run(script: Path, env: dict, *args: str, cwd: Path | None = None, timeout: int = 120) -> tuple[int, str]:
    proc = subprocess.run(
        ["bash", str(script), *args],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )
    return proc.returncode, proc.stdout + proc.stderr


def calls(tmp_path: Path) -> list[str]:
    log = tmp_path / "fake" / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


def airflow_state(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "fake" / "airflow.json").read_text())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
