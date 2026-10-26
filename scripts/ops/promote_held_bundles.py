"""Promote bundles the promote gate sent to a human (MANUAL DECISION NEEDED),
through the gate's own merge + sidecar code path. Backs production up first.

Operator tool. Approved 2026-10-26 for the four NBA/NFL spread/total bundles of
run manual__2026-10-26T1544231974660000 (evidence in
scripts/ops/evidence_20261026_held_bundles.json: none is distinguishable from
its incumbent on the challenger's held-out rows the incumbent never saw):

    cd /opt/auspex
    docker compose exec -T api python /app/scripts/ops/promote_held_bundles.py \\
        manual__2026-10-26T1544231974660000 nba_spread,nba_total,nfl_spread,nfl_total \\
        /app/scripts/ops/evidence_20261026_held_bundles.json
    docker compose -f /opt/auspex/docker-compose.yml restart api

Undo: copy the printed production-prev-*-manual backup back over production and
restart the api.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/app/scripts")

import model_metrics_store as mstore  # noqa: E402
from pull_modal_artifacts import _merge_bundle_into_staging  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    run_id, bundles_arg, evidence_path = argv[1], argv[2], argv[3]
    bundles = [b.strip() for b in bundles_arg.split(",") if b.strip()]
    evidence = json.loads(Path(evidence_path).read_text())

    models = Path("/app/models")
    prod = models / "production"
    incoming = models / "modal-incoming" / run_id
    summary = json.loads((prod / "promote_decisions.json").read_text())
    if summary.get("run_id") != run_id:
        print(f"production/promote_decisions.json is for run {summary.get('run_id')!r}, not {run_id!r} — refusing")
        return 1
    decisions = {d["bundle"]: d for d in summary["decisions"]}
    for b in bundles:
        if b not in decisions or not (incoming / b).is_dir():
            print(f"{b}: no decision row or no artifacts under {incoming} — refusing")
            return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = models / f"production-prev-{stamp}-manual"
    shutil.copytree(prod, backup, symlinks=True)
    print("backup:", backup)

    for b in bundles:
        row = decisions[b]
        _merge_bundle_into_staging(incoming / b, prod)  # rm-then-cp per item; gate.json skipped
        payload = mstore.build_payload(
            row["ensemble_name"],
            served_brier=row["challenger_brier"],
            kept=row["kept_calibration"],
            n=row["n"],
            run_id=run_id,
            fingerprint=row["challenger_fingerprint"],
            previous=mstore.read_incumbent(str(backup), row["ensemble_name"]),
            derived=row.get("derived_metric"),
        )
        mstore.write_sidecar(str(prod), payload)
        mstore.mirror_to_db(os.environ.get("DATABASE_URL"), payload)
        print("promoted", b, "->", row["ensemble_name"])

    record = {
        "run_id": run_id,
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "why_manual": "the gate could not build a shared set (frames gained rows inside the incumbent date range)",
        "rule": "modal_bundles.paired_decision on the challenger's held-out rows the incumbent never saw",
        "evidence": {b: evidence.get(b) for b in bundles},
        "backup": str(backup),
    }
    (prod / f"manual_promote_{stamp}.json").write_text(json.dumps(record, indent=2))
    print("record written; restart the api to serve the new artifacts")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
