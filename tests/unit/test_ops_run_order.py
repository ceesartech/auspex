"""Every RUN ORDER header in scripts/ops/*.sh and the tools' docstrings names
the steps in the order scripts/ops/soccer_repair_runbook.sh runs them."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "scripts" / "ops"
RUNBOOK = OPS / "soccer_repair_runbook.sh"

# The step each wrapper name stands for, in the runbook's order.
STEPS = [
    ("repoint_team_matches", r"repoint_team_matches(\.sh)? --pass 1"),
    ("migrate_027_outcome_trigger", r"migrate_027_outcome_trigger"),
    ("merge_soccer_teams", r"merge_soccer_teams\.sh"),
    ("repair_football_data_kickoffs", r"repair_football_data_kickoffs"),
    ("repoint_team_matches", r"repoint_team_matches(\.sh)? --pass 2"),
    ("repair_stranded_recs_soccer", r"repair_stranded_recs_soccer"),
    ("cleanup_soccer_orphans", r"cleanup_soccer_orphans"),
    ("backfill_soccer_regulation_scores", r"backfill_soccer_regulation_scores"),
    ("backfill_season_labels", r"backfill_season_labels"),
    ("compute_features", r"compute_features(\.py)? --backfill-finished soccer"),
]
HEADERS = [
    OPS / "repoint_team_matches.sh",
    OPS / "migrate_027_outcome_trigger.sh",
    OPS / "merge_soccer_teams.sh",
    OPS / "repair_football_data_kickoffs.sh",
    OPS / "repair_stranded_recs_soccer.sh",
    OPS / "cleanup_soccer_orphans.sh",
    OPS / "backfill_soccer_regulation_scores.sh",
    OPS / "backfill_season_labels.sh",
    ROOT / "scripts" / "merge_soccer_teams.py",
    ROOT / "scripts" / "repair_football_data_kickoffs.py",
    ROOT / "scripts" / "repair_fixture_orphans.py",
]


def test_the_runbook_runs_the_steps_in_this_order():
    calls = re.findall(r"^run_step (\d+) \"[^\"]*\" (.*)$", RUNBOOK.read_text(), re.M)
    assert [int(n) for n, _ in calls] == list(range(1, 12))
    for (_, cmd), (name, _pat) in zip(calls, STEPS + [("step11", "")]):
        if name in ("compute_features", "step11"):
            continue
        if name == "backfill_soccer_regulation_scores":
            assert cmd == "step8"
            continue
        assert f"$OPS/{name}.sh" in cmd, cmd
    assert '"$OPS/repoint_team_matches.sh" --pass 1' in calls[0][1]
    assert '"$OPS/repoint_team_matches.sh" --pass 2' in calls[4][1]


@pytest.mark.parametrize("path", HEADERS, ids=lambda p: p.name)
def test_each_run_order_header_names_the_runbooks_order(path):
    text = path.read_text()
    start = re.search(r"^(# )?RUN ORDER", text, re.M).start()  # the header, not a mention in passing
    block = text[start : start + 2500]
    self_name = path.stem
    block = re.sub(r"\bTHIS\b", f"{self_name}.sh", block)
    pos = 0
    for _name, pat in STEPS:
        m = re.compile(pat).search(block, pos)
        assert m, f"{path.name}: {pat!r} missing or out of order in its RUN ORDER"
        pos = m.end()
