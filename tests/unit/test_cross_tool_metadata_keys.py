"""The matches.metadata keys three tools share, written and read on separate
code paths: fetch_upcoming records ESPN's postponements, repair_fixture_orphans
marks ESPN-verified postponed stale rows and retires fixture forks
(orphan_of), and grade_completed_matches voids recs by those keys. A key
spelled differently on one side would leave recs open without a word, so the
spellings are pinned against the grader's constants."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fu = _load("fetch_upcoming_metadata_keys", "fetch_upcoming.py")
gc = _load("grade_completed_matches_metadata_keys", "grade_completed_matches.py")
rfo = _load("repair_fixture_orphans_metadata_keys", "repair_fixture_orphans.py")


def test_fetch_upcoming_stamps_the_keys_the_grader_reads():
    assert (fu.POSTPONED_KICKOFF_KEY, fu.POSTPONED_AT_KEY) == (gc.POSTPONED_KICKOFF_KEY, gc.POSTPONED_AT_KEY)


def test_repair_fixture_orphans_marks_postponements_with_the_graders_keys():
    sql = rfo.MARK_POSTPONED_SQL
    assert f"'{gc.POSTPONED_KICKOFF_KEY}', %(postponed_kickoff)s" in sql
    assert f"'{gc.POSTPONED_AT_KEY}', %(postponed_at)s" in sql
    assert f"? '{gc.POSTPONED_KICKOFF_KEY}'" in sql  # stamped once, as fetch_upcoming does


def test_a_retired_fork_carries_the_key_the_grader_holds_its_recs_by():
    assert f"'{gc.ORPHAN_OF_KEY}', %(twin)s" in rfo.CANCEL_PINNED_SQL
