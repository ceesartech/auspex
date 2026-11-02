"""scripts/settle_mma_held_rows.py without a database: the reviewed list, the
plan keys, --expect-plan and the CLI's refusals. The end-to-end runs against a
real Postgres are in test_settle_mma_held_rows_pg.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


st = _load("settle_mma_held_rows", "settle_mma_held_rows.py")

REVIEWED_PLAN = (
    "PLAN record=1 record_done=0 record_predictions=2 record_recs_void=0 delete=1 delete_done=0 "
    "delete_predictions=2 delete_odds=0 delete_snapshots=0 delete_features=2 delete_other_rows=0 teams_delete=1 "
    "refuse=0"
)
DONE_PLAN = (
    "PLAN record=0 record_done=1 record_predictions=0 record_recs_void=0 delete=0 delete_done=1 "
    "delete_predictions=0 delete_odds=0 delete_snapshots=0 delete_features=0 delete_other_rows=0 teams_delete=0 "
    "refuse=0"
)


def _totals(line: str) -> dict[str, int]:
    return {k: int(v) for k, v in (kv.split("=") for kv in line.split()[1:])}


class TestTheReviewedList:
    def test_it_holds_exactly_the_two_held_rows(self):
        entries = st.load_list()
        assert [(e.match_id, e.action, e.expect) for e in entries] == [
            ("b7f16d27-42a1-41b0-a6f6-6e8e9837da47", "record_no_winner", "draw"),
            ("787bd68e-2ce8-4beb-b346-c9f422c42574", "delete_replaced", "replaced"),
        ]
        gatto, machado = entries
        assert (gatto.espn_event_id, gatto.espn_competition_id, gatto.kickoff.isoformat()) == (
            "600061541",
            "401924511",
            "2026-10-10T21:00:00+00:00",
        )
        assert (machado.replaced_name, machado.replaced_by, machado.opponent_name) == (
            "Valesca Machado",
            "Tina Black",
            "Melissa Amaya",
        )
        assert machado.replacement_match_id == "588fc028-7a2d-4fa1-a061-6ebd2e9aa15e"
        assert len(gatto.prediction_ids) == len(machado.prediction_ids) == 2

    def test_they_are_the_rows_run_b_holds(self):
        hold = (SCRIPTS_DIR / "data" / "mma_stale_rows_hold_20261101.csv").read_text()
        for e in st.load_list():
            assert e.match_id in hold and e.espn_competition_id in hold

    def test_none_of_them_is_an_identity_streams_row(self):
        # Bruno Silva, Raffael/Rafael Cerqueira and José Luiz / José Montanha belong to the mma_identity stream.
        text = (SCRIPTS_DIR / "data" / "mma_held_rows_settle_20261102.csv").read_text().lower()
        for name in ("bruno silva", "cerqueira", "jose luiz", "josé luiz", "montanha"):
            assert name not in text


HEADER = ",".join(st.LIST_FIELDS)
GOOD_RECORD = (
    "b7f16d27-42a1-41b0-a6f6-6e8e9837da47,record_no_winner,draw,600061541,401924511,2026-10-10,"
    "2026-10-10T21:00:00+00:00,A,B,,,,,eeb51fd3-2b59-407a-85c4-638a40930834,evidence"
)
GOOD_DELETE = (
    "787bd68e-2ce8-4beb-b346-c9f422c42574,delete_replaced,replaced,600061266,401914464,2026-09-26,"
    "2026-09-27T00:00:00+00:00,C,D,home,108423d1-7f54-43ba-8039-43e5ef23b8bf,E,"
    "588fc028-7a2d-4fa1-a061-6ebd2e9aa15e,f9fa6232-90a1-4de6-a679-c70d946e33ce,evidence"
)


class TestListRefusals:
    def _load(self, tmp_path, *rows):
        p = tmp_path / "list.csv"
        p.write_text("# comment\n" + HEADER + "\n" + "\n".join(rows) + "\n")
        return st.load_list(p)

    def test_good_rows_load(self, tmp_path):
        assert len(self._load(tmp_path, GOOD_RECORD, GOOD_DELETE)) == 2

    @pytest.mark.parametrize(
        "row,why",
        [
            (GOOD_RECORD.replace("record_no_winner", "merge"), "unknown action"),
            (GOOD_RECORD.replace(",draw,", ",win,"), "expects one of"),
            (GOOD_RECORD.replace("b7f16d27-42a1", "zzz"), "not a uuid"),
            (GOOD_RECORD.replace(",evidence", ","), "no evidence"),
            (GOOD_RECORD.replace("+00:00", ""), "no time zone"),
            (GOOD_RECORD.replace(",,,,,", ",home,,,,"), "takes no replaced_"),
            (GOOD_DELETE.replace(",home,", ",,"), "needs replaced_side"),
            (GOOD_DELETE.replace("108423d1-7f54", "nope"), "not a uuid"),
            (GOOD_RECORD.replace(",600061541,", ",,"), "empty espn_event_id"),
            (GOOD_DELETE.replace(",E,", ",C,"), "is one of the row's own fighters"),
            (GOOD_DELETE.replace(",E,", ",D,"), "is one of the row's own fighters"),
        ],
    )
    def test_a_bad_row_refuses(self, tmp_path, row, why):
        with pytest.raises(st.ListError, match=why):
            self._load(tmp_path, row)

    def test_a_row_listed_twice_refuses(self, tmp_path):
        with pytest.raises(st.ListError, match="listed twice"):
            self._load(tmp_path, GOOD_RECORD, GOOD_RECORD)

    def test_an_empty_list_refuses(self, tmp_path):
        with pytest.raises(st.ListError, match="no rows"):
            self._load(tmp_path)


class TestPlanKeys:
    def test_the_plan_line_shape(self):
        assert st.plan_line(_totals(REVIEWED_PLAN)) == REVIEWED_PLAN

    def test_done(self):
        entries = st.load_list()
        assert st.is_done(_totals(DONE_PLAN), entries)
        assert not st.is_done(_totals(REVIEWED_PLAN), entries)
        for key in ("record_predictions", "record_recs_void", "teams_delete", "refuse"):
            assert not st.is_done({**_totals(DONE_PLAN), key: 1}, entries)
        assert not st.is_done({**_totals(DONE_PLAN), "delete_done": 0}, entries)

    def test_expect_plan_needs_every_key(self):
        full = ",".join(REVIEWED_PLAN.split()[1:])
        assert st.parse_expect(full) == _totals(REVIEWED_PLAN)
        with pytest.raises(ValueError, match="every key"):
            st.parse_expect("record=1")
        with pytest.raises(ValueError, match="bad entry"):
            st.parse_expect(full + ",bogus=1")
        with pytest.raises(ValueError, match="bad entry"):
            st.parse_expect(full.replace("record=1", "record=x"))

    def test_an_unsettled_entry_plan_counts(self):
        e = st.load_list()[1]
        p = st.EntryPlan(
            e, delete=True, children={"predictions": ["a", "b"], "features_cache": ["c"], "weather": ["d"]}
        )
        t = st.summarize([p])
        assert (t["delete"], t["delete_predictions"], t["delete_features"], t["delete_other_rows"]) == (1, 2, 1, 1)
        assert st.summarize([st.EntryPlan(e, refusal="x", delete=True)])["refuse"] == 1


class TestCli:
    def test_apply_needs_expect_plan_and_the_reverse(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql:///nowhere")
        assert st.main(["--apply"]) == 2
        assert st.main(["--expect-plan", "record=1"]) == 2

    def test_check_after_takes_no_apply(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql:///nowhere")
        assert st.main(["--check-after", "--apply"]) == 2

    def test_a_bad_expect_plan_refuses_before_connecting(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql:///nowhere")
        assert st.main(["--apply", "--expect-plan", "record=1"]) == 2

    def test_write_once_audit_files(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DATABASE_URL", "postgresql:///nowhere")
        taken = tmp_path / "a.csv"
        taken.write_text("x")
        assert st.main(["--out-csv", str(taken)]) == 2
        full = ",".join(REVIEWED_PLAN.split()[1:])
        arch = tmp_path / "a.jsonl.gz"
        arch.write_text("x")
        assert (
            st.main(["--apply", "--expect-plan", full, "--out-csv", str(tmp_path / "b.csv"), "--archive", str(arch)])
            == 2
        )

    def test_no_database_url(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert st.main([]) == 2
