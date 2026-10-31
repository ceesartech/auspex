"""restore_cascaded_recs: which deleted recs may be re-inserted, and with
what. Pure pieces only — no DB, no pg_restore."""

from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rcr = _load("restore_cascaded_recs", "restore_cascaded_recs.py")


def _audit(**kw):
    row = {
        "rec_id": "r1",
        "match_id": "orphan",
        "new_match_id": "twin",
        "twin_home_score": "0",
        "twin_away_score": "1",
        "outcome": "won",
        "status": "won",
        "profit_loss": "50.284800",
        "action": "settle",
    }
    return {**row, **kw}


def _rec(**kw):
    row = {
        "id": "r1",
        "match_id": "orphan",
        "prediction_id": "p1",
        "status": "pending",
        "bet_type": "moneyline",
        "selection": "away",
        "recommended_stake": Decimal("46.56"),
        "odds_at_recommendation": Decimal("2.08"),
    }
    return {**row, **kw}


PRED = {
    "id": "p1",
    "match_id": "orphan",
    "model_name": "ensemble_tennis_ml",
    "model_version": "v",
    "prediction_type": "moneyline",
}
TWIN = {"id": "twin", "status": "finished", "home_score": 0, "away_score": 1}


class TestCopyBlock:
    DUMP = (
        "SET x = 1;\n"
        "COPY public.betting_recommendations (id, match_id, reasoning) FROM stdin;\n"
        "r1\torphan\tline one\\nline two\n"
        "r2\tother\t\\N\n"
        "\\.\n"
        "COPY public.other (id) FROM stdin;\nr1\n\\.\n"
    )

    def test_keeps_only_wanted_rows_verbatim(self):
        cols, kept = rcr.copy_block(self.DUMP, "betting_recommendations", {"r1"})
        assert cols == ["id", "match_id", "reasoning"]
        assert kept == ["r1\torphan\tline one\\nline two"]  # escapes left for COPY to parse

    def test_missing_table_is_loud(self):
        with pytest.raises(RuntimeError, match="no COPY section"):
            rcr.copy_block("SET x = 1;\n", "betting_recommendations", {"r1"})

    def test_truncated_section_is_loud(self):
        with pytest.raises(RuntimeError, match="truncated"):
            rcr.copy_block(
                "COPY public.betting_recommendations (id) FROM stdin;\nr1\n", "betting_recommendations", {"r1"}
            )


class TestCheckRow:
    def test_a_faithful_restore_passes(self):
        assert rcr.check_row(_audit(), _rec(), PRED, TWIN) is None

    def test_lost_side_regrades_too(self):
        audit = _audit(outcome="lost", status="lost", profit_loss="-46.56")
        assert rcr.check_row(audit, _rec(selection="home"), PRED, TWIN) is None

    @pytest.mark.parametrize(
        "rec, pred, twin, reason",
        [
            (None, PRED, TWIN, "not_in_dump"),
            (_rec(status="won"), PRED, TWIN, "dump_row_not_the_repaired_one"),
            (_rec(match_id="elsewhere"), PRED, TWIN, "dump_row_not_the_repaired_one"),
            (_rec(), None, TWIN, "prediction_not_archived"),
            (_rec(), {**PRED, "match_id": "elsewhere"}, TWIN, "prediction_not_archived"),
            (_rec(), PRED, None, "twin_not_finished"),
            (_rec(), PRED, {**TWIN, "status": "scheduled"}, "twin_not_finished"),
            (_rec(), PRED, {**TWIN, "home_score": 2}, "twin_score_changed"),
        ],
    )
    def test_refusals(self, rec, pred, twin, reason):
        assert rcr.check_row(_audit(), rec, pred, twin) == reason

    def test_a_settlement_grading_no_longer_reproduces_is_refused(self):
        assert rcr.check_row(_audit(), _rec(selection="home"), PRED, TWIN) == "regrade_disagrees"
        assert rcr.check_row(_audit(profit_loss="99.00"), _rec(), PRED, TWIN) == "regrade_disagrees"


class TestInsertSql:
    def test_overrides_only_the_settlement_and_links(self):
        cols = [
            "id",
            "prediction_id",
            "match_id",
            "bet_type",
            "status",
            "actual_result",
            "profit_loss",
            "settled_at",
            "created_at",
            "updated_at",
        ]
        sql = rcr.insert_sql(cols)
        select = sql.split("SELECT ", 1)[1].split(" FROM restore_recs", 1)[0].split(", ")
        assert select == [
            "id",
            "%(prediction_id)s::uuid",
            "%(match_id)s::uuid",
            "bet_type",
            "%(status)s",
            "%(outcome)s",
            "%(profit_loss)s::numeric",
            "NOW()",
            "created_at",
            "NOW()",
        ]
        assert sql.endswith("ON CONFLICT (id) DO NOTHING")


class TestAuditCsv:
    def test_write_once(self, tmp_path):
        out = tmp_path / "restored.csv"
        rcr.write_csv(str(out), [])
        with pytest.raises(FileExistsError):
            rcr.write_csv(str(out), [])

    def test_default_follows_backup_local_dir(self, monkeypatch):
        monkeypatch.setenv("BACKUP_LOCAL_DIR", "/srv/b")
        assert rcr.default_out_csv().startswith("/srv/b/restored_recs_")


class TestArgs:
    def test_dry_run_is_the_default(self):
        args = rcr.parse_args(["--audit-csv", "a.csv", "--dump", "d.dump", "--archive", "x.jsonl.gz"])
        assert args.apply is False

    def test_all_three_inputs_are_required(self):
        with pytest.raises(SystemExit):
            rcr.parse_args(["--audit-csv", "a.csv", "--dump", "d.dump"])
