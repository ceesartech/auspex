"""scripts/mma_orphan_cleanup.py without a database: the reviewed list (its
shape, and that it names no row another MMA list holds), the per-row plan
(delete only a row exactly as reviewed: name, sport, ESPN id, no metadata, no
reference of any kind, no alias claim), the plan digest and the comparison
with the dry run's plan file, and the guarded DELETE's SQL."""

from __future__ import annotations

import csv
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("psycopg2")

REPO = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
_spec = importlib.util.spec_from_file_location("mma_orphan_cleanup_unit", SCRIPTS_DIR / "mma_orphan_cleanup.py")
moc = importlib.util.module_from_spec(_spec)
sys.modules["mma_orphan_cleanup_unit"] = moc
_spec.loader.exec_module(moc)

LIST = SCRIPTS_DIR / "data" / "mma_orphan_fighters_20261103.csv"
APPROVED = {
    "Rodrigo Vera",
    "Max Gimenis",
    "Ramiro Jimenez",
    "Uran Satybaldiev",
    "Henrique da Silva Lopes",
    "Ethyn Ewing",
    "TBA",
    "Opponent TBA",
}
TID = "1de9e9ce-308e-4543-8cf7-5537c1642616"
LEAGUE = "b62c2537-e36b-4c8a-966f-41fc7af2d07f"
CREATED = datetime(2026, 8, 30, 0, 0, 7, 299191, tzinfo=timezone.utc)


def _row(**kw):
    r = {
        "id": TID,
        "name": "Rodrigo Vera",
        "normalized_name": "rodrigo vera",
        "sport": "mma",
        "league_id": LEAGUE,
        "country": None,
        "is_active": True,
        "external_ids": {},
        "metadata": {},
        "created_at": CREATED,
        "updated_at": CREATED,
    }
    r.update(kw)
    return r


def _state(row=None, refs=None, claims=None, other=None):
    rows = {} if row is None else {row["id"]: row}
    base = {"matches.home_team_id": 0, "matches.away_team_id": 0, "teams.json": 0, "matches.json": 0}
    return moc.State(
        rows=rows,
        team_fks=(("matches", "away_team_id"), ("matches", "home_team_id")),
        refs={TID: {**base, **(refs or {})}} if row is not None else {},
        alias_claims={TID: claims or []},
        other_boutless=other or [],
    )


ORPHAN = moc.Orphan(TID, "Rodrigo Vera", "")


def _write(tmp_path, rows, header="team_id,team_name,espn_athlete_id,why"):
    p = tmp_path / "list.csv"
    p.write_text("# reviewed\n" + header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return p


# --- the committed list -------------------------------------------------------------------------------------------


def test_the_committed_list_holds_exactly_the_eight_approved_rows():
    orphans = moc.load_list(LIST)
    assert len(orphans) == 8 and {o.team_name for o in orphans} == APPROVED
    assert all(o.espn == "" for o in orphans)  # none carries an ESPN athlete id (reviewed 2026-11-02)
    with LIST.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(ln for ln in fh if not ln.startswith("#")))
    assert all(len(r["why"]) > 60 for r in rows)  # each row carries its evidence


def test_the_committed_list_names_no_row_of_another_mma_list():
    others = moc.other_list_ids()
    assert len(others) > 1300  # the identity clean-up's lists (1,299 ids, split, alias, orphan) and the merge map
    moc.check_no_overlap(moc.load_list(LIST), others)  # raises on an overlap


def test_a_row_of_another_mma_list_is_refused():
    others = moc.other_list_ids()
    stamped = next(k for k, v in others.items() if "athlete-id" in v)
    with pytest.raises(moc.ListError, match="athlete-id list"):
        moc.check_no_overlap([moc.Orphan(stamped, "Someone", "")], others)
    merged = next(k for k, v in others.items() if "merge map" in v)
    with pytest.raises(moc.ListError, match="merge map"):
        moc.check_no_overlap([moc.Orphan(merged, "Someone", "")], others)


@pytest.mark.parametrize(
    "rows, header, match",
    [
        ([f"{TID},Rodrigo Vera,,why"], "team_id,team_name,why", "missing column"),
        ([f"{TID},Rodrigo Vera,,"], None, "empty why"),
        ([f"{TID},,,why"], None, "empty team_name"),
        (["not-a-uuid,Rodrigo Vera,,why"], None, "not a UUID"),
        ([f"{TID},Rodrigo Vera,12a,why"], None, "not digits"),
        ([f"{TID},Rodrigo Vera,,why", f"{TID},Other,,why"], None, "listed twice"),
        ([f"{TID},Rodrigo Vera,,why", "39ed8231-51d5-414c-b92e-68d0dfcebb21,RODRIGO VERA,,why"], None, "twice"),
        ([], None, "no data rows"),
    ],
)
def test_a_malformed_list_is_refused(tmp_path, rows, header, match):
    p = _write(tmp_path, rows, header or "team_id,team_name,espn_athlete_id,why")
    with pytest.raises(moc.ListError, match=match):
        moc.load_list(p)


def test_a_listed_espn_id_is_kept(tmp_path):
    (o,) = moc.load_list(_write(tmp_path, [f"{TID},Rodrigo Vera,5012345,why"]))
    assert o.espn == "5012345"


# --- the plan -----------------------------------------------------------------------------------------------------


def test_a_row_as_reviewed_is_deleted_with_its_observed_values():
    p = moc.plan_row(ORPHAN, _state(_row()))
    assert p.action == "delete"
    assert p.observed == {k: _row()[k] for k in moc.OBSERVED}


def test_a_row_that_is_gone_is_done():
    assert moc.plan_row(ORPHAN, _state()).action == "done"


@pytest.mark.parametrize(
    "row, refs, claims, why",
    [
        (_row(sport="tennis"), None, None, "a tennis row"),
        (_row(name="Rodrigo Vera Jr"), None, None, "is named"),
        (_row(external_ids={"espn": "5000001"}), None, None, "ESPN athlete id 5000001"),
        (_row(metadata={"aliases": ["R. Vera"]}), None, None, "metadata ['aliases']"),
        (_row(metadata={"merged_from": []}), None, None, "metadata ['merged_from']"),
        (_row(metadata=["x"]), None, None, "not an object"),
        (_row(external_ids={"espn": "", "odds_api": "x"}), None, None, "external ids ['odds_api']"),
        (_row(), {"matches.home_team_id": 1}, None, "referenced: {'matches.home_team_id': 1}"),
        (_row(), {"players.team_id": 2}, None, "players.team_id"),
        (_row(), {"teams.json": 1}, None, "teams.json"),
        (_row(), {"matches.json": 1}, None, "matches.json"),
        (_row(), {"features_cache.json": 3}, None, "features_cache.json"),
        (_row(), None, ["0d6e0000-0000-0000-0000-000000000000"], "claim 'Rodrigo Vera' as an alias"),
    ],
)
def test_a_row_not_as_reviewed_is_refused(row, refs, claims, why):
    p = moc.plan_row(ORPHAN, _state(row, refs=refs, claims=claims))
    assert p.action == "refuse" and why in p.reason, p.reason
    assert p.observed == {}


def test_a_listed_espn_id_must_match():
    o = moc.Orphan(TID, "Rodrigo Vera", "5000001")
    assert moc.plan_row(o, _state(_row(external_ids={"espn": "5000001"}))).action == "delete"
    assert moc.plan_row(o, _state(_row())).action == "refuse"
    assert moc.plan_row(o, _state(_row(external_ids={"espn": "5000002"}))).action == "refuse"


def test_null_metadata_and_external_ids_are_as_reviewed():
    assert moc.plan_row(ORPHAN, _state(_row(metadata=None, external_ids=None))).action == "delete"


def test_the_plan_line_and_summary():
    plan = [
        moc.RowPlan(TID, "a", "delete", ""),
        moc.RowPlan("x", "b", "done", ""),
        moc.RowPlan("y", "c", "refuse", "why"),
    ]
    assert moc.plan_line(moc.summarize(plan)) == "PLAN listed=3 delete=1 done=1 refuse=1"
    assert moc.refusals(plan) == ["y ('c'): why"]


# --- digest and the dry run's plan file ---------------------------------------------------------------------------


def _doc(row=None, other=None):
    state = _state(row or _row(), other=other)
    return moc.plan_document(moc.plan_run((ORPHAN,), state), state, "dry-run")


def test_the_digest_is_stable_and_ignores_the_unlisted_bout_less_rows():
    a, b = _doc(), _doc(other=[{"id": "z", "name": "Someone"}])
    assert a["digest"] == b["digest"] and a["summary"]["delete"] == 1
    assert b["other_boutless"] == [{"id": "z", "name": "Someone"}]
    assert moc.plan_difference(a, b) is None


@pytest.mark.parametrize(
    "change",
    [
        {"updated_at": datetime(2026, 11, 3, tzinfo=timezone.utc)},
        {"metadata": None},
        {"normalized_name": "rodrigo  vera"},
        {"league_id": None},
        {"is_active": False},
    ],
)
def test_a_row_that_changed_since_the_dry_run_changes_the_digest(change):
    a, b = _doc(), _doc(_row(**change))
    assert a["digest"] != b["digest"]
    assert "changed since the dry run" in moc.plan_difference(a, b)


def test_plan_difference_names_an_action_change_and_a_foreign_plan():
    a = _doc()
    refused = moc.plan_document(
        moc.plan_run((ORPHAN,), _state(_row(), refs={"matches.away_team_id": 1})), _state(_row()), "apply"
    )
    assert "refuse, the expected plan says delete" in moc.plan_difference(a, refused)
    assert "not a mma_orphan_cleanup plan" in moc.plan_difference({"tool": "mma_identity_cleanup"}, a)
    assert moc.plan_difference({**a, "digest": "0" * 64}, a) == "the plan digest differs from the expected plan"
    assert "is not in the expected plan" in moc.plan_difference({**a, "digest": "0", "rows": []}, a)


# --- the guarded DELETE -------------------------------------------------------------------------------------------


def _as_text(composed) -> str:
    """The composed SQL with identifiers bare (no connection needed)."""
    parts = []
    for p in composed.seq if hasattr(composed, "seq") else [composed]:
        if hasattr(p, "seq"):
            parts.append(_as_text(p))
        elif hasattr(p, "strings"):
            parts.append(".".join(p.strings))
        else:
            parts.append(p.string)
    return "".join(parts)


def test_the_delete_is_guarded_on_every_observed_column_and_every_foreign_key():
    fks = (("matches", "home_team_id"), ("matches", "away_team_id"), ("players", "team_id"))
    text = _as_text(moc.delete_sql(fks))
    for col in ("name", "normalized_name", "league_id", "country", "is_active", "external_ids", "metadata"):
        assert f"t.{col}" in text, col
    assert "t.created_at IS NOT DISTINCT FROM" in text and "t.updated_at IS NOT DISTINCT FROM" in text
    assert "t.sport = 'mma'" in text
    for tbl, col in fks:
        assert f"NOT EXISTS (SELECT 1 FROM {tbl} x WHERE x.{col} = t.id)" in text
    params = moc.delete_params(moc.plan_row(ORPHAN, _state(_row())))
    assert params["external_ids"] == "{}" and params["metadata"] == "{}" and params["created_at"] == CREATED
    assert moc.delete_params(moc.plan_row(ORPHAN, _state(_row(metadata=None))))["metadata"] is None


def test_the_audit_rows_carry_the_observed_row():
    (r,) = moc.csv_rows([moc.plan_row(ORPHAN, _state(_row()))])
    assert r["unit"] == "orphan" and r["action"] == "delete" and r["row_id"] == TID and '"rodrigo vera"' in r["before"]


def test_expect_plan_without_apply_is_refused(capsys):
    assert moc.main(["--expect-plan", "/nonexistent.json", "--database-url", "postgresql:///x"]) == 2


def test_no_database_url_is_refused(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert moc.main(["--database-url", ""]) == 2


def test_main_refuses_a_list_naming_a_row_of_another_mma_list_before_connecting(tmp_path):
    stamped = next(k for k, v in moc.other_list_ids().items() if "athlete-id" in v)
    lst = _write(tmp_path, [f"{stamped},Someone,,why"])
    # an unreachable database: the refusal comes before any connection
    assert moc.main(["--list", str(lst), "--database-url", "postgresql://nobody@127.0.0.1:1/none"]) == 2
