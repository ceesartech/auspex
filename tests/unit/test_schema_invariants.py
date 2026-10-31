"""Schema decisions that must not quietly revert. Static — no DB."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = ROOT / "services" / "data-ingestion" / "db" / "migrations"


def _last_definition(constraint: str) -> str:
    """The FK clause of the highest-numbered migration that (re)defines it."""
    last = ""
    for path in sorted(MIGRATIONS.glob("[0-9][0-9][0-9]_*.sql")):
        sql = re.sub(r"--[^\n]*", "", path.read_text())
        for m in re.finditer(rf"ADD CONSTRAINT {constraint}\s+(FOREIGN KEY[^;,]*)", sql):
            last = m.group(1)
    return last


def _strip_sql_comments(sql: str) -> str:
    # One left-to-right pass, so whichever comment opens first wins.
    return re.sub(r"--[^\n]*|/\*.*?\*/", "", sql, flags=re.S)


def _loose_definitions(sql: str, name: str) -> int:
    """How many statements (re)define function `name`, however spelled: any
    case, any whitespace, any schema, quoted or not, with or without OR
    REPLACE. CREATE TRIGGER ... EXECUTE FUNCTION name() is a use, not one."""
    ident = rf'(?:"?\w+"?\s*\.\s*)?"?{name}"?'
    return len(re.findall(rf"\bCREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+{ident}\s*\(", sql, re.I))


def _strict_definitions(sql: str, name: str) -> list[str]:
    """The whitespace-collapsed body of every plpgsql trigger function `name`
    defined in `sql` (comments already stripped), in file order. Case-
    insensitive; `name` or `public.name`; any dollar-quote tag; LANGUAGE
    plpgsql before AS or after the body, exactly once. Anything else is left
    to _loose_definitions to notice."""
    pattern = (
        rf"\bCREATE\s+OR\s+REPLACE\s+FUNCTION\s+(?:public\s*\.\s*)?{name}\s*\(\s*\)\s*"
        r"RETURNS\s+TRIGGER\s+(LANGUAGE\s+plpgsql\s+)?AS\s+(\$(?:[A-Za-z_]\w*)?\$)(.*?)(?-i:\2)"
        r"\s*(LANGUAGE\s+plpgsql\s*)?;"
    )
    return [
        " ".join(m.group(3).split())
        for m in re.finditer(pattern, sql, re.I | re.S)
        if bool(m.group(1)) != bool(m.group(4))
    ]


def _function_definitions(name: str) -> tuple[list[tuple[str, str]], list[str]]:
    """((migration file, body) for every definition of trigger function `name`
    the strict parse reads, and the file of every definition the loose pattern
    sees, one entry each), both in migration order. The two must agree, or a
    redefinition the strict parse cannot read would leave the guard checks
    reading an older body."""
    strict: list[tuple[str, str]] = []
    loose: list[str] = []
    for path in sorted(MIGRATIONS.glob("[0-9][0-9][0-9]_*.sql")):
        sql = _strip_sql_comments(path.read_text())
        strict += [(path.name, body) for body in _strict_definitions(sql, name)]
        loose += [path.name] * _loose_definitions(sql, name)
    return strict, loose


def _str_constant(path: Path, name: str) -> str:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
            return node.value.value
    raise AssertionError(f"{name} not assigned in {path}")


def test_deleting_a_prediction_never_cascades_to_its_recs():
    # 2026-10-31: the CASCADE silently deleted 45 settled tennis recs (migration 025)
    clause = _last_definition("betting_recommendations_prediction_id_fkey")
    assert "REFERENCES predictions(id)" in clause and "ON DELETE NO ACTION" in clause
    model = (ROOT / "services" / "api" / "src" / "models" / "database_models.py").read_text()
    assert 'ForeignKey("predictions.id", ondelete="NO ACTION")' in model


def test_deleting_a_match_never_cascades_to_its_recs():
    # migration 026: a fixture with recs must have them re-pointed or removed first, explicitly
    clause = _last_definition("betting_recommendations_match_id_fkey")
    assert "REFERENCES matches(id)" in clause and "ON DELETE NO ACTION" in clause
    model = (ROOT / "services" / "api" / "src" / "models" / "database_models.py").read_text()
    assert 'ForeignKey("matches.id", ondelete="NO ACTION")' in model


def test_outcome_trigger_never_settles_rows_written_after_kickoff():
    # migration 027: the trigger used to mark post-kickoff recs won/lost and grade
    # post-kickoff predictions, which the Python grader voids / skips. Each guard
    # must sit in its own UPDATE, with the grader's comparison (> for
    # predictions, >= for recs).
    strict, loose = _function_definitions("calculate_match_outcome")
    assert [
        f for f, _ in strict
    ] == loose, "a migration defines calculate_match_outcome() in a form the parser cannot read"
    assert strict, "no migration defines calculate_match_outcome()"
    migration, body = strict[-1]
    preds = re.search(r"UPDATE predictions p .*?;", body)
    recs = re.search(r"UPDATE betting_recommendations br .*?;", body)
    assert preds and recs, migration
    assert "AND (p.created_at > NEW.match_date) IS NOT TRUE" in preds.group(0), migration
    assert "AND (br.created_at >= NEW.match_date) IS NOT TRUE" in recs.group(0), migration


def test_accuracy_rollup_drops_the_graders_post_kickoff_void_label():
    # The API image cannot import scripts/, so the actual_result prefix is
    # duplicated; a drift would quietly put hindsight voids back into ROI.
    grader = _str_constant(ROOT / "scripts" / "grade_completed_matches.py", "POST_KICKOFF_VOID_RESULT")
    api = _str_constant(ROOT / "services" / "api" / "src" / "routes" / "accuracy.py", "POST_KICKOFF_VOID_PREFIX")
    assert api == grader == "void: written after kickoff"


# ── the definition parser itself ─────────────────────────────────────

_FN = "calculate_match_outcome"
_GUARDED = "BEGIN UPDATE predictions p SET x = 1 WHERE (p.created_at > NEW.match_date) IS NOT TRUE; RETURN NEW; END;"
_CREATE = "CREATE OR REPLACE FUNCTION"


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(f"{_CREATE} {_FN}()\nRETURNS TRIGGER AS $$\n{_GUARDED}\n$$ LANGUAGE plpgsql;", id="repo-style"),
        pytest.param(
            f"create or replace function {_FN}() returns trigger as $$ {_GUARDED} $$ language plpgsql;", id="lowercase"
        ),
        pytest.param(
            f"{_CREATE} public.{_FN}() RETURNS TRIGGER AS $$ {_GUARDED} $$ LANGUAGE plpgsql;", id="public-prefix"
        ),
        pytest.param(
            f"{_CREATE} {_FN}() RETURNS TRIGGER AS $function$ {_GUARDED} $function$ LANGUAGE plpgsql;",
            id="function-tag",
        ),
        pytest.param(
            f"{_CREATE} {_FN} ( ) RETURNS trigger LANGUAGE plpgsql AS $body$ {_GUARDED} $body$;", id="language-first"
        ),
    ],
)
def test_definition_parser_reads_every_spelling(sql):
    assert _loose_definitions(sql, _FN) == 1
    assert _strict_definitions(sql, _FN) == [_GUARDED]


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(f"CREATE FUNCTION {_FN}() RETURNS TRIGGER AS $$ x $$ LANGUAGE plpgsql;", id="no-or-replace"),
        pytest.param(f"{_CREATE} auspex.{_FN}() RETURNS TRIGGER AS $$ x $$ LANGUAGE plpgsql;", id="other-schema"),
        pytest.param(f'{_CREATE} "{_FN}"() RETURNS TRIGGER AS $$ x $$ LANGUAGE plpgsql;', id="quoted"),
        pytest.param(f"{_CREATE} {_FN}() RETURNS TRIGGER AS 'x' LANGUAGE plpgsql;", id="single-quoted-body"),
        pytest.param(
            f"{_CREATE} {_FN}() RETURNS TRIGGER SECURITY DEFINER AS $$ x $$ LANGUAGE plpgsql;", id="extra-attribute"
        ),
        pytest.param(
            f"{_CREATE} {_FN}() RETURNS TRIGGER LANGUAGE plpgsql AS $$ x $$ LANGUAGE plpgsql;", id="language-twice"
        ),
        pytest.param(f"{_CREATE} {_FN}() RETURNS TRIGGER AS $a$ x $b$ LANGUAGE plpgsql;", id="mismatched-tags"),
        pytest.param(f"{_CREATE} {_FN}() RETURNS TRIGGER AS $a$ x $A$ LANGUAGE plpgsql;", id="tag-case-differs"),
    ],
)
def test_definition_parser_counts_what_it_cannot_read(sql):
    # The invariant test then fails on loose != strict instead of checking an older body.
    assert _loose_definitions(sql, _FN) == 1
    assert _strict_definitions(sql, _FN) == []


def test_definition_parser_ignores_uses_and_comments():
    sql = _strip_sql_comments(
        "CREATE TRIGGER trg_match_outcome AFTER UPDATE ON matches FOR EACH ROW\n"
        f"    EXECUTE FUNCTION {_FN}();\n"
        f"-- {_CREATE} {_FN}() RETURNS TRIGGER AS $$ x $$ LANGUAGE plpgsql;\n"
        f"/* {_CREATE} {_FN}()\n RETURNS TRIGGER AS $$ x $$ LANGUAGE plpgsql; */\n"
        f"{_CREATE} {_FN}_v2() RETURNS TRIGGER AS $$ x $$ LANGUAGE plpgsql;\n"
    )
    assert _loose_definitions(sql, _FN) == 0
    assert _strict_definitions(sql, _FN) == []


def test_definition_parser_reads_every_definition_in_a_file_in_order():
    sql = (
        f"{_CREATE} {_FN}() RETURNS TRIGGER AS $$ old $$ LANGUAGE plpgsql;\n"
        f"create or replace function public.{_FN}() returns trigger as $fn$ new $fn$ language plpgsql;"
    )
    assert _loose_definitions(sql, _FN) == 2
    assert _strict_definitions(sql, _FN) == ["old", "new"]
