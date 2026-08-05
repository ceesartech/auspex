"""Repo-wide guard against bare % inside parameterized SQL strings.

psycopg2 %-interpolates the ENTIRE query string — SQL comments included.
A literal % that isn't %s, %% or %(name)s therefore parses as a printf
placeholder and blows up at execute() time ("IndexError: tuple index out
of range" when the arg counts diverge). A prose comment saying
"93% dead space" inside the horse-racing upsert broke
precompute_predictions_horse_racing (and everything downstream of it in
the auspex_pipeline DAG) from 2026-07-14 to 2026-08-05.

Heuristic: any string literal that contains a %s / %(name)s placeholder
AND an uppercase SQL keyword is treated as a parameterized query, and
every other % in it must be escaped as %%. Uppercase-keyword matching
keeps log-format strings ("delete errors (%d)") out of scope — queries
in this repo write their keywords uppercase.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = ("scripts", "services")
SKIP_PARTS = {".venv", "node_modules", "__pycache__", "frontend"}

SQL_KEYWORD = re.compile(r"\b(SELECT|INSERT INTO|UPDATE|DELETE FROM|WITH)\b")
HAS_PLACEHOLDER = re.compile(r"%s|%\(\w+\)s")


def _bare_percent(s: str) -> bool:
    """True if s contains a % that is not part of %s, %% or %(name)s."""
    stripped = re.sub(r"%\(\w+\)s", "", s).replace("%%", "").replace("%s", "")
    return "%" in stripped


def _violations_in(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        s = node.value
        if HAS_PLACEHOLDER.search(s) and SQL_KEYWORD.search(s) and _bare_percent(s):
            out.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    return out


def test_no_bare_percent_in_parameterized_sql():
    violations: list[str] = []
    for root in SCAN_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            violations.extend(_violations_in(path))
    assert not violations, (
        "Bare % inside a parameterized SQL string (psycopg2 interpolates "
        "comments too — escape as %% or move prose to a Python comment): " + ", ".join(violations)
    )


def test_scanner_catches_the_original_bug():
    """The exact string that broke the horse-racing task must be flagged."""
    bad = """
        INSERT INTO race_predictions (race_id) VALUES (%s)
        -- bloated race_predictions to 93% dead space
        """
    assert HAS_PLACEHOLDER.search(bad) and SQL_KEYWORD.search(bad)
    assert _bare_percent(bad)


def test_scanner_allows_escaped_and_named_placeholders():
    ok = "SELECT * FROM t WHERE name LIKE 'vc_%%' AND id = %s AND k = %(key)s"
    assert not _bare_percent(ok)
