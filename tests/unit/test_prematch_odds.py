"""Training frames and feature builders read PRE-KICKOFF odds only.

fetch_live_odds used to insert any odds key it had not seen before even after
kickoff, so almost every finished live-era fixture carried in-play prices that
every training frame and post-game recompute then read (audit §7, 2026-10-25).
utils.training_data.prematch_odds() rewrites `FROM odds` to a pre-kickoff-only
derived table; these tests pin the rewrite and guard every call site.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ML_SRC = REPO / "services" / "ml-models" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from utils import training_data as td  # noqa: E402

RAW_FROM_ODDS = re.compile(r"\bFROM\s+odds\b(?![_.])", re.IGNORECASE)


@pytest.mark.parametrize(
    "sql, alias",
    [
        ("SELECT AVG(o.odds_decimal) FROM odds o WHERE o.match_id = m.id", "o"),
        ("SELECT * FROM odds\n        WHERE match_id = %s", "odds"),
        ("SELECT od2.odds_decimal FROM odds od2 WHERE od2.match_id = o.match_id", "od2"),
        ("SELECT 1 FROM odds AS x WHERE x.match_id = 1", "x"),
        ("SELECT 1 FROM odds ORDER BY timestamp", "odds"),
    ],
)
def test_rewrite_keeps_the_alias(sql, alias):
    out = td.prematch_odds(sql)
    assert f"{td.PREMATCH_ODDS_SQL} {alias}" in out
    # the only remaining raw reads are the two inside the derived table itself
    assert len(RAW_FROM_ODDS.findall(out)) == 2 * out.count(td.PREMATCH_ODDS_SQL)


def test_rewrite_never_touches_odds_snapshots_and_is_idempotent():
    sql = "SELECT * FROM odds_snapshots s JOIN (SELECT 1 FROM odds o) q ON true"
    once = td.prematch_odds(sql)
    assert "FROM odds_snapshots s" in once
    assert td.prematch_odds(once) == once


def test_derived_table_keeps_pre_kickoff_rows_or_backfill_only_matches():
    s = td.PREMATCH_ODDS_SQL
    assert "< pm.match_date" in s and "NOT EXISTS" in s


def test_every_training_query_reads_odds_through_the_rewrite():
    for name in [n for n in dir(td) if n.endswith("_QUERY")]:
        q = getattr(td, name)
        wrapped = q.count(td.PREMATCH_ODDS_SQL)
        assert len(RAW_FROM_ODDS.findall(q)) == 2 * wrapped, name


FEATURE_BUILDERS = [
    "compute_features.py",
    "compute_features_mma.py",
    "compute_features_tennis.py",
    "compute_features_nfl.py",
    "compute_features_nba.py",
    "compute_features_nhl.py",
]


@pytest.mark.parametrize("filename", FEATURE_BUILDERS)
def test_every_feature_builder_odds_query_is_wrapped(filename):
    src = (REPO / "scripts" / filename).read_text()
    calls = re.findall(r"cur\.execute\((.*?)\n\s*\)", src, re.S)
    reading_odds = [c for c in calls if RAW_FROM_ODDS.search(c)]
    assert reading_odds, f"{filename}: expected at least one odds query"
    for call in reading_odds:
        assert "prematch_odds(" in call, f"{filename}: unwrapped odds query:\n{call[:200]}"
