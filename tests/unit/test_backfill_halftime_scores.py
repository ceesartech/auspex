"""backfill_halftime_scores: which ESPN competitions may stamp half-time
scores onto which existing rows. Pure planning — no DB or HTTP."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from datetime import date
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


bh = _load("backfill_halftime_scores", "backfill_halftime_scores.py")


def _comp(home_score=2, away_score=1, details=None, completed=True, espn_id="7001"):
    return {
        "id": espn_id,
        "date": "2026-08-01T18:00:00Z",
        "status": {"type": {"state": "post" if completed else "in", "completed": completed}},
        "competitors": [
            {"homeAway": "home", "id": "10", "team": {"displayName": "Home FC"}, "score": str(home_score)},
            {"homeAway": "away", "id": "20", "team": {"displayName": "Away FC"}, "score": str(away_score)},
        ],
        "details": (
            details
            if details is not None
            else [
                {"scoringPlay": True, "team": {"id": "10"}, "clock": {"displayValue": "20'"}},
                {"scoringPlay": True, "team": {"id": "20"}, "clock": {"displayValue": "60'"}},
                {"scoringPlay": True, "team": {"id": "10"}, "clock": {"displayValue": "90'+4'"}},
            ]
        ),
    }


def _row(home_score=2, away_score=1, metadata=None):
    return {"id": "m-1", "home_score": home_score, "away_score": away_score, "metadata": metadata or {}}


@pytest.fixture
def found(monkeypatch):
    """Make find_row return a chosen row without touching a DB."""
    state = {"row": _row(), "how": "espn"}
    monkeypatch.setattr(bh, "find_row", lambda *a: (state["row"], state["how"]))
    return state


class TestPlanCompetition:
    def test_plans_halftime_for_a_matching_row(self, found):
        counts = Counter()
        entry = bh.plan_competition(None, "lg", _comp(), counts)
        assert (entry["match_id"], entry["halftime_home"], entry["halftime_away"]) == ("m-1", 1, 0)
        assert counts["matched_by_espn"] == 1

    def test_row_with_a_different_final_score_is_left_alone(self, found):
        found["row"] = _row(home_score=3, away_score=1)
        counts = Counter()
        assert bh.plan_competition(None, "lg", _comp(), counts) is None
        assert counts["score_mismatch"] == 1

    def test_row_already_stamped_is_skipped(self, found):
        found["row"] = _row(metadata={"halftime_home_score": 1, "halftime_away_score": 0})
        counts = Counter()
        assert bh.plan_competition(None, "lg", _comp(), counts) is None
        assert counts["already_set"] == 1

    def test_incomplete_plays_are_not_planned(self, found):
        counts = Counter()
        details = [{"scoringPlay": True, "team": {"id": "10"}, "clock": {"displayValue": "20'"}}]
        assert bh.plan_competition(None, "lg", _comp(details=details), counts) is None
        assert counts["plays_do_not_add_up"] == 1

    def test_unfinished_competition_is_not_planned(self, found):
        counts = Counter()
        assert bh.plan_competition(None, "lg", _comp(completed=False), counts) is None
        assert counts["not_completed"] == 1

    def test_no_existing_row_is_counted_not_created(self, found):
        found["row"], found["how"] = None, "unmatched"
        counts = Counter()
        assert bh.plan_competition(None, "lg", _comp(), counts) is None
        assert counts["unmatched"] == 1


class _Cur:
    def __init__(self, espn_rows=(), triple_rows=()):
        self.espn_rows, self.triple_rows, self.sql = list(espn_rows), list(triple_rows), []
        self._last = []

    def execute(self, sql, params=None):
        self.sql.append(sql)
        self._last = self.espn_rows if "external_ids" in sql else self.triple_rows

    def fetchall(self):
        return self._last


class TestFindRow:
    def test_espn_id_match_wins(self):
        row, how = bh.find_row(_Cur(espn_rows=[_row()]), "lg", "7001", "Home FC", "Away FC", None)
        assert (row["id"], how) == ("m-1", "espn")

    def test_two_rows_on_one_espn_id_is_ambiguous(self):
        row, how = bh.find_row(_Cur(espn_rows=[_row(), _row()]), "lg", "7001", "Home FC", "Away FC", None)
        assert row is None and how == "ambiguous"

    def test_falls_back_to_the_exact_triple_via_find_team(self, monkeypatch):
        monkeypatch.setattr(bh.fu, "find_team", lambda cur, cfg, name, lg: f"id-{name}")
        cur = _Cur(triple_rows=[_row()])
        row, how = bh.find_row(cur, "lg", "7001", "Home FC", "Away FC", "2026-08-01")
        assert how == "triple" and row["id"] == "m-1"
        assert not any("INSERT" in s for s in cur.sql)

    def test_unknown_team_is_unmatched(self, monkeypatch):
        monkeypatch.setattr(bh.fu, "find_team", lambda *a: None)
        row, how = bh.find_row(_Cur(), "lg", None, "Home FC", "Away FC", "2026-08-01")
        assert row is None and how == "unmatched"


class TestHelpers:
    def test_scoreboard_days_include_the_previous_us_date(self):
        assert bh.scoreboard_days({date(2026, 8, 2), date(2026, 8, 3)}) == [
            date(2026, 8, 1),
            date(2026, 8, 2),
            date(2026, 8, 3),
        ]

    def test_every_soccer_league_maps_back_to_its_slug(self):
        slugs = bh.league_slugs()
        for slug, (_code, name, country) in bh.SOCCER.leagues.items():
            assert slugs[(name, country)] == slug

    def test_csv_is_write_once(self, tmp_path):
        target = tmp_path / "ht.csv"
        bh.write_csv(str(target), [])
        with pytest.raises(FileExistsError):
            bh.write_csv(str(target), [])

    def test_default_csv_is_absolute(self, monkeypatch):
        monkeypatch.delenv("BACKUP_LOCAL_DIR", raising=False)
        assert bh.default_out_csv().startswith("/app/backups/halftime_backfill_")
