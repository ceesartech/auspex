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


def _row(home_score=2, away_score=1, half_time_home=None):
    return {"id": "m-1", "home_score": home_score, "away_score": away_score, "half_time_home": half_time_home}


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
        found["row"] = _row(half_time_home=1)
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
        assert bh.default_out_csv("halftime_from_metadata").startswith("/app/backups/halftime_from_metadata_")

    def test_writes_go_to_the_columns_and_only_fill_an_empty_pair(self):
        assert "SET half_time_home = %s, half_time_away = %s" in bh.UPDATE_SQL
        assert "AND half_time_home IS NULL" in bh.UPDATE_SQL
        assert "m.half_time_home IS NULL" in bh.PENDING_DAYS_SQL


def _meta_row(meta=("1", "0"), cols=(None, None), final=(2, 1)):
    return {
        "id": "m-1",
        "match_date": "2026-09-01",
        "home_score": final[0],
        "away_score": final[1],
        "half_time_home": cols[0],
        "half_time_away": cols[1],
        "meta_home": meta[0],
        "meta_away": meta[1],
    }


class TestPlanMetadataMove:
    def test_keys_move_into_empty_columns(self):
        counts = Counter()
        entry = bh.plan_metadata_move(_meta_row(), counts)
        assert (entry["match_id"], entry["halftime_home"], entry["halftime_away"]) == ("m-1", 1, 0)
        assert counts["columns_empty"] == 1

    def test_columns_already_holding_the_same_score_just_lose_the_keys(self):
        counts = Counter()
        assert bh.plan_metadata_move(_meta_row(cols=(1, 0)), counts) is not None
        assert counts["columns_already_equal"] == 1

    @pytest.mark.parametrize(
        "row, reason",
        [
            (_meta_row(cols=(0, 0)), "columns_disagree"),
            (_meta_row(meta=("x", "0")), "meta_unparseable"),
            (_meta_row(meta=("1", None)), "meta_unparseable"),
            (_meta_row(meta=("3", "0")), "meta_exceeds_final"),
        ],
    )
    def test_refusals_keep_the_keys(self, row, reason):
        counts = Counter()
        assert bh.plan_metadata_move(row, counts) is None
        assert counts[reason] == 1

    def test_the_move_is_guarded_on_the_planned_values(self):
        sql = bh.MOVE_FROM_METADATA_SQL
        assert "metadata - 'halftime_home_score' - 'halftime_away_score'" in sql
        assert "metadata->>'halftime_home_score' = %(h)s::text" in sql
        assert "half_time_home IS NULL OR (half_time_home = %(h)s AND half_time_away = %(a)s)" in sql

    def test_from_metadata_is_a_dry_run_by_default(self):
        args = bh.parse_args(["--from-metadata"])
        assert args.from_metadata is True and args.apply is False
