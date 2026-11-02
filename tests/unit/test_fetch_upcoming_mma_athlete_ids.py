"""fetch_upcoming resolves an MMA fighter by ESPN athlete id first
(SportConfig.athlete_ids, since 2026-11-02).

ESPN changed 9 UFC fighters' display names and the name-only resolver gave
each a second row; it also put two ESPN athletes named 'Bruno Silva' on one
row. Now: the athlete id (teams.external_ids.espn) first; the name tiers only
when no row carries that id, and the id is stamped onto the row a name finds;
a name that finds only rows of OTHER athletes is refused (team_conflicts,
exit 3), never given a second row. On rows without an id it resolves exactly
as by name alone — the code is live before the backfill runs. Tennis keeps
resolving by name."""

from __future__ import annotations

import importlib.util
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_soccer_identity_fakes", HERE / "test_team_identity_soccer.py")
_fakes = importlib.util.module_from_spec(_spec)
sys.modules["_soccer_identity_fakes"] = _fakes
_spec.loader.exec_module(_fakes)

fu = _fakes.fu
FakeDB = _fakes.FakeDB
MMA = fu.SPORT_CONFIGS["mma"]
TENNIS = fu.SPORT_CONFIGS["tennis"]
UFC = "lg-ufc"


def fighter(team_id, name, espn=None, aliases=None):
    return _fakes.team(team_id, name, sport="mma", espn=espn, aliases=aliases, country=None)


@pytest.fixture(autouse=True)
def identity_stats(monkeypatch):
    for key in list(fu.IDENTITY_STATS):
        monkeypatch.setitem(fu.IDENTITY_STATS, key, 0)
    return fu.IDENTITY_STATS


def test_only_mma_resolves_by_athlete_id():
    assert MMA.athlete_ids is True
    assert TENNIS.athlete_ids is False
    assert not any(cfg.athlete_ids for name, cfg in fu.SPORT_CONFIGS.items() if name != "mma")


class TestCompetitorEspnId:
    def test_mma_reads_the_competitor_id_the_ufc_scoreboard_carries(self):
        # The UFC scoreboard leaves athlete.id out; the competitor's own id is the athlete id.
        assert fu._competitor_espn_id({"id": "4333158", "athlete": {"displayName": "Bruno Silva"}}, True, True) == (
            "4333158"
        )
        assert fu._competitor_espn_id({"id": 5212598}, True, True) == "5212598"

    def test_mma_prefers_athlete_id(self):
        assert fu._competitor_espn_id({"id": "1", "athlete": {"id": "3895544"}}, True, True) == "3895544"

    def test_mma_without_an_id_carries_none(self):
        assert fu._competitor_espn_id({"athlete": {"displayName": "X"}}, True, True) is None
        assert fu._competitor_espn_id({"id": ""}, True, True) is None

    def test_tennis_and_team_sports_unchanged(self):
        assert fu._competitor_espn_id({"id": "3623", "athlete": {"id": "3623"}}, True) is None
        assert fu._competitor_espn_id({"id": "3623", "athlete": {"id": "3623"}}, True, False) is None
        assert fu._competitor_espn_id({"id": "382", "team": {"id": "382"}}, False, True) == "382"


class TestEnsureTeamByAthleteId:
    def test_the_athlete_id_wins_over_a_changed_display_name(self):
        db = FakeDB([fighter("zach", "Zach Reese", espn="5143223")])
        assert fu.ensure_team(db, MMA, "Zachary Reese", UFC, espn_id="5143223") == "zach"
        assert db.writes() == []

    def test_a_row_found_by_name_gets_the_athlete_id(self):
        db = FakeDB([fighter("raf", "Rafael Cerqueira", aliases=["Raffael Cerqueira"])])
        assert fu.ensure_team(db, MMA, "Raffael Cerqueira", UFC, espn_id="5212598") == "raf"
        assert db._row("raf")["external_ids"] == {"espn": "5212598"}
        assert [q for q in db.writes() if q.startswith("INSERT")] == []

    def test_a_new_fighter_is_inserted_and_stamped(self):
        db = FakeDB([fighter("a", "Alex Pereira", espn="4705658")])
        new_id = fu.ensure_team(db, MMA, "Debut Fighter", UFC, espn_id="5999999")
        assert new_id not in (None, "a")
        assert db._row(new_id)["external_ids"] == {"espn": "5999999"}
        assert fu.IDENTITY_STATS["team_conflicts"] == 0

    @pytest.mark.parametrize(
        "rows, name",
        [
            ([fighter("a", "Joe Kropschot")], "Joe Kropschot"),  # exact
            ([fighter("a", "Joe Kropschot", aliases=["Joseph Kropschot"])], "Joseph Kropschot"),  # alias
            ([fighter("a", "Adrian Yañez")], "Adrian Yanez"),  # accent-folded
            ([fighter("a", "Rose Conceição")], "Rose Conceição"),  # decomposed name, composed row
            ([fighter("a", "Someone Else")], "A New Name"),  # a miss: an insert
        ],
    )
    def test_on_rows_without_an_id_it_resolves_as_by_name_alone(self, rows, name):
        """The code goes live before the backfill: the same row (or the same insert) as without an athlete id."""
        by_name = FakeDB(rows)
        by_id = FakeDB(rows)
        want = fu.ensure_team(by_name, MMA, name, UFC)
        got = fu.ensure_team(by_id, MMA, name, UFC, espn_id="777")
        assert got == want
        assert [t["id"] for t in by_id.teams] == [t["id"] for t in by_name.teams]
        assert by_id._row(got)["external_ids"] == {"espn": "777"}
        assert fu.IDENTITY_STATS["team_conflicts"] == 0

    def test_a_name_holding_another_athlete_is_refused_loudly(self, caplog):
        db = FakeDB([fighter("bulldog", "Bruno Silva", espn="3895544")])
        with caplog.at_level(logging.ERROR):
            assert fu.ensure_team(db, MMA, "Bruno Silva", UFC, espn_id="4333158") is None
        assert fu.IDENTITY_STATS["team_conflicts"] == 1
        assert db.writes() == [] and len(db.teams) == 1
        assert "Refusing mma athlete 'Bruno Silva' (ESPN athlete id 4333158)" in caplog.text
        assert "team bulldog (exact match, ESPN athlete 3895544)" in caplog.text
        assert fu.identity_exit_code(MMA) == fu.IDENTITY_EXIT_CODE

    def test_an_alias_holding_another_athlete_is_refused_too(self):
        db = FakeDB([fighter("sk", "Sean King III", espn="5401058", aliases=["Sean King"])])
        assert fu.ensure_team(db, MMA, "Sean King", UFC, espn_id="9999999") is None
        assert fu.IDENTITY_STATS["team_conflicts"] == 1

    def test_one_athlete_id_on_two_rows_is_refused(self):
        db = FakeDB([fighter("a", "A One", espn="1"), fighter("b", "A Two", espn="1")])
        assert fu.ensure_team(db, MMA, "A One", UFC, espn_id="1") is None
        assert fu.IDENTITY_STATS["team_conflicts"] == 1

    def test_the_split_bruno_silvas_resolve_to_their_own_rows(self):
        rows = [
            fighter("bulldog", "Bruno Silva", espn="3895544"),
            fighter("blindado", "Bruno Silva (ESPN 4333158)", espn="4333158", aliases=["Bruno Silva"]),
        ]
        db = FakeDB(rows)
        assert fu.ensure_team(db, MMA, "Bruno Silva", UFC, espn_id="3895544") == "bulldog"
        assert fu.ensure_team(db, MMA, "Bruno Silva", UFC, espn_id="4333158") == "blindado"
        assert fu.find_team(db, MMA, "Bruno Silva", UFC) == "bulldog"  # by name alone: the plain name's row
        assert fu.ensure_team(db, MMA, "Bruno Silva", UFC, espn_id="1234567") is None  # a third one: refused
        assert db.writes() == [] and fu.IDENTITY_STATS["team_conflicts"] == 1

    def test_tennis_still_resolves_by_name(self):
        db = FakeDB([_fakes.team("p1", "Carlos Alcaraz", sport="tennis", espn="999")])
        assert fu.ensure_team(db, TENNIS, "Carlos Alcaraz", "lg") == "p1"


def _comp(home, away, state="pre", completed=False, comp_id="401"):
    return {
        "id": comp_id,
        "date": (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%dT%H:%MZ"),
        "status": {"type": {"state": state, "completed": completed, "name": "STATUS_FINAL" if completed else "X"}},
        "competitors": [home, away],
    }


class TestCallSites:
    @pytest.fixture
    def calls(self, monkeypatch):
        seen: list[tuple] = []

        def fake_ensure(cur, cfg, name, league_id, espn_id=None, alt_names=()):
            seen.append((name, espn_id))
            return f"id-{name}"

        monkeypatch.setattr(fu, "ensure_team", fake_ensure)
        monkeypatch.setattr(fu, "_listed_reversed", lambda *a, **k: False)
        monkeypatch.setattr(fu, "_reconcile_match_row", lambda *a, **k: True)
        monkeypatch.setattr(fu, "insert_scheduled_match", lambda *a, **k: 1)
        monkeypatch.setattr(fu, "insert_finished_match", lambda *a, **k: 1)
        monkeypatch.setattr(fu, "_stamp_identity", lambda *a, **k: True)
        return seen

    def test_the_fixtures_path_passes_mma_athlete_ids(self, calls):
        comp = _comp(
            {"id": "3895544", "athlete": {"displayName": "Bruno Silva"}},
            {"id": "4895760", "athlete": {"displayName": "Édgar Cháirez"}},
        )
        assert fu._process_competition(None, MMA, UFC, comp)
        assert calls == [("Bruno Silva", "3895544"), ("Édgar Cháirez", "4895760")]

    def test_the_fixtures_path_passes_no_tennis_id(self, calls):
        comp = _comp(
            {"id": "3623", "athlete": {"id": "3623", "displayName": "Carlos Alcaraz"}},
            {"id": "3782", "athlete": {"id": "3782", "displayName": "Jannik Sinner"}},
        )
        assert fu._process_competition(None, TENNIS, "lg", comp)
        assert calls == [("Carlos Alcaraz", None), ("Jannik Sinner", None)]

    def test_the_results_path_passes_mma_athlete_ids(self, calls):
        comp = _comp(
            {"id": "4333158", "athlete": {"displayName": "Bruno Silva"}, "winner": False},
            {"id": "4294504", "athlete": {"displayName": "Ismail Naurdiev"}, "winner": True},
            state="post",
            completed=True,
        )
        fu._record_result(None, MMA, UFC, comp)
        assert calls == [("Bruno Silva", "4333158"), ("Ismail Naurdiev", "4294504")]

    def test_the_status_path_resolves_by_athlete_id_and_refuses_another_athletes_row(self):
        db = FakeDB([fighter("bulldog", "Bruno Silva", espn="3895544")])
        side = fu._existing_side(db, MMA, "Bruno Silva", UFC, {"id": "3895544"})
        assert side == "bulldog" and fu.IDENTITY_STATS["team_conflicts"] == 0
        assert fu._existing_side(db, MMA, "Bruno Silva", UFC, {"id": "4333158"}) is None
        assert fu.IDENTITY_STATS["team_conflicts"] == 1
        assert db.writes() == []

    def test_the_dropped_bout_check_resolves_fighters_by_athlete_id(self, monkeypatch):
        seen: list[tuple] = []

        def fake_find(cur, cfg, name, league_id, espn_id=None):
            seen.append((name, espn_id))
            return None

        monkeypatch.setattr(fu, "find_team", fake_find)

        class Cur:
            def execute(self, *a, **k):
                pass

            def fetchall(self):
                return []

        comp = _comp(
            {"id": "5143223", "athlete": {"displayName": "Zach Reese"}},
            {"id": "5000001", "athlete": {"displayName": "Opponent One"}},
            state="post",
            completed=True,
        )
        assert fu.cancel_dropped_bouts(Cur(), MMA, UFC, [{"id": "600", "competitions": [comp]}]) == 0
        assert seen == [("Zach Reese", "5143223"), ("Opponent One", "5000001")]


class TestOpponentPlaceholders:
    """A UFC slot with no booked opponent ('Opponent TBA') is skipped before resolution, like TBD/TBA: resolved by
    athlete id it would stamp the slot's id onto the 'Opponent TBA' row, and the next such slot (another id) would
    be refused every sweep."""

    @pytest.mark.parametrize("slot", ["Opponent TBA", "opponent tbd", " TBA  Opponent ", "TBD Opponent"])
    def test_the_slot_is_skipped(self, slot):
        comp = _comp({"id": "3895544", "athlete": {"displayName": "Bruno Silva"}}, {"id": "-1", "athlete": {}})
        comp["competitors"][1]["athlete"]["displayName"] = slot
        assert fu._is_placeholder_name(slot)
        assert fu._competition_parts(comp, MMA) is None

    def test_the_fixtures_path_never_resolves_it(self, monkeypatch):
        seen: list = []
        monkeypatch.setattr(fu, "ensure_team", lambda *a, **k: seen.append(a) or "x")
        comp = _comp(
            {"id": "3895544", "athlete": {"displayName": "Bruno Silva"}},
            {"id": "9", "athlete": {"displayName": "Opponent TBA"}},
        )
        assert not fu._process_competition(None, MMA, UFC, comp)
        assert seen == []

    def test_a_real_name_containing_the_words_is_kept(self):
        assert not fu._is_placeholder_name("Tba Opponentov")


SCRIPTS = HERE.parents[1] / "scripts"


@pytest.fixture
def lmh(monkeypatch):
    """scripts/load_mma_historical.py, bound to the fetch_upcoming these tests drive (other suites re-load it)."""
    monkeypatch.setitem(sys.modules, "fetch_upcoming", fu)
    spec = importlib.util.spec_from_file_location("load_mma_historical_athlete_ids", SCRIPTS / "load_mma_historical.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ufc_fight(home: tuple[str, str], away: tuple[str, str], when="2024-10-26T14:00Z"):
    """A finished bout as ESPN's UFC scoreboard carries it: the athlete id on the competitor, not on .athlete."""
    return {
        "date": when,
        "status": {"type": {"state": "post"}},
        "competitors": [
            {"id": home[1], "winner": False, "athlete": {"displayName": home[0]}},
            {"id": away[1], "winner": True, "athlete": {"displayName": away[0]}},
        ],
    }


class TestHistoricalLoader:
    """load_mma_historical resolves fighters as the pipeline does: by ESPN athlete id first. By name alone a re-load
    of 2022-2024 would put Blindado's bouts back on Bulldog's row and undo the split."""

    def _split_db(self):
        return FakeDB(
            [
                fighter("bulldog", "Bruno Silva", espn="3895544"),
                fighter("blindado", "Bruno Silva (ESPN 4333158)", espn="4333158", aliases=["Bruno Silva"]),
                fighter("naurdiev", "Ismail Naurdiev", espn="4294504"),
                fighter("durden", "Cody Durden", espn="4217396"),
            ]
        )

    def test_the_parsed_fight_carries_both_athlete_ids(self, lmh):
        parsed = lmh.extract_finished_fight(_ufc_fight(("Bruno Silva", "4333158"), ("Ismail Naurdiev", "4294504")))
        assert (parsed["home_espn"], parsed["away_espn"]) == ("4333158", "4294504")
        assert parsed["home_name"] == "Bruno Silva" and parsed["away_score"] == 1

    def test_each_bruno_silva_resolves_to_his_own_row(self, lmh):
        db = self._split_db()
        blindado = lmh.extract_finished_fight(_ufc_fight(("Bruno Silva", "4333158"), ("Ismail Naurdiev", "4294504")))
        bulldog = lmh.extract_finished_fight(
            _ufc_fight(("Bruno Silva", "3895544"), ("Cody Durden", "4217396"), "2024-07-21T00:00Z")
        )
        assert lmh.resolve_fighters(db, MMA, UFC, blindado) == ("blindado", "naurdiev")
        assert lmh.resolve_fighters(db, MMA, UFC, bulldog) == ("bulldog", "durden")
        by_name = dict(blindado, home_espn=None, away_espn=None)  # what the loader did before
        assert lmh.resolve_fighters(db, MMA, UFC, by_name)[0] == "bulldog"
        assert db.writes() == [] and fu.IDENTITY_STATS["team_conflicts"] == 0

    def test_a_third_bruno_silva_is_refused_and_the_run_exits_3(self, lmh, monkeypatch):
        db = self._split_db()
        third = lmh.extract_finished_fight(_ufc_fight(("Bruno Silva", "7777777"), ("Ismail Naurdiev", "4294504")))
        assert lmh.resolve_fighters(db, MMA, UFC, third)[0] is None
        assert fu.IDENTITY_STATS["team_conflicts"] == 1 and db.writes() == []
        monkeypatch.setattr(lmh, "run", lambda *a, **k: {})
        argv = ["--start-date", "2024-10-26", "--end-date", "2024-10-26", "--database-url", "postgresql:///x"]
        assert lmh.main(argv) == fu.IDENTITY_EXIT_CODE

    def test_a_clean_run_exits_0(self, lmh, monkeypatch):
        monkeypatch.setattr(lmh, "run", lambda *a, **k: {})
        assert lmh.main(["--start-date", "2024-10-26", "--end-date", "2024-10-26", "--database-url", "x"]) == 0
