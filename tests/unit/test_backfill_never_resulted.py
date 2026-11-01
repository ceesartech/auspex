"""scripts/backfill_never_resulted.py: which listed rows may be healed (screen,
assess), which share one ESPN competition (fork_reasons), the scoreboard days
read, and the write-once ESPN cache the apply replays. The whole tool ran
against a replica of prod with real ESPN scoreboards
(scripts/ops/backfill_soccer_never_resulted.sh header)."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
_spec = importlib.util.spec_from_file_location("backfill_never_resulted", SCRIPTS_DIR / "backfill_never_resulted.py")
bnr = importlib.util.module_from_spec(_spec)
sys.modules["backfill_never_resulted"] = bnr
_spec.loader.exec_module(bnr)

UTC = timezone.utc
AS_OF = datetime(2026, 11, 1, 3, 0, tzinfo=UTC)
KICKOFF = datetime(2026, 6, 13, 16, 30, tzinfo=UTC)


def _row(**over):
    row = {
        "id": "aaaaaaaa-0000-0000-0000-000000000001",
        "league_id": "lg",
        "league_name": "Primera División",
        "country": "Chile",
        "sport": "soccer",
        "match_date": KICKOFF,
        "home_id": "h",
        "away_id": "a",
        "home": "Coquimbo Unido",
        "away": "O'Higgins",
        "status": "scheduled",
        "home_score": None,
        "away_score": None,
        "espn_id": None,
    }
    row.update(over)
    return row


def _comp(home=0, away=0, state="post", completed=True, name="STATUS_FULL_TIME", detail="FT", when=KICKOFF, plays=()):
    return {
        "id": "401850590",
        "date": when.strftime("%Y-%m-%dT%H:%MZ"),
        "status": {"type": {"state": state, "completed": completed, "name": name, "detail": detail}},
        "competitors": [
            {"homeAway": "home", "id": "10", "team": {"id": "10", "displayName": "Coquimbo Unido"}, "score": str(home)},
            {"homeAway": "away", "id": "20", "team": {"id": "20", "displayName": "O'Higgins"}, "score": str(away)},
        ],
        "details": [{"scoringPlay": True, "team": {"id": t}, "clock": {"displayValue": f"{m}'"}} for t, m in plays],
    }


def _assess(row, comp):
    return bnr.assess(row, comp, bnr.fu._competition_parts(comp, bnr.SOCCER))


class TestScreen:
    def test_a_stale_scheduled_soccer_row_goes_to_espn(self):
        assert bnr.screen(_row(), AS_OF) is None

    def test_a_finished_row_is_done(self):
        assert bnr.screen(_row(status="finished", home_score=1, away_score=0), AS_OF) == (bnr.DONE, "already finished")

    @pytest.mark.parametrize(
        "over, why",
        [
            ({"sport": "tennis"}, "not a soccer row"),
            ({"status": "postponed"}, "status 'postponed'"),
            ({"status": "finished"}, "status 'finished'"),  # finished without a score
            ({"match_date": datetime(2026, 10, 29, 12, 0, tzinfo=UTC)}, "within 96 h"),
        ],
    )
    def test_anything_else_is_refused(self, over, why):
        action, reason = bnr.screen(_row(**over), AS_OF)
        assert action == bnr.REFUSE and why in reason


class TestAssess:
    def test_a_completed_final_whose_plays_agree_is_healed(self):
        comp = _comp(2, 1, plays=(("10", 12), ("20", 40), ("10", 88)))
        assert _assess(_row(), comp) == (bnr.HEAL, "", (2, 1), "FT")

    def test_a_goalless_final_lists_no_plays_and_is_healed(self):
        assert _assess(_row(), _comp(0, 0))[:3] == (bnr.HEAL, "", (0, 0))

    @pytest.mark.parametrize(
        "comp, why",
        [
            (_comp(name="STATUS_POSTPONED", state="post", completed=False), "ESPN status STATUS_POSTPONED"),
            (_comp(name="STATUS_ABANDONED"), "ESPN status STATUS_ABANDONED"),
            (_comp(state="in", completed=False, name="STATUS_SECOND_HALF"), "not a completed competition"),
            (_comp(2, 1, plays=(("10", 12),)), "scoring plays add up to 1-0"),
            (_comp(when=datetime(2026, 6, 17, 16, 31, tzinfo=UTC)), "more than"),
        ],
        ids=["postponed", "abandoned", "in-play", "plays-disagree", "kickoff-too-far"],
    )
    def test_anything_else_is_refused(self, comp, why):
        action, reason, final, _ = _assess(_row(), comp)
        assert action == bnr.REFUSE and why in reason and final is None


def _found(*rows_and_kickoffs, espn="401850590"):
    """{row id: {competition key: (comp, ctx, parts)}}: every given row matched to one competition."""
    comp = _comp(when=KICKOFF)
    comp["id"] = espn
    parts = bnr.fu._competition_parts(comp, bnr.SOCCER)
    return {r["id"]: {espn: (comp, {}, parts)} for r in rows_and_kickoffs}


class TestForkReasons:
    A = _row(id="row-a")  # at ESPN's kickoff
    B = _row(id="row-b", match_date=datetime(2026, 6, 15, 0, 0, tzinfo=UTC))  # the fork, 31.5 h later

    def test_the_row_at_espns_kickoff_owns_the_competition_and_the_other_is_its_fork(self):
        reasons = bnr.fork_reasons([self.A, self.B], _found(self.A, self.B))
        assert set(reasons) == {"row-b"} and "a fork of row row-a" in reasons["row-b"]

    def test_the_row_carrying_the_espn_id_owns_it_whatever_its_kickoff(self):
        b_with_id = {**self.B, "espn_id": "401850590"}
        reasons = bnr.fork_reasons([self.A, b_with_id], _found(self.A, b_with_id))
        assert set(reasons) == {"row-a"} and "a fork of row row-b" in reasons["row-a"]

    def test_no_single_owner_refuses_them_all_as_ambiguous(self):
        c = _row(id="row-c", match_date=datetime(2026, 6, 14, 0, 0, tzinfo=UTC))
        reasons = bnr.fork_reasons([self.B, c], _found(self.B, c))
        assert set(reasons) == {"row-b", "row-c"} and all("ambiguous" in r for r in reasons.values())

    def test_one_row_per_competition_has_no_fork(self):
        assert bnr.fork_reasons([self.A], _found(self.A)) == {}


def test_scoreboard_days_are_each_kickoffs_utc_day_and_its_neighbours():
    late = datetime(2026, 6, 15, 0, 0, tzinfo=UTC)
    assert bnr.scoreboard_days([KICKOFF, late]) == [date(2026, 6, d) for d in (12, 13, 14, 15, 16)]


class TestEspnCache:
    def test_fetch_mode_records_each_day_and_a_failed_request_as_none(self, tmp_path, monkeypatch):
        calls = []

        def fetch_day(cfg, slug, day):
            calls.append((slug, day))
            if day == date(2026, 6, 14):
                bnr.fu.FETCH_STATS["failed"] += 1
                return []
            return [{"id": f"ev-{day:%d}"}]

        monkeypatch.setattr(bnr.fu, "fetch_day", fetch_day)
        cache = bnr.EspnCache(str(tmp_path / "espn.json"))
        assert not cache.replay
        assert cache.events("chi.1", date(2026, 6, 13)) == [{"id": "ev-13"}]
        assert cache.events("chi.1", date(2026, 6, 13)) == [{"id": "ev-13"}]  # read once
        assert cache.events("chi.1", date(2026, 6, 14)) is None
        assert calls == [("chi.1", date(2026, 6, 13)), ("chi.1", date(2026, 6, 14))]
        cache.save()
        saved = json.loads((tmp_path / "espn.json").read_text())["days"]
        assert saved == {"chi.1|20260613": [{"id": "ev-13"}], "chi.1|20260614": None}

    def test_replay_mode_never_fetches_and_refuses_a_day_it_does_not_hold(self, tmp_path, monkeypatch):
        (tmp_path / "espn.json").write_text(json.dumps({"days": {"chi.1|20260613": [{"id": "ev"}]}}))
        monkeypatch.setattr(bnr.fu, "fetch_day", lambda *a: pytest.fail("a replay must not call ESPN"))
        cache = bnr.EspnCache(str(tmp_path / "espn.json"))
        assert cache.replay and cache.events("chi.1", date(2026, 6, 13)) == [{"id": "ev"}]
        with pytest.raises(bnr.Refused, match="does not hold chi.1\\|20260614"):
            cache.events("chi.1", date(2026, 6, 14))
        cache.save()  # a no-op: the file stays as the dry run wrote it
        assert json.loads((tmp_path / "espn.json").read_text())["days"] == {"chi.1|20260613": [{"id": "ev"}]}

    def test_an_unreadable_cache_refuses(self, tmp_path):
        (tmp_path / "espn.json").write_text("{not json")
        with pytest.raises(bnr.Refused, match="cannot read the ESPN cache"):
            bnr.EspnCache(str(tmp_path / "espn.json"))


class TestArguments:
    def test_expect_plan_needs_every_key(self):
        full = ",".join(f"{k}=0" for k in bnr.PLAN_KEYS)
        assert bnr.parse_expect_plan(full) == {k: 0 for k in bnr.PLAN_KEYS}
        with pytest.raises(bnr.Refused, match="missing espn_failed"):
            bnr.parse_expect_plan(full.replace(",espn_failed=0", ""))
        with pytest.raises(bnr.Refused, match="bad entry"):
            bnr.parse_expect_plan(full + ",bogus=1")

    def test_as_of_needs_a_time_zone(self):
        assert bnr.parse_as_of("2026-11-01T03:00:00Z") == AS_OF
        with pytest.raises(bnr.Refused, match="time zone"):
            bnr.parse_as_of("2026-11-01T03:00:00")

    def test_an_apply_without_a_pin_refuses(self, tmp_path):
        ids = tmp_path / "ids.txt"
        ids.write_text("aaaaaaaa-0000-0000-0000-000000000001\n")
        argv = ["--match-ids-file", str(ids), "--espn-cache", str(tmp_path / "e.json"), "--apply"]
        assert bnr.main(argv + ["--out-csv", str(tmp_path / "a.csv"), "--database-url", "x"]) == 1
