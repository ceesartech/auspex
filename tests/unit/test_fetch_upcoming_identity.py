"""Unit tests for fixture IDENTITY + SEASON TYPE in fetch_upcoming.py.

Two verified prod defects are locked down here:

  1. FIXTURE IDENTITY — `matches` is keyed UNIQUE (home, away, match_date)
     and no ESPN id was ever stored, so a kickoff time slid by ESPN forked
     one real fixture into several rows (prod: 19,008 stale tennis rows,
     1,727 recs that could never settle). The fix stores the ESPN
     COMPETITION id and resolves against it before inserting.
  2. SEASON TYPE — every ESPN event carries season.type (1 preseason /
     2 regular / 3 post-season) and it was discarded, putting 147 NFL and
     147 NBA preseason games into the corpus untagged.

Everything here is a pure unit test against a fake cursor that actually
honours the predicates in the SQL (status, espn-id-is-null, the time
window), so the resolution ORDER and the healing REFUSAL are behaviour,
not string matching. No DB, no network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fu = _load("fetch_upcoming_identity", "fetch_upcoming.py")

SOCCER = fu.SPORT_CONFIGS["soccer"]
TENNIS = fu.SPORT_CONFIGS["tennis"]

KICKOFF = datetime(2026, 9, 4, 18, 0, tzinfo=timezone.utc)
HOME = "home-uuid"
AWAY = "away-uuid"


def _row(rid, match_date, *, status="scheduled", espn=None, home=HOME, away=AWAY):
    return {
        "id": rid,
        "match_date": match_date,
        "status": status,
        "espn": espn,
        "home_team_id": home,
        "away_team_id": away,
    }


class FakeCursor:
    """A `matches` table for one team pair, queried the way the resolver
    queries it. Each branch applies the SAME predicates the real SQL does,
    so a test that passes here would pass against Postgres."""

    def __init__(self, rows=None, sport="soccer"):
        self.rows = list(rows or [])
        self.sport = sport
        self.statements: list[tuple[str, tuple]] = []
        self.updates: list[tuple[str, tuple]] = []
        self._result: list[dict] = []
        self.rowcount = 0

    def execute(self, sql, params=None):
        sql = " ".join(sql.split())
        params = tuple(params or ())
        self.statements.append((sql, params))
        self._result = []
        self.rowcount = 0

        if "JOIN leagues" in sql:  # step 1 — espn id, scoped to the sport
            espn_id, sport = params
            self._result = [{"id": r["id"]} for r in self.rows if r["espn"] == espn_id and self.sport == sport][:1]
        elif sql.startswith("SELECT id FROM matches WHERE home_team_id"):  # step 2 — exact triple
            _h, _a, when = params
            self._result = [{"id": r["id"]} for r in self.rows if r["match_date"] == when]
        elif "AND status = 'scheduled'" in sql:  # step 3 — healing candidates
            _h, _a, lo, hi = params
            self._result = [
                {"id": r["id"], "match_date": r["match_date"]}
                for r in sorted(self.rows, key=lambda r: r["match_date"])
                if r["status"] == "scheduled" and r["espn"] is None and lo <= r["match_date"] <= hi
            ]
        elif sql.startswith("SELECT home_team_id, away_team_id, match_date FROM matches WHERE id"):
            self._result = [
                {"home_team_id": r["home_team_id"], "away_team_id": r["away_team_id"], "match_date": r["match_date"]}
                for r in self.rows
                if r["id"] == params[0]
            ]
        elif sql.startswith("UPDATE matches"):
            self.updates.append((sql, params))
            if "dup.match_date" in sql:  # realignment — honour the NOT EXISTS guard
                new_date, rid = params[0], params[1]
                blocked = any(r["match_date"] == new_date and r["id"] != rid for r in self.rows)
                self.rowcount = 0 if blocked else 1
                if not blocked:
                    for r in self.rows:
                        if r["id"] == rid:
                            r["match_date"] = new_date
            else:
                self.rowcount = 1

        return None

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)


# ── Season type ───────────────────────────────────────────────────────


class TestSeasonType:
    def test_espn_codes_map_to_labels(self):
        assert fu._event_season_type({"season": {"type": 1}}) == "preseason"
        assert fu._event_season_type({"season": {"type": 2}}) == "regular"
        assert fu._event_season_type({"season": {"type": 3}}) == "postseason"

    def test_string_codes_map_too(self):
        # ESPN returns the enum as a number on the scoreboard but as a
        # string on some sibling endpoints — accept both.
        assert fu._event_season_type({"season": {"type": "2"}}) == "regular"

    def test_unknown_code_maps_to_none(self):
        assert fu._event_season_type({"season": {"type": 9}}) is None
        assert fu._event_season_type({"season": {"type": "all-star"}}) is None

    def test_missing_season_maps_to_none(self):
        # None means "unknown" and must never be written — a
        # COALESCE(..., 'regular') downstream would re-admit preseason.
        assert fu._event_season_type({}) is None
        assert fu._event_season_type({"season": {}}) is None
        assert fu._event_season_type(None) is None


class TestIterEventCompetitionsCarriesContext:
    def test_team_event_yields_competition_and_season_type(self):
        event = {"id": "401671789", "season": {"type": 1}, "competitions": [{"id": "c1"}]}
        out = list(fu._iter_event_competitions(event, SOCCER))
        assert len(out) == 1
        comp, ctx = out[0]
        assert comp["id"] == "c1"
        assert ctx["season_type"] == "preseason"
        assert ctx["event_id"] == "401671789"

    def test_nested_tennis_event_keeps_the_two_layer_walk(self):
        event = {
            "id": "ev-9",
            "season": {"type": 2},
            "groupings": [
                {"competitions": [{"id": "m1"}, {"id": "m2"}]},
                {"competitions": [{"id": "m3"}]},
            ],
        }
        out = list(fu._iter_event_competitions(event, TENNIS))
        assert [c["id"] for c, _ in out] == ["m1", "m2", "m3"]
        assert {ctx["season_type"] for _, ctx in out} == {"regular"}

    def test_unknown_season_type_yields_none_context(self):
        event = {"id": "ev-1", "season": {"type": 7}, "competitions": [{"id": "c1"}]}
        (_comp, ctx) = list(fu._iter_event_competitions(event, SOCCER))[0]
        assert ctx["season_type"] is None


# ── Competition id capture ────────────────────────────────────────────


def _comp(comp_id="401671789", when="2026-09-04T18:00:00Z", state="pre"):
    return {
        "id": comp_id,
        "competitors": [
            {"homeAway": "home", "team": {"displayName": "Home FC"}, "score": "2"},
            {"homeAway": "away", "team": {"displayName": "Away FC"}, "score": "1"},
        ],
        "date": when,
        "status": {"type": {"state": state, "completed": state == "post", "detail": "Final"}},
        "venue": {"fullName": "Test Park"},
    }


class TestCompetitionParts:
    def test_returns_competition_id_as_string(self):
        parts = fu._competition_parts(_comp(comp_id=401671789), SOCCER)
        assert parts is not None
        assert parts[-1] == "401671789"

    def test_missing_id_is_none(self):
        comp = _comp()
        del comp["id"]
        assert fu._competition_parts(comp, SOCCER)[-1] is None


# ── Resolution order ──────────────────────────────────────────────────


class TestResolveMatchId:
    def test_espn_id_wins_over_the_date_triple(self):
        cur = FakeCursor(
            [
                _row("by-espn", KICKOFF - timedelta(hours=3), espn="401"),
                _row("by-date", KICKOFF),
            ]
        )
        assert fu.resolve_match_id(cur, SOCCER, "401", HOME, AWAY, KICKOFF) == "by-espn"

    def test_espn_lookup_is_scoped_to_the_sport(self):
        # An ESPN id is only unique WITHIN a sport; an NFL competition
        # carrying the same number must not resolve a soccer fixture.
        cur = FakeCursor([_row("nfl-row", KICKOFF - timedelta(hours=3), espn="401")], sport="nfl")
        assert fu.resolve_match_id(cur, SOCCER, "401", HOME, AWAY, KICKOFF) is None

    def test_exact_triple_used_when_no_espn_id_known(self):
        cur = FakeCursor([_row("by-date", KICKOFF), _row("older", KICKOFF - timedelta(hours=2))])
        assert fu.resolve_match_id(cur, SOCCER, None, HOME, AWAY, KICKOFF) == "by-date"

    def test_exact_triple_preferred_over_healing(self):
        # The twin already sits at ESPN's time: resolve it, and never heal
        # the older orphan into a slot that is taken (that UPDATE would
        # raise a unique violation and kill the whole ingest run).
        cur = FakeCursor([_row("orphan", KICKOFF - timedelta(hours=2)), _row("twin", KICKOFF, status="finished")])
        assert fu.resolve_match_id(cur, SOCCER, None, HOME, AWAY, KICKOFF) == "twin"

    def test_healing_resolves_a_single_shifted_row(self):
        cur = FakeCursor([_row("orphan", KICKOFF - timedelta(hours=2))])
        assert fu.resolve_match_id(cur, SOCCER, "401", HOME, AWAY, KICKOFF) == "orphan"

    def test_healing_refuses_when_two_rows_qualify(self):
        # A same-pair rematch inside the window is rare but real (two-legged
        # ties, tournament re-draws); guessing would corrupt a fixture.
        cur = FakeCursor(
            [
                _row("leg-1", KICKOFF - timedelta(hours=48)),
                _row("leg-2", KICKOFF - timedelta(hours=2)),
            ]
        )
        assert fu.resolve_match_id(cur, SOCCER, "401", HOME, AWAY, KICKOFF) is None

    def test_healing_ignores_rows_outside_the_window(self):
        cur = FakeCursor([_row("old", KICKOFF - timedelta(hours=100))])
        assert fu.resolve_match_id(cur, SOCCER, "401", HOME, AWAY, KICKOFF) is None

    def test_healing_ignores_non_scheduled_rows(self):
        cur = FakeCursor([_row("done", KICKOFF - timedelta(hours=2), status="finished")])
        assert fu.resolve_match_id(cur, SOCCER, "401", HOME, AWAY, KICKOFF) is None

    def test_a_row_with_a_different_espn_id_is_never_healed_into(self):
        cur = FakeCursor([_row("other-fixture", KICKOFF - timedelta(hours=2), espn="999")])
        assert fu.resolve_match_id(cur, SOCCER, "401", HOME, AWAY, KICKOFF) is None

    def test_window_hours_is_honoured(self):
        cur = FakeCursor([_row("orphan", KICKOFF - timedelta(hours=10))])
        assert fu.resolve_match_id(cur, SOCCER, "401", HOME, AWAY, KICKOFF, window_hours=4) is None
        assert fu.resolve_match_id(cur, SOCCER, "401", HOME, AWAY, KICKOFF, window_hours=24) == "orphan"


# ── Realignment (the de-fragmentation step) ───────────────────────────


class TestRealign:
    def test_moves_the_resolved_row_to_espns_time(self):
        cur = FakeCursor([_row("orphan", KICKOFF - timedelta(hours=2))])
        assert fu._realign_match_date(cur, "orphan", HOME, AWAY, KICKOFF) is True
        sql, params = cur.updates[0]
        assert sql.startswith("UPDATE matches SET match_date = %s")
        assert params[0] == KICKOFF and params[1] == "orphan"

    def test_already_aligned_row_is_left_alone(self):
        # True == "the row sits at ESPN's kickoff", which it already did.
        cur = FakeCursor([_row("row", KICKOFF)])
        assert fu._realign_match_date(cur, "row", HOME, AWAY, KICKOFF) is True
        assert cur.updates == []

    def test_update_guards_against_an_existing_duplicate(self):
        # Turns a would-be unique violation (which aborts the whole ingest
        # run) into a logged no-op — and reports False so the caller knows
        # the slot belongs to another row and must NOT be stamped.
        cur = FakeCursor([_row("orphan", KICKOFF - timedelta(hours=2)), _row("twin", KICKOFF)])
        assert fu._realign_match_date(cur, "orphan", HOME, AWAY, KICKOFF) is False
        assert "NOT EXISTS" in cur.updates[0][0]

    def test_guard_is_keyed_on_the_resolved_rows_own_pair(self):
        # The UPDATE writes (row.home, row.away, match_dt); the guard must
        # check THAT slot, not the incoming pair's.
        cur = FakeCursor([_row("orphan", KICKOFF - timedelta(hours=2))])
        fu._realign_match_date(cur, "orphan", HOME, AWAY, KICKOFF)
        _sql, params = cur.updates[0]
        assert params[2] == HOME and params[3] == AWAY

    def test_refuses_when_the_resolved_rows_pair_drifted(self):
        # Step 1 resolves by ESPN id alone, so the row can carry a different
        # pair than this parse (MMA competitors have no homeAway and are
        # ordered by array position). Moving it would rewrite another
        # fixture and can raise a unique violation that kills the run.
        cur = FakeCursor([_row("other", KICKOFF - timedelta(hours=2), home="someone-else", away=AWAY)])
        assert fu._realign_match_date(cur, "other", HOME, AWAY, KICKOFF) is False
        assert cur.updates == []

    def test_missing_row_is_refused_not_moved(self):
        cur = FakeCursor([])
        assert fu._realign_match_date(cur, "gone", HOME, AWAY, KICKOFF) is False
        assert cur.updates == []


# ── Payload-aware healing (two fixtures for one pair in a sweep) ──────


def _event(*comps, season_type=2):
    return {"id": "e", "season": {"type": season_type}, "competitions": list(comps)}


class TestPairIndexHealingGuard:
    def test_index_counts_every_competition_for_a_pair(self):
        idx = fu.build_pair_index(
            SOCCER,
            [
                _event(_comp("1", "2026-09-04T18:00:00Z")),
                _event(_comp("2", "2026-09-06T18:00:00Z")),
            ],
        )
        assert len(idx[("Home FC", "Away FC")]) == 2

    def test_two_competitions_inside_the_window_block_healing(self):
        idx = fu.build_pair_index(
            SOCCER,
            [
                _event(_comp("1", "2026-09-04T18:00:00Z")),
                _event(_comp("2", "2026-09-06T17:00:00Z")),  # 47 h later
            ],
        )
        assert fu.healing_allowed(idx, "Home FC", "Away FC", KICKOFF) is False

    def test_a_lone_competition_still_heals(self):
        idx = fu.build_pair_index(SOCCER, [_event(_comp("1", "2026-09-04T18:00:00Z"))])
        assert fu.healing_allowed(idx, "Home FC", "Away FC", KICKOFF) is True

    def test_the_second_leg_outside_the_window_does_not_block(self):
        idx = fu.build_pair_index(
            SOCCER,
            [
                _event(_comp("1", "2026-09-04T18:00:00Z")),
                _event(_comp("2", "2026-09-11T18:00:00Z")),  # a week later
            ],
        )
        assert fu.healing_allowed(idx, "Home FC", "Away FC", KICKOFF) is True

    def test_no_index_keeps_healing_enabled(self):
        assert fu.healing_allowed(None, "Home FC", "Away FC", KICKOFF) is True

    def test_resolver_skips_step_3_when_healing_is_disallowed(self):
        cur = FakeCursor([_row("orphan", KICKOFF - timedelta(hours=47))])
        assert fu.resolve_match_id(cur, SOCCER, "401", HOME, AWAY, KICKOFF) == "orphan"
        cur = FakeCursor([_row("orphan", KICKOFF - timedelta(hours=47))])
        assert fu.resolve_match_id(cur, SOCCER, "401", HOME, AWAY, KICKOFF, allow_heal=False) is None

    def test_a_second_same_pair_fixture_is_not_merged_onto_the_first(self):
        # The prod-shaped case: game A's row exists and is unidentified,
        # game B's row does not. B must NOT capture A's row.
        idx = fu.build_pair_index(
            SOCCER,
            [
                _event(_comp("1", "2026-09-02T19:00:00Z")),  # game A, 47 h earlier
                _event(_comp("2", "2026-09-04T18:00:00Z")),  # game B — this one
            ],
        )
        cur = FakeCursor([_row("game-a", KICKOFF - timedelta(hours=47))])
        with _RecordingPath():
            fu._process_competition(cur, SOCCER, "league", _comp("2"), {"season_type": "regular", "pair_index": idx})
        assert not any(s.startswith("UPDATE matches SET match_date") for s, _ in cur.updates)


# ── Identity stamping: MERGE, never replace ───────────────────────────


class TestStampIdentity:
    def test_merges_both_columns(self):
        cur = FakeCursor()
        assert fu._stamp_identity(cur, HOME, AWAY, KICKOFF, "401671789", "preseason") is True
        sql, params = cur.updates[0]
        assert "external_ids = COALESCE(external_ids, '{}'::jsonb) || %s::jsonb" in sql
        assert "metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb" in sql
        assert json.loads(params[0]) == {"espn": "401671789"}
        assert json.loads(params[1]) == {"season_type": "preseason"}
        assert params[2:] == (HOME, AWAY, KICKOFF)

    def test_only_the_known_halves_are_written(self):
        cur = FakeCursor()
        fu._stamp_identity(cur, HOME, AWAY, KICKOFF, "401", None)
        _sql, params = cur.updates[0]
        # Unknown season type writes an EMPTY patch, not 'regular': the
        # merge must leave a missing marker missing.
        assert json.loads(params[1]) == {}

    def test_no_statement_at_all_when_espn_gave_us_neither(self):
        cur = FakeCursor()
        assert fu._stamp_identity(cur, HOME, AWAY, KICKOFF, None, None) is False
        assert cur.statements == []


# ── End to end through the two insert paths ───────────────────────────


class _RecordingPath:
    """Stubs the team/league writes so the two handlers can be driven
    against FakeCursor while still exercising reconcile + stamp."""

    def __enter__(self):
        self._team, self._sched, self._fin = fu.ensure_team, fu.insert_scheduled_match, fu.insert_finished_match
        fu.ensure_team = lambda cur, cfg, name, league_id: HOME if name == "Home FC" else AWAY
        fu.insert_scheduled_match = lambda *a, **k: 1
        fu.insert_finished_match = lambda *a, **k: 1
        return self

    def __exit__(self, *exc):
        fu.ensure_team, fu.insert_scheduled_match, fu.insert_finished_match = self._team, self._sched, self._fin
        return False


class TestBothInsertPathsCaptureIdentity:
    def test_fixtures_path_heals_the_orphan_then_stamps_it(self):
        cur = FakeCursor([_row("orphan", KICKOFF - timedelta(hours=2))])
        with _RecordingPath():
            assert fu._process_competition(cur, SOCCER, "league", _comp(), {"season_type": "regular"}) is True
        # realigned the orphan (no twin forked) ...
        assert any(s.startswith("UPDATE matches SET match_date") for s, _ in cur.updates)
        # ... and stamped identity onto it.
        stamp = [p for s, p in cur.updates if "external_ids" in s][0]
        assert json.loads(stamp[0]) == {"espn": "401671789"}
        assert json.loads(stamp[1]) == {"season_type": "regular"}

    def test_results_path_reconciles_too(self):
        # 20 of the 119 prod twins were inserted by the RESULTS path.
        cur = FakeCursor([_row("orphan", KICKOFF - timedelta(hours=2))])
        with _RecordingPath():
            ok = fu._record_result(cur, SOCCER, "league", _comp(state="post"), {"season_type": "postseason"})
        assert ok is True
        assert any(s.startswith("UPDATE matches SET match_date") for s, _ in cur.updates)
        stamp = [p for s, p in cur.updates if "external_ids" in s][0]
        assert json.loads(stamp[1]) == {"season_type": "postseason"}

    def test_espn_id_is_not_stamped_when_realignment_was_refused(self):
        # A pre-fix twin holds the target slot: the row that owns the
        # history stays unidentified, and the twin must NOT be given the
        # same ESPN id (that is what makes the fragmentation permanent).
        cur = FakeCursor([_row("orphan", KICKOFF - timedelta(hours=2), espn="401671789"), _row("twin", KICKOFF)])
        with _RecordingPath():
            fu._process_competition(cur, SOCCER, "league", _comp(), {"season_type": "regular"})
        stamp = [p for s, p in cur.updates if "external_ids" in s][0]
        assert json.loads(stamp[0]) == {}
        assert json.loads(stamp[1]) == {"season_type": "regular"}

    def test_missing_event_context_is_tolerated(self):
        cur = FakeCursor()
        with _RecordingPath():
            assert fu._process_competition(cur, SOCCER, "league", _comp()) is True
        stamp = [p for s, p in cur.updates if "external_ids" in s][0]
        assert json.loads(stamp[1]) == {}


class TestProcessEventThreadsTheContext:
    def test_handler_receives_the_event_season_type(self):
        seen: list[dict | None] = []

        def handler(cur, cfg, league_id, comp, context=None):
            seen.append(context)
            return True

        event = {"id": "e", "season": {"type": 1}, "competitions": [_comp()]}
        assert fu.process_event(None, SOCCER, "league", event, handler=handler) == 1
        assert seen == [{"season_type": "preseason", "event_id": "e"}]


class TestPlaceholderCompetitors:
    """ESPN publishes future tennis rounds as draw slots with no players yet.

    Each one became a `matches` row that could never be updated when the
    players were announced (the team ids change, so a NEW row appears and the
    placeholder is orphaned). Prod 2026-09-04: 3,061 such rows, 3,022 stale,
    holding 3,528 predictions. They also make the healing check useless, since
    every placeholder shares the same (home, away) identity.
    """

    def _comp(self, home_name: str, away_name: str) -> dict:
        return {
            "id": "c1",
            "date": "2026-09-09T04:00Z",
            "competitors": [
                {"homeAway": "home", "athlete": {"displayName": home_name}},
                {"homeAway": "away", "athlete": {"displayName": away_name}},
            ],
            "status": {"type": {"state": "pre"}},
        }

    def test_tbd_pair_is_skipped(self):
        assert fu._competition_parts(self._comp("TBD", "TBD"), TENNIS) is None

    def test_one_placeholder_side_is_enough_to_skip(self):
        assert fu._competition_parts(self._comp("Carlos Alcaraz", "TBD"), TENNIS) is None
        assert fu._competition_parts(self._comp("Qualifier", "Jannik Sinner"), TENNIS) is None

    def test_placeholder_match_is_case_and_space_insensitive(self):
        assert fu._competition_parts(self._comp(" tbd ", "Jannik Sinner"), TENNIS) is None
        assert fu._competition_parts(self._comp("To Be Determined", "Jannik Sinner"), TENNIS) is None

    def test_real_players_still_parse(self):
        parts = fu._competition_parts(self._comp("Carlos Alcaraz", "Jannik Sinner"), TENNIS)
        assert parts is not None
        assert parts[2] == "Carlos Alcaraz"
        assert parts[3] == "Jannik Sinner"

    def test_a_name_merely_containing_a_placeholder_word_is_kept(self):
        # Guard against an over-broad substring match: only the WHOLE name is a
        # placeholder. "Qualifier" alone is a draw slot; "Qualifier Cup" is not.
        parts = fu._competition_parts(self._comp("Qualifier Cup Winner", "Jannik Sinner"), TENNIS)
        assert parts is not None
        assert parts[2] == "Qualifier Cup Winner"


class TestSampleTimes:
    """A tennis sweep found 192 competitions for one placeholder pair; dumping
    every timestamp into the warning buries real errors in the log."""

    def test_short_list_renders_in_full(self):
        times = [KICKOFF, KICKOFF + timedelta(hours=1)]
        rendered = fu._sample_times(times)
        assert rendered.count(":") >= 2
        assert "more)" not in rendered

    def test_long_list_is_truncated_with_a_count(self):
        times = [KICKOFF + timedelta(hours=i) for i in range(50)]
        rendered = fu._sample_times(times)
        assert rendered.endswith("(44 more)")
        assert len(rendered) < 400
