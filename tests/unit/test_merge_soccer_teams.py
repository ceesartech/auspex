"""merge_soccer_teams: the committed map and its agreement with the ESPN seed,
the deployed resolvers, the foreign-key discovery, football-data kickoffs,
how a twin cluster is folded (kickoff and ESPN id included), what moves off a
dropped twin, the per-pair plan (sequential pairs, forks, refusals, predicted
resolution), that a pair writes in one guarded transaction that commits only
after the resolvers resolve the merged spelling, the run (audit CSV, plan
file, archive before the first write, --expect-plan) and the after-state
check. No DB."""

from __future__ import annotations

import csv
import gzip
import importlib
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
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


ms = _load("merge_soccer_teams", "merge_soccer_teams.py")
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
ti = sys.modules.get("team_identity") or importlib.import_module("team_identity")

T0 = datetime(2026, 7, 6, 17, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 10, 31, tzinfo=timezone.utc)
NEW, OLD, OPP, NEW2, OLD2 = (f"00000000-0000-0000-0000-00000000000{i}" for i in range(1, 6))
THIRD = "00000000-0000-0000-0000-000000000009"


def _pair(new=NEW, old=OLD, new_name="BK Häcken", old_name="Hacken", espn="7834") -> object:
    return ms.MapRow(new, old, new_name, old_name, espn, "exact-alias", "key_equal 1.000")


def _team(tid, name, **kw) -> dict:
    return {
        "id": tid,
        "name": name,
        "normalized_name": ms.normalize_name(name),
        "sport": "soccer",
        "league_id": "L1",
        "external_ids": {},
        "metadata": {},
        **kw,
    }


def _teams() -> dict:
    return {
        NEW: _team(NEW, "BK Häcken"),
        OLD: _team(OLD, "Hacken"),
        OPP: _team(OPP, "Djurgarden"),
        NEW2: _team(NEW2, "Djurgården"),
        OLD2: _team(OLD2, "Djurgarden IF"),
    }


def _m(mid, home, away, when=T0, status="finished", score=(2, 4), ht=(None, None), espn=None, **kw) -> dict:
    return {
        "id": mid,
        "home_team_id": home,
        "away_team_id": away,
        "match_date": when,
        "status": status,
        "home_score": score[0] if status == "finished" else None,
        "away_score": score[1] if status == "finished" else None,
        "half_time_home": ht[0],
        "half_time_away": ht[1],
        "external_ids": {"espn": espn} if espn else {},
        "metadata": kw.pop("metadata", {}),
        **kw,
    }


def _espn_row(mid, home, away, when=T0, **kw) -> dict:
    """A row the ESPN path wrote (it carries the competition id)."""
    return _m(mid, home, away, when, espn=kw.pop("espn", "401" + mid), **kw)


def _ctx(teams, aliases=None, seed=None, extra=()) -> object:
    index = dict(aliases or {})
    for t in teams.values():
        for a in ms.aliases_of(t) or []:
            index.setdefault(ti.alias_form(a), set()).add(t["id"])
    return ms.build_ctx(list(teams.values()) + list(extra), index, seed or {}, ti.alias_form)


def _plan(matches, deps=None, teams=None, pairs=None, refs=None, ctx=None):
    state = {m["id"]: m for m in matches}
    teams = teams or _teams()
    return ms.plan_all(pairs or [_pair()], teams, refs or {}, state, deps or {}, NOW, ctx or _ctx(teams))


def _seed(*rows) -> dict:
    return {r[0]: ti.SeedEntry(r[0], r[1], "x", "evidence") for r in rows}


# ---------------------------------------------------------------------------
# The committed map
# ---------------------------------------------------------------------------


class TestCommittedMap:
    rows = ms.load_map(ms.MAP_PATH)

    def test_loads_and_holds_the_reviewed_pairs(self):
        assert len(self.rows) == 171
        assert {r.method for r in self.rows} == {"exact-alias", "fuzzy", "proven-ambiguous"}

    @pytest.mark.parametrize("name", ["Kalamata", "Barcelona SC", "Independiente del Valle", "Sparta Prague"])
    def test_known_wrong_pairs_are_excluded_and_explained(self, name):
        assert name not in {r.new_name for r in self.rows}
        assert f"#   {name}:" in ms.MAP_PATH.read_text()

    def test_every_row_carries_its_evidence(self):
        for r in self.rows:
            assert "same country" in r.evidence, r

    def test_the_map_agrees_with_the_committed_espn_seed(self):
        seed = ti.load_espn_seed()
        assert seed, "the seed CSV must be committed"
        assert ms.map_seed_conflicts(self.rows, seed) == {}

    def test_the_seed_check_would_catch_a_contradicting_row(self):
        seed = ti.load_espn_seed()
        espn, entry = next(iter(seed.items()))
        bad = ms.MapRow(NEW, OLD, "Somebody", "Not " + entry.team_name, espn, "fuzzy", "same country X")
        assert NEW in ms.map_seed_conflicts(self.rows + [bad], seed)

    def test_wrapper_runs_the_reviewed_sequence(self):
        wrapper = (SCRIPTS_DIR / "ops" / "merge_soccer_teams.sh").read_text()
        assert wrapper.startswith("#!/bin/bash\n# NOT YET APPROVED")
        assert f"EXPECTED={len(self.rows)}" in wrapper
        assert "compute_features.py --backfill-finished soccer" in wrapper
        assert "--apply --expect-plan $PLAN" in wrapper and "--check-after $PLAN" in wrapper
        assert "GRADE_AFTER" in wrapper and "/tmp/merge_teams_dry.log" in wrapper
        order = [
            "repoint_team_matches.sh",
            "THIS script",
            "repair_football_data_kickoffs.sh",
            "cleanup_soccer_orphans.sh",
        ]
        positions = [wrapper.index(s) for s in order]
        assert positions == sorted(positions)
        assert "promote_raw stays blocked until" in wrapper

    def test_wrapper_pauses_the_writers_and_always_unpauses_what_it_paused(self):
        """The behaviour (drain first, Airflow 3.3's frozen paused runs) is run for real in
        tests/unit/test_merge_soccer_teams_wrapper.py; this pins the shape."""
        wrapper = (SCRIPTS_DIR / "ops" / "merge_soccer_teams.sh").read_text()
        lib = (SCRIPTS_DIR / "ops" / "_airflow_lib.sh").read_text()
        assert 'DAGS="auspex_pipeline fetch_live_odds"' in wrapper
        assert "trap af_resume_on_exit EXIT" in wrapper and "trap 'exit 130' INT TERM HUP" in wrapper
        assert "trap '' PIPE" in lib  # a dead tee cannot kill the exit path (run for real in the wrapper tests)
        assert "airflow dags pause -y" in lib and "airflow dags unpause -y" in lib
        assert 'AF_PAUSED_BY_US="$AF_PAUSED_BY_US $d"' in lib  # only what it paused; an already paused DAG stays paused
        backup, pause = wrapper.index('log "backup landed'), wrapper.index("af_drain_and_pause 5 40 $DAGS")
        dry, merged_at, record, apply_, resume, grade = (
            wrapper.index('log "2/5 dry run"'),
            wrapper.index("MERGED_AT=$(date -u +%FT%TZ)"),
            wrapper.index('write_state_file "$PLAN" "$MERGED_AT" "$grade_ids" applying'),
            wrapper.index("$TOOL --apply --expect-plan $PLAN"),
            wrapper.index("\nfi\nresume_ok=1\naf_resume || resume_ok=0\n"),
            wrapper.index('log "4/5 grade"'),
        )
        # MERGED_AT and the 'applying' record come before the apply: a run that ends during it still leaves them.
        assert backup < pause < dry < merged_at < record < apply_ < resume < grade
        assert wrapper.count("MERGED_AT=$(date") == 1
        # What a run that is gone left paused is released before step 0, so no exit (ALREADY DONE included) leaves it
        # paused; and no DONE is printed while a DAG this run paused is still paused.
        release, step0 = wrapper.index("af_release_orphans $DAGS ||"), wrapper.index('log "0/5 already done?')
        assert wrapper.index("af_lock merge_soccer_teams") < release < step0
        assert wrapper.index('if [ "$resume_ok" != "1" ]; then') < wrapper.index('log "DONE: $EXPECTED/$EXPECTED')

    def test_wrapper_gates_on_the_reviewed_kickoff_moves_and_checks_the_odds_afterwards(self):
        wrapper = (SCRIPTS_DIR / "ops" / "merge_soccer_teams.sh").read_text()
        assert "EXPECTED_KICKOFFS_ADOPTED=0" in wrapper
        assert 'if [ "$kickoffs" != "$EXPECTED_KICKOFFS_ADOPTED" ]; then' in wrapper
        assert "kickoffs_adopted=[0-9]*" in wrapper
        assert "--check-after $PLAN --odds-since $MERGED_AT" in wrapper

    def test_ids_lead_each_line_unquoted_for_the_wrapper(self):
        data = [ln for ln in ms.MAP_PATH.read_text().splitlines() if not ln.startswith("#")][1:]
        for ln in data:
            new, old = ln.split(",")[:2]
            assert ms.UUID_RE.match(new) and ms.UUID_RE.match(old), ln


def _write_map(tmp_path, rows: list[dict]) -> Path:
    path = tmp_path / "map.csv"
    with path.open("w", newline="") as fh:
        fh.write("# a comment\n")
        w = csv.DictWriter(fh, fieldnames=ms.MAP_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def _row(new=NEW, old=OLD, espn="1", **kw) -> dict:
    return {
        "new_team_id": new,
        "old_team_id": old,
        "new_name": "N",
        "old_name": "O",
        "espn_team_id": espn,
        "espn_league": "x",
        "method": "exact-alias",
        "evidence": "same country X",
        **kw,
    }


class TestLoadMap:
    def test_valid(self, tmp_path):
        rows = ms.load_map(_write_map(tmp_path, [_row(), _row(new=NEW2, old=OLD2, espn="")]))
        assert [r.new_team_id for r in rows] == [NEW, NEW2] and rows[1].espn_team_id == ""

    @pytest.mark.parametrize(
        "rows, match",
        [
            ([_row(), _row(old=OLD2)], "more than once"),
            ([_row(), _row(new=OLD, old=OLD2)], "chain"),
            ([_row(old=NEW)], "onto itself"),
            ([_row(new="not-a-uuid")], "uuid"),
            ([_row(espn="abc")], "numeric"),
            ([_row(), _row(new=NEW2, espn="2")], "two ESPN team ids"),
            ([_row(), _row(new=NEW2, old=OLD2, espn="1")], "one ESPN team id on two pairs"),
            ([_row(evidence=" ")], "evidence"),
        ],
    )
    def test_unsafe_maps_are_refused(self, tmp_path, rows, match):
        with pytest.raises(ms.MapError, match=match):
            ms.load_map(_write_map(tmp_path, rows))

    def test_wrong_header_is_refused(self, tmp_path):
        path = tmp_path / "m.csv"
        path.write_text("new_team_id,old_team_id\n")
        with pytest.raises(ms.MapError, match="header"):
            ms.load_map(path)

    def test_only_narrows_and_refuses_ids_outside_the_map(self):
        rows = [_pair(), _pair(new=NEW2, old=OLD2)]
        assert ms.select_pairs(rows, NEW2) == [rows[1]]
        assert ms.select_pairs(rows, None) == rows
        with pytest.raises(ms.MapError, match="not in the merge map"):
            ms.select_pairs(rows, f"{NEW},{OPP}")


# The odds feed's spellings of merged clubs (reviewed, from prod's fetch_live_odds logs).
REVIEWED_ODDS_SPELLINGS = {
    ("Brighton & Hove Albion", "Brighton and Hove Albion"),
    ("Paris Saint-Germain", "Paris Saint Germain"),
    ("FC Twente", "FC Twente Enschede"),
    ("Kyoto Sanga", "Kyoto Purple Sanga"),
    ("Deportivo", "Deportivo La Coruña"),
    ("Racing Santander", "Real Racing Club de Santander"),
    ("Vitória de Guimaraes", "Vitória SC"),
    ("Henan", "Henan FC"),
    ("Huracán", "Atlético Huracán"),
    ("Belgrano (Córdoba)", "Belgrano de Cordoba"),
    ("OH Leuven", "Leuven"),
}


def _write_spellings(tmp_path, rows, header="new_team_id,new_name,odds_name,evidence") -> Path:
    path = tmp_path / "spellings.csv"
    with path.open("w", newline="") as fh:
        fh.write("# comment\n" + header + "\n")
        w = csv.writer(fh)
        for r in rows:
            w.writerow(r)
    return path


class TestOddsSpellings:
    rows = ms.load_map(ms.MAP_PATH)

    def test_the_committed_list_loads_against_the_committed_map(self):
        names = ms.load_odds_spellings(ms.ODDS_SPELLINGS_PATH, self.rows)
        by_id = {r.new_team_id: r.new_name for r in self.rows}
        got = {(by_id[nid], s) for nid, ss in names.items() for s in ss}
        assert REVIEWED_ODDS_SPELLINGS <= got
        for nid, ss in names.items():  # never the twin's own alias: the merged-spelling probe covers that one
            assert all(ti.alias_form(s) != ti.alias_form(by_id[nid]) for s in ss), (by_id[nid], ss)

    def test_a_valid_row(self, tmp_path):
        path = _write_spellings(tmp_path, [[NEW, "BK Häcken", "Hacken FC", "log"], [NEW, "BK Häcken", "Häcken", "log"]])
        assert ms.load_odds_spellings(path, [_pair()]) == {NEW: ["Hacken FC", "Häcken"]}

    @pytest.mark.parametrize(
        "row, match",
        [
            ([OLD, "BK Häcken", "Hacken FC", "log"], "not a pair of the merge map"),
            ([NEW, "BK Hacken", "Hacken FC", "log"], "in the merge map, not"),
            ([NEW, "BK Häcken", " ", "log"], "needs an odds_name"),
            ([NEW, "BK Häcken", "Hacken FC", ""], "needs an odds_name and its evidence"),
        ],
    )
    def test_a_row_nobody_can_place_refuses(self, tmp_path, row, match):
        with pytest.raises(ms.MapError, match=match):
            ms.load_odds_spellings(_write_spellings(tmp_path, [row]), [_pair()])

    def test_a_duplicate_or_a_wrong_header_refuses(self, tmp_path):
        row = [NEW, "BK Häcken", "Hacken FC", "log"]
        with pytest.raises(ms.MapError, match="twice"):
            ms.load_odds_spellings(_write_spellings(tmp_path, [row, row]), [_pair()])
        with pytest.raises(ms.MapError, match="header"):
            ms.load_odds_spellings(_write_spellings(tmp_path, [row], header="a,b,c,d"), [_pair()])


class TestSeedConflict:
    seed = _seed(("225", "Santos Laguna"), ("2674", "Santos"))

    def test_agreeing_or_unseeded_rows_pass(self):
        assert ms.seed_conflict("Santos Laguna FC", "Santos Laguna", "225", self.seed) is None
        assert ms.seed_conflict("BK Häcken", "Hacken", "7834", self.seed) is None
        assert ms.seed_conflict("BK Häcken", "Hacken", "", self.seed) is None

    def test_a_seeded_id_goes_onto_the_row_the_seed_names(self):
        assert "pins ESPN team 225" in ms.seed_conflict("Club Santos", "Santos", "225", self.seed)

    def test_a_seeded_row_is_never_merged_away(self):
        assert "never merged away" in ms.seed_conflict("Santos", "Santos FC", "", self.seed)

    def test_a_seeded_row_never_takes_another_id(self):
        assert "pins 'Santos' to ESPN team 2674" in ms.seed_conflict("Santos SP", "Santos", "999", self.seed)


# ---------------------------------------------------------------------------
# The deployed resolvers
# ---------------------------------------------------------------------------


class TestNormalizeName:
    def test_it_is_fetch_upcomings_normalized_name_composed_too(self):
        import fetch_upcoming as fu

        decomposed = "O\u0308rgryte IS Sa\u0303o Paulo"  # as ESPN sends some names: letters plus combining marks
        assert ms.normalize_name(decomposed) == fu.normalize_team(decomposed) == "\u00f6rgryte is s\u00e3o paulo"
        assert ms.normalize_name("  Santos   FC ") == fu.normalize_team("  Santos   FC ") == "santos fc"


class TestLoadResolvers:
    def test_the_scripts_tree_provides_them(self):
        resolvers, reason = ms.load_resolvers(SCRIPTS_DIR)
        assert reason is None and resolvers is not None
        assert resolvers.alias_form("BK Häcken ") == "bk hacken"

    def test_a_missing_function_is_a_refusal(self, tmp_path, monkeypatch):
        for name in ms.RESOLVER_API:
            monkeypatch.delitem(sys.modules, name, raising=False)
        (tmp_path / "team_identity.py").write_text(
            "def alias_form(n): return n\ndef load_alias_index(c, s): return {}\n"
            "def resolve_alias(i, s, n): return None\ndef espn_seed(): return {}\n"
            "def stamp_espn_id(c, t, e, label=''): return True\n"
        )
        (tmp_path / "fetch_upcoming.py").write_text("SPORT_CONFIGS = {'soccer': object()}\ndef find_team(*a): pass\n")
        (tmp_path / "fetch_live_odds.py").write_text("def something_else(): pass\n")
        monkeypatch.setattr(sys, "path", [str(tmp_path)] + [p for p in sys.path if p != str(SCRIPTS_DIR)])
        resolvers, reason = ms.load_resolvers(tmp_path)
        assert resolvers is None and "fetch_live_odds.py lacks find_match_id" in reason

    def test_an_import_error_is_a_refusal_not_a_crash(self, tmp_path, monkeypatch):
        for name in ms.RESOLVER_API:
            monkeypatch.delitem(sys.modules, name, raising=False)
        (tmp_path / "team_identity.py").write_text("import no_such_module_xyz\n")
        monkeypatch.setattr(sys, "path", [str(tmp_path)] + [p for p in sys.path if p != str(SCRIPTS_DIR)])
        resolvers, reason = ms.load_resolvers(tmp_path)
        assert resolvers is None and "cannot import scripts/team_identity.py" in reason


# ---------------------------------------------------------------------------
# Foreign keys
# ---------------------------------------------------------------------------


def _fk(tbl, col, ref):
    return {"tbl": tbl, "col": col, "ref": ref}


PROD_FKS = [
    _fk("matches", "home_team_id", "teams"),
    _fk("matches", "away_team_id", "teams"),
    _fk("match_stats", "team_id", "teams"),
    _fk("players", "team_id", "teams"),
    _fk("transfers", "from_team_id", "teams"),
    _fk("transfers", "to_team_id", "teams"),
    _fk("team_managers", "team_id", "teams"),
    _fk("nhl_match_stats", "team_id", "teams"),
    _fk("nhl_match_advanced_stats", "team_id", "teams"),
] + [
    _fk(t, "match_id", "matches")
    for t in (
        "match_stats",
        "odds",
        "player_match_stats",
        "weather",
        "predictions",
        "betting_recommendations",
        "betting_history",
        "features_cache",
        "nhl_match_stats",
        "nhl_match_advanced_stats",
        "match_weather",
        "odds_snapshots",
    )
]


class TestSchema:
    def test_prod_keys_are_all_known(self):
        schema, unknown = ms.read_schema(PROD_FKS)
        assert unknown == [] and schema is not None
        assert ("nfl_players", "team_id") not in schema.team_refs  # not on prod: never queried
        assert "nfl_injury_reports" not in schema.match_children
        assert "predictions" in schema.match_children and "betting_recommendations" not in schema.match_children

    def test_an_unknown_key_refuses(self):
        schema, unknown = ms.read_schema(PROD_FKS + [_fk("match_notes", "match_id", "matches")])
        assert schema is None and unknown == ["match_notes.match_id -> matches"]

    def test_the_matches_keys_must_be_there(self):
        schema, unknown = ms.read_schema([f for f in PROD_FKS if f["tbl"] != "matches"])
        assert schema is None and sorted(unknown) == [
            "expected foreign key matches.away_team_id -> teams is missing",
            "expected foreign key matches.home_team_id -> teams is missing",
        ]

    def test_every_reference_guards_the_team_delete_and_the_checks(self):
        schema, _ = ms.read_schema(PROD_FKS + [_fk("nfl_players", "team_id", "teams")])
        delete = ms.team_delete_sql(schema)
        refs = ms.new_team_refs_sql(schema)
        plan_refs = ms.team_refs_sql(schema)
        for tbl, col in schema.team_refs:
            assert f"NOT EXISTS (SELECT 1 FROM {tbl} x WHERE" in delete and f"x.{col} = t.id" in delete
            assert f"FROM {tbl} x WHERE" in refs
            assert (f"FROM {tbl} x WHERE" in plan_refs) == (tbl != "matches")
        assert "x.from_team_id = t.id OR x.to_team_id = t.id" in delete


# ---------------------------------------------------------------------------
# football-data kickoffs
# ---------------------------------------------------------------------------


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


class TestFootballDataKickoffs:
    def test_summer_kickoffs_move_back_an_hour_winter_ones_do_not(self):
        assert ms.corrected_kickoff(_utc("2026-07-05 13:00")) == _utc("2026-07-05 12:00")
        assert ms.corrected_kickoff(_utc("2026-01-10 15:00")) == _utc("2026-01-10 15:00")
        # a 00:00 UK kickoff (Bragantino v Gremio): 23:00Z the day before, where the ESPN row sits
        assert ms.corrected_kickoff(_utc("2026-07-18 00:00")) == _utc("2026-07-17 23:00")

    def test_a_midnight_row_gets_the_timed_window_inside_the_date_only_one(self):
        lo, hi = ms.promote_window(_utc("2026-07-18 00:00"))
        assert (lo, hi) == (_utc("2026-07-17 23:00"), _utc("2026-07-18 02:00"))
        assert not ms.promote_finds(_utc("2026-07-18 05:00"), _utc("2026-07-18 00:00"))  # whole-day would say yes

    @pytest.mark.parametrize(
        "espn, legacy",
        [
            ("2026-07-26 00:46", "2026-07-26 00:30"),  # Orlando City v Nashville SC (corrects to 23:30Z the day before)
            ("2026-08-02 00:10", "2026-08-02 00:30"),  # Inter Miami v Columbus Crew
            ("2026-07-26 01:06", "2026-07-26 02:00"),  # Club Leon v Pachuca
        ],
    )
    def test_the_reviewers_examples_are_found_on_the_kept_espn_row(self, espn, legacy):
        assert ms.promote_finds(_utc(espn), _utc(legacy))

    def test_a_kept_row_off_the_window_or_the_uk_date_is_not_found(self):
        assert not ms.promote_finds(_utc("2026-07-05 16:30"), _utc("2026-07-05 13:00"))  # 4.5 h after 12:00Z
        assert not ms.promote_finds(_utc("2026-07-25 22:50"), _utc("2026-07-26 00:30"))  # before the UK date starts

    def test_flags_only_for_football_data_rows(self):
        fd = _m("d", OLD, OPP, _utc("2026-07-26 00:30"))
        flags = ms.football_data_flags(fd, _utc("2026-07-26 00:46"))
        assert flags["fd_corrected_kickoff"] == "2026-07-25T23:30:00+00:00"
        assert flags["kickoff_differs"] is True and flags["promote_finds_kept"] is True
        assert ms.football_data_flags(_espn_row("e", OLD, OPP), T0) == {}
        assert ms.football_data_flags(_m("s", OLD, OPP, status="scheduled"), T0) == {}
        espn_result = _m("r", OLD, OPP, metadata={"result_source": "espn_scoreboard"})
        assert ms.football_data_flags(espn_result, T0) == {}

    def test_agrees_with_promote_raws_own_window_when_it_has_one(self):
        pr = _load("_promote_raw_probe", "promote_raw.py")
        if not hasattr(pr, "existing_match_window"):
            pytest.skip("promote_raw has no existing_match_window on this branch (football-data branch adds it)")
        for legacy in ("2026-07-26 00:30", "2026-07-05 13:00", "2026-01-10 15:00"):
            wall = _utc(legacy).replace(tzinfo=None)
            match_dt = pr.parse_match_date(wall.strftime("%d/%m/%Y"), wall.strftime("%H:%M"))
            assert pr.existing_match_window(match_dt, wall.date()) == ms.promote_window(_utc(legacy))


# ---------------------------------------------------------------------------
# Clusters and the keep decision
# ---------------------------------------------------------------------------


class TestFindClusters:
    def test_twin_within_the_window_after_the_repoint(self):
        state = {m["id"]: m for m in [_m("a", NEW, OPP), _m("b", OLD, OPP, T0 + timedelta(hours=36))]}
        assert [[m["id"] for m in c] for c in ms.find_clusters(NEW, OLD, state)] == [["a", "b"]]

    def test_outside_the_window_or_other_orientation_is_no_twin(self):
        state = {
            m["id"]: m for m in [_m("a", NEW, OPP), _m("b", OLD, OPP, T0 + timedelta(hours=37)), _m("c", OPP, OLD)]
        }
        assert ms.find_clusters(NEW, OLD, state) == []

    def test_clusters_chain_and_only_start_from_the_new_team(self):
        state = {
            m["id"]: m
            for m in [
                _m("a", NEW, OPP, T0, status="scheduled"),
                _m("b", NEW, OPP, T0 + timedelta(hours=20)),
                _m("c", OLD, OPP, T0 + timedelta(hours=21)),
                _m("x", OLD, OPP, T0 - timedelta(days=1), score=(1, 1)),  # old-only, 23h before a: chained via a
                _m("y", OLD2, OPP, T0),  # unrelated pair
            ]
        }
        ids = [sorted(m["id"] for m in c) for c in ms.find_clusters(NEW, OLD, state)]
        assert ids == [["a", "b", "c", "x"]]

    def test_dependents_are_loaded_for_every_row_any_run_can_cluster(self):
        pairs = [_pair(), _pair(new=NEW2, old=OLD2, new_name="Djurgården", old_name="Djurgarden IF")]
        rows = {
            m["id"]: m
            for m in [
                _m("e", NEW, NEW2),  # both spellings: a twin of f only once both pairs fold
                _m("f", OLD, OLD2, T0 + timedelta(hours=1)),
                _m("g", OLD, OPP, T0 + timedelta(days=3)),  # alone in its window
                _m("h", OLD, OPP, T0 + timedelta(days=3, hours=2)),  # twin of g, but no merged team in it
            ]
        }
        assert ms.cluster_member_ids(pairs, rows) == {"e", "f"}


def _deps(**kw) -> object:
    return ms.Deps(**kw)


class TestDecideCluster:
    def test_rec_holder_is_kept(self):
        a, b = _m("a", NEW, OPP, espn="1"), _m("b", OLD, OPP)
        v = ms.decide_cluster([a, b], {"b": _deps(holds_recs=True)})
        assert v["keep"]["id"] == "b" and [d["id"] for d in v["drops"]] == ["a"]

    def test_else_the_finished_row_with_espn_identity_then_more_predictions(self):
        a, b, c = _m("a", NEW, OPP), _m("b", OLD, OPP, espn="9"), _m("c", NEW, OPP, status="scheduled")
        assert ms.decide_cluster([a, b, c], {"a": _deps(n_preds=50)})["keep"]["id"] == "b"
        assert ms.decide_cluster([a, c], {"a": _deps(n_preds=5), "c": _deps(n_preds=50)})["keep"]["id"] == "a"

    def test_an_espn_scoreboard_result_beats_a_football_data_row(self):
        # Inter Miami v Columbus Crew: ESPN's result row (no competition id) at 00:10, football-data's at 00:30.
        espn = _m("e", NEW, OPP, _utc("2026-08-02 00:10"), metadata={"result_source": "espn_scoreboard"})
        fd = _m("f", OLD, OPP, _utc("2026-08-02 00:30"))
        v = ms.decide_cluster([espn, fd], {"f": _deps(n_preds=9, odds=[{}] * 30)})
        assert v["keep"]["id"] == "e" and [d["id"] for d in v["drops"]] == ["f"]

    @pytest.mark.parametrize(
        "rows, deps, match",
        [
            ([_m("a", NEW, OPP), _m("b", OLD, OPP, score=(1, 4))], {}, "disagree on the score"),
            (
                [_m("a", NEW, OPP), _m("b", OLD, OPP)],
                {"a": _deps(holds_recs=True), "b": _deps(holds_recs=True)},
                "2 twins hold",
            ),
            ([_m("a", NEW, OPP), _m("b", OLD, OPP, status="live")], {}, "in play"),
            ([_m("a", NEW, OPP), _m("b", OLD, OPP, status="cancelled")], {}, "cancelled"),
            ([_m("a", NEW, OPP, espn="1"), _m("b", OLD, OPP, espn="2")], {}, "different ESPN"),
            ([_m("a", NEW, OPP), _m("b", OLD, OPP, T0 + timedelta(hours=40))], {}, "spans"),
            ([_m("a", NEW, OPP), {**_m("b", OLD, OPP), "home_score": None}], {}, "no full-time score"),
        ],
    )
    def test_refusals(self, rows, deps, match):
        assert match in ms.decide_cluster(rows, deps)["refuse"]

    def test_stale_rec_holder_adopts_an_espn_result_and_its_kickoff_when_its_recs_are_settled(self):
        stale = _m("s", NEW, OPP, T0, status="scheduled")
        played = _espn_row("p", OLD, OPP, T0 + timedelta(hours=20), ht=(1, 2))
        v = ms.decide_cluster([stale, played], {"s": _deps(holds_recs=True)})
        assert v["keep"]["id"] == "s" and v["adopt_from"]["id"] == "p" and v["ht_from"]["id"] == "p"
        assert v["kickoff"] == played["match_date"] and v["espn_from"]["id"] == "p"

    def test_the_kickoff_comes_with_an_espn_scoreboard_result_too(self):
        # ESPN identity without a competition id: no ESPN id to copy, the result alone moves the kickoff
        stale = _m("s", NEW, OPP, T0, status="scheduled")
        played = _m("p", OLD, OPP, T0 + timedelta(hours=20), metadata={"result_source": "espn_scoreboard"})
        v = ms.decide_cluster([stale, played], {"s": _deps(holds_recs=True)})
        assert v["adopt_from"]["id"] == "p" and v["espn_from"] is None and v["kickoff"] == played["match_date"]

    def test_open_recs_adopt_only_on_the_same_kickoff(self):
        stale = _m("s", OLD, OPP, T0, status="scheduled")
        moved = _espn_row("p", NEW, OPP, T0 + timedelta(hours=1))
        deps = {"s": _deps(holds_recs=True, open_recs=1)}
        assert "differs" in ms.decide_cluster([stale, moved], deps)["refuse"]
        same = _m("p", NEW, OPP, T0, espn="77")
        v = ms.decide_cluster([stale, same], deps)
        assert v["adopt_from"]["id"] == "p" and v["espn_from"]["id"] == "p" and v["kickoff"] is None

    def test_a_football_data_result_never_lends_its_stored_kickoff(self):
        # Hacken v Djurgarden: the stale ESPN fixture at 17:00Z, football-data's copy at 18:00 (UK time read as UTC).
        stale = _m("s", NEW, OPP, _utc("2026-07-06 17:00"), status="scheduled")
        fd = _m("f", OLD, OPP, _utc("2026-07-06 18:00"))
        v = ms.decide_cluster([stale, fd], {"s": _deps(holds_recs=True)})
        assert v["adopt_from"]["id"] == "f" and v["kickoff"] is None

    def test_a_football_data_result_far_from_the_rec_holder_is_refused(self):
        stale = _m("s", NEW, OPP, _utc("2026-07-05 10:00"), status="scheduled")  # postponed?
        fd = _m("f", OLD, OPP, _utc("2026-07-06 18:00"))
        assert (
            "not near football-data's corrected kickoff"
            in ms.decide_cluster([stale, fd], {"s": _deps(holds_recs=True)})["refuse"]
        )

    def test_open_recs_beside_a_football_data_result_need_its_corrected_kickoff_exactly(self):
        stale = _m("s", NEW, OPP, _utc("2026-07-06 17:30"), status="scheduled")
        fd = _m("f", OLD, OPP, _utc("2026-07-06 18:00"))
        deps = {"s": _deps(holds_recs=True, open_recs=2)}
        assert "corrected kickoff" in ms.decide_cluster([stale, fd], deps)["refuse"]
        exact = _m("s", NEW, OPP, _utc("2026-07-06 17:00"), status="scheduled")
        assert ms.decide_cluster([exact, fd], deps)["adopt_from"]["id"] == "f"

    def test_an_espn_id_comes_only_with_its_kickoff(self):
        fd_holder = _m("k", OLD, OPP, _utc("2026-07-06 18:00"))
        espn = _espn_row("e", NEW, OPP, _utc("2026-07-06 17:00"))
        v = ms.decide_cluster([fd_holder, espn], {"k": _deps(holds_recs=True)})
        assert v["espn_from"]["id"] == "e" and v["kickoff"] == espn["match_date"]
        v = ms.decide_cluster([fd_holder, espn], {"k": _deps(holds_recs=True, open_recs=1)})
        assert v["espn_from"] is None and v["kickoff"] is None and "not copied" in v["notes"][0]
        stale_espn = _espn_row("e", NEW, OPP, _utc("2026-07-06 17:00"), status="scheduled")
        v = ms.decide_cluster([fd_holder, stale_espn], {"k": _deps(holds_recs=True)})
        assert v["espn_from"] is None and v["kickoff"] is None and "not finished" in v["notes"][0]

    def test_a_dropped_football_data_twin_promote_raw_would_not_find_refuses(self):
        keep = _espn_row("k", NEW, OPP, _utc("2026-07-06 21:30"))
        fd = _m("f", OLD, OPP, _utc("2026-07-06 18:00"))  # corrects to 17:00Z: the kept row is 4.5 h off
        assert "promote_raw would insert" in ms.decide_cluster([keep, fd], {})["refuse"]

    def test_a_football_data_keep_never_takes_an_espn_kickoff_promote_raw_would_miss(self):
        fd_holder = _m("k", OLD, OPP, _utc("2026-07-05 13:00"))  # promote_raw searches 09:00-15:00Z on 07-05
        espn = _espn_row("e", NEW, OPP, _utc("2026-07-06 06:00"))
        v = ms.decide_cluster([fd_holder, espn], {"k": _deps(holds_recs=True)})
        assert "outside the window promote_raw searches" in v["refuse"]
        near = _espn_row("e", NEW, OPP, _utc("2026-07-05 14:30"))
        v = ms.decide_cluster([fd_holder, near], {"k": _deps(holds_recs=True)})
        assert v["espn_from"]["id"] == "e" and v["kickoff"] == near["match_date"]

    def test_kept_row_takes_a_missing_half_time(self):
        keep = _m("k", NEW, OPP, espn="1")
        v = ms.decide_cluster([keep, _m("d", OLD, OPP, ht=(0, 3))], {})
        assert v["keep"]["id"] == "k" and v["ht_from"]["id"] == "d" and v["adopt_from"] is None


# ---------------------------------------------------------------------------
# What moves off a dropped twin
# ---------------------------------------------------------------------------


def _odds(oid, ts, book="Pinnacle", sel="home", live=False):
    return {
        "id": oid,
        "bookmaker": book,
        "market_type": "1x2",
        "selection": sel,
        "line": None,
        "is_opening": False,
        "is_live": live,
        "timestamp": ts,
        "created_at": ts,
    }


def _snap(sid, ts, book="Pinnacle"):
    return {"id": sid, "bookmaker": book, "market_type": "1x2", "selection": "home", "line": None, "captured_at": ts}


def _pred(pid, created, updated=None, model="dixon_coles", version="v1", ptype="match_result"):
    return {
        "id": pid,
        "model_name": model,
        "model_version": version,
        "prediction_type": ptype,
        "created_at": created,
        "updated_at": updated,
    }


class TestPlanMoves:
    def test_only_the_freshest_pre_kickoff_price_per_key_and_never_in_play(self):
        drop = _deps(
            odds=[
                _odds("o1", T0 - timedelta(hours=5)),
                _odds("o2", T0 - timedelta(hours=1)),
                _odds("o3", T0 + timedelta(minutes=5)),
                _odds("o4", T0 - timedelta(hours=2), sel="away", live=True),
            ]
        )
        odds, _, _, _ = ms.plan_moves(drop, _deps(), T0, {OLD, OPP})
        assert [o["id"] for o in odds] == ["o2"]

    def test_the_kept_rows_own_prices_win(self):
        keep = _deps(odds=[_odds("k1", T0 - timedelta(hours=9))])
        odds, _, _, _ = ms.plan_moves(_deps(odds=[_odds("o1", T0 - timedelta(hours=1))]), keep, T0, {OLD})
        assert odds == []

    def test_snapshots_before_kickoff_the_kept_row_lacks(self):
        t = T0 - timedelta(hours=3)
        keep = _deps(snapshots=[_snap("k1", t)])
        drop = _deps(snapshots=[_snap("s1", t), _snap("s2", t - timedelta(hours=1)), _snap("s3", T0)])
        _, snaps, _, _ = ms.plan_moves(drop, keep, T0, {OLD})
        assert [s["id"] for s in snaps] == ["s2"]

    def test_match_stats_only_for_a_team_of_the_kept_row_it_lacks(self):
        keep = _deps(match_stats=[{"id": "k", "team_id": OPP}])
        drop = _deps(
            match_stats=[{"id": "a", "team_id": OLD}, {"id": "b", "team_id": OPP}, {"id": "c", "team_id": NEW}]
        )
        _, _, stats, _ = ms.plan_moves(drop, keep, T0, {OLD, OPP})
        assert [s["id"] for s in stats] == ["a"]

    def test_predictions_the_latest_pre_kickoff_write_per_free_key(self):
        drop = _deps(
            predictions=[
                _pred("p1", T0 - timedelta(days=2)),
                _pred("p2", T0 - timedelta(days=1)),  # the latest dixon_coles match_result: moves
                _pred("p3", T0 - timedelta(days=3), updated=T0 + timedelta(minutes=1), model="elo"),  # rewritten later
                _pred("p4", T0 - timedelta(days=1), ptype="over_under"),  # key held by the kept row
                _pred("p5", T0 + timedelta(hours=1), model="poisson"),  # written after kickoff
                _pred("p6", T0 - timedelta(hours=2), model="xgb", version=None),
            ]
        )
        keep = _deps(predictions=[_pred("k1", T0 - timedelta(days=5), ptype="over_under")])
        _, _, _, preds = ms.plan_moves(drop, keep, T0, {OLD})
        assert [p["id"] for p in preds] == ["p2", "p6"]

    def test_a_tie_moves_the_smallest_id(self):
        drop = _deps(predictions=[_pred("p9", T0 - timedelta(days=1)), _pred("p1", T0 - timedelta(days=1))])
        _, _, _, preds = ms.plan_moves(drop, _deps(), T0, {OLD})
        assert [p["id"] for p in preds] == ["p1"]

    def test_predictions_stay_put_when_the_kept_row_is_still_to_be_played(self):
        drop = _deps(predictions=[_pred("p1", T0 - timedelta(days=1))], odds=[_odds("o1", T0 - timedelta(hours=1))])
        odds, _, _, preds = ms.plan_moves(drop, _deps(), T0, {OLD}, move_predictions=False)
        assert preds == [] and [o["id"] for o in odds] == ["o1"]

    def test_a_second_twin_sees_what_the_first_moved(self):
        keep = _deps()
        ms.plan_moves(
            _deps(odds=[_odds("o1", T0 - timedelta(hours=1))], predictions=[_pred("p1", T0 - timedelta(days=1))]),
            keep,
            T0,
            {OLD},
        )
        odds, _, _, preds = ms.plan_moves(
            _deps(odds=[_odds("o2", T0 - timedelta(hours=2))], predictions=[_pred("p2", T0 - timedelta(days=2))]),
            keep,
            T0,
            {OLD},
        )
        assert odds == [] and preds == [] and [o["id"] for o in keep.odds] == ["o1"]


# ---------------------------------------------------------------------------
# Planning pairs
# ---------------------------------------------------------------------------


class TestPlanPair:
    def test_repoints_every_match_and_records_the_alias_and_the_espn_id(self):
        (p,) = _plan([_m("a", NEW, OPP), _m("b", OPP, NEW, T0 + timedelta(days=7))])
        assert p.action == "merge" and p.drops == []
        assert [(r["match_id"], r["side"]) for r in p.repoints] == [("a", "home"), ("b", "away")]
        t = p.team_patch
        assert t["aliases"] == ["bk häcken"] and t["espn"] == "7834" and t["stamp"] is True
        assert t["new_team_espn"] == "" and t["merged_from"][0]["team_id"] == NEW

    def test_the_twins_stamped_espn_id_moves_to_the_kept_team(self):
        teams = _teams()
        teams[NEW]["external_ids"] = {"espn": "7834"}
        (p,) = _plan([_m("a", NEW, OPP)], teams=teams)
        assert p.team_patch["new_team_espn"] == "7834" and p.team_patch["espn"] == "7834" and p.team_patch["stamp"]
        teams[NEW]["external_ids"] = {"espn": "5555"}
        (q,) = _plan([_m("a", NEW, OPP)], teams=teams, pairs=[_pair(espn="")])
        assert q.action == "merge" and q.team_patch["espn"] == "5555"

    def test_a_kept_team_already_holding_the_id_is_not_stamped_again(self):
        teams = _teams()
        teams[OLD]["external_ids"] = {"espn": "7834"}
        (p,) = _plan([_m("a", NEW, OPP)], teams=teams)
        assert p.action == "merge" and p.team_patch["stamp"] is False and p.team_patch["espn"] == "7834"
        # FC Tokyo: the sweep stamped the kept row, the twin brings nothing; the kept row keeps its id
        (q,) = _plan([_m("a", NEW, OPP)], teams=teams, pairs=[_pair(espn="")])
        assert q.team_patch["stamp"] is False and q.team_patch["espn"] == "7834"
        assert json.loads(ms.csv_rows([q])[-2]["after"])["espn"] == "7834"

    def test_exact_kickoff_collision_drops_the_twin_before_any_repoint(self):
        espn_row = _m("a", NEW, OPP, espn="401")
        fd_row = _m("b", OLD, OPP)
        (p,) = _plan([espn_row, fd_row], deps={"a": _deps(n_preds=57)})
        assert [(d.match_id, d.keep_id) for d in p.drops] == [("b", "a")]
        assert [r["match_id"] for r in p.repoints] == ["a"]
        assert p.drops[0].flags["kickoff_differs"] is True  # 17:00 stored = 16:00Z corrected, kept at 17:00

    def test_a_dropped_new_side_row_is_not_repointed(self):
        (p,) = _plan([_m("a", NEW, OPP, status="scheduled"), _m("b", OLD, OPP, T0 + timedelta(hours=1))])
        assert [d.match_id for d in p.drops] == ["a"] and p.repoints == []

    def test_a_kept_row_whose_kickoff_moves_is_repointed_at_its_new_kickoff(self):
        holder = _m("h", NEW, OPP, T0, status="scheduled")
        played = _espn_row("p", OLD, OPP, T0 + timedelta(hours=3))
        (p,) = _plan([holder, played], deps={"h": _deps(holds_recs=True)})
        (e,) = p.enrich
        assert e.sets["match_date"] == played["match_date"] and e.observed["match_date"] == T0
        (r,) = p.repoints
        assert r["match_id"] == "h" and r["observed"]["match_date"] == played["match_date"]

    def test_no_prediction_moves_onto_an_upcoming_fixture(self):
        # trg_predictions_updated_at would stamp the moved row updated_at = NOW() and generate_recommendations
        # takes the newest ensemble row by updated_at: an older model version would price recs until the next tick
        ahead = NOW + timedelta(days=5)
        keep = _espn_row("k", NEW, OPP, ahead, status="scheduled")
        stale = _m("s", OLD, OPP, ahead - timedelta(hours=1), status="scheduled")
        deps = {"s": _deps(predictions=[_pred("p1", NOW - timedelta(days=1))])}
        (p,) = _plan([keep, stale], deps=deps)
        assert [d.match_id for d in p.drops] == ["s"] and p.drops[0].prediction_ids == []
        assert not any(e.takes_predictions for e in p.enrich) and ms.summarize([p])["predictions_moved"] == 0

    def test_moved_predictions_onto_a_finished_row_are_listed_for_grading(self):
        keep = _espn_row("k", NEW, OPP, T0)
        stale = _m("s", OLD, OPP, T0 - timedelta(hours=1), status="scheduled")
        deps = {"s": _deps(predictions=[_pred("p1", T0 - timedelta(days=1))])}
        plans = _plan([keep, stale], deps=deps)
        assert plans[0].drops[0].prediction_ids == ["p1"] and plans[0].enrich[0].takes_predictions
        assert ms.grade_after(plans) == ["k"]
        assert ms.summarize(plans)["predictions_moved"] == 1

    def test_pairs_plan_on_the_state_the_earlier_ones_leave(self):
        pairs = [_pair(), _pair(new=NEW2, old=OLD2, new_name="Djurgården", old_name="Djurgarden IF", espn="2339")]
        espn_row = _m("e", NEW, NEW2, espn="5")
        fd_row = _m("f", OLD, OLD2)
        p1, p2 = _plan([espn_row, fd_row], pairs=pairs)
        assert p1.drops == [] and [r["side"] for r in p1.repoints] == ["home"]  # no twin until both are mapped
        assert [d.match_id for d in p2.drops] == ["f"]
        assert p2.repoints[0]["observed"]["home_team_id"] == OLD  # guarded on what pair 1 leaves

    def test_two_spellings_onto_one_kept_team_accumulate_aliases(self):
        pairs = [_pair(espn=""), _pair(new=NEW2, new_name="BK Hacken FF", espn="7834")]
        teams = {**_teams(), NEW2: _team(NEW2, "BK Hacken FF")}
        rows = [_m("a", NEW, OPP), _m("b", NEW2, OPP, T0 + timedelta(days=9))]
        p1, p2 = _plan(rows, pairs=pairs, teams=teams)
        assert p1.action == p2.action == "merge"
        assert p2.team_patch["observed_aliases"] == ["bk häcken"]
        assert p2.team_patch["aliases"] == ["bk hacken ff", "bk häcken"]
        assert p2.old_team["metadata"]["aliases"] == ["bk häcken"]

    def test_an_alias_already_there_in_another_spelling_is_not_added_twice(self):
        teams = _teams()
        teams[OLD]["metadata"] = {"aliases": ["BK Hacken"]}  # alias_form folds the accent: the same alias
        (p,) = _plan([_m("a", NEW, OPP)], teams=teams)
        assert p.action == "merge" and p.team_patch["aliases"] == ["BK Hacken"]

    def test_a_pre_merge_fork_is_folded(self):
        stale = _m("s", NEW, OPP, T0 - timedelta(hours=2), status="scheduled")
        (p,) = _plan([stale, _m("p", NEW, OPP)])
        assert [(d.match_id, d.keep_id, d.fork) for d in p.drops] == [("s", "p", True)]

    def test_a_fork_the_rules_refuse_is_left_alone_and_the_pair_still_merges(self):
        a = _m("a", NEW, OPP, T0)
        b = _m("b", NEW, OPP, T0 + timedelta(hours=30), score=(0, 0))
        (p,) = _plan([a, b])
        assert p.action == "merge" and p.drops == [] and p.forks[0]["ids"] == ("a", "b")

    def test_a_merge_made_cluster_the_rules_refuse_refuses_the_whole_pair(self):
        (p,) = _plan([_m("a", NEW, OPP), _m("b", OLD, OPP, score=(0, 0)), _m("c", NEW, OPP, T0 + timedelta(days=7))])
        assert p.action == "refuse" and "disagree" in p.reason and p.repoints == [] and p.drops == []

    def test_a_refused_pair_leaves_the_state_for_the_next_pair_untouched(self):
        pairs = [_pair(), _pair(new=NEW2, old=OLD2, new_name="Djurgården", old_name="Djurgarden IF", espn="")]
        rows = [_m("a", NEW, OPP), _m("b", OLD, OPP, score=(0, 0)), _m("c", NEW2, OPP, T0 + timedelta(days=3))]
        p1, p2 = _plan(rows, pairs=pairs)
        assert p1.action == "refuse" and p2.action == "merge"
        assert p2.old_team["metadata"] == {}

    def test_a_match_against_itself_is_refused(self):
        (p,) = _plan([_m("a", NEW, OLD)])
        assert p.action == "refuse" and "against itself" in p.reason

    def test_a_merge_made_swapped_twin_is_refused_but_an_old_one_is_ignored(self):
        (p,) = _plan([_m("a", NEW, OPP), _m("b", OPP, OLD, T0 + timedelta(hours=2))])
        assert p.action == "refuse" and "swapped" in p.reason
        (q,) = _plan([_m("a", NEW, OPP), _m("b", OPP, NEW, T0 + timedelta(hours=2), status="scheduled")])
        assert q.action == "merge"

    def test_done_when_the_new_team_is_gone_and_the_merge_is_recorded(self):
        teams = _teams()
        del teams[NEW]
        teams[OLD]["metadata"] = {"aliases": ["BK Hacken"], "merged_from": [{"team_id": NEW}]}
        (p,) = _plan([], teams=teams)
        assert p.action == "done"
        teams[OLD]["metadata"] = {}
        (q,) = _plan([], teams=teams)
        assert q.action == "refuse" and "does not record" in q.reason

    @pytest.mark.parametrize(
        "change, match",
        [
            (lambda t: t[NEW].update(name="Häcken"), "the map says"),
            (lambda t: t[OLD].update(name="Hacken BK"), "the map says"),
            (lambda t: t[NEW].update(external_ids={"espn": "1"}), "new team carries ESPN team 1"),
            (lambda t: t[OLD].update(external_ids={"espn": "1"}), "already carries ESPN team 1"),
            (lambda t: t[OLD].update(metadata={"aliases": "hacken"}), "not an array of names"),
            (lambda t: t[OLD].update(metadata={"aliases": ["hacken", 7]}), "not an array of names"),
            (lambda t: t.pop(OLD), "kept team row not found"),
        ],
    )
    def test_team_drift_is_refused(self, change, match):
        teams = _teams()
        change(teams)
        (p,) = _plan([_m("a", NEW, OPP)], teams=teams)
        assert p.action == "refuse" and match in p.reason

    def test_new_team_referenced_outside_matches_is_refused(self):
        (p,) = _plan([_m("a", NEW, OPP)], refs={NEW: {"match_stats": 2, "players": 0}})
        assert p.action == "refuse" and "match_stats" in p.reason

    def test_adopted_result_is_listed_for_grading(self):
        holder = _m("s", OLD, OPP, T0, status="scheduled")
        played = _m("p", NEW, OPP, T0, espn="9", ht=(0, 0))
        plans = _plan([holder, played], deps={"s": _deps(holds_recs=True, open_recs=2)})
        (e,) = plans[0].enrich
        assert e.sets == {
            "status": "finished",
            "home_score": 2,
            "away_score": 4,
            "half_time_home": 0,
            "half_time_away": 0,
        }
        assert e.external_patch == {"espn": "9"} and e.metadata_patch == {"result_adopted_from": "p"}
        assert ms.adopted_with_open_recs(plans) == ["s"] and ms.grade_after(plans) == ["s"]

    def test_plan_all_leaves_its_inputs_alone(self):
        rows = [_m("a", NEW, OPP, espn="1"), _m("b", OLD, OPP)]
        state = {m["id"]: m for m in rows}
        teams = _teams()
        deps = {"b": _deps(odds=[_odds("o", T0 - timedelta(hours=1))])}
        ctx = _ctx(teams)
        ms.plan_all([_pair()], teams, {}, state, deps, NOW, ctx)
        assert state["a"]["home_team_id"] == NEW and NEW in teams and deps.get("a") is None
        assert NEW in ctx.names["bk häcken"] and "bk hacken" not in ctx.aliases


class TestPredictedResolution:
    """The dry run refuses a pair the resolvers would not resolve after the merge."""

    def test_another_row_holding_the_exact_name_refuses(self):
        teams = _teams()
        ctx = _ctx(teams, extra=[_team(THIRD, "BK Häcken")])
        (p,) = _plan([_m("a", NEW, OPP)], teams=teams, ctx=ctx)
        assert p.action == "refuse" and "exact name" in p.reason

    def test_another_team_claiming_the_alias_refuses(self):
        teams = _teams()
        ctx = _ctx(teams, aliases={"bk hacken": {THIRD}})
        (p,) = _plan([_m("a", NEW, OPP)], teams=teams, ctx=ctx)
        assert p.action == "refuse" and "already claim the alias 'bk hacken'" in p.reason

    def test_two_pairs_bringing_one_alias_to_two_kept_teams_refuse_the_second(self):
        pairs = [_pair(espn=""), _pair(new=NEW2, old=OLD2, new_name="BK Hacken", old_name="Djurgarden IF", espn="")]
        teams = {**_teams(), NEW2: _team(NEW2, "BK Hacken")}  # alias_form folds the accent: one alias
        p1, p2 = _plan([_m("a", NEW, OPP), _m("b", NEW2, OPP, T0 + timedelta(days=4))], pairs=pairs, teams=teams)
        assert p1.action == "merge" and p2.action == "refuse" and "already claim" in p2.reason

    def test_a_third_row_holding_the_espn_id_refuses(self):
        teams = _teams()
        ctx = _ctx(teams, extra=[_team(THIRD, "Someone", external_ids={"espn": "7834"})])
        (p,) = _plan([_m("a", NEW, OPP)], teams=teams, ctx=ctx)
        assert p.action == "refuse" and "also stamped on" in p.reason

    def test_a_third_row_holding_the_kept_teams_own_id_refuses(self):
        teams = _teams()
        teams[OLD]["external_ids"] = {"espn": "4242"}
        ctx = _ctx(teams, extra=[_team(THIRD, "Someone", external_ids={"espn": "4242"})])
        (p,) = _plan([_m("a", NEW, OPP)], teams=teams, pairs=[_pair(espn="")], ctx=ctx)
        assert p.action == "refuse" and "ESPN team 4242 is also stamped on" in p.reason

    def test_the_seed_must_agree(self):
        teams = _teams()
        ctx = _ctx(teams, seed=_seed(("7834", "BK Hacken Women")))
        (p,) = _plan([_m("a", NEW, OPP)], teams=teams, ctx=ctx)
        assert p.action == "refuse" and "the ESPN seed pins" in p.reason
        ok = _ctx(teams, seed=_seed(("7834", "Hacken")))
        (q,) = _plan([_m("a", NEW, OPP)], teams=teams, ctx=ok)
        assert q.action == "merge"


# ---------------------------------------------------------------------------
# Writing one pair
# ---------------------------------------------------------------------------

SCHEMA, _ = ms.read_schema(PROD_FKS)


class _Cursor:
    def __init__(self, conn):
        self.conn, self.rowcount, self._rows = conn, 0, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        c = self.conn
        c.log.append((sql, params))
        self._rows = []
        if sql == ms.LOCK_TEAMS_SQL:
            self._rows = [dict(t) for t in c.teams]
        elif sql == ms.LOCK_NEW_TEAM_MATCHES_SQL:
            self._rows = [{"id": i} for i in c.new_matches]
        elif sql == ms.new_team_refs_sql(SCHEMA):
            self._rows = [dict(c.refs)]
        elif sql == ms.TWINS_AFTER_SQL:
            self._rows = list(c.twins)
        elif sql == ms.twin_children_sql(SCHEMA):
            self._rows = [{"tbl": t, "id": i} for t, i in c.children.get(params["m"], [])]
        elif sql == ms.REPOINTED_FIXTURE_SQL:
            self._rows = [
                {
                    "match_date": T0 + timedelta(days=7),
                    "status": "scheduled",
                    "home_name": "Djurgarden",
                    "away_name": "Hacken",
                }
            ]
        elif sql.lstrip().startswith(("UPDATE", "DELETE")):
            self.rowcount = c.rowcount(sql, params)
            if c.on_write:
                c.on_write(sql)

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _Conn:
    def __init__(
        self, plan, rowcount=None, new_matches=None, teams=None, refs=None, twins=(), on_write=None, children=None
    ):
        self.log: list = []
        self.children = children or {}  # twin id -> [(table, row id)] it holds at delete time
        self.teams = teams if teams is not None else [plan.new_team, plan.old_team]
        self.new_matches = new_matches if new_matches is not None else plan.new_match_ids
        self.refs = refs or {"matches": 0, "match_stats": 0, "players": 0}
        self.twins = twins
        self.rowcount = rowcount or (lambda sql, params: len(params.get("ids", [1])) if "ids" in params else 1)
        self.on_write = on_write
        self.committed = self.rolled_back = 0

    def cursor(self, cursor_factory=None):
        return _Cursor(self)

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1

    def writes(self):
        return [s for s, _ in self.log if s.lstrip().startswith(("UPDATE", "DELETE"))]

    def team_deleted(self):
        return any(s == ms.team_delete_sql(SCHEMA) for s, _ in self.log)


class _Resolvers:
    """Stand-ins for the deployed resolvers, logging into the connection."""

    def __init__(self, conn, team=OLD, team_espn=OLD, match="c", stamp=True, odds=None):
        self.conn, self.team, self.team_espn, self.match, self.stamp = conn, team, team_espn, match, stamp
        self.odds = odds or {}  # odds-feed spelling -> (fixture found before the team delete, after it)

    def build(self):
        def find_team(cur, name, league, espn):
            self.conn.log.append((f"find_team {name} {espn}", None))
            return self.team_espn if espn else self.team

        def find_match_id(cur, home, away, when):
            self.conn.log.append((f"find_match_id {home} v {away}", None))
            for spelling, (before, after) in self.odds.items():
                if spelling in (home, away):
                    return after if self.conn.team_deleted() else before
            return self.match

        def stamp_espn_id(cur, team_id, espn, label=""):
            self.conn.log.append((f"stamp {team_id} {espn}", None))
            return self.stamp

        return ms.Resolvers(ti.alias_form, lambda c, s: {}, lambda: {}, stamp_espn_id, find_team, find_match_id)


def _merge_plan(twin_espn=""):
    espn_row = _m("a", NEW, OPP, espn="401")
    fd_row = _m("b", OLD, OPP)
    later = _m("c", OPP, NEW, T0 + timedelta(days=7), status="scheduled")
    deps = {
        "b": _deps(
            odds=[_odds("o1", T0 - timedelta(hours=1))],
            match_stats=[{"id": "s1", "team_id": OLD}],
            predictions=[_pred("p1", T0 - timedelta(days=1))],
        )
    }
    teams = _teams()
    teams[NEW]["external_ids"] = {"espn": twin_espn} if twin_espn else {}
    (p,) = _plan([espn_row, fd_row, later], deps=deps, teams=teams)
    return p


def _apply(p, conn=None, archived=None, **kw):
    conn = conn or _Conn(p)
    return conn, ms.apply_pair(conn, p, _Resolvers(conn, **kw).build(), SCHEMA, archived or {})


class TestApplyPair:
    def test_writes_in_order_in_one_transaction_and_checks_the_resolvers_last(self):
        p = _merge_plan(twin_espn="7834")
        conn, done = _apply(p)
        # moves before the twin is dropped, the drop before any re-point, the ESPN id off the twin before the
        # kept team is stamped, the team row deleted, and only then the resolvers, then COMMIT
        assert [s for s, _ in conn.log if s in conn.writes() or not s.lstrip().startswith(("SELECT", "SET"))] == [
            ms.MOVE_ODDS_SQL,
            ms.MOVE_MATCH_STATS_SQL,
            ms.MOVE_PREDICTIONS_SQL,
            ms.DROP_TWIN_SQL,
            ms.REPOINT_SQL["home"],
            ms.REPOINT_SQL["away"],
            ms.CLEAR_TWIN_ESPN_SQL,
            ms.TEAM_RECORD_SQL,
            f"stamp {OLD} 7834",
            ms.team_delete_sql(SCHEMA),
            "find_team BK Häcken None",
            "find_team BK Häcken 7834",
            "find_match_id Djurgarden v BK Häcken",
        ]
        assert conn.committed == 1 and conn.rolled_back == 0
        assert done["twins_dropped"] == 1 and done["matches_repointed"] == 2 and done["teams_deleted"] == 1
        assert done["moved_predictions"] == 1 and done["espn_stamped"] == 1 and done["resolver_checks"] == 3

    def test_the_moves_and_the_drop_are_guarded_on_the_plan(self):
        p = _merge_plan()
        conn, _ = _apply(p)
        (params,) = [prm for s, prm in conn.log if s == ms.DROP_TWIN_SQL]
        assert params["id"] == "b" and params["home_team_id"] == OLD and params["status"] == "finished"
        (moved,) = [prm for s, prm in conn.log if s == ms.MOVE_PREDICTIONS_SQL]
        assert moved == {"keep": "a", "drop": "b", "ids": ["p1"], "cutoff": T0}
        assert "NOT EXISTS (SELECT 1 FROM betting_recommendations" in ms.DROP_TWIN_SQL

    def test_no_twin_stamp_means_no_clear(self):
        p = _merge_plan()
        conn, _ = _apply(p)
        assert ms.CLEAR_TWIN_ESPN_SQL not in conn.writes() and any(s == f"stamp {OLD} 7834" for s, _ in conn.log)

    @pytest.mark.parametrize(
        "kw, match",
        [
            ({"team": OPP}, "find_team\\('BK Häcken'\\) resolves to"),
            ({"team_espn": None}, "ESPN 7834\\) resolves to None"),
            ({"match": None}, "find_match_id"),
            ({"stamp": False}, "guarded stamp refused"),
        ],
    )
    def test_a_resolver_miss_or_a_refused_stamp_rolls_the_pair_back(self, kw, match):
        p = _merge_plan()
        conn = _Conn(p)
        with pytest.raises(ms.GuardError, match=match):
            _apply(p, conn, **kw)
        assert conn.committed == 0 and conn.rolled_back == 1

    def test_a_pair_without_a_resolvable_repoint_skips_only_the_odds_check(self):
        p = _merge_plan()
        p.repoints = [r for r in p.repoints if r["observed"]["status"] == "cancelled"]
        conn, done = _apply(p, match=None)
        assert done["resolver_checks"] == 2 and conn.committed == 1

    def test_a_row_that_moved_rolls_the_pair_back(self):
        p = _merge_plan()
        conn = _Conn(p, rowcount=lambda sql, params: 0 if sql == ms.REPOINT_SQL["away"] else 1)
        with pytest.raises(ms.GuardError, match="repoint away"):
            _apply(p, conn)
        assert conn.committed == 0 and conn.rolled_back == 1

    def test_a_fixture_added_since_the_plan_rolls_the_pair_back_before_any_write(self):
        p = _merge_plan()
        conn = _Conn(p, new_matches=p.new_match_ids + ["zzz"])
        with pytest.raises(ms.GuardError, match="gained"):
            _apply(p, conn)
        assert conn.writes() == [] and conn.committed == 0

    def test_a_team_changed_since_the_plan_rolls_back_before_any_write(self):
        p = _merge_plan()
        conn = _Conn(p, teams=[{**p.new_team, "external_ids": {"espn": "7834"}}, p.old_team])
        with pytest.raises(ms.GuardError, match="changed since the plan"):
            _apply(p, conn)
        assert conn.writes() == []

    def test_a_reference_left_on_the_new_team_rolls_back(self):
        p = _merge_plan()
        conn = _Conn(p, refs={"matches": 1, "players": 0})
        with pytest.raises(ms.GuardError, match="still referenced"):
            _apply(p, conn)
        assert ms.team_delete_sql(SCHEMA) not in conn.writes() and conn.committed == 0

    def test_a_twin_left_after_the_repoint_rolls_back(self):
        p = _merge_plan()
        conn = _Conn(p, twins=[{"a": "a", "b": "q"}])
        with pytest.raises(ms.GuardError, match="twin"):
            _apply(p, conn)
        assert conn.committed == 0

    def test_a_kickoff_move_is_written_guarded_on_the_observed_kickoff(self):
        holder = _m("h", NEW, OPP, T0, status="scheduled")
        played = _espn_row("p", OLD, OPP, T0 + timedelta(hours=3))
        (p,) = _plan([holder, played], deps={"h": _deps(holds_recs=True)})
        conn, _ = _apply(p, match="h")
        (params,) = [prm for s, prm in conn.log if s == ms.ENRICH_SQL]
        assert params["match_date"] == played["match_date"] and params["o_match_date"] == T0
        (rp,) = [prm for s, prm in conn.log if s == ms.REPOINT_SQL["home"]]
        assert rp["match_date"] == played["match_date"]
        assert "match_date = %(o_match_date)s" in ms.ENRICH_SQL


class TestOddsSpellingChecks:
    """Every odds-feed spelling that found the probe fixture before the pair's
    first write must still find it before COMMIT."""

    SPELLING = "Hacken FC"

    def _plan(self):
        p = _merge_plan()
        p.odds_names = [self.SPELLING]
        return p

    def test_a_spelling_that_found_the_fixture_must_still_find_it(self):
        p = self._plan()
        conn, done = _apply(p, odds={self.SPELLING: ("c", "c")})
        assert conn.committed == 1 and done["odds_spelling_checks"] == 1 and done["resolver_checks"] == 3
        probes = [i for i, (s, _) in enumerate(conn.log) if s == "find_match_id Djurgarden v Hacken FC"]
        first_write = next(i for i, (s, _) in enumerate(conn.log) if s in conn.writes())
        assert len(probes) == 2 and probes[0] < first_write < probes[1]  # once before the first write, once after

    def test_one_that_stops_finding_it_rolls_the_pair_back(self):
        p = self._plan()
        conn = _Conn(p)
        with pytest.raises(ms.GuardError, match="the odds feed's spelling"):
            _apply(p, conn, odds={self.SPELLING: ("c", None)})
        assert conn.committed == 0 and conn.rolled_back == 1

    def test_one_that_missed_before_is_not_held_to_anything(self, caplog):
        p = self._plan()
        with caplog.at_level("WARNING", logger="merge_soccer_teams"):
            conn, done = _apply(p, odds={self.SPELLING: (None, None)})
        assert conn.committed == 1 and done["odds_spellings_unmatched_before"] == 1 and not done["odds_spelling_checks"]
        (warning,) = [r for r in caplog.records if r.levelname == "WARNING" and "did not find" in r.getMessage()]
        assert "'Hacken FC'" in warning.getMessage() and "BK Häcken -> Hacken" in warning.getMessage()

    def test_a_hit_on_a_dropped_twin_counts_as_its_kept_row(self):
        p = self._plan()
        p.repoints = [r for r in p.repoints if r["match_id"] == "a"]  # the probe is 'a', which keeps twin 'b'
        conn, done = _apply(p, match="a", odds={self.SPELLING: ("b", "a")})
        assert conn.committed == 1 and done["odds_spelling_checks"] == 1
        conn = _Conn(p)
        with pytest.raises(ms.GuardError, match="found the fixture a before the merge"):
            _apply(p, conn, match="a", odds={self.SPELLING: ("b", "b")})


class TestArchivedChildren:
    def test_a_row_written_on_a_twin_after_the_archive_rolls_the_pair_back(self):
        p = _merge_plan()
        conn = _Conn(p, children={"b": [("odds", "o1"), ("odds", "o9")]})
        with pytest.raises(ms.GuardError, match="written after the archive.*odds:o9"):
            _apply(p, conn, archived={"b": {"odds": {"o1"}}})
        assert ms.DROP_TWIN_SQL not in conn.writes() and conn.committed == 0 and conn.rolled_back == 1

    def test_rows_all_in_the_archive_let_the_twin_go(self):
        p = _merge_plan()
        conn = _Conn(p, children={"b": [("odds", "o1"), ("predictions", "p7")]})
        _, done = _apply(p, conn, archived={"b": {"odds": {"o1"}, "predictions": {"p7"}}})
        assert done["twins_dropped"] == 1 and conn.committed == 1

    def test_pair_j_drops_the_row_pair_i_kept_after_rows_were_moved_onto_it(self):
        """Pair i folds fd twin 'b' into ESPN row 'a' and moves b's price o1
        onto 'a'; pair j (a second spelling of the opponent) then drops 'a'
        into the rec holder 'd'. o1 was archived under 'b', yet it is in the
        archive: pair j must go through, not roll back as 'written after the
        archive'."""
        teams = {k: v for k, v in _teams().items() if k != NEW2}
        pair_j = ms.MapRow(OPP, OLD2, "Djurgarden", "Djurgarden IF", "", "exact-alias", "key_equal 1.000")
        a, b, d = _espn_row("a", NEW, OPP), _m("b", OLD, OPP), _m("d", OLD, OLD2)
        deps = {
            "b": _deps(odds=[_odds("o1", T0 - timedelta(hours=1))]),
            "a": _deps(odds=[_odds("oa", T0 - timedelta(hours=2), book="Bet365")]),
            "d": _deps(holds_recs=True),
        }
        plan_i, plan_j = _plan([a, b, d], deps=deps, teams=teams, pairs=[_pair(), pair_j])
        assert [(x.match_id, x.keep_id) for x in plan_i.drops] == [("b", "a")]
        assert [(x.match_id, x.keep_id) for x in plan_j.drops] == [("a", "d")]
        assert plan_i.drops[0].odds_ids == ["o1"]
        # write_archive files each child under the twin it hung off at archive time, before any write
        archived = {"b": {"odds": {"o1"}}, "a": {"odds": {"oa"}}}
        conn_i = _Conn(plan_i, children={"b": []})  # o1 left 'b' in pair i's own move
        _, done_i = _apply(plan_i, conn_i, archived=archived, match="a")
        assert done_i["moved_odds"] == 1 and conn_i.committed == 1
        conn_j = _Conn(plan_j, children={"a": [("odds", "o1"), ("odds", "oa")]})
        _, done_j = _apply(plan_j, conn_j, archived=archived, team=OLD2, team_espn=OLD2, match="d")
        assert done_j["twins_dropped"] == 1 and conn_j.committed == 1 and conn_j.rolled_back == 0
        # a row no twin's archive holds still stops it
        conn_x = _Conn(plan_j, children={"a": [("odds", "o1"), ("odds", "oa"), ("odds", "o9")]})
        with pytest.raises(ms.GuardError, match="written after the archive.*odds:o9"):
            _apply(plan_j, conn_x, archived=archived, team=OLD2, team_espn=OLD2, match="d")

    def test_archived_ids_unions_every_twin(self):
        got = ms.archived_ids({"b": {"odds": {"o1"}}, "a": {"odds": {"oa"}, "predictions": {"p1"}}, "c": {}})
        assert got == {"odds": {"o1", "oa"}, "predictions": {"p1"}}

    def test_the_children_query_covers_every_child_table_the_schema_has(self):
        sql = ms.twin_children_sql(SCHEMA)
        for t in SCHEMA.match_children:
            assert f"FROM {t} WHERE match_id = %(m)s::uuid" in sql


def test_sql_guards_reference_observed_values():
    assert "home_team_id = %(new)s::uuid" in ms.REPOINT_SQL["home"]
    assert "match_date = %(match_date)s" in ms.REPOINT_SQL["away"]
    assert "status = %(o_status)s" in ms.ENRICH_SQL
    assert "observed_aliases" in ms.TEAM_RECORD_SQL and "observed_espn" in ms.TEAM_RECORD_SQL
    assert "external_ids" not in ms.TEAM_RECORD_SQL.split("WHERE")[0]  # the ESPN id goes on only via the stamp
    assert "external_ids->>'espn' = %(espn)s" in ms.CLEAR_TWIN_ESPN_SQL
    assert 'o."timestamp" < %(cutoff)s' in ms.MOVE_ODDS_SQL and "NOT COALESCE(o.is_live, false)" in ms.MOVE_ODDS_SQL
    assert "s.captured_at < %(cutoff)s" in ms.MOVE_SNAPSHOTS_SQL
    assert "COALESCE(p.updated_at, p.created_at) < %(cutoff)s" in ms.MOVE_PREDICTIONS_SQL
    assert "p.match_id = %(drop)s::uuid" in ms.MOVE_PREDICTIONS_SQL
    assert "t.model_version IS NOT DISTINCT FROM p.model_version" in ms.MOVE_PREDICTIONS_SQL


# ---------------------------------------------------------------------------
# The run: audit trail first, archive before the first write, --expect-plan
# ---------------------------------------------------------------------------


class _RunConn:
    """Hands back fixed rows per SQL; records writes and, at the first write,
    whether the archive was already on disk."""

    def __init__(self, teams, matches, fks=PROD_FKS, archive=None, children=None):
        self.teams, self.matches, self.fks = teams, matches, fks
        self.archive, self.children = archive, children or {}
        self.writes, self.committed = [], 0
        self.archive_before_first_write = None

    def cursor(self, cursor_factory=None):
        conn = self

        class C:
            rowcount = 1

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, params=None):
                self.sql, self.params = sql, params
                if sql.lstrip().startswith(("UPDATE", "DELETE")):
                    if conn.archive_before_first_write is None:
                        conn.archive_before_first_write = conn.archive is not None and conn.archive.exists()
                    conn.writes.append(sql)
                    self.rowcount = len(params.get("ids", [1])) if isinstance(params, dict) and "ids" in params else 1

            def fetchall(self):
                s = self.sql
                if s == ms.FOREIGN_KEYS_SQL:
                    return [dict(f) for f in conn.fks]
                if s in (ms.TEAMS_SQL, ms.LOCK_TEAMS_SQL):
                    return [dict(t) for t in conn.teams]
                if s == ms.LOCK_NEW_TEAM_MATCHES_SQL:
                    return [{"id": m["id"]} for m in conn.matches if NEW in (m["home_team_id"], m["away_team_id"])]
                if s.startswith("SELECT t.id::text AS id"):
                    return [{"id": NEW, "match_stats": 0, "players": 0}]
                if s == ms.MATCHES_SQL:
                    return [dict(m) for m in conn.matches]
                if s == ms.ALL_SOCCER_TEAMS_SQL:
                    return [{"id": t["id"], "normalized_name": t["normalized_name"], "espn": None} for t in conn.teams]
                if s.startswith("SELECT * FROM teams"):
                    return [dict(t) for t in conn.teams if t["id"] in self.params[0]]
                if s.startswith("SELECT * FROM matches"):
                    return [dict(m, id=m["id"]) for m in conn.matches if m["id"] in self.params[0]]
                for table, rows in conn.children.items():
                    if s.startswith(f"SELECT * FROM {table} WHERE match_id"):
                        return [r for r in rows if r["match_id"] in self.params[0]]
                if s == ms.twin_children_sql(SCHEMA):
                    return [
                        {"tbl": t, "id": r["id"]}
                        for t, rows in conn.children.items()
                        for r in rows
                        if r["match_id"] == self.params["m"]
                    ]
                return []

            def fetchone(self):
                s = self.sql
                if s == ms.new_team_refs_sql(SCHEMA):
                    return {"matches": 0}
                if s == ms.REPOINTED_FIXTURE_SQL:
                    return {"match_date": T0, "status": "finished", "home_name": "Hacken", "away_name": "Djurgarden"}
                return None

        return C()

    def commit(self):
        self.committed += 1

    def rollback(self):
        pass


def _csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _run_resolvers(seed=None, team=OLD, match="a"):
    return ms.Resolvers(
        ti.alias_form,
        lambda cur, sport: {},
        lambda: seed or {},
        lambda cur, t, e, label="": True,
        lambda cur, name, league, espn: team,
        lambda cur, home, away, when: match,
    )


class TestRun:
    def _conn(self, **kw):
        rows = [_m("a", NEW, OPP), _m("b", OPP, NEW, T0 + timedelta(days=7))]
        return _RunConn(list(_teams().values()), rows, **kw)

    def test_dry_run_writes_the_audit_csv_and_the_plan_file_and_nothing_else(self, tmp_path):
        conn = self._conn()
        rc = ms.run(conn, False, [_pair()], tmp_path / "a.csv", tmp_path / "a.jsonl.gz", _run_resolvers())
        rows = _csv(tmp_path / "a.csv")
        assert rc == 0 and conn.writes == [] and conn.committed == 0
        assert [r["action"] for r in rows] == ["repoint_home", "repoint_away", "record_alias", "delete_team"]
        assert json.loads(rows[0]["after"]) == {"home_team_id": OLD}
        doc = json.loads((tmp_path / "a.plan.json").read_text())
        assert doc["mode"] == "dry-run" and doc["pairs"][0]["action"] == "merge" and len(doc["digest"]) == 64
        assert not (tmp_path / "a.jsonl.gz").exists()

    def test_without_the_resolvers_nothing_is_read_or_written(self, tmp_path):
        conn = self._conn()
        for apply in (True, False):
            rc = ms.run(conn, apply, [_pair()], tmp_path / "a.csv", tmp_path / "a.gz", None, "no resolver")
            assert rc == 2 and conn.writes == [] and not (tmp_path / "a.csv").exists()

    def test_an_unknown_foreign_key_refuses_the_run(self, tmp_path):
        conn = self._conn(fks=PROD_FKS + [_fk("match_notes", "match_id", "matches")])
        rc = ms.run(conn, True, [_pair()], tmp_path / "a.csv", tmp_path / "a.gz", _run_resolvers())
        assert rc == 2 and conn.writes == [] and not (tmp_path / "a.csv").exists()

    def test_a_map_row_the_seed_contradicts_refuses_the_run(self, tmp_path):
        conn = self._conn()
        seed = _seed(("7834", "Somebody Else"))
        rc = ms.run(conn, True, [_pair()], tmp_path / "a.csv", tmp_path / "a.gz", _run_resolvers(seed=seed))
        assert rc == 2 and conn.writes == [] and not (tmp_path / "a.csv").exists()

    def test_the_audit_csv_and_the_plan_file_are_write_once(self, tmp_path):
        out = tmp_path / "a.csv"
        out.write_text("x")
        with pytest.raises(FileExistsError):
            ms.run(self._conn(), True, [_pair()], out, tmp_path / "a.jsonl.gz", _run_resolvers())
        (tmp_path / "b.plan.json").write_text("x")
        conn = self._conn()
        with pytest.raises(FileExistsError):
            ms.run(conn, True, [_pair()], tmp_path / "b.csv", tmp_path / "b.jsonl.gz", _run_resolvers())
        assert conn.writes == []

    def test_a_refused_pair_fails_the_dry_run(self, tmp_path):
        conn = _RunConn([_team(OLD, "Hacken"), _team(OPP, "Djurgarden")], [])
        rc = ms.run(conn, False, [_pair()], tmp_path / "a.csv", tmp_path / "a.gz", _run_resolvers())
        assert rc == 1 and _csv(tmp_path / "a.csv")[0]["action"] == "pair_refuse"

    def _twin_run(self, tmp_path):
        rows = [_m("a", NEW, OPP, espn="401"), _m("b", OLD, OPP)]
        archive = tmp_path / "a.jsonl.gz"
        children = {"predictions": [{"id": "p1", "match_id": "b"}], "odds": [{"id": "o1", "match_id": "b"}]}
        return _RunConn(list(_teams().values()), rows, archive=archive, children=children), archive

    def test_apply_writes_the_archive_before_the_first_write(self, tmp_path):
        conn, archive = self._twin_run(tmp_path)
        rc = ms.run(conn, True, [_pair()], tmp_path / "a.csv", archive, _run_resolvers())
        assert rc == 0 and conn.writes and conn.archive_before_first_write is True and conn.committed == 1
        recs = [json.loads(ln) for ln in gzip.open(archive, "rt")]
        roles = {(r["table"], r["role"]) for r in recs}
        assert ("matches", "dropped_twin") in roles and ("teams", "team_before") in roles
        assert ("predictions", "dropped_twin_child") in roles and ("odds", "dropped_twin_child") in roles
        assert next(r for r in recs if r["role"] == "dropped_twin")["keep_id"] == "a"

    def test_the_archive_is_write_once(self, tmp_path):
        conn, archive = self._twin_run(tmp_path)
        archive.write_bytes(b"x")
        with pytest.raises(FileExistsError):
            ms.run(conn, True, [_pair()], tmp_path / "a.csv", archive, _run_resolvers())
        assert conn.writes == []

    def test_the_plan_line_carries_kickoffs_adopted(self, tmp_path, caplog):
        with caplog.at_level("INFO", logger="merge_soccer_teams"):
            ms.run(self._conn(), False, [_pair()], tmp_path / "a.csv", tmp_path / "a.gz", _run_resolvers())
        assert "PLAN pairs_merge=1 pairs_done=0 pairs_refuse=0 kickoffs_adopted=0 digest=" in caplog.text

    def test_the_odds_spellings_reach_the_in_transaction_check(self, tmp_path):
        conn, archive = self._twin_run(tmp_path)
        asked = []

        def find_match_id(cur, home, away, when):
            asked.append((home, away))
            return "a"

        resolvers = ms.Resolvers(
            ti.alias_form, lambda c, s: {}, lambda: {}, lambda *a, **k: True, lambda *a: OLD, find_match_id
        )
        rc = ms.run(conn, True, [_pair()], tmp_path / "a.csv", archive, resolvers, odds_names={NEW: ["Hacken FC"]})
        assert rc == 0 and asked.count(("Hacken FC", "Djurgarden")) == 2  # before the first write and after

    def test_apply_refuses_a_plan_other_than_the_dry_runs(self, tmp_path):
        conn, archive = self._twin_run(tmp_path)
        ms.run(conn, False, [_pair()], tmp_path / "d.csv", archive, _run_resolvers())
        expected = json.loads((tmp_path / "d.plan.json").read_text())
        same, _ = self._twin_run(tmp_path)
        assert (
            ms.run(
                same,
                True,
                [_pair()],
                tmp_path / "s.csv",
                tmp_path / "s.gz",
                _run_resolvers(),
                None,
                expect_plan=expected,
            )
            == 0
        )
        moved, _ = self._twin_run(tmp_path)
        moved.matches.append(_m("c", NEW, OPP, T0 + timedelta(days=9), status="scheduled"))
        rc = ms.run(
            moved, True, [_pair()], tmp_path / "x.csv", tmp_path / "x.gz", _run_resolvers(), None, expect_plan=expected
        )
        assert rc == 2 and moved.writes == [] and not (tmp_path / "x.gz").exists()
        assert "1 added" in ms.plan_difference(expected, json.loads((tmp_path / "x.plan.json").read_text()))


# ---------------------------------------------------------------------------
# The after-state
# ---------------------------------------------------------------------------


class _AfterConn:
    """Answers check_after's reads from an in-memory database."""

    def __init__(self, teams, matches, recs=(), odds_last_run=None, odds_stopped=()):
        self.teams, self.matches, self.recs = teams, matches, list(recs)
        self.odds_last_run, self.odds_stopped = odds_last_run, list(odds_stopped)
        self.odds_params: list = []

    def cursor(self, cursor_factory=None):
        conn = self

        class C:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, params=None):
                self.sql, self.params = sql, params

            def _ids(self):
                p = self.params
                return set(p[0] if isinstance(p, tuple) else p["t"])

            def fetchone(self):
                s = self.sql
                if s.startswith("SELECT count(*) AS n FROM teams"):
                    return {"n": sum(1 for t in conn.teams if t["id"] in self._ids())}
                if s.startswith("SELECT count(*) AS n FROM matches WHERE id"):
                    return {"n": sum(1 for m in conn.matches if m["id"] in self._ids())}
                if s.startswith("SELECT count(*) AS n FROM matches"):
                    ids = self._ids()
                    return {"n": sum(1 for m in conn.matches if {m["home_team_id"], m["away_team_id"]} & ids)}
                if s == ms.UNSETTLED_RECS_SQL:
                    return {"n": sum(1 for r in conn.recs if r["match_id"] in self._ids() and r["open"])}
                if s == ms.ODDS_LAST_RUN_SQL:
                    conn.odds_params.append(self.params)
                    return {"last_run": conn.odds_last_run}
                raise AssertionError(s)

            def fetchall(self):
                s = self.sql
                if s.startswith("SELECT id::text AS id, metadata"):
                    return [t for t in conn.teams if t["id"] in self._ids()]
                if s.startswith("SELECT id::text AS id, home_team_id"):
                    return [
                        {"id": m["id"], "home": m["home_team_id"], "away": m["away_team_id"]}
                        for m in conn.matches
                        if m["id"] in self._ids()
                    ]
                if s == ms.FINISHED_TWINS_SQL:
                    return []
                if s == ms.ODDS_STOPPED_SQL:
                    conn.odds_params.append(self.params)
                    return [
                        dict(r, merged=bool({r["home_id"], r["away_id"]} & set(self.params["teams"])))
                        for r in conn.odds_stopped
                    ]
                if s == ms.ALL_SOCCER_TEAMS_SQL:
                    return [
                        {"id": t["id"], "normalized_name": t["normalized_name"], "espn": _team_espn(t)}
                        for t in conn.teams
                    ]
                return []

        return C()

    def rollback(self):
        pass


def _team_espn(t):
    return (t.get("external_ids") or {}).get("espn")


def _after_doc():
    return {
        "tool": "merge_soccer_teams",
        "pairs": [
            {
                "new_team_id": NEW,
                "old_team_id": OLD,
                "new_name": "BK Häcken",
                "action": "merge",
                "league_id": "L1",
                "espn": "7834",
                "drops": [["b", "a"], ["r", "a"]],
                "repoints": [["a", "home"], ["r", "away"]],
                "forks": [],
            }
        ],
        "adopted_with_open_recs": ["a"],
    }


def _merged_teams():
    kept = _team(
        OLD,
        "Hacken",
        external_ids={"espn": "7834"},
        metadata={"aliases": ["bk häcken"], "merged_from": [{"team_id": NEW}]},
    )
    return [kept, _team(OPP, "Djurgarden")]


class TestCheckAfter:
    def _resolvers(self, team=OLD):
        return _run_resolvers(team=team)

    def test_a_clean_merge_passes(self):
        conn = _AfterConn(_merged_teams(), [_m("a", OLD, OPP)], recs=[{"match_id": "a", "open": False}])
        out = ms.check_after(conn, _after_doc(), self._resolvers())
        assert ms.after_state_ok(out), out
        assert out["repoints_missing"] == 0  # 'r' was re-pointed, then dropped by a later pair: gone on purpose

    @pytest.mark.parametrize(
        "mutate, key",
        [
            (lambda t, m: t.append(_team(THIRD, "BK Hacken")), "teams_reminted"),
            (lambda t, m: t.append(_team(NEW, "BK Häcken")), "merged_team_rows_left"),
            (lambda t, m: m.append(_m("b", OLD, OPP, T0 + timedelta(hours=1))), "dropped_twins_left"),
            (lambda t, m: m.__setitem__(0, _m("a", NEW, OPP)), "repoints_not_landed"),
            (lambda t, m: t.append(_team(THIRD, "X", external_ids={"espn": "7834"})), "espn_misplaced"),
            (lambda t, m: t[0]["metadata"].update(aliases=[]), "pairs_recorded"),
        ],
    )
    def test_each_broken_expectation_fails(self, mutate, key):
        teams, matches = _merged_teams(), [_m("a", OLD, OPP)]
        mutate(teams, matches)
        out = ms.check_after(_AfterConn(teams, matches), _after_doc(), self._resolvers())
        assert not ms.after_state_ok(out)
        assert out[key] == (0 if key == "pairs_recorded" else out[key]) and (key == "pairs_recorded" or out[key] > 0)

    def test_after_later_steps_reports_rows_later_repairs_deleted_and_holds_the_rest(self):
        """Runbook step 11 runs after the fd repair and the orphan cleanup, which
        delete some re-pointed rows on purpose: repoints_missing is reported then,
        every other count still fails."""
        teams = _merged_teams()
        out = ms.check_after(_AfterConn(teams, []), _after_doc(), self._resolvers())  # 'a' was deleted later
        assert out["repoints_missing"] == 1
        assert not ms.after_state_ok(out) and ms.after_state_ok(out, after_later_steps=True)
        teams.append(_team(THIRD, "BK Hacken"))  # a re-minted team still fails
        out = ms.check_after(_AfterConn(teams, []), _after_doc(), self._resolvers())
        assert not ms.after_state_ok(out, after_later_steps=True)

    def test_a_resolver_miss_and_an_unsettled_rec_fail(self):
        conn = _AfterConn(_merged_teams(), [_m("a", OLD, OPP)], recs=[{"match_id": "a", "open": True}])
        out = ms.check_after(conn, _after_doc(), self._resolvers(team=THIRD))
        assert out["resolver_misses"] == 2 and out["unsettled_recs_on_adopted"] == 1 and not ms.after_state_ok(out)


class TestOddsAfterTheMerge:
    SINCE = NOW
    RUN = NOW + timedelta(minutes=95)

    def _out(self, **kw):
        conn = _AfterConn(_merged_teams(), [_m("a", OLD, OPP)], recs=[{"match_id": "a", "open": False}], **kw)
        return conn, ms.check_after(conn, _after_doc(), _run_resolvers(), odds_since=self.SINCE)

    @staticmethod
    def _fixture(home, away):
        return {
            "id": "f1",
            "match_date": NOW + timedelta(days=2),
            "home": "x",
            "away": "y",
            "home_id": home,
            "away_id": away,
        }

    def test_without_odds_since_there_is_no_odds_check(self):
        conn = _AfterConn(_merged_teams(), [_m("a", OLD, OPP)], recs=[{"match_id": "a", "open": False}])
        out = ms.check_after(conn, _after_doc(), _run_resolvers())
        assert not any(k.startswith("odds_") for k in out) and conn.odds_params == []

    def test_every_merged_clubs_fixture_still_quoted_passes(self):
        conn, out = self._out(odds_last_run=self.RUN, odds_stopped=[self._fixture(OPP, THIRD)])
        assert ms.after_state_ok(out), out
        assert out["odds_fixtures_stopped"] == 0 and out["odds_control_stopped"] == 1
        assert conn.odds_params[1]["teams"] == [OLD] and conn.odds_params[1]["last_run"] == self.RUN

    def test_a_merged_clubs_fixture_the_feed_stopped_quoting_fails(self):
        _, out = self._out(odds_last_run=self.RUN, odds_stopped=[self._fixture(OPP, OLD)])
        assert out["odds_fixtures_stopped"] == 1 and not ms.after_state_ok(out)

    def test_no_odds_run_since_the_merge_is_not_a_pass(self):
        _, out = self._out(odds_last_run=None)
        assert out["odds_not_yet_observed"] == 1 and not ms.after_state_ok(out)

    def test_the_sql_reads_odds_through_upcoming_fixtures_only(self):
        for sql in (ms.ODDS_LAST_RUN_SQL, ms.ODDS_STOPPED_SQL):
            assert "JOIN odds o ON o.match_id = m.id" in sql or "WHERE o.match_id = m.id" in sql
        assert "m.match_date > %(last_run)s" in ms.ODDS_STOPPED_SQL
        assert "o.last_seen_at < %(since)s" in ms.ODDS_STOPPED_SQL
        assert "NOT EXISTS (SELECT 1 FROM odds o WHERE o.match_id = m.id AND o.last_seen_at >= %(since)s)" in (
            ms.ODDS_STOPPED_SQL
        )


class TestMain:
    def test_main_hands_the_committed_odds_spellings_to_the_run(self, tmp_path, monkeypatch):
        asked = []
        resolvers = _run_resolvers()
        monkeypatch.setattr(ms, "load_resolvers", lambda: (resolvers, None))
        monkeypatch.setattr(ms.psycopg2, "connect", lambda url: type("C", (), {"close": lambda self: None})())
        monkeypatch.setattr(ms, "run", lambda c, *a, **kw: asked.append(kw.get("odds_names")) or 0)
        spellings = _write_spellings(tmp_path, [[NEW, "BK Häcken", "Hacken FC", "log"]])
        mapfile = _write_map(tmp_path, [_row(new_name="BK Häcken", old_name="Hacken")])
        rc = ms.main(["--database-url", "x", "--map", str(mapfile), "--odds-spellings", str(spellings)])
        assert rc == 0 and asked == [{NEW: ["Hacken FC"]}]

    def test_odds_since_needs_check_after(self):
        assert ms.main(["--database-url", "postgresql://nowhere/x", "--odds-since", "2026-10-31T10:00:00Z"]) == 2
        assert ms.main(["--database-url", "postgresql://nowhere/x", "--after-later-steps"]) == 2

    def test_a_missing_odds_spellings_file_refuses_before_connecting(self, tmp_path):
        rc = ms.main(["--database-url", "postgresql://nowhere/x", "--odds-spellings", str(tmp_path / "none.csv")])
        assert rc == 2

    def test_odds_since_reads_as_utc(self):
        assert ms._utc_arg("2026-10-31T10:00:00Z") == datetime(2026, 10, 31, 10, tzinfo=timezone.utc)
        assert ms._utc_arg("2026-10-31T10:00:00") == datetime(2026, 10, 31, 10, tzinfo=timezone.utc)
