"""merge_soccer_teams: the committed map, the resolver precondition, how a
twin cluster is folded, what moves off a dropped twin, the per-pair plan
(sequential pairs, forks, refusals), and that a pair writes in one guarded
transaction or not at all. No DB."""

from __future__ import annotations

import csv
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

T0 = datetime(2026, 7, 6, 17, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 10, 31, tzinfo=timezone.utc)
NEW, OLD, OPP, NEW2, OLD2 = (f"00000000-0000-0000-0000-00000000000{i}" for i in range(1, 6))


def _pair(new=NEW, old=OLD, new_name="BK Häcken", old_name="Hacken", espn="7834") -> object:
    return ms.MapRow(new, old, new_name, old_name, espn, "exact-alias", "key_equal 1.000")


def _team(tid, name, **kw) -> dict:
    return {
        "id": tid,
        "name": name,
        "normalized_name": ms.normalize_name(name),
        "sport": "soccer",
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


def _plan(matches, deps=None, teams=None, pairs=None, refs=None):
    state = {m["id"]: m for m in matches}
    return ms.plan_all(pairs or [_pair()], teams or _teams(), refs or {}, state, deps or {}, NOW)


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

    def test_wrapper_expects_the_map_count(self):
        wrapper = (SCRIPTS_DIR / "ops" / "merge_soccer_teams.sh").read_text()
        assert f"EXPECTED={len(self.rows)}" in wrapper
        assert "compute_features.py --backfill-finished soccer" in wrapper

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


# ---------------------------------------------------------------------------
# The precondition
# ---------------------------------------------------------------------------


def _scripts(tmp_path, identity: str, upcoming: str = "find_team_by_alias", odds: str = "find_team_by_alias") -> Path:
    (tmp_path / "team_identity.py").write_text(identity)
    (tmp_path / "fetch_upcoming.py").write_text(f"# calls team_identity.{upcoming}\n")
    (tmp_path / "fetch_live_odds.py").write_text(f"# calls team_identity.{odds}\n")
    return tmp_path


RESOLVER = 'def find_team_by_alias(cur, sport, name):\n    return "aliases"\n'


class TestResolverMarker:
    def test_live_resolver_passes(self, tmp_path):
        assert ms.resolver_refusal(_scripts(tmp_path, RESOLVER)) is None

    def test_no_alias_callable_refuses(self, tmp_path):
        reason = ms.resolver_refusal(_scripts(tmp_path, "def strip_accents(t):\n    return t\n"))
        assert "no alias resolver" in reason

    def test_resolver_must_read_the_aliases_key(self, tmp_path):
        reason = ms.resolver_refusal(_scripts(tmp_path, "def find_team_by_alias(cur, s, n):\n    return None\n"))
        assert "'aliases'" in reason

    @pytest.mark.parametrize("caller", ["fetch_upcoming.py", "fetch_live_odds.py"])
    def test_every_caller_must_use_it(self, tmp_path, caller):
        kw = {"upcoming": "nothing"} if caller == "fetch_upcoming.py" else {"odds": "nothing"}
        assert caller in ms.resolver_refusal(_scripts(tmp_path, RESOLVER, **kw))

    def test_import_error_is_a_refusal_not_a_crash(self, tmp_path):
        assert "cannot import" in ms.resolver_refusal(_scripts(tmp_path, "import no_such_module_xyz\n"))


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

    def test_else_the_finished_row_with_an_espn_id_then_more_predictions(self):
        a, b, c = _m("a", NEW, OPP), _m("b", OLD, OPP, espn="9"), _m("c", NEW, OPP, status="scheduled")
        assert ms.decide_cluster([a, b, c], {"a": _deps(n_preds=50)})["keep"]["id"] == "b"
        assert ms.decide_cluster([a, c], {"a": _deps(n_preds=5), "c": _deps(n_preds=50)})["keep"]["id"] == "a"

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

    def test_stale_rec_holder_adopts_the_result_when_its_recs_are_settled(self):
        stale = _m("s", NEW, OPP, T0, status="scheduled")
        played = _m("p", OLD, OPP, T0 + timedelta(hours=1), ht=(1, 2))
        v = ms.decide_cluster([stale, played], {"s": _deps(holds_recs=True)})
        assert v["keep"]["id"] == "s" and v["adopt_from"]["id"] == "p" and v["ht_from"]["id"] == "p"

    def test_open_recs_adopt_only_on_the_same_kickoff(self):
        stale = _m("s", OLD, OPP, T0, status="scheduled")
        moved = _m("p", NEW, OPP, T0 + timedelta(hours=1))
        deps = {"s": _deps(holds_recs=True, open_recs=1)}
        assert "differs" in ms.decide_cluster([stale, moved], deps)["refuse"]
        same = _m("p", NEW, OPP, T0, espn="77")
        v = ms.decide_cluster([stale, same], deps)
        assert v["adopt_from"]["id"] == "p" and v["espn_from"]["id"] == "p"

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
        odds, _, _ = ms.plan_moves(drop, _deps(), T0, {OLD, OPP})
        assert [o["id"] for o in odds] == ["o2"]

    def test_the_kept_rows_own_prices_win(self):
        keep = _deps(odds=[_odds("k1", T0 - timedelta(hours=9))])
        odds, _, _ = ms.plan_moves(_deps(odds=[_odds("o1", T0 - timedelta(hours=1))]), keep, T0, {OLD})
        assert odds == []

    def test_snapshots_before_kickoff_the_kept_row_lacks(self):
        t = T0 - timedelta(hours=3)
        keep = _deps(snapshots=[_snap("k1", t)])
        drop = _deps(snapshots=[_snap("s1", t), _snap("s2", t - timedelta(hours=1)), _snap("s3", T0)])
        _, snaps, _ = ms.plan_moves(drop, keep, T0, {OLD})
        assert [s["id"] for s in snaps] == ["s2"]

    def test_match_stats_only_for_a_team_of_the_kept_row_it_lacks(self):
        keep = _deps(match_stats=[{"id": "k", "team_id": OPP}])
        drop = _deps(
            match_stats=[{"id": "a", "team_id": OLD}, {"id": "b", "team_id": OPP}, {"id": "c", "team_id": NEW}]
        )
        _, _, stats = ms.plan_moves(drop, keep, T0, {OLD, OPP})
        assert [s["id"] for s in stats] == ["a"]

    def test_a_second_twin_sees_what_the_first_moved(self):
        keep = _deps()
        ms.plan_moves(_deps(odds=[_odds("o1", T0 - timedelta(hours=1))]), keep, T0, {OLD})
        odds, _, _ = ms.plan_moves(_deps(odds=[_odds("o2", T0 - timedelta(hours=2))]), keep, T0, {OLD})
        assert odds == [] and [o["id"] for o in keep.odds] == ["o1"]


# ---------------------------------------------------------------------------
# Planning pairs
# ---------------------------------------------------------------------------


class TestPlanPair:
    def test_repoints_every_match_and_records_the_alias(self):
        (p,) = _plan([_m("a", NEW, OPP), _m("b", OPP, NEW, T0 + timedelta(days=7))])
        assert p.action == "merge" and p.drops == []
        assert [(r["match_id"], r["side"]) for r in p.repoints] == [("a", "home"), ("b", "away")]
        assert p.team_patch["aliases"] == ["bk häcken"] and p.team_patch["external_patch"] == {"espn": "7834"}
        assert p.team_patch["merged_from"][0]["team_id"] == NEW

    def test_exact_kickoff_collision_drops_the_twin_before_any_repoint(self):
        espn_row = _m("a", NEW, OPP, espn="401")
        fd_row = _m("b", OLD, OPP)
        (p,) = _plan([espn_row, fd_row], deps={"a": _deps(n_preds=57)})
        assert [(d.match_id, d.keep_id) for d in p.drops] == [("b", "a")]
        assert [r["match_id"] for r in p.repoints] == ["a"]

    def test_a_dropped_new_side_row_is_not_repointed(self):
        (p,) = _plan([_m("a", NEW, OPP, status="scheduled"), _m("b", OLD, OPP, T0 + timedelta(hours=1))])
        assert [d.match_id for d in p.drops] == ["a"] and p.repoints == []

    def test_pairs_plan_on_the_state_the_earlier_ones_leave(self):
        pairs = [_pair(), _pair(new=NEW2, old=OLD2, new_name="Djurgården", old_name="Djurgarden IF", espn="2339")]
        espn_row = _m("e", NEW, NEW2, espn="5")
        fd_row = _m("f", OLD, OLD2)
        p1, p2 = _plan([espn_row, fd_row], pairs=pairs)
        assert p1.drops == [] and [r["side"] for r in p1.repoints] == ["home"]  # no twin until both are mapped
        assert [d.match_id for d in p2.drops] == ["f"]
        assert p2.repoints[0]["observed"]["home_team_id"] == OLD  # guarded on what pair 1 leaves

    def test_two_spellings_onto_one_kept_team_accumulate_aliases(self):
        pairs = [_pair(espn=""), _pair(new=NEW2, new_name="BK Hacken", espn="7834")]
        teams = {**_teams(), NEW2: _team(NEW2, "BK Hacken")}
        rows = [_m("a", NEW, OPP), _m("b", NEW2, OPP, T0 + timedelta(days=9))]
        p1, p2 = _plan(rows, pairs=pairs, teams=teams)
        assert p1.action == p2.action == "merge"
        assert p2.team_patch["observed_aliases"] == ["bk häcken"]
        assert p2.team_patch["aliases"] == ["bk hacken", "bk häcken"]
        assert p2.old_team["metadata"]["aliases"] == ["bk häcken"]

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
        teams[OLD]["metadata"] = {"aliases": ["bk häcken"], "merged_from": [{"team_id": NEW}]}
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
        assert ms.adopted_with_open_recs(plans) == ["s"]

    def test_plan_all_leaves_its_inputs_alone(self):
        rows = [_m("a", NEW, OPP, espn="1"), _m("b", OLD, OPP)]
        state = {m["id"]: m for m in rows}
        teams = _teams()
        deps = {"b": _deps(odds=[_odds("o", T0 - timedelta(hours=1))])}
        ms.plan_all([_pair()], teams, {}, state, deps, NOW)
        assert state["a"]["home_team_id"] == NEW and NEW in teams and deps.get("a") is None


# ---------------------------------------------------------------------------
# Writing one pair
# ---------------------------------------------------------------------------


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
        elif sql == ms.NEW_TEAM_REFS_SQL:
            self._rows = [dict(c.refs)]
        elif sql == ms.TWINS_AFTER_SQL:
            self._rows = list(c.twins)
        elif sql.lstrip().startswith(("UPDATE", "DELETE")):
            self.rowcount = c.rowcount(sql, params)

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0]


class _Conn:
    def __init__(self, plan, rowcount=None, new_matches=None, teams=None, refs=None, twins=()):
        self.log: list = []
        self.teams = teams if teams is not None else [plan.new_team, plan.old_team]
        self.new_matches = new_matches if new_matches is not None else plan.new_match_ids
        self.refs = refs or {k: 0 for k in ("matches", "match_stats", "players")}
        self.twins = twins
        self.rowcount = rowcount or (lambda sql, params: len(params.get("ids", [1])) if "ids" in params else 1)
        self.committed = self.rolled_back = 0

    def cursor(self, cursor_factory=None):
        return _Cursor(self)

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1

    def writes(self):
        return [s for s, _ in self.log if s.lstrip().startswith(("UPDATE", "DELETE"))]


def _merge_plan():
    espn_row = _m("a", NEW, OPP, espn="401")
    fd_row = _m("b", OLD, OPP)
    later = _m("c", OPP, NEW, T0 + timedelta(days=7), status="scheduled")
    deps = {"b": _deps(odds=[_odds("o1", T0 - timedelta(hours=1))], match_stats=[{"id": "s1", "team_id": OLD}])}
    (p,) = _plan([espn_row, fd_row, later], deps=deps)
    p.new_team = {**_team(NEW, "BK Häcken"), "id": NEW}
    p.old_team = {**_team(OLD, "Hacken"), "id": OLD}
    return p


class TestApplyPair:
    def test_writes_in_order_in_one_transaction(self):
        p = _merge_plan()
        conn = _Conn(p)
        done = ms.apply_pair(conn, p)
        # moves before the twin is dropped, the drop before any re-point, the team row last
        assert conn.writes() == [
            ms.MOVE_ODDS_SQL,
            ms.MOVE_MATCH_STATS_SQL,
            ms.DROP_TWIN_SQL,
            ms.REPOINT_SQL["home"],
            ms.REPOINT_SQL["away"],
            ms.TEAM_RECORD_SQL,
            ms.TEAM_DELETE_SQL,
        ]
        assert conn.committed == 1 and conn.rolled_back == 0
        assert done["twins_dropped"] == 1 and done["matches_repointed"] == 2 and done["teams_deleted"] == 1

    def test_the_drop_is_guarded_on_the_observed_row(self):
        p = _merge_plan()
        conn = _Conn(p)
        ms.apply_pair(conn, p)
        (params,) = [prm for s, prm in conn.log if s == ms.DROP_TWIN_SQL]
        assert params["id"] == "b" and params["home_team_id"] == OLD and params["status"] == "finished"
        assert "NOT EXISTS (SELECT 1 FROM betting_recommendations" in ms.DROP_TWIN_SQL

    def test_a_row_that_moved_rolls_the_pair_back(self):
        p = _merge_plan()
        conn = _Conn(p, rowcount=lambda sql, params: 0 if sql == ms.REPOINT_SQL["away"] else 1)
        with pytest.raises(ms.GuardError, match="repoint away"):
            ms.apply_pair(conn, p)
        assert conn.committed == 0 and conn.rolled_back == 1

    def test_a_fixture_added_since_the_plan_rolls_the_pair_back_before_any_write(self):
        p = _merge_plan()
        conn = _Conn(p, new_matches=p.new_match_ids + ["zzz"])
        with pytest.raises(ms.GuardError, match="gained"):
            ms.apply_pair(conn, p)
        assert conn.writes() == [] and conn.committed == 0

    def test_a_team_changed_since_the_plan_rolls_back_before_any_write(self):
        p = _merge_plan()
        conn = _Conn(p, teams=[p.new_team, {**p.old_team, "metadata": {"aliases": ["x"]}}])
        with pytest.raises(ms.GuardError, match="changed since the plan"):
            ms.apply_pair(conn, p)
        assert conn.writes() == []

    def test_a_reference_left_on_the_new_team_rolls_back(self):
        p = _merge_plan()
        conn = _Conn(p, refs={"matches": 1, "players": 0})
        with pytest.raises(ms.GuardError, match="still referenced"):
            ms.apply_pair(conn, p)
        assert ms.TEAM_DELETE_SQL not in conn.writes() and conn.committed == 0

    def test_a_twin_left_after_the_repoint_rolls_back(self):
        p = _merge_plan()
        conn = _Conn(p, twins=[{"a": "a", "b": "q"}])
        with pytest.raises(ms.GuardError, match="twin"):
            ms.apply_pair(conn, p)
        assert conn.committed == 0

    def test_the_team_delete_requires_zero_references(self):
        for table in ("matches m", "match_stats", "players", "team_managers", "transfers", "nhl_match_stats"):
            assert f"NOT EXISTS (SELECT 1 FROM {table}" in ms.TEAM_DELETE_SQL


# ---------------------------------------------------------------------------
# The run: audit trail first, refusal without the resolver
# ---------------------------------------------------------------------------


class _RunConn:
    """load_state reads: hand back fixed rows per SQL."""

    def __init__(self, teams, matches):
        self.teams, self.matches = teams, matches
        self.writes, self.committed = [], 0

    def cursor(self, cursor_factory=None):
        conn = self

        class C:
            rowcount = 1

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, params=None):
                self.sql = sql
                if sql.lstrip().startswith(("UPDATE", "DELETE")):
                    conn.writes.append(sql)

            def fetchall(self):
                if self.sql == ms.TEAMS_SQL:
                    return [dict(t) for t in conn.teams]
                if self.sql == ms.TEAM_REFS_SQL:
                    return [{"id": NEW, "match_stats": 0, "players": 0}]
                if self.sql == ms.MATCHES_SQL:
                    return [dict(m) for m in conn.matches]
                return []

        return C()

    def commit(self):
        self.committed += 1

    def rollback(self):
        pass


def _csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


class TestRun:
    def _conn(self):
        return _RunConn(list(_teams().values()), [_m("a", NEW, OPP), _m("b", OPP, NEW, T0 + timedelta(days=7))])

    def test_dry_run_writes_the_audit_csv_and_nothing_else(self, tmp_path):
        conn = self._conn()
        rc = ms.run(conn, False, [_pair()], tmp_path / "a.csv", tmp_path / "a.jsonl.gz", None)
        rows = _csv(tmp_path / "a.csv")
        assert rc == 0 and conn.writes == [] and conn.committed == 0
        assert [r["action"] for r in rows] == ["repoint_home", "repoint_away", "record_alias", "delete_team"]
        assert json.loads(rows[0]["after"]) == {"home_team_id": OLD}
        assert not (tmp_path / "a.jsonl.gz").exists()

    def test_apply_without_the_resolver_reads_and_writes_nothing(self, tmp_path):
        conn = self._conn()
        rc = ms.run(conn, True, [_pair()], tmp_path / "a.csv", tmp_path / "a.jsonl.gz", "no resolver")
        assert rc == 2 and conn.writes == [] and not (tmp_path / "a.csv").exists()

    def test_dry_run_without_the_resolver_plans_but_fails(self, tmp_path):
        rc = ms.run(self._conn(), False, [_pair()], tmp_path / "a.csv", tmp_path / "x.gz", "no resolver")
        assert rc == 3 and (tmp_path / "a.csv").exists()

    def test_the_audit_csv_is_write_once(self, tmp_path):
        out = tmp_path / "a.csv"
        out.write_text("x")
        with pytest.raises(FileExistsError):
            ms.run(self._conn(), True, [_pair()], out, tmp_path / "a.jsonl.gz", None)

    def test_a_refused_pair_fails_the_dry_run(self, tmp_path):
        conn = _RunConn([_team(OLD, "Hacken"), _team(OPP, "Djurgarden")], [])
        rc = ms.run(conn, False, [_pair()], tmp_path / "a.csv", tmp_path / "a.gz", None)
        assert rc == 1 and _csv(tmp_path / "a.csv")[0]["action"] == "pair_refuse"


def test_sql_guards_reference_observed_values():
    assert "home_team_id = %(new)s::uuid" in ms.REPOINT_SQL["home"]
    assert "match_date = %(match_date)s" in ms.REPOINT_SQL["away"]
    assert "status = %(o_status)s" in ms.ENRICH_SQL
    assert "observed_aliases" in ms.TEAM_RECORD_SQL and "observed_espn" in ms.TEAM_RECORD_SQL
    assert 'o."timestamp" < %(cutoff)s' in ms.MOVE_ODDS_SQL and "NOT COALESCE(o.is_live, false)" in ms.MOVE_ODDS_SQL
    assert "s.captured_at < %(cutoff)s" in ms.MOVE_SNAPSHOTS_SQL
