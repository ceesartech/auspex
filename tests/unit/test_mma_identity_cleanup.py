"""scripts/mma_identity_cleanup.py without a database: the reviewed lists
(validated, cross-checked, never naming the rows the parallel stream
mma_draws_held owns), and the plan for each unit in each state — the split
(todo, done, every refusal), the alias, the orphan, the athlete ids — and the
plan digest, which covers the id each row ends up with, so a row a sweep
stamped between the dry run and the apply (done instead of stamp) does not
change it."""

from __future__ import annotations

import csv
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mic = _load("mma_identity_cleanup_unit", "mma_identity_cleanup.py")
fu = sys.modules.get("fetch_upcoming") or _load("fetch_upcoming", "fetch_upcoming.py")
ti = sys.modules.get("team_identity") or _load("team_identity", "team_identity.py")

T, N = "69e14391-f07b-489c-8ce4-d1d0e95d29d0", "d31648b1-b3f3-4c55-9d63-e8b112a498ed"
KEEP, MOVE = "3895544", "4333158"
M1, M2, M3 = (f"00000000-0000-4000-8000-00000000000{i}" for i in (1, 2, 3))
OPP = "00000000-0000-4000-8000-0000000000aa"
RAF, JOSE, ZACH, PLAIN = (f"00000000-0000-4000-8000-0000000001{i}0" for i in (1, 2, 3, 4))
K1 = datetime(2024, 3, 31, 2, 0, tzinfo=timezone.utc)
K2 = datetime(2024, 7, 21, 0, 0, tzinfo=timezone.utc)
K3 = datetime(2024, 10, 26, 14, 0, tzinfo=timezone.utc)


def team(tid, name, espn="", aliases=None, **meta):
    md = dict(meta)
    if aliases is not None:
        md["aliases"] = aliases
    return {
        "id": tid,
        "name": name,
        "normalized_name": fu.normalize_team(name),
        "league_id": "lg",
        "country": None,
        "is_active": True,
        "espn": espn,
        "external_ids": {"espn": espn} if espn else {},
        "metadata": md,
    }


def resolvers(find=None):
    return mic.Resolvers(
        normalize=fu.normalize_team,
        alias_form=ti.alias_form,
        find_team=find or (lambda cur, name, espn: None),
        stamp_sql=ti.STAMP_ESPN_SQL,
    )


def split_list():
    bouts = (
        mic.SplitBout(M1, K1, "finished", "away", "Chris Weidman", "401619308", MOVE, "move"),
        mic.SplitBout(M2, K2, "finished", "home", "Cody Durden", "401678841", KEEP, "keep"),
        mic.SplitBout(M3, K3, "finished", "home", "Ismail Naurdiev", "401720687", MOVE, "move"),
    )
    return mic.SplitList(T, "Bruno Silva", N, f"Bruno Silva (ESPN {MOVE})", KEEP, MOVE, bouts)


def lists(ids=None):
    return mic.Lists(
        split_list(),
        (mic.AliasRow(RAF, "Rafael Cerqueira", "Raffael Cerqueira", "5212598"),),
        (mic.OrphanRow(JOSE, "José Luiz"),),
        tuple(ids or (mic.IdRow(RAF, "Rafael Cerqueira", "5212598"), mic.IdRow(ZACH, "Zach Reese", "5143223"))),
    )


def bouts_on(t_id=T, moved_to=None):
    """The three bouts; moved_to puts the move bouts on that row."""
    m1 = moved_to or t_id
    return {
        M1: {"id": M1, "home": OPP, "away": m1, "match_date": K1, "status": "finished"},
        M2: {"id": M2, "home": t_id, "away": OPP, "match_date": K2, "status": "finished"},
        M3: {"id": M3, "home": m1, "away": OPP, "match_date": K3, "status": "finished"},
    }


def state(teams=None, matches=None, holders=None, refs=None):
    rows = (
        teams
        if teams is not None
        else [
            team(T, "Bruno Silva"),
            team(RAF, "Rafael Cerqueira"),
            team(JOSE, "José Luiz"),
            team(ZACH, "Zach Reese", espn="5143223", aliases=["Zachary Reese"]),
            team(OPP, "Opponent"),
        ]
    )
    return mic.State(
        teams={t["id"]: t for t in rows},
        split_matches=matches if matches is not None else bouts_on(),
        holders=holders or {"recs": 0, "history": 0, "predictions": 2},
        team_fks=(("matches", "away_team_id"), ("matches", "home_team_id")),
        orphan_refs=refs or {JOSE: {"matches.home_team_id": 0, "matches.away_team_id": 0}},
    )


# ── the lists ────────────────────────────────────────────────────────


def test_the_committed_lists_load_and_cross_check():
    ls = mic.load_lists()
    assert len(ls.split.bouts) == 11 and len(ls.split.of("move")) == 7 and len(ls.split.of("keep")) == 4
    assert (ls.split.keep_athlete, ls.split.move_athlete) == (KEEP, MOVE)
    assert (ls.split.team_id, ls.split.new_team_id, ls.split.new_team_name) == (T, N, "Bruno Silva (ESPN 4333158)")
    assert [a.alias for a in ls.aliases] == ["Raffael Cerqueira"]
    assert [o.team_name for o in ls.orphans] == ["José Luiz"]
    assert len(ls.ids) == 1299
    named = {ls.split.team_id, ls.split.new_team_id} | {o.team_id for o in ls.orphans} | {r.team_id for r in ls.ids}
    assert not named & (mic.HELD_TEAMS | mic.HELD_MATCHES)
    assert not {b.match_id for b in ls.split.bouts} & mic.HELD_MATCHES


def _write(path, fields, rows, comments=("reviewed",)):
    with path.open("w", newline="", encoding="utf-8") as fh:
        for c in comments:
            fh.write(f"# {c}\n")
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def _split_rows(**over):
    base = {
        "team_id": T,
        "team_name": "Bruno Silva",
        "new_team_id": N,
        "new_team_name": "Bruno Silva (ESPN 4333158)",
        "status": "finished",
        "opponent": "X",
        "espn_competition_id": "1",
        "espn_event_id": "2",
        "evidence": "ESPN",
    }
    rows = [
        {**base, "match_id": M1, "kickoff": K1.isoformat(), "side": "away", "espn_athlete_id": MOVE, "assign": "move"},
        {**base, "match_id": M2, "kickoff": K2.isoformat(), "side": "home", "espn_athlete_id": KEEP, "assign": "keep"},
    ]
    for i, patch in over.items():
        rows[int(i[1:])].update(patch)
    return rows


@pytest.mark.parametrize(
    "over, why",
    [
        ({"r1": {"match_id": M1}}, "listed twice"),
        ({"r1": {"espn_athlete_id": MOVE}}, "the same athlete"),
        ({"r1": {"assign": "move"}}, "exactly one ESPN athlete id"),
        ({"r0": {"kickoff": "2024-03-31T02:00:00"}}, "no time zone"),
        ({"r0": {"side": "left"}}, "home|away"),
        ({"r0": {"match_id": "b7f16d27-42a1-41b0-a6f6-6e8e9837da47"}}, "mma_draws_held"),
        ({"r0": {"evidence": ""}}, "empty evidence"),
        ({"r1": {"new_team_id": "00000000-0000-4000-8000-000000000999"}}, "different (team, new team) heads"),
    ],
)
def test_a_malformed_split_list_is_refused(tmp_path, over, why):
    path = _write(tmp_path / "s.csv", mic.SPLIT_FIELDS, _split_rows(**over))
    with pytest.raises(mic.ListError, match=why.replace("|", r"\|").replace("(", r"\(").replace(")", r"\)")):
        mic.load_split(path)


def test_a_list_naming_the_held_team_is_refused(tmp_path):
    path = _write(
        tmp_path / "ids.csv",
        mic.IDS_FIELDS,
        [
            {
                "team_id": "108423d1-7f54-43ba-8039-43e5ef23b8bf",
                "team_name": "Valesca Machado",
                "espn_athlete_id": "1",
                "espn_names": "x",
                "bouts": "1",
                "espn_competition_ids": "1",
                "method": "names",
            }
        ],
    )
    with pytest.raises(mic.ListError, match="mma_draws_held"):
        mic.load_ids(path)


def test_an_athlete_listed_twice_is_refused(tmp_path):
    row = {"team_name": "A", "espn_athlete_id": "1", "espn_names": "A", "bouts": "1", "espn_competition_ids": "1"}
    path = _write(
        tmp_path / "ids.csv",
        mic.IDS_FIELDS,
        [{**row, "team_id": RAF, "method": "names"}, {**row, "team_id": ZACH, "method": "names"}],
    )
    with pytest.raises(mic.ListError, match="listed twice"):
        mic.load_ids(path)


def test_the_cross_checks(tmp_path):
    split = _write(tmp_path / "s.csv", mic.SPLIT_FIELDS, _split_rows())
    alias = _write(
        tmp_path / "a.csv",
        mic.ALIAS_FIELDS,
        [
            {
                "team_id": RAF,
                "team_name": "Rafael Cerqueira",
                "alias": "Raffael Cerqueira",
                "espn_athlete_id": "5212598",
                "evidence": "e",
            }
        ],
    )
    orphans = _write(
        tmp_path / "o.csv",
        mic.ORPHAN_FIELDS,
        [{"team_id": JOSE, "team_name": "José Luiz", "created_at": "2026-06-07", "evidence": "e"}],
    )

    def ids(rows):
        return _write(
            tmp_path / "i.csv",
            mic.IDS_FIELDS,
            [{"espn_names": "x", "bouts": "1", "espn_competition_ids": "1", "method": "names", **r} for r in rows],
        )

    ok = ids([{"team_id": RAF, "team_name": "Rafael Cerqueira", "espn_athlete_id": "5212598"}])
    assert len(mic.load_lists(split, alias, orphans, ok).ids) == 1
    for rows, why in (
        ([{"team_id": T, "team_name": "Bruno Silva", "espn_athlete_id": "9"}], "split row"),
        ([{"team_id": ZACH, "team_name": "Zach Reese", "espn_athlete_id": MOVE}], "split athlete"),
        ([{"team_id": JOSE, "team_name": "José Luiz", "espn_athlete_id": "9"}], "orphan"),
        ([{"team_id": RAF, "team_name": "Rafael Cerqueira", "espn_athlete_id": "9"}], "alias row"),
    ):
        with pytest.raises(mic.ListError, match=why):
            mic.load_lists(split, alias, orphans, ids(rows))


# ── the split ────────────────────────────────────────────────────────


def test_the_split_moves_only_the_listed_bouts_of_the_other_athlete():
    sp = mic.plan_split(split_list(), state(), resolvers())
    assert sp.action == "split", sp.reason
    assert [(m["match_id"], m["side"], m["other"]) for m in sp.moves] == [(M1, "away", OPP), (M3, "home", OPP)]
    assert sp.new_row["espn"] == MOVE and sp.new_row["aliases"] == ["Bruno Silva"]
    assert sp.new_row["normalized_name"] == "bruno silva (espn 4333158)" and sp.predictions == 2


def test_a_split_row_already_stamped_with_the_kept_athlete_still_splits():
    teams = [team(T, "Bruno Silva", espn=KEEP), team(OPP, "Opponent")]
    assert mic.plan_split(split_list(), state(teams=teams), resolvers()).action == "split"


def test_the_done_split_is_recognised_and_tolerates_new_bouts():
    teams = [
        team(T, "Bruno Silva", espn=KEEP),
        team(N, "Bruno Silva (ESPN 4333158)", espn=MOVE, aliases=["Bruno Silva"]),
        team(OPP, "Opponent"),
    ]
    matches = bouts_on(moved_to=N)
    matches["later"] = {"id": "later", "home": N, "away": OPP, "match_date": K3, "status": "scheduled"}
    assert mic.plan_split(split_list(), state(teams=teams, matches=matches), resolvers()).action == "done"


@pytest.mark.parametrize(
    "case, why",
    [
        ("row_holds_move_athlete", "carries ESPN athlete 4333158"),
        ("unlisted_bout", "does not assign"),
        ("rec_on_moved", "reference a bout to move"),
        ("history_on_moved", "reference a bout to move"),
        ("new_name_taken", "already carries the new name"),
        ("alias_claimed", "already claimed"),
        ("move_athlete_elsewhere", "already on team"),
        ("bout_side_changed", "not as reviewed"),
        ("bout_kickoff_changed", "not as reviewed"),
        ("bout_status_changed", "not as reviewed"),
        ("bout_missing", "not on either row"),
        ("renamed", "is named"),
        ("half_done", "not complete"),
    ],
)
def test_the_split_refuses(case, why):
    teams = [team(T, "Bruno Silva"), team(OPP, "Opponent")]
    matches = bouts_on()
    holders = None
    if case == "row_holds_move_athlete":
        teams[0] = team(T, "Bruno Silva", espn=MOVE)
    elif case == "unlisted_bout":
        matches["new"] = {"id": "new", "home": T, "away": OPP, "match_date": K3, "status": "scheduled"}
    elif case == "rec_on_moved":
        holders = {"recs": 1, "history": 0, "predictions": 2}
    elif case == "history_on_moved":
        holders = {"recs": 0, "history": 1, "predictions": 2}
    elif case == "new_name_taken":
        teams.append(team("x", "Bruno Silva (ESPN 4333158)"))
    elif case == "alias_claimed":
        teams.append(team("x", "B. Silva", aliases=["Bruno Silva"]))
    elif case == "move_athlete_elsewhere":
        teams.append(team("x", "Someone", espn=MOVE))
    elif case == "bout_side_changed":
        matches[M1] = {**matches[M1], "home": T, "away": OPP}
    elif case == "bout_kickoff_changed":  # a kept bout: only the per-bout check reads it
        matches[M2] = {**matches[M2], "match_date": K2 + timedelta(hours=1)}
    elif case == "bout_status_changed":
        matches[M2] = {**matches[M2], "status": "cancelled"}
    elif case == "bout_missing":
        del matches[M3]
    elif case == "renamed":
        teams[0] = team(T, "Bruno Silva Jr")
    elif case == "half_done":
        teams.append(team(N, "Bruno Silva (ESPN 4333158)", espn=MOVE, aliases=["Bruno Silva"]))
    sp = mic.plan_split(split_list(), state(teams=teams, matches=matches, holders=holders), resolvers())
    assert sp.action == "refuse" and why in sp.reason, sp.reason
    if case in ("rec_on_moved", "history_on_moved"):
        assert mic.summarize(mic.RunPlan(sp, [], [], []))["recs_on_moved"] == 1


def test_a_half_done_split_with_a_moved_bout_still_on_the_old_row_refuses_not_done():
    teams = [
        team(T, "Bruno Silva", espn=KEEP),
        team(N, "Bruno Silva (ESPN 4333158)", espn=MOVE, aliases=["Bruno Silva"]),
        team(OPP, "Opponent"),
    ]
    matches = bouts_on(moved_to=N)
    matches[M1] = {**matches[M1], "away": T}  # one Blindado bout was never re-pointed
    sp = mic.plan_split(split_list(), state(teams=teams, matches=matches), resolvers())
    assert sp.action == "refuse" and "1 moved bout(s) still on the split row" in sp.reason, sp.reason


# ── alias, orphan, ids ───────────────────────────────────────────────


def test_the_alias_is_added_then_done():
    a = mic.AliasRow(RAF, "Rafael Cerqueira", "Raffael Cerqueira", "5212598")
    u = mic.plan_alias(a, state(), resolvers())
    assert (u.action, u.detail["aliases"]) == ("add", ["Raffael Cerqueira"])
    done = state(teams=[team(RAF, "Rafael Cerqueira", espn="5212598", aliases=["Raffael Cerqueira"])])
    assert mic.plan_alias(a, done, resolvers()).action == "done"


@pytest.mark.parametrize(
    "extra, why",
    [
        (team("x", "Raffael Cerqueira"), "already carry the name"),
        (team("x", "Raffael Cerqueirá"), "already carry the name"),  # accent-folded
        (team("x", "R. Cerqueira", aliases=["Raffael Cerqueira"]), "already claimed"),
        (team("x", "Somebody", espn="5212598"), "ESPN athlete 5212598 is on"),
    ],
)
def test_the_alias_refuses(extra, why):
    st = state(teams=[team(RAF, "Rafael Cerqueira"), extra])
    u = mic.plan_alias(mic.AliasRow(RAF, "Rafael Cerqueira", "Raffael Cerqueira", "5212598"), st, resolvers())
    assert u.action == "refuse" and why in u.reason, u.reason


def test_the_orphan_is_deleted_then_done():
    o = mic.OrphanRow(JOSE, "José Luiz")
    u = mic.plan_orphan(o, state())
    assert u.action == "delete" and u.detail == {"name": "José Luiz", "normalized_name": "josé luiz"}
    assert mic.plan_orphan(o, state(teams=[team(OPP, "Opponent")])).action == "done"


@pytest.mark.parametrize(
    "row, refs, why",
    [
        (team(JOSE, "José Luiz"), {JOSE: {"matches.away_team_id": 1}}, "is referenced"),
        (team(JOSE, "José Luiz", espn="5351808"), None, "carries an ESPN id"),
        (team(JOSE, "José Luiz", aliases=["Jose Luiz"]), None, "carries an ESPN id or aliases"),
        (team(JOSE, "José Montanha"), None, "is named"),
    ],
)
def test_the_orphan_refuses(row, refs, why):
    u = mic.plan_orphan(mic.OrphanRow(JOSE, "José Luiz"), state(teams=[row], refs=refs))
    assert u.action == "refuse" and why in u.reason, u.reason


@pytest.mark.parametrize(
    "rows, action, why",
    [
        ([team(RAF, "Rafael Cerqueira")], "stamp", ""),
        ([team(RAF, "Rafael Cerqueira", espn="5212598")], "done", ""),
        ([team(RAF, "Rafael Cerqueira", espn="1")], "refuse", "carries ESPN id 1"),
        ([team(RAF, "Rafael Cerqueira"), team("x", "Raffael Cerqueira", espn="5212598")], "refuse", "merge it first"),
        ([team(RAF, "Rafael C.")], "refuse", "not an MMA row of that name"),
        ([], "refuse", "not an MMA row of that name"),
    ],
)
def test_the_athlete_id_plan(rows, action, why):
    u = mic.plan_id(mic.IdRow(RAF, "Rafael Cerqueira", "5212598"), state(teams=rows))
    assert u.action == action and why in u.reason, u.reason


# ── the plan document ────────────────────────────────────────────────


def _doc(st):
    ls = lists()
    return mic.plan_document(mic.plan_run(ls, st, resolvers()), ls, "dry-run")


def test_the_digest_covers_the_ids_rows_end_up_with_not_stamp_or_done():
    before = _doc(state())
    teams = [
        team(T, "Bruno Silva"),
        team(RAF, "Rafael Cerqueira"),
        team(JOSE, "José Luiz"),
        team(ZACH, "Zach Reese", espn="5143223"),
        team(OPP, "Opponent"),
    ]
    assert before["summary"]["ids_stamp"] == 1 and before["summary"]["ids_done"] == 1
    swept = _doc(state(teams=[team(ZACH, "Zach Reese", espn="5143223", aliases=["Zachary Reese"])] + teams[:3]))
    assert swept["digest"] == before["digest"]
    teams[1] = team(RAF, "Rafael Cerqueira", espn="5212598")  # a sweep stamped him between dry run and apply
    after_sweep = _doc(state(teams=teams + [team(ZACH + "x", "Zz")]))
    assert after_sweep["summary"]["ids_stamp"] == 0 and after_sweep["summary"]["ids_done"] == 2
    assert after_sweep["digest"] == before["digest"]
    assert mic.plan_difference(before, after_sweep) is None


def test_the_digest_moves_with_a_decision():
    before = _doc(state())
    moved = bouts_on()
    moved[M1] = {**moved[M1], "status": "cancelled"}
    changed = _doc(state(matches=moved))
    assert changed["digest"] != before["digest"]
    assert mic.plan_difference(before, changed) is not None
    assert mic.plan_difference({"tool": "merge_mma_fighters", "digest": "x"}, before).startswith("the expected plan")


def test_a_difference_only_the_digest_sees_still_refuses():
    """Same actions everywhere, but the split row's metadata (which the guarded stamp writes back) changed: only
    the digest differs, and that alone must refuse the apply."""
    before = _doc(state())
    teams = [
        team(T, "Bruno Silva", note="edited"),
        team(RAF, "Rafael Cerqueira"),
        team(JOSE, "José Luiz"),
        team(ZACH, "Zach Reese", espn="5143223", aliases=["Zachary Reese"]),
        team(OPP, "Opponent"),
    ]
    changed = _doc(state(teams=teams))
    assert changed["summary"] == before["summary"] and changed["digest"] != before["digest"]
    assert mic.plan_difference(before, changed) == "the plan digest differs from the expected plan"
    assert (
        mic.plan_difference(dict(before, digest="0" * 64), before) == "the plan digest differs from the expected plan"
    )


def test_plan_difference_names_an_id_that_changed():
    before = _doc(state())
    teams = [
        team(T, "Bruno Silva"),
        team(RAF, "Rafael Cerqueira"),
        team(JOSE, "José Luiz"),
        team(ZACH, "Zach Reese", espn="1", aliases=["Zachary Reese"]),
        team(OPP, "Opponent"),
    ]
    assert mic.plan_difference(before, _doc(state(teams=teams))) == (
        f"ids {ZACH}: the planned athlete id differs from the expected plan"
    )


def test_summary_and_refusals():
    ls = lists()
    plan = mic.plan_run(ls, state(), resolvers())
    assert mic.plan_line(mic.summarize(plan)) == (
        "PLAN split=1 split_done=0 split_refuse=0 bouts_moved=2 predictions_on_moved=2 recs_on_moved=0 "
        "alias_add=1 alias_done=0 alias_refuse=0 orphan_delete=1 orphan_done=0 orphan_refuse=0 "
        "ids_stamp=1 ids_done=1 ids_refuse=0"
    )
    assert mic.refusals(plan) == []


def test_the_resolver_must_be_the_athlete_id_one(monkeypatch):
    res, why = mic.load_resolvers()
    assert why is None and res is not None
    import dataclasses
    import importlib

    deployed = importlib.import_module("fetch_upcoming")  # the module load_resolvers reads (other tests re-load it)
    plain = dataclasses.replace(deployed.SPORT_CONFIGS["mma"], athlete_ids=False)
    monkeypatch.setitem(deployed.SPORT_CONFIGS, "mma", plain)
    res, why = mic.load_resolvers()
    assert res is None and "athlete id" in why


def test_after_state_ok_needs_every_id_recorded():
    out = {k: 0 for k in mic.AFTER_KEYS}
    out.update(ids_expected=3, ids_recorded=3)
    assert mic.after_state_ok(out)
    assert not mic.after_state_ok({**out, "ids_recorded": 2})
    assert not mic.after_state_ok({**out, "split_resolver_misses": 1})
