"""scripts/merge_mma_fighters.py, the pure parts: the reviewed lists, the
clusters a merge creates (either order), what a fold moves (the freshest
pre-game price per key, pre-game snapshots, the latest pre-game prediction per
key; a reversed copy's prices with home/away swapped and none of its
predictions), the refusals (a live row, two finished rows, a copy not yet
stale, a re-point that collides, an alias or a name another row holds, an
athlete id on a third row), the rec-held copy left in place (a listed one
refused), a rec's prediction never moved, 'done' detection, the NFC renames,
the plan digest and its comparison, the after-state rule, and apply_fold's
refusal to delete a copy holding a child row the archive lacks.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
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


mm = _load("merge_mma_fighters_t", "merge_mma_fighters.py")
import fetch_upcoming as fu  # noqa: E402
import team_identity as ti  # noqa: E402

NOW = datetime(2026, 11, 1, 21, 0, tzinfo=timezone.utc)
CARD = datetime(2026, 7, 11, 21, 0, tzinfo=timezone.utc)
RES = mm.Resolvers(
    normalize=fu.normalize_team,
    alias_form=ti.alias_form,
    find_team=lambda cur, name, league: None,
    find_team_by_alias=lambda cur, name: None,
    stamp_espn_id=lambda *a, **k: True,
)

KEPT, MERGED, OPP, OPP2, OTHER = (f"00000000-0000-4000-8000-00000000000{i}" for i in range(1, 6))


def _uid(n: int) -> str:
    return f"10000000-0000-4000-8000-{n:012d}"


def team(tid, name, espn="", aliases=None, merged_from=None):
    meta = {}
    if aliases is not None:
        meta["aliases"] = aliases
    if merged_from is not None:
        meta["merged_from"] = merged_from
    return {
        "id": tid,
        "name": name,
        "normalized_name": fu.normalize_team(name),
        "sport": "mma",
        "league_id": "L",
        "external_ids": {"espn": espn} if espn else {},
        "metadata": meta,
    }


def match(mid, home, away, when, status="finished", espn="", score=(1, 0)):
    fin = status == "finished"
    return {
        "id": mid,
        "league_id": "L",
        "home_team_id": home,
        "away_team_id": away,
        "match_date": when,
        "status": status,
        "home_score": score[0] if fin else None,
        "away_score": score[1] if fin else None,
        "espn": espn,
    }


def odd(oid, mid, sel, ts, book="FanDuel", market="moneyline", live=False, price="2.00"):
    return {
        "id": oid,
        "match_id": mid,
        "bookmaker": book,
        "market_type": market,
        "selection": sel,
        "line": None,
        "is_opening": False,
        "is_live": live,
        "timestamp": ts,
        "created_at": ts,
        "odds_decimal": price,
    }


def snap(sid, mid, sel, ts, book="FanDuel"):
    return {
        "id": sid,
        "match_id": mid,
        "bookmaker": book,
        "market_type": "moneyline",
        "selection": sel,
        "line": None,
        "captured_at": ts,
    }


def pred(pid, mid, version, written, rec=False):
    return {
        "id": pid,
        "match_id": mid,
        "model_name": "ensemble_mma_ml",
        "model_version": version,
        "prediction_type": "moneyline",
        "created_at": written - timedelta(days=1),
        "updated_at": written,
        "rec_referenced": rec,
    }


PAIR = mm.MergePair(MERGED, "Zachary Reese", KEPT, "Zach Reese", "5143223", "espn_athlete_id", "evidence")


def state(matches, teams=None, deps=None, sport_teams=None, holders=None, refs=None):
    teams = teams or [team(KEPT, "Zach Reese"), team(MERGED, "Zachary Reese")]
    others = [team(OPP, "Ryan Gandra"), team(OPP2, "Eli Aronov")]
    sport = sport_teams if sport_teams is not None else teams + others
    st = [
        {
            "id": t["id"],
            "name": t["name"],
            "normalized_name": t["normalized_name"],
            "espn": (t.get("external_ids") or {}).get("espn", ""),
            "aliases": (t.get("metadata") or {}).get("aliases"),
        }
        for t in sport
    ]
    ms = {m["id"]: m for m in matches}
    d = {mid: mm.Deps() for mid in ms}
    for mid, dep in (deps or {}).items():
        d[mid] = dep
    return mm.State({t["id"]: t for t in teams}, st, ms, d, holders or {}, refs or {})


def plan(st, pairs=(PAIR,), listed=()):
    return mm.plan_run(list(pairs), list(listed), st, NOW, RES)


# ── the reviewed lists ────────────────────────────────────────────────


def test_the_committed_lists_load():
    pairs = mm.load_map(mm.MAP_PATH)
    assert len(pairs) == 9
    assert {p.merged_name: p.kept_name for p in pairs}["Zachary Reese"] == "Zach Reese"
    assert all(p.espn_athlete_id.isdigit() for p in pairs)
    folds = mm.load_folds(mm.FOLDS_PATH)
    assert [(f.stale_home, f.stale_away) for f in folds] == [("Marcus McGhee", "Anthony Romero")]
    assert folds[0].kickoff == datetime(2026, 10, 3, 22, 0, tzinfo=timezone.utc)


def _write(tmp_path, header, rows):
    p = tmp_path / "list.csv"
    p.write_text("# a comment\n" + header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return p


HDR = ",".join(mm.MAP_FIELDS)


@pytest.mark.parametrize(
    "rows, why",
    [
        ([f"{MERGED},A,{KEPT},B,1,m,"], "empty evidence"),
        ([f"{MERGED},A,{MERGED},B,1,m,e"], "merges a row into itself"),
        ([f"{MERGED},A,{KEPT},A,1,m,e"], "merges a row into itself"),
        ([f"{MERGED},A,{KEPT},B,x1,m,e"], "not digits"),
        ([f"{MERGED},A,{KEPT},B,1,m,e", f"{OPP},C,{KEPT},B,2,m,e"], "listed twice"),
        ([f"{MERGED},A,{KEPT},B,1,m,e", f"{OPP},C,{OPP2},D,1,m,e"], "listed twice"),
        ([f"not-a-uuid,A,{KEPT},B,1,m,e"], "not a team id"),
    ],
)
def test_a_bad_map_is_refused(tmp_path, rows, why):
    with pytest.raises(mm.ListError, match=why):
        mm.load_map(_write(tmp_path, HDR, rows))


def test_a_map_without_a_column_is_refused(tmp_path):
    with pytest.raises(mm.ListError, match="missing column"):
        mm.load_map(_write(tmp_path, "merged_team_id,merged_name", [f"{MERGED},A"]))


def test_a_fold_kickoff_without_a_zone_is_refused(tmp_path):
    hdr = ",".join(mm.FOLD_FIELDS)
    with pytest.raises(mm.ListError, match="no time zone"):
        mm.load_folds(_write(tmp_path, hdr, [f"{_uid(1)},{_uid(2)},2026-10-03T22:00:00,A,B,1,e"]))


def test_spellings_are_the_composed_and_decomposed_forms():
    assert mm.spellings("Zach Reese") == ["Zach Reese"]
    nfc, nfd = mm.spellings("Adrian Yañez")
    assert nfc == "Adrian Yañez" and nfd == "Adrian Yañez"


def test_an_unknown_foreign_key_refuses_the_schema():
    rows = [
        {"tbl": "matches", "col": "home_team_id", "ref": "teams"},
        {"tbl": "matches", "col": "away_team_id", "ref": "teams"},
        {"tbl": "odds", "col": "match_id", "ref": "matches"},
        {"tbl": "fighter_bios", "col": "team_id", "ref": "teams"},
    ]
    schema, unknown = mm.read_schema(rows)
    assert schema is None and unknown == ["fighter_bios.team_id -> teams"]
    schema, unknown = mm.read_schema(rows[:3])
    assert unknown == [] and schema.match_children == ("odds",)


# ── clusters ──────────────────────────────────────────────────────────


def test_a_cluster_is_the_same_two_fighters_in_either_order_within_the_window():
    rows = [
        match("a", KEPT, OPP, CARD),
        match("b", MERGED, OPP, CARD, "scheduled"),
        match("c", OPP, MERGED, CARD + timedelta(hours=3), "scheduled"),
        match("d", MERGED, OPP, CARD + timedelta(days=60)),  # a rematch, not this bout
        match("e", KEPT, OPP2, CARD),  # no merged row
        match("f", KEPT, OPP2, CARD + timedelta(hours=1), "scheduled"),
    ]
    clusters = mm.find_clusters(rows, {MERGED: KEPT}, {MERGED})
    assert [[m["id"] for m in cl] for cl in clusters] == [["a", "b", "c"]]


# ── folds ─────────────────────────────────────────────────────────────


def _fold_state():
    stale, keep = match("s", MERGED, OPP, CARD, "scheduled", espn="401874061"), match("k", KEPT, OPP, CARD)
    t0 = CARD - timedelta(hours=10)
    deps = {
        "s": mm.Deps(
            odds=[
                odd(_uid(1), "s", "home", t0),
                odd(_uid(2), "s", "home", t0 + timedelta(hours=1)),  # fresher: this one moves
                odd(_uid(3), "s", "away", CARD + timedelta(minutes=5)),  # in play by time
                odd(_uid(4), "s", "away", t0, live=True),  # live
                odd(_uid(5), "s", "away", t0, book="DraftKings"),
                odd(_uid(6), "s", "home", t0, book="Caesars"),  # the finished row holds this key
            ],
            snapshots=[
                snap(_uid(11), "s", "home", t0),
                snap(_uid(12), "s", "home", t0),  # same key, later id: not moved twice
                snap(_uid(13), "s", "away", CARD),  # at kickoff: not pre-game
                snap(_uid(14), "s", "away", t0, book="Caesars"),  # the finished row holds it
            ],
            predictions=[
                pred(_uid(21), "s", "v1", t0),
                pred(_uid(22), "s", "v1", t0 + timedelta(hours=1)),  # latest for v1: moves
                pred(_uid(23), "s", "v2", CARD + timedelta(hours=1)),  # written after kickoff
                pred(_uid(24), "s", "v3", t0),  # the finished row holds v3
            ],
        ),
        "k": mm.Deps(
            odds=[odd(_uid(31), "k", "home", t0, book="Caesars")],
            snapshots=[snap(_uid(32), "k", "away", t0, book="Caesars")],
            predictions=[pred(_uid(33), "k", "v3", t0)],
        ),
    }
    return stale, keep, deps


def test_a_same_order_fold_moves_the_freshest_pregame_rows_the_finished_row_lacks():
    stale, keep, deps = _fold_state()
    f = mm.plan_fold(stale, keep, deps, False, "pair 1", {"401874061": {"s"}})
    assert f.odds == [[_uid(2), "home"], [_uid(5), "away"]]
    assert f.snapshots == [[_uid(11), "home"]]
    assert f.predictions == [_uid(22)]
    assert f.predictions_dropped == 3
    assert f.cutoff == CARD and f.adopt_espn == "401874061"
    # the finished row now holds what moved: a second copy of the bout moves none of it again
    again = mm.plan_fold(dict(stale, id="s2"), keep, {**deps, "s2": deps["s"]}, False, "pair 1", {})
    assert again.odds == [] and again.snapshots == [] and again.predictions == []


def test_a_reversed_fold_swaps_the_selection_and_moves_no_prediction():
    stale, keep, deps = _fold_state()
    deps["s"].odds = [o for o in deps["s"].odds if o["id"] in (_uid(1), _uid(5))]
    f = mm.plan_fold(stale, keep, deps, True, "listed 1", {})
    assert f.odds == [[_uid(1), "away"], [_uid(5), "home"]]
    assert f.snapshots == [[_uid(11), "away"], [_uid(14), "home"]]  # 14 swaps to a key the row lacks
    assert f.predictions == [] and f.predictions_dropped == 4


def test_a_fold_never_moves_a_prediction_a_rec_references():
    stale, keep, deps = _fold_state()
    deps["s"].predictions = [
        pred(_uid(21), "s", "v1", CARD - timedelta(hours=10)),
        pred(_uid(22), "s", "v1", CARD - timedelta(hours=9), rec=True),  # the latest, but a rec's: it stays
    ]
    f = mm.plan_fold(stale, keep, deps, False, "pair 1", {})
    assert f.predictions == [_uid(21)] and f.predictions_dropped == 1


def test_a_reversed_copy_with_a_price_that_has_no_side_refuses():
    stale, keep, deps = _fold_state()
    deps["s"].odds.append(odd(_uid(7), "s", "over", CARD - timedelta(hours=2), market="total"))
    assert "total/over" in mm.plan_fold(stale, keep, deps, True, "listed 1", {})


@pytest.mark.parametrize(
    "keep_espn, holders, keep_when, adopted",
    [
        ("", {"401874061": {"s"}}, CARD, "401874061"),
        ("401000000", {"401874061": {"s"}}, CARD, ""),  # the finished row has its own
        ("", {"401874061": {"s", "x"}}, CARD, ""),  # another row holds it
        ("", {"401874061": {"s"}}, CARD + timedelta(hours=3), ""),  # a different kickoff
    ],
)
def test_the_espn_competition_id_moves_only_onto_an_unidentified_row_at_the_same_kickoff(
    keep_espn, holders, keep_when, adopted
):
    stale, keep, deps = _fold_state()
    keep = dict(keep, espn=keep_espn, match_date=keep_when)
    assert mm.plan_fold(stale, keep, deps, False, "pair 1", holders).adopt_espn == adopted


# ── the run ───────────────────────────────────────────────────────────


def _bout_rows():
    return [
        match("hist", MERGED, OPP2, CARD - timedelta(days=400)),  # history: re-pointed
        match("stale", MERGED, OPP, CARD, "scheduled"),  # folded
        match("fin", KEPT, OPP, CARD),
        match("next", KEPT, OPP2, NOW + timedelta(days=10), "scheduled"),  # the kept row's own
    ]


def test_a_merge_folds_the_copy_repoints_the_history_and_records_the_alias():
    p = plan(state(_bout_rows()))
    (pl,) = p.pairs
    assert pl.action == "merge", pl.reason
    assert [(f.stale_id, f.keep_id, f.reversed) for f in pl.folds] == [("stale", "fin", False)]
    assert [(r.match_id, r.side) for r in pl.repoints] == [("hist", "home")]
    assert pl.team_patch["aliases"] == ["Zachary Reese"] and pl.team_patch["stamp"] is True
    s = mm.summarize(p)
    assert (s["pairs_merge"], s["folds"], s["folds_reversed"], s["repoints"], s["athlete_ids_stamped"]) == (
        1,
        1,
        0,
        1,
        1,
    )
    assert mm.plan_line(s).startswith("PLAN pairs_merge=1 pairs_done=0 pairs_refuse=0 folds=1 ")


def test_a_copy_the_other_way_round_is_folded_reversed():
    rows = [match("stale", OPP, MERGED, CARD, "scheduled"), match("fin", KEPT, OPP, CARD)]
    (pl,) = plan(state(rows)).pairs
    assert [(f.stale_id, f.reversed) for f in pl.folds] == [("stale", True)]
    assert pl.folds[0].keep_teams == [KEPT, OPP]


def test_a_copy_a_rec_holds_is_left_in_place_and_repointed():
    rows = [match("stale", MERGED, OPP, CARD - timedelta(hours=1), "scheduled"), match("fin", KEPT, OPP, CARD)]
    p = plan(state(rows, deps={"stale": mm.Deps(recs=1)}))
    (pl,) = p.pairs
    assert pl.action == "merge" and pl.folds == [] and pl.left == ["stale"]
    assert [(r.match_id, r.side) for r in pl.repoints] == [("stale", "home")]
    assert mm.summarize(p)["left_rec_held"] == 1


def test_a_copy_a_rec_holds_through_a_prediction_is_left_too():
    rows = [match("stale", MERGED, OPP, CARD - timedelta(hours=1), "scheduled"), match("fin", KEPT, OPP, CARD)]
    (pl,) = plan(state(rows, deps={"stale": mm.Deps(pred_recs=1)})).pairs
    assert pl.left == ["stale"] and pl.folds == []


@pytest.mark.parametrize(
    "rows, why",
    [
        ([match("a", MERGED, OPP, CARD, "live"), match("b", KEPT, OPP, CARD)], "in play"),
        ([match("a", MERGED, OPP, CARD), match("b", KEPT, OPP, CARD + timedelta(hours=1))], "two finished rows"),
        (
            [
                match("a", MERGED, OPP, NOW - timedelta(hours=20), "scheduled"),
                match("b", KEPT, OPP, NOW - timedelta(hours=20)),
            ],
            "results sweep may still write it",
        ),
        ([match("a", MERGED, OPP, CARD, "postponed"), match("b", KEPT, OPP, CARD)], "'postponed' beside"),
        # a rec-held copy at the finished row's kickoff, same order: its re-point would duplicate the row
        ([match("a", MERGED, OPP, CARD, "scheduled"), match("b", KEPT, OPP, CARD)], "would duplicate"),
    ],
)
def test_what_a_merge_refuses(rows, why):
    deps = {"a": mm.Deps(recs=1)} if why == "would duplicate" else None
    (pl,) = plan(state(rows, deps=deps)).pairs
    assert pl.action == "refuse" and why in pl.reason, pl.reason
    assert pl.folds == [] and pl.repoints == []


def test_a_cluster_without_a_finished_row_is_only_repointed():
    rows = [match("a", MERGED, OPP, CARD, "scheduled"), match("b", KEPT, OPP, CARD + timedelta(hours=3), "scheduled")]
    (pl,) = plan(state(rows)).pairs
    assert pl.action == "merge" and pl.folds == [] and [r.match_id for r in pl.repoints] == ["a"]


@pytest.mark.parametrize(
    "teams, sport_extra, refs, why",
    [
        ([team(KEPT, "Zach Reese"), team(MERGED, "Zachary Reese ")], [], {}, "not mma 'Zachary Reese'"),
        ([team(KEPT, "Zach Reese", espn="999"), team(MERGED, "Zachary Reese")], [], {}, "holds ESPN id 999"),
        (None, [team(OTHER, "Somebody", espn="5143223")], {}, "on another MMA row"),
        (None, [team(OTHER, "Z. Reese", aliases=["zachary  reese"])], {}, "already claimed"),
        (None, [team(OTHER, "ZACHARY REESE")], {}, "still names team(s)"),
        (None, [], {MERGED: {"players": 1}}, "referenced outside matches"),
        ([team(KEPT, "Zach Reese", aliases="Zachary"), team(MERGED, "Zachary Reese")], [], {}, "malformed"),
        ([team(MERGED, "Zachary Reese")], [], {}, "does not exist"),
    ],
)
def test_what_a_pair_refuses_before_its_fixtures(teams, sport_extra, refs, why):
    base = teams or [team(KEPT, "Zach Reese"), team(MERGED, "Zachary Reese")]
    sport = base + [team(OPP, "Ryan Gandra"), team(OPP2, "Eli Aronov")] + sport_extra
    if why == "still names team(s)":
        sport[-1]["normalized_name"] = "zachary reese"
    (pl,) = plan(state(_bout_rows(), teams=base, sport_teams=sport, refs=refs)).pairs
    assert pl.action == "refuse" and why in pl.reason, pl.reason


def test_a_merged_row_recorded_on_the_kept_row_is_done():
    kept = team(KEPT, "Zach Reese", espn="5143223", aliases=["Zachary Reese"], merged_from=[{"team_id": MERGED}])
    p = plan(state([match("fin", KEPT, OPP, CARD)], teams=[kept]))
    assert [pl.action for pl in p.pairs] == ["done"]
    s = mm.summarize(p)
    assert s["pairs_done"] == 1 and s["folds"] == 0 and s["repoints"] == 0


def test_a_merged_row_gone_without_a_record_refuses():
    (pl,) = plan(state([match("fin", KEPT, OPP, CARD)], teams=[team(KEPT, "Zach Reese")])).pairs
    assert pl.action == "refuse" and "does not record" in pl.reason


def _listed(stale_id="s", fin_id="f", kickoff=None, espn="401927906"):
    return mm.ListedFold(stale_id, fin_id, kickoff or CARD, "A", "B", espn, "e")


def test_a_listed_reversed_copy_folds_and_then_reads_done():
    rows = [match("s", OPP, OPP2, CARD, "scheduled", espn="401927906"), match("f", OPP2, OPP, CARD)]
    p = plan(state(rows), pairs=(), listed=(_listed(),))
    (lp,) = p.listed
    assert lp.action == "fold" and lp.fold.reversed and lp.fold.adopt_espn == "401927906"
    p2 = plan(state([rows[1]]), pairs=(), listed=(_listed(),))
    assert p2.listed[0].action == "done" and mm.summarize(p2)["folds_done"] == 1


@pytest.mark.parametrize(
    "rows, listed, why",
    [
        ([match("s", OPP, OPP2, CARD, "scheduled"), match("f", OPP, OPP2, CARD)], _listed(), "other way round"),
        ([match("s", OPP, OPP2, CARD, "scheduled")], _listed(), "is gone"),
        (
            [match("s", OPP, OPP2, CARD, "scheduled"), match("f", OPP2, OPP, CARD)],
            _listed(kickoff=CARD + timedelta(hours=1)),
            "the list says",
        ),
        (
            [match("s", OPP, OPP2, CARD, "scheduled", espn="1"), match("f", OPP2, OPP, CARD)],
            _listed(),
            "carries ESPN id 1",
        ),
    ],
)
def test_what_a_listed_fold_refuses(rows, listed, why):
    (lp,) = plan(state(rows), pairs=(), listed=(listed,)).listed
    assert lp.action == "refuse" and why in lp.reason, lp.reason


@pytest.mark.parametrize("dep", [mm.Deps(recs=1), mm.Deps(history=1), mm.Deps(pred_recs=1)])
def test_a_listed_copy_a_rec_holds_is_refused_not_folded(dep):
    rows = [match("s", OPP, OPP2, CARD, "scheduled", espn="401927906"), match("f", OPP2, OPP, CARD)]
    p = plan(state(rows, deps={"s": dep}), pairs=(), listed=(_listed(),))
    (lp,) = p.listed
    assert lp.action == "refuse" and "left in place" in lp.reason and lp.fold is None, lp.reason
    assert mm.summarize(p)["folds_refuse"] == 1 and p.folds() == []


def test_a_decomposed_name_is_renamed_unless_its_composed_form_is_taken():
    nfd = team(OTHER, "Rose Conceição")
    nfd["name"] = "Rose Conceição"
    nfd["normalized_name"] = "rose conceição"
    p = plan(state([], sport_teams=[nfd]), pairs=())
    (r,) = p.nfc
    assert r.new_name == "Rose Conceição" and r.new_normalized_name == "rose conceição"
    clash = team(OPP, "Rose Conceição")
    p = plan(state([], sport_teams=[nfd, clash]), pairs=())
    assert p.nfc == [] and "is held by" in p.nfc_refused[0]
    assert mm.summarize(p)["nfc_refuse"] == 1


# ── plan file ─────────────────────────────────────────────────────────


def test_the_digest_covers_every_moved_row_and_the_difference_names_it():
    stale, keep, deps = _fold_state()
    rows = [dict(stale), keep]
    st = state(rows, deps=deps)
    doc = mm.plan_document(plan(st), "dry-run")
    assert doc["digest"] == mm.plan_document(plan(st), "apply")["digest"]  # mode and time do not count
    assert mm.plan_difference(doc, doc) is None
    deps2 = {k: mm.Deps(list(v.odds), list(v.snapshots), list(v.predictions)) for k, v in deps.items()}
    deps2["s"].odds = [o for o in deps2["s"].odds if o["id"] != _uid(5)]
    doc2 = mm.plan_document(plan(state(rows, deps=deps2)), "apply")
    assert doc2["digest"] != doc["digest"]
    assert "counts differ" in mm.plan_difference(doc, doc2)
    deps3 = {k: mm.Deps(list(v.odds), list(v.snapshots), list(v.predictions)) for k, v in deps.items()}
    deps3["s"].odds = [dict(o, id=_uid(9)) if o["id"] == _uid(5) else o for o in deps3["s"].odds]
    doc3 = mm.plan_document(plan(state(rows, deps=deps3)), "apply")
    assert mm.plan_difference(doc, doc3) == f"pairs {MERGED}: folds differs from the expected plan"
    assert "not a merge_mma_fighters plan" in mm.plan_difference({"tool": "other"}, doc)


def test_grade_after_lists_the_finished_rows_that_receive_predictions():
    stale, keep, deps = _fold_state()
    p = plan(state([stale, keep], deps=deps))
    assert mm.grade_after(p) == ["k"]
    assert mm.plan_document(p, "dry-run")["grade_after"] == ["k"]


def test_the_audit_rows_name_every_write():
    stale, keep, deps = _fold_state()
    rows = [stale, keep, match("hist", MERGED, OPP2, CARD - timedelta(days=400))]
    actions = [r["action"] for r in mm.csv_rows(plan(state(rows, deps=deps)))]
    assert actions.count("move") == 4 and "fold_stale" in actions and "adopt_espn_id" in actions
    assert actions[-3:] == ["repoint_home", "record_merge", "delete_team"]


def test_the_after_state_rule():
    ok = {k: 0 for k in mm.AFTER_KEYS}
    ok.update(pairs_expected=9, pairs_recorded=9)
    assert mm.after_state_ok(ok)
    assert not mm.after_state_ok({**ok, "pairs_recorded": 8})
    assert not mm.after_state_ok({**ok, "resolver_misses": 1})
    assert not mm.after_state_ok({**ok, "repoints_missing": 1})
    assert mm.after_state_ok({**ok, "repoints_missing": 1}, after_later_steps=True)
    assert mm.after_line(ok).startswith("pairs_expected=9 pairs_recorded=9 merged_rows_left=0")


def test_normalize_team_composes_a_decomposed_name():
    assert fu.normalize_team("Rose Conceição ") == "rose conceição"
    assert fu.normalize_team("  Zach   REESE ") == "zach reese"


# ── apply_fold: the archive guard ─────────────────────────────────────


class _ChildCursor:
    """Just enough cursor for apply_fold on a copy with nothing to move: the
    child-row read answers `children`; every other statement changes 1 row."""

    def __init__(self, schema, children):
        self.children_sql, self.children = mm.children_sql(schema), children
        self.sql: list[str] = []
        self.rowcount = 0
        self._rows: list[dict] = []

    def execute(self, sql, params=None):
        self.sql.append(sql)
        self._rows, self.rowcount = (self.children, len(self.children)) if sql == self.children_sql else ([], 1)

    def fetchall(self):
        return self._rows


def _bare_fold():
    return mm.Fold(
        "s", "k", False, CARD, "pair 1", match("s", MERGED, OPP, CARD, "scheduled"), match("k", KEPT, OPP, CARD)
    )


def test_a_copy_holding_a_child_row_the_archive_lacks_is_never_deleted():
    schema = mm.Schema(team_refs=(("matches", "home_team_id"), ("matches", "away_team_id")), match_children=("odds",))
    cur = _ChildCursor(schema, [{"tbl": "odds", "id": "o1"}, {"tbl": "odds", "id": "o2"}])  # o2 arrived after
    with pytest.raises(mm.GuardError, match="1 row\\(s\\) the archive lacks.*odds:o2"):
        mm.apply_fold(cur, _bare_fold(), schema, {"odds": {"o1"}}, Counter())
    assert mm.DELETE_STALE_SQL not in cur.sql
    cur, done = _ChildCursor(schema, [{"tbl": "odds", "id": "o1"}]), Counter()
    mm.apply_fold(cur, _bare_fold(), schema, {"odds": {"o1"}}, done)
    assert mm.DELETE_STALE_SQL in cur.sql and done["stale_copies_deleted"] == 1
