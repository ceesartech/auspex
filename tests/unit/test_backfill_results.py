"""scripts/backfill_results.py without a database: how a row's change is
classified into the PLAN line, the --dates and --expect-plan guards, and the
rule that --apply needs every key pinned. The transaction itself runs against
a real Postgres in test_backfill_results_pg.py."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

_spec = importlib.util.spec_from_file_location("backfill_results_unit", SCRIPTS_DIR / "backfill_results.py")
bf = importlib.util.module_from_spec(_spec)
sys.modules["backfill_results_unit"] = bf
_spec.loader.exec_module(bf)

KICKOFF = datetime(2026, 6, 6, 21, tzinfo=timezone.utc)


def _match(**kw):
    row = {
        "status": "scheduled",
        "match_date": KICKOFF,
        "home_team_id": "h",
        "away_team_id": "a",
        "home_score": None,
        "away_score": None,
        "external_ids": {},
        "metadata": {},
    }
    row.update(kw)
    return row


def _rec(**kw):
    row = {"match_id": "m", "status": "pending", "actual_result": None, "profit_loss": None, "settled_at": None}
    row.update(kw)
    return row


FINAL = {"status": "finished", "home_score": 0, "away_score": 1, "external_ids": {"espn": "401874287"}}


@pytest.mark.parametrize(
    "old, new, key",
    [
        (_match(), _match(), None),
        (None, _match(**FINAL), "inserted"),
        (_match(), None, "other"),
        (_match(), _match(**FINAL), "flipped"),
        (_match(), _match(**FINAL, match_date=datetime(2026, 6, 7, tzinfo=timezone.utc)), "flipped"),
        (_match(), _match(status="cancelled", metadata={"espn_status": "DROPPED_FROM_CARD"}), "cancelled"),
        (_match(**FINAL), _match(**{**FINAL, "metadata": {"season_type": "regular"}}), "stamped"),
        (_match(**{**FINAL, "external_ids": {}}), _match(**FINAL), "stamped"),
        (_match(**FINAL), _match(**{**FINAL, "home_score": 1, "away_score": 0}), "rescored"),
        (_match(**FINAL), _match(**{**FINAL, "match_date": datetime(2026, 6, 7, tzinfo=timezone.utc)}), "other"),
        (_match(), _match(match_date=datetime(2026, 6, 7, tzinfo=timezone.utc)), "other"),
        (_match(status="cancelled"), _match(**FINAL), "other"),
    ],
)
def test_a_match_change_lands_in_one_plan_key(old, new, key):
    assert bf.classify_match(old, new) == key


@pytest.mark.parametrize(
    "old, new, key",
    [
        (_rec(), _rec(), None),
        (_rec(), _rec(status="won", actual_result="won", profit_loss=Decimal("20.00")), "recs_settled"),
        (_rec(), _rec(status="lost", actual_result="lost", profit_loss=Decimal("-10.00")), "recs_settled"),
        (_rec(), _rec(status="won", actual_result="home"), "recs_other"),  # the trigger's status, no money
        (_rec(), _rec(status="void", actual_result="void: match cancelled", profit_loss=Decimal("0")), "recs_voided"),
        (_rec(status="won", profit_loss=Decimal("1")), _rec(status="lost", profit_loss=Decimal("-1")), "recs_other"),
        (_rec(), _rec(match_id="twin"), "recs_other"),
        (None, _rec(), "recs_other"),
        (_rec(), None, "recs_other"),
    ],
)
def test_a_rec_change_lands_in_one_plan_key(old, new, key):
    assert bf.classify_rec(old, new) == key


def test_diff_counts_and_lists_every_changed_row():
    before = ({"m1": _match(), "m2": _match(**FINAL)}, {"r1": _rec(match_id="m1")})
    after = (
        {"m1": _match(**FINAL), "m2": _match(**FINAL), "m3": _match(**FINAL)},
        {"r1": _rec(match_id="m1", status="won", actual_result="won", profit_loss=Decimal("5"))},
    )
    counts, entries = bf.diff(before, after)
    assert dict(counts) == {"flipped": 1, "inserted": 1, "recs_settled": 1}
    assert [(e["kind"], e["id"], e["change"]) for e in entries] == [
        ("match", "m1", "flipped"),
        ("match", "m3", "inserted"),
        ("rec", "r1", "recs_settled"),
    ]


def test_the_plan_line_names_every_key_in_order():
    line = bf.plan_line({"results": 56, "flipped": 44})
    assert line.startswith("PLAN results=56 flipped=44 inserted=0 ")
    assert [p.split("=")[0] for p in line.split()[1:]] == list(bf.PLAN_KEYS)


TODAY = date(2026, 11, 1)


@pytest.mark.parametrize(
    "text, problem",
    [
        ("", "empty"),
        ("2026-06-06,2026-06-06", "twice"),
        ("2026-6-6x", "YYYY-MM-DD"),
        ("2026-10-29", "within 4 days"),
        ("2026-11-02", "within 4 days"),
        (",".join(f"2026-05-{d:02d}" for d in range(1, 32)) + ",2026-06-01", "more than 31"),
    ],
)
def test_bad_dates_are_refused(text, problem):
    with pytest.raises(bf.UnsafeRequest, match=problem):
        bf.parse_dates(text, TODAY)


def test_dates_are_sorted_and_the_window_edge_is_allowed():
    assert bf.parse_dates("2026-10-28,2026-06-06", TODAY) == [date(2026, 6, 6), date(2026, 10, 28)]


@pytest.mark.parametrize("text", ["flipped=1,flipped=1", "flipped", "nope=1", "flipped=-1", "flipped=x"])
def test_a_malformed_pin_is_refused(text):
    with pytest.raises(bf.UnsafeRequest):
        bf.parse_expect_plan(text)


def test_apply_needs_every_key_pinned(capsys):
    full = ",".join(f"{k}=0" for k in bf.PLAN_KEYS)
    args = bf.parse_args(["--sport", "mma", "--dates", "2026-06-06", "--apply", "--expect-plan", full])
    assert args.apply and args.expect_plan == dict.fromkeys(bf.PLAN_KEYS, 0)
    for argv in (
        ["--sport", "mma", "--dates", "2026-06-06", "--apply"],
        ["--sport", "mma", "--dates", "2026-06-06", "--apply", "--expect-plan", "flipped=0"],
    ):
        with pytest.raises(SystemExit):
            bf.parse_args(argv)
    assert "missing" in capsys.readouterr().err


def test_only_mma_is_allowed():
    with pytest.raises(SystemExit):
        bf.parse_args(["--sport", "soccer", "--dates", "2026-06-06"])


# ── --expect-recs ─────────────────────────────────────────────────────

REVIEWED = SCRIPTS_DIR / "data" / "mma_stranded_recs_20261101.csv"


def test_the_reviewed_file_reads_as_the_rec_pin():
    want = bf.parse_expect_recs(REVIEWED)
    assert len(want) == 21
    statuses = [w["status"] for w in want.values()]
    assert statuses.count("won") == 6 and statuses.count("lost") == 14 and statuses.count("void") == 1
    assert sum(w["profit_loss"] for w in want.values()) == Decimal("-100.80")
    assert want["d9da951f-c6a9-47ce-bb26-e04112e0f443"] == {
        "status": "void",
        "profit_loss": Decimal("0.00"),
        "actual_result": "void: match cancelled",
    }


@pytest.mark.parametrize(
    "body, problem",
    [
        ("rec_id,status\nr1,won\n", "want columns"),
        ("rec_id,status,profit_loss,actual_result\nr1,won,1.00,won\nr1,lost,-1.00,lost\n", "twice"),
        ("rec_id,status,profit_loss,actual_result\nr1,won,x,won\n", "profit_loss"),
        ("rec_id,status,profit_loss,actual_result\n", "no rec"),
    ],
)
def test_a_malformed_rec_pin_is_refused(tmp_path, body, problem):
    f = tmp_path / "pin.csv"
    f.write_text(body)
    with pytest.raises(bf.UnsafeRequest, match=problem):
        bf.parse_expect_recs(f)
    with pytest.raises(bf.UnsafeRequest, match="No such file"):
        bf.parse_expect_recs(tmp_path / "missing.csv")


def _entry(rid, **after):
    return {"kind": "rec", "id": rid, "change": "recs_settled", "before": _rec(), "after": _rec(**after)}


WON = {"status": "won", "profit_loss": Decimal("20.00"), "actual_result": "won"}


@pytest.mark.parametrize(
    "entries, bad",
    [
        ([_entry("r1", **WON)], []),
        ([_entry("r1", **{**WON, "profit_loss": Decimal("20")})], []),  # the same amount
        ([_entry("r1", **{**WON, "status": "lost"})], ["r1 lost/20.00/'won', expected won/20.00/'won'"]),
        ([_entry("r1", **{**WON, "profit_loss": Decimal("19.99")})], ["r1 won/19.99/'won', expected won/20.00/'won'"]),
        ([_entry("r1", **{**WON, "profit_loss": None})], ["r1 won/None/'won', expected won/20.00/'won'"]),
        ([], ["r1 not changed"]),
        ([_entry("r1", **WON), _entry("r2", **WON)], ["r2 changed but not expected"]),
    ],
)
def test_rec_outcomes_must_be_exactly_the_pinned_ones(entries, bad):
    match_entry = {"kind": "match", "id": "m1", "change": "flipped", "before": _match(), "after": _match(**FINAL)}
    assert bf.rec_outcome_mismatch([match_entry, *entries], {"r1": WON}) == bad


def test_the_rec_pin_is_read_by_the_cli():
    args = bf.parse_args(["--sport", "mma", "--dates", "2026-06-06", "--expect-recs", str(REVIEWED)])
    assert len(args.expect_recs) == 21


@pytest.mark.parametrize("key", sorted(bf.fu.IDENTITY_STATS))
def test_identity_refused_counts_every_counter_the_sweep_exits_3_on(monkeypatch, key):
    # The no-winner counters (refused, unconfirmed) were added after this count was written: a key it skipped
    # would let a pinned apply plan identity_refused=0 while the results path refused a write.
    for k in list(bf.fu.IDENTITY_STATS):
        monkeypatch.setitem(bf.fu.IDENTITY_STATS, k, 0)
    monkeypatch.setitem(bf.fu.IDENTITY_STATS, key, 1)
    assert bf.identity_refused() == 1
    assert "no_winner_unconfirmed" in bf.fu.IDENTITY_STATS and "no_winner_refused" in bf.fu.IDENTITY_STATS
