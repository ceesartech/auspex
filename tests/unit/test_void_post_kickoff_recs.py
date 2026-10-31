"""Unit tests for scripts/void_post_kickoff_recs.py — voiding settled recs
that were written at or after kickoff.

A FakeCursor stands in for psycopg2 and APPLIES the candidate query's
filters (created_at >= kickoff, settled status, sport) and the UPDATE's
guards (match_id, old status / profit_loss / actual_result, created_at
re-check) to a small in-memory table, so a guard test exercises the guard's
parameters rather than a hand-picked rowcount. The fixture rows are shaped
on the prod rows found read-only on 2026-10-30.
"""

from __future__ import annotations

import csv
import importlib.util
import itertools
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


vp = _load("void_post_kickoff_recs", REPO / "scripts" / "void_post_kickoff_recs.py")

UTC = timezone.utc
# Independiente v Platense: ESPN moved the kickoff from 08-09 18:00 to
# 08-09 00:30; recs were written on the stale orphan at 17:45.
KICKOFF_SOCCER = datetime(2026, 8, 9, 0, 30, tzinfo=UTC)
# Neil Magny v Ramiz Brahimaj: early prelims from 21:30; rec at 22:45.
KICKOFF_MMA = datetime(2026, 8, 15, 21, 30, tzinfo=UTC)
SETTLED_AT = datetime(2026, 9, 4, 17, 36, 11, tzinfo=UTC)

MATCHES = {
    "m-soccer": {"match_date": KICKOFF_SOCCER, "status": "finished", "sport": "soccer"},
    "m-mma": {"match_date": KICKOFF_MMA, "status": "finished", "sport": "mma"},
}


def make_rec(
    rec_id,
    match_id="m-soccer",
    status="won",
    profit_loss="59.02",
    actual_result=None,
    created_offset=timedelta(hours=17, minutes=15),
    stake="50.00",
    bet_type="1x2",
    selection="away",
):
    kickoff = MATCHES[match_id]["match_date"]
    return {
        "id": rec_id,
        "match_id": match_id,
        "status": status,
        "profit_loss": None if profit_loss is None else Decimal(profit_loss),
        "actual_result": actual_result if actual_result is not None else status,
        "created_at": kickoff + created_offset,
        "settled_at": SETTLED_AT,
        "bet_type": bet_type,
        "selection": selection,
        "recommended_stake": Decimal(stake),
        "odds_at_recommendation": Decimal("2.1800"),
    }


def _same(a, b) -> bool:
    """IS NOT DISTINCT FROM."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, Decimal) or isinstance(b, Decimal):
        return Decimal(str(a)) == Decimal(str(b))
    return a == b


class FakeCursor:
    def __init__(self, recs, matches, before_update=None):
        self.recs = {r["id"]: r for r in recs}
        self.matches = matches
        self.selects: list[tuple[str, tuple]] = []
        self.updates: list[dict] = []
        self.before_update = before_update
        self._rows: list[dict] = []
        self.rowcount = -1

    def execute(self, sql, params=None):
        flat = " ".join(sql.split())
        if flat.startswith("SELECT br.id::text AS rec_id"):
            self.selects.append((flat, params))
            sports = params[0]
            rows = []
            for r in self.recs.values():
                m = self.matches[r["match_id"]]
                if r["created_at"] < m["match_date"]:
                    continue
                if r["status"] not in ("won", "lost", "void"):
                    continue
                if m["sport"] not in sports:
                    continue
                rows.append(
                    {
                        "rec_id": r["id"],
                        "sport": m["sport"],
                        "match_id": r["match_id"],
                        "match_status": m["status"],
                        "match_date": m["match_date"],
                        "created_at": r["created_at"],
                        "bet_type": r["bet_type"],
                        "selection": r["selection"],
                        "recommended_stake": r["recommended_stake"],
                        "odds_at_recommendation": r["odds_at_recommendation"],
                        "status": r["status"],
                        "actual_result": r["actual_result"],
                        "profit_loss": r["profit_loss"],
                        "settled_at": r["settled_at"],
                    }
                )
            self._rows = sorted(rows, key=lambda x: (x["sport"], x["created_at"], x["rec_id"]))
            return
        if flat.startswith("UPDATE betting_recommendations"):
            if self.before_update:
                self.before_update(self, params)
            self.updates.append(dict(params))
            r = self.recs.get(params["rec_id"])
            ok = (
                r is not None
                and r["match_id"] == params["match_id"]
                and r["status"] == params["old_status"]
                and _same(r["profit_loss"], params["old_pl"])
                and _same(r["actual_result"], params["old_result"])
                and r["created_at"] >= self.matches[params["match_id"]]["match_date"]
            )
            if ok:
                r.update(status="void", profit_loss=Decimal("0"), actual_result=params["new_result"])
            self.rowcount = 1 if ok else 0
            return
        raise AssertionError(f"Unexpected query: {flat[:120]}")

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, recs, matches=MATCHES, before_update=None):
        self.cur = FakeCursor(recs, matches, before_update=before_update)
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, cursor_factory=None):
        return self.cur

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


_SEQ = itertools.count()


def _run(conn, tmp_path, **kw):
    kw.setdefault("sports", ["all"])
    kw.setdefault("apply_changes", True)
    kw.setdefault("out_csv", str(tmp_path / f"void-{next(_SEQ)}.csv"))
    return vp.run(conn, **kw)


def _csv_rows(path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def prod_shaped_recs():
    """Two soccer recs on one moved fixture (+59.02 / -72.44), a soccer push
    already void at 0.00, one MMA loss, plus rows that must NOT be touched:
    a settled pre-kickoff bet and a pending post-kickoff rec."""
    return [
        make_rec("s-won", status="won", profit_loss="59.02"),
        make_rec("s-lost", status="lost", profit_loss="-72.44", bet_type="over_under", selection="over_2.5"),
        make_rec("s-push", status="void", profit_loss="0.00", actual_result="push", bet_type="asian_handicap"),
        make_rec("m-lost", match_id="m-mma", status="lost", profit_loss="-40.21", created_offset=timedelta(minutes=75)),
        make_rec("s-pre", status="won", profit_loss="25.00", created_offset=timedelta(hours=-3)),
        make_rec("s-pending", status="pending", profit_loss=None, actual_result=""),
    ]


# ── the correction ────────────────────────────────────────────────────


def test_apply_voids_post_kickoff_settled_recs_and_leaves_the_rest(tmp_path):
    conn = FakeConn(prod_shaped_recs())
    result = _run(conn, tmp_path)

    recs = conn.cur.recs
    assert recs["s-won"]["status"] == "void" and recs["s-won"]["profit_loss"] == Decimal("0")
    assert recs["s-won"]["actual_result"] == "void: written after kickoff (was won 59.02)"
    assert recs["s-lost"]["actual_result"] == "void: written after kickoff (was lost -72.44)"
    assert recs["m-lost"]["actual_result"] == "void: written after kickoff (was lost -40.21)"
    # already right: a push, void at 0.00 — counted, not rewritten
    assert recs["s-push"]["actual_result"] == "push"
    # never touched: a pre-kickoff bet and an unsettled rec
    assert recs["s-pre"]["status"] == "won" and recs["s-pre"]["profit_loss"] == Decimal("25.00")
    assert recs["s-pending"]["status"] == "pending"

    assert {u["rec_id"] for u in conn.cur.updates} == {"s-won", "s-lost", "m-lost"}
    assert conn.commits == 1  # ONE transaction
    assert result["to_void"] == 3 and result["already_void"] == 1 and result["written"] == 3

    soccer, mma = result["per_sport"]["soccer"], result["per_sport"]["mma"]
    assert (soccer["void_won"], soccer["void_lost"], soccer["already_void"]) == (1, 1, 1)
    assert soccer["pl_removed"] == Decimal("59.02") + Decimal("-72.44")
    assert soccer["won_pl"] == Decimal("59.02") and soccer["lost_pl"] == Decimal("-72.44")
    assert soccer["stake"] == Decimal("100.00")
    assert mma["void_lost"] == 1 and mma["pl_removed"] == Decimal("-40.21")


def test_update_is_guarded_on_every_observed_value(tmp_path):
    conn = FakeConn([make_rec("s-won", status="won", profit_loss="59.02")])
    _run(conn, tmp_path)
    (params,) = conn.cur.updates
    assert params == {
        "rec_id": "s-won",
        "match_id": "m-soccer",
        "old_status": "won",
        "old_pl": Decimal("59.02"),
        "old_result": "won",
        "new_result": "void: written after kickoff (was won 59.02)",
    }
    flat = " ".join(vp.UPDATE_SQL.split())
    for guard in (
        "WHERE id = %(rec_id)s::uuid",
        "AND match_id = %(match_id)s::uuid",
        "AND status = %(old_status)s",
        "AND profit_loss IS NOT DISTINCT FROM %(old_pl)s",
        "AND actual_result IS NOT DISTINCT FROM %(old_result)s",
        "AND created_at >= (SELECT m.match_date FROM matches m WHERE m.id = %(match_id)s::uuid)",
    ):
        assert guard in flat
    assert "SET status = 'void', profit_loss = 0" in flat


def test_candidate_query_filters_on_kickoff_settled_status_and_sport(tmp_path):
    conn = FakeConn([])
    _run(conn, tmp_path, sports=["soccer", "mma"], apply_changes=False)
    ((flat, params),) = conn.cur.selects
    assert "WHERE br.created_at >= m.match_date" in flat
    assert "AND br.status IN ('won', 'lost', 'void')" in flat
    assert "AND l.sport = ANY(%s)" in flat
    assert params == (["soccer", "mma"],)


def test_a_row_that_moved_between_plan_and_write_rolls_everything_back(tmp_path):
    recs = [make_rec("s-won"), make_rec("s-lost", status="lost", profit_loss="-72.44")]

    def regrade_second(cur, params):
        # something re-settled s-lost after the plan read it
        if params["rec_id"] == "s-lost":
            cur.recs["s-lost"]["profit_loss"] = Decimal("-36.22")

    conn = FakeConn(recs, before_update=regrade_second)
    out = tmp_path / "audit.csv"
    with pytest.raises(RuntimeError, match="changed between plan and write"):
        vp.run(conn, sports=["soccer"], apply_changes=True, out_csv=str(out))
    assert conn.commits == 0 and conn.rollbacks >= 2  # snapshot close + the abort
    assert out.exists()  # the audit trail landed before the first UPDATE
    assert len(_csv_rows(out)) == 2


def test_a_rec_repointed_since_the_plan_is_refused(tmp_path):
    def repoint(cur, params):
        cur.recs[params["rec_id"]]["match_id"] = "m-mma"

    conn = FakeConn([make_rec("s-won")], before_update=repoint)
    with pytest.raises(RuntimeError):
        _run(conn, tmp_path)
    assert conn.commits == 0


# ── dry run, CSV, re-run ──────────────────────────────────────────────


def test_dry_run_writes_the_audit_csv_and_nothing_else(tmp_path):
    conn = FakeConn(prod_shaped_recs())
    out = tmp_path / "nested" / "dry.csv"
    result = vp.run(conn, sports=["all"], apply_changes=False, out_csv=str(out))

    assert conn.cur.updates == [] and conn.commits == 0
    assert result["written"] == 0 and result["to_void"] == 3
    rows = {r["rec_id"]: r for r in _csv_rows(out)}
    assert set(rows) == {"s-won", "s-lost", "s-push", "m-lost"}
    won = rows["s-won"]
    assert won["action"] == vp.ACTION_VOID
    assert (won["old_status"], Decimal(won["old_profit_loss"]), won["old_actual_result"]) == (
        "won",
        Decimal("59.02"),
        "won",
    )
    assert won["old_settled_at"] == str(SETTLED_AT)
    assert (won["new_status"], Decimal(won["new_profit_loss"])) == ("void", Decimal("0"))
    assert won["new_actual_result"] == "void: written after kickoff (was won 59.02)"
    assert float(won["minutes_after_kickoff"]) == 17 * 60 + 15
    assert rows["s-push"]["action"] == vp.ACTION_ALREADY_VOID
    assert rows["s-push"]["new_actual_result"] == "push"


def test_csv_header_is_stable_and_written_when_empty(tmp_path):
    out = tmp_path / "empty.csv"
    result = vp.run(FakeConn([]), sports=["all"], apply_changes=True, out_csv=str(out))
    assert out.read_text(encoding="utf-8").splitlines()[0].split(",") == list(vp.CSV_FIELDS)
    assert result["written"] == 0


def test_csv_is_write_once(tmp_path):
    out = tmp_path / "audit.csv"
    vp.run(FakeConn([make_rec("s-won")]), sports=["all"], apply_changes=False, out_csv=str(out))
    before = out.read_text(encoding="utf-8")
    conn = FakeConn([make_rec("s-won")])
    with pytest.raises(FileExistsError):
        vp.run(conn, sports=["all"], apply_changes=True, out_csv=str(out))
    assert out.read_text(encoding="utf-8") == before
    assert conn.cur.updates == []


def test_rerun_after_apply_writes_nothing(tmp_path):
    conn = FakeConn(prod_shaped_recs())
    _run(conn, tmp_path)
    n = len(conn.cur.updates)
    again = _run(conn, tmp_path)
    assert len(conn.cur.updates) == n
    assert again["to_void"] == 0 and again["already_void"] == 4


def test_sport_scope_is_respected(tmp_path):
    conn = FakeConn(prod_shaped_recs())
    result = _run(conn, tmp_path, sports=["mma"])
    assert {u["rec_id"] for u in conn.cur.updates} == {"m-lost"}
    assert conn.cur.recs["s-won"]["status"] == "won"
    assert set(result["per_sport"]) == {"mma"}


def test_max_rows_guard_aborts_before_the_csv_and_yes_overrides(tmp_path):
    recs = [make_rec(f"r{i}") for i in range(4)]
    out = tmp_path / "guarded.csv"
    conn = FakeConn(recs)
    with pytest.raises(vp.MaxRowsExceeded):
        vp.run(conn, sports=["all"], apply_changes=True, out_csv=str(out), max_rows=3)
    assert conn.cur.updates == [] and not out.exists()

    result = _run(FakeConn(recs), tmp_path, max_rows=3, yes=True)
    assert result["written"] == 4


# ── pure pieces ───────────────────────────────────────────────────────


def test_resolve_sports():
    assert vp.resolve_sports(["all"]) == list(vp.ALLOWED_SPORTS)
    assert vp.resolve_sports(["mma", "soccer"]) == ["soccer", "mma"]
    assert "horse_racing" not in vp.ALLOWED_SPORTS  # race_recommendations is another table
    with pytest.raises(vp.UnsafeRequest, match="Unknown sport"):
        vp.resolve_sports(["socer"])
    with pytest.raises(vp.UnsafeRequest, match="No sport"):
        vp.resolve_sports([])


@pytest.mark.parametrize(
    "status,pl,expected",
    [
        ("won", "31.41", "void"),
        ("lost", "-40.21", "void"),
        ("won", None, "void"),  # trigger-settled, P&L never filled: still a verdict
        ("void", "0.00", "already_void"),
        ("void", "12.50", "void"),  # a void carrying money is not void
        ("void", None, "void"),
    ],
)
def test_classify(status, pl, expected):
    row = {
        "rec_id": "r",
        "status": status,
        "profit_loss": None if pl is None else Decimal(pl),
        "created_at": KICKOFF_SOCCER,
        "match_date": KICKOFF_SOCCER,  # exactly at kickoff counts
    }
    assert vp.classify(row) == expected


def test_classify_refuses_a_pre_kickoff_or_unsettled_row():
    pre = {"rec_id": "r", "status": "won", "profit_loss": Decimal("1"), "match_date": KICKOFF_SOCCER}
    with pytest.raises(ValueError, match="not provably post-kickoff"):
        vp.classify({**pre, "created_at": KICKOFF_SOCCER - timedelta(seconds=1)})
    with pytest.raises(ValueError, match="not provably post-kickoff"):
        vp.classify({**pre, "created_at": None})
    with pytest.raises(ValueError, match="not settled"):
        vp.classify({**pre, "status": "pending", "created_at": KICKOFF_SOCCER})


def test_new_actual_result_label_and_length_guard(monkeypatch):
    assert vp.new_actual_result("won", Decimal("31.4088")) == "void: written after kickoff (was won 31.41)"
    assert vp.new_actual_result("lost", Decimal("-150.01")) == "void: written after kickoff (was lost -150.01)"
    assert vp.new_actual_result("won", None) == "void: written after kickoff (was won null)"
    # the prefix the settlement paths write, so one LIKE finds every row
    assert vp.new_actual_result("won", 1).startswith(vp.POST_KICKOFF_VOID_RESULT)
    monkeypatch.setattr(vp, "ACTUAL_RESULT_MAX_LEN", 20)
    with pytest.raises(ValueError, match="exceeds"):
        vp.new_actual_result("won", 1)


def test_default_out_csv(monkeypatch):
    monkeypatch.delenv("BACKUP_LOCAL_DIR", raising=False)
    path = vp.default_out_csv(list(vp.ALLOWED_SPORTS))
    assert path.startswith("/app/backups/void_post_kickoff_recs_all_") and path.endswith("Z.csv")
    monkeypatch.setenv("BACKUP_LOCAL_DIR", "/mnt/elsewhere")
    assert vp.default_out_csv(["soccer", "mma"]).startswith("/mnt/elsewhere/void_post_kickoff_recs_soccer-mma_")


# ── CLI ───────────────────────────────────────────────────────────────


def test_sport_is_required():
    with pytest.raises(SystemExit):
        vp.parse_args([])
    assert vp.parse_args(["--sport", "all"]).apply is False  # dry run by default


def test_main_returns_2_without_a_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert vp.main(["--sport", "all"]) == 2


def test_main_refuses_an_unknown_sport_before_connecting(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused/db")

    def no_connect(*a, **k):
        raise AssertionError("must not connect")

    monkeypatch.setattr(vp.psycopg2, "connect", no_connect)
    assert vp.main(["--sport", "tenis"]) == 2
