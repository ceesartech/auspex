"""Unit tests for scripts/repair_stranded_recs.py — the twin-settlement
repair for recs stranded on duplicate ("orphan") fixture rows.

Pure unit tests: a FakeCursor stands in for psycopg2 and actually applies
the twin query's filters (orientation, status, scores, window) to a small
in-memory `matches` table, so the swapped-orientation test really exercises
the WHERE clause's parameters rather than a hand-picked return value.
Expected P&L is always DERIVED from grading_outcomes, never hardcoded.
"""

from __future__ import annotations

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


rs = _load("repair_stranded_recs", REPO / "scripts" / "repair_stranded_recs.py")
go = _load("grading_outcomes", REPO / "scripts" / "grading_outcomes.py")


HOME = "team-home-uuid"
AWAY = "team-away-uuid"
KICKOFF = datetime(2026, 8, 20, 18, 0, tzinfo=timezone.utc)
ORPHAN = "match-orphan-uuid"
TWIN = "match-twin-uuid"

STAKE = Decimal("50.00")
ODDS = Decimal("2.5000")


def make_rec(rec_id="rec-1", bet_type="1x2", selection="home", sport="tennis", match_id=ORPHAN):
    return {
        "rec_id": rec_id,
        "match_id": match_id,
        "bet_type": bet_type,
        "selection": selection,
        "recommended_stake": STAKE,
        "odds_at_recommendation": ODDS,
        "home_team_id": HOME,
        "away_team_id": AWAY,
        "match_date": KICKOFF,
        "sport": sport,
    }


def make_match(match_id, home_team_id=HOME, away_team_id=AWAY, home_score=2, away_score=1, offset_hours=1.0):
    return {
        "id": match_id,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "status": "finished",
        "home_score": home_score,
        "away_score": away_score,
        "match_date": KICKOFF + timedelta(hours=offset_hours),
    }


class FakeCursor:
    """Routes by query shape; applies the twin query's real filters.

    `rowcount` mirrors psycopg2: 1 for an UPDATE that matched. Set
    `update_rowcount = 0` to simulate the `AND status = 'pending'` guard
    firing because something settled the row concurrently.
    """

    def __init__(self, recs, matches, update_rowcount=1):
        self.recs = recs
        self.matches = matches
        self.updates: list[tuple[str, tuple]] = []
        self._rows: list[dict] = []
        self.update_rowcount = update_rowcount
        self.rowcount = -1

    def execute(self, sql, params=None):
        flat = " ".join(sql.split())
        if flat.startswith("UPDATE betting_recommendations"):
            self.updates.append((flat, params))
            self._rows = []
            self.rowcount = self.update_rowcount
            return
        if "FROM betting_recommendations" in flat:
            sport = params[1]
            self._rows = [dict(r) for r in self.recs if r["sport"] == sport]
            return
        if "FROM matches t" in flat:
            home, away, orphan_id, mdate, hours = params[0], params[1], params[2], params[3], int(params[4])
            window = timedelta(hours=hours)
            self._rows = [
                {
                    "twin_id": m["id"],
                    "home_team_id": m["home_team_id"],
                    "away_team_id": m["away_team_id"],
                    "home_score": m["home_score"],
                    "away_score": m["away_score"],
                    "match_date": m["match_date"],
                }
                for m in self.matches
                if m["home_team_id"] == home
                and m["away_team_id"] == away
                and m["id"] != orphan_id
                and m["status"] == "finished"
                and m["home_score"] is not None
                and m["away_score"] is not None
                and abs(m["match_date"] - mdate) <= window
            ]
            return
        raise AssertionError(f"Unexpected query: {flat[:120]}")

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, recs, matches, update_rowcount=1):
        self.cur = FakeCursor(recs, matches, update_rowcount=update_rowcount)
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, cursor_factory=None):
        return self.cur

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


_CSV_SEQ = itertools.count()


def _run(conn, tmp_path, **kw):
    """The audit CSV is write-once (exclusive create), so every run inside
    a test gets its own path — mirroring the timestamped default name."""
    kw.setdefault("sports", ["tennis"])
    kw.setdefault("apply_changes", True)
    kw.setdefault("out_csv", str(tmp_path / "out" / f"repaired-{next(_CSV_SEQ)}.csv"))
    return rs.run(conn, **kw)


def _csv_rows(path: Path) -> list[dict]:
    import csv

    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ── settlement ────────────────────────────────────────────────────────


def test_clean_twin_settles_won_with_derived_pl(tmp_path):
    conn = FakeConn([make_rec()], [make_match(TWIN, home_score=2, away_score=1)])
    result = _run(conn, tmp_path)

    outcome = go.grade_rec_selection("1x2", "home", 2, 1)
    exp_status, exp_pl = go.rec_status_and_pl(outcome, STAKE, ODDS)
    assert exp_status == "won"

    assert result["applied"] == 1
    assert conn.commits == 1
    assert len(conn.cur.updates) == 1
    _, params = conn.cur.updates[0]
    assert params[0] == "won"
    assert params[1] == outcome
    assert params[2] == exp_pl
    # The rec is REPOINTED onto the finished twin — without that it stays
    # on a status='scheduled' row and accuracy.py's ROI query never sees it.
    assert params[3] == TWIN
    assert params[4] == "rec-1"

    counts = result["per_sport"]["tennis"]
    assert counts["settled_won"] == 1
    assert counts["net_pl"] == exp_pl
    assert counts["total_stake"] == STAKE


def test_losing_rec_settles_lost_with_negative_stake(tmp_path):
    conn = FakeConn([make_rec(selection="away")], [make_match(TWIN, home_score=2, away_score=1)])
    result = _run(conn, tmp_path)

    outcome = go.grade_rec_selection("1x2", "away", 2, 1)
    exp_status, exp_pl = go.rec_status_and_pl(outcome, STAKE, ODDS)
    assert exp_status == "lost"
    assert exp_pl == -STAKE

    _, params = conn.cur.updates[0]
    assert params[0] == "lost"
    assert params[2] == exp_pl
    assert result["per_sport"]["tennis"]["settled_lost"] == 1


def test_asian_handicap_quarter_line_settles_half_won(tmp_path):
    # home +0.25 on a 1-1 draw: half the stake pushes, half wins.
    rec = make_rec(bet_type="asian_handicap", selection="home_0.25", sport="soccer")
    conn = FakeConn([rec], [make_match(TWIN, home_score=1, away_score=1)])
    result = _run(conn, tmp_path, sports=["soccer"])

    outcome = go.grade_rec_selection("asian_handicap", "home_0.25", 1, 1)
    assert outcome == go.REC_HALF_WON
    exp_status, exp_pl = go.rec_status_and_pl(outcome, STAKE, ODDS)

    _, params = conn.cur.updates[0]
    assert params[0] == exp_status == "won"
    assert params[1] == go.REC_HALF_WON
    assert params[2] == exp_pl
    assert params[3] == TWIN
    # half-stake money, derived not hardcoded
    assert exp_pl == STAKE / 2 * (ODDS - 1)
    assert result["per_sport"]["soccer"]["settled_won"] == 1


# ── refusals ──────────────────────────────────────────────────────────


def test_two_twins_with_different_scores_are_ambiguous_and_skipped(tmp_path):
    matches = [
        make_match("twin-a", home_score=2, away_score=1, offset_hours=1),
        make_match("twin-b", home_score=0, away_score=3, offset_hours=5),
    ]
    conn = FakeConn([make_rec()], matches)
    result = _run(conn, tmp_path)

    assert conn.cur.updates == []
    counts = result["per_sport"]["tennis"]
    assert counts["ambiguous_skipped"] == 1
    assert counts["settled_won"] == counts["settled_lost"] == 0
    assert result["affected"] == 0


def test_tied_delta_twins_are_ambiguous_even_with_equal_scores(tmp_path):
    matches = [
        make_match("twin-a", offset_hours=2),
        make_match("twin-b", offset_hours=-2),
    ]
    conn = FakeConn([make_rec()], matches)
    result = _run(conn, tmp_path)
    assert conn.cur.updates == []
    assert result["per_sport"]["tennis"]["ambiguous_skipped"] == 1


def test_swapped_orientation_twin_is_not_matched(tmp_path):
    swapped = make_match(TWIN, home_team_id=AWAY, away_team_id=HOME)
    conn = FakeConn([make_rec()], [swapped])
    result = _run(conn, tmp_path)

    assert conn.cur.updates == []
    counts = result["per_sport"]["tennis"]
    assert counts["no_twin"] == 1
    assert counts["settled_won"] == counts["settled_lost"] == 0


def test_verify_orientation_drops_a_swapped_row_defensively():
    """Even if the WHERE clause were loosened, orientation is re-checked."""
    swapped = {"twin_id": TWIN, "home_team_id": AWAY, "away_team_id": HOME}
    assert rs.verify_orientation([swapped], ORPHAN, HOME, AWAY) == []


def test_ungradable_bet_type_is_skipped_not_guessed(tmp_path):
    rec = make_rec(bet_type="team_total", selection="home_over_1.5")
    conn = FakeConn([rec], [make_match(TWIN)])
    result = _run(conn, tmp_path)

    assert go.grade_rec_selection("team_total", "home_over_1.5", 2, 1) is None
    assert conn.cur.updates == []
    assert result["per_sport"]["tennis"]["ungradable_skipped"] == 1


# ── no-twin voiding ───────────────────────────────────────────────────


def test_no_twin_rows_are_left_alone_by_default(tmp_path):
    conn = FakeConn([make_rec()], [])
    result = _run(conn, tmp_path)

    assert conn.cur.updates == []
    assert result["per_sport"]["tennis"]["no_twin"] == 1
    assert result["per_sport"]["tennis"]["no_twin_voided"] == 0
    assert result["affected"] == 0


def test_no_twin_rows_are_voided_with_the_flag(tmp_path):
    conn = FakeConn([make_rec()], [])
    result = _run(conn, tmp_path, void_no_twin=True)

    assert len(conn.cur.updates) == 1
    sql, params = conn.cur.updates[0]
    assert "SET status = 'void'" in sql
    assert params[0] == rs.VOID_ACTUAL_RESULT
    assert params[1] == "rec-1"
    assert result["per_sport"]["tennis"]["no_twin_voided"] == 1


def test_default_sports_exclude_nba_and_nhl():
    """nba/nhl results are recoverable by a backfill — never void them here."""
    assert "nba" not in rs.DEFAULT_SPORTS
    assert "nhl" not in rs.DEFAULT_SPORTS
    assert set(rs.DEFAULT_SPORTS) == {"soccer", "tennis", "mma"}
    assert rs.parse_args([]).sports is None


# ── dry-run, CSV, guards ──────────────────────────────────────────────


def test_dry_run_writes_csv_but_performs_no_updates(tmp_path):
    conn = FakeConn([make_rec()], [make_match(TWIN, home_score=2, away_score=1)])
    out = tmp_path / "nested" / "repaired.csv"
    result = rs.run(conn, sports=["tennis"], apply_changes=False, out_csv=str(out))

    assert conn.cur.updates == []
    assert conn.commits == 0
    assert out.exists()

    rows = _csv_rows(out)
    assert len(rows) == 1
    row = rows[0]
    assert row["rec_id"] == "rec-1"
    assert row["match_id"] == ORPHAN
    assert row["twin_id"] == TWIN
    assert row["sport"] == "tennis"
    assert row["bet_type"] == "1x2"
    assert row["selection"] == "home"
    assert row["status"] == "won"
    assert row["outcome"] == go.REC_WON
    assert Decimal(row["profit_loss"]) == go.rec_status_and_pl(go.REC_WON, STAKE, ODDS)[1]
    assert Decimal(row["stake"]) == STAKE
    assert Decimal(row["odds"]) == ODDS
    assert result["applied"] == 0

    # The row must be re-gradable OFFLINE: the scores the settlement was
    # computed from are in the file, not only in the live DB.
    assert row["new_match_id"] == TWIN
    assert int(row["twin_home_score"]) == 2
    assert int(row["twin_away_score"]) == 1
    assert row["twin_match_date"]
    assert int(row["twin_count"]) == 1
    assert int(row["duplicate_of_group"]) == 0
    replayed = go.grade_rec_selection(
        row["bet_type"],
        row["selection"],
        int(row["twin_home_score"]),
        int(row["twin_away_score"]),
    )
    assert replayed == row["outcome"]


def test_csv_columns_are_stable_and_written_even_when_empty(tmp_path):
    conn = FakeConn([], [])
    out = tmp_path / "repaired.csv"
    rs.run(conn, sports=["tennis"], apply_changes=False, out_csv=str(out))
    header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == rs.CSV_COLUMNS


def test_max_rows_guard_aborts_and_yes_overrides(tmp_path):
    recs = [make_rec(rec_id=f"rec-{i}") for i in range(5)]
    matches = [make_match(TWIN, home_score=2, away_score=1)]

    conn = FakeConn(recs, matches)
    with pytest.raises(rs.MaxRowsExceeded):
        _run(conn, tmp_path, max_rows=3)
    assert conn.cur.updates == []

    conn2 = FakeConn(recs, matches)
    result = _run(conn2, tmp_path, max_rows=3, yes=True)
    assert result["applied"] == 5


def test_update_statement_carries_the_pending_status_guard(tmp_path):
    conn = FakeConn([make_rec()], [make_match(TWIN)])
    _run(conn, tmp_path)
    sql, _ = conn.cur.updates[0]
    assert "WHERE id = %s AND status = 'pending'" in sql

    conn2 = FakeConn([make_rec()], [])
    _run(conn2, tmp_path, void_no_twin=True)
    void_sql, _ = conn2.cur.updates[0]
    assert "WHERE id = %s AND status = 'pending'" in void_sql


def test_main_returns_2_without_a_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert rs.main([]) == 2


def test_default_out_csv_shape():
    assert rs.default_out_csv().startswith("backups/repaired_rec_ids_")
    assert rs.default_out_csv().endswith(".csv")


def test_settle_repoints_the_rec_onto_the_finished_twin(tmp_path):
    """Settling without repointing leaves the rec on a status='scheduled'
    row, where accuracy.py's recs P&L query (`WHERE m.status = 'finished'`)
    can never see it — the readout this repair exists to produce."""
    conn = FakeConn([make_rec()], [make_match(TWIN)])
    _run(conn, tmp_path)

    sql, params = conn.cur.updates[0]
    assert "match_id = %s" in sql
    assert params[3] == TWIN


def test_void_does_not_repoint(tmp_path):
    conn = FakeConn([make_rec()], [])
    _run(conn, tmp_path, void_no_twin=True)
    sql, _ = conn.cur.updates[0]
    assert "match_id" not in sql


# ── concurrent settlement (rowcount 0) ────────────────────────────────


def test_row_settled_concurrently_is_not_counted_as_applied(tmp_path):
    """The `AND status = 'pending'` guard firing means the UPDATE hit 0
    rows. Counting it as applied would put a settlement this tool never
    made into the P&L summary and the reversal CSV."""
    conn = FakeConn([make_rec()], [make_match(TWIN)], update_rowcount=0)
    result = _run(conn, tmp_path)

    assert len(conn.cur.updates) == 1  # attempted
    assert result["applied"] == 0  # but not counted
    assert result["skipped_not_pending"] == 1

    counts = result["per_sport"]["tennis"]
    assert counts["skipped_not_pending"] == 1
    assert counts["settled_won"] == 0
    assert counts["net_pl"] == Decimal("0")
    assert counts["total_stake"] == Decimal("0")

    not_applied = Path(result["not_applied_csv_path"])
    assert not_applied.exists()
    assert [r["rec_id"] for r in _csv_rows(not_applied)] == ["rec-1"]


def test_concurrently_settled_void_is_backed_out_too(tmp_path):
    conn = FakeConn([make_rec()], [], update_rowcount=0)
    result = _run(conn, tmp_path, void_no_twin=True)

    assert result["applied"] == 0
    assert result["per_sport"]["tennis"]["no_twin_voided"] == 0
    assert result["per_sport"]["tennis"]["skipped_not_pending"] == 1


# ── fan-in / collapse factor ──────────────────────────────────────────


def test_duplicate_recs_on_sibling_orphans_are_flagged_not_double_counted(tmp_path):
    """Two orphan rows of ONE real fixture each carry their own rec. Both
    settle correctly, but the aggregate counts one real bet twice — the
    operator must see that before quoting an ROI n."""
    orphan_b = "match-orphan-2"
    recs = [make_rec(rec_id="rec-1"), make_rec(rec_id="rec-2", match_id=orphan_b)]
    conn = FakeConn(recs, [make_match(TWIN)])
    result = _run(conn, tmp_path)

    counts = result["per_sport"]["tennis"]
    assert counts["settled_won"] == 2
    assert counts["distinct_twins"] == 1
    assert counts["duplicate_settles"] == 1

    rows = _csv_rows(Path(result["csv_path"]))
    assert {r["twin_id"] for r in rows} == {TWIN}
    assert sorted(int(r["duplicate_of_group"]) for r in rows) == [0, 1]


def test_distinct_fixtures_are_not_reported_as_duplicates(tmp_path):
    other = "team-other-uuid"
    rec_b = make_rec(rec_id="rec-2", match_id="match-orphan-2")
    rec_b["home_team_id"] = other
    match_b = make_match("twin-b", home_team_id=other)
    conn = FakeConn([make_rec(), rec_b], [make_match(TWIN), match_b])
    result = _run(conn, tmp_path)

    counts = result["per_sport"]["tennis"]
    assert counts["settled_won"] == 2
    assert counts["distinct_twins"] == 2
    assert counts["duplicate_settles"] == 0


# ── enforced refusals ─────────────────────────────────────────────────


def test_void_no_twin_is_refused_for_backfill_recoverable_sports(tmp_path):
    """The nba/nhl rule was documentation-only: `--sport nhl --void-no-twin`
    would have permanently voided 18 Finals recs a backfill can settle."""
    assert rs.RECOVERABLE_BY_BACKFILL == frozenset({"nba", "nhl"})
    conn = FakeConn([make_rec(sport="nhl")], [])
    with pytest.raises(rs.UnsafeRequest, match="backfill"):
        _run(conn, tmp_path, sports=["nhl"], void_no_twin=True)
    assert conn.cur.updates == []


def test_nba_nhl_may_still_be_inspected_without_the_void_flag(tmp_path):
    conn = FakeConn([make_rec(sport="nhl")], [])
    result = _run(conn, tmp_path, sports=["nhl"])
    assert conn.cur.updates == []
    assert result["per_sport"]["nhl"]["no_twin"] == 1


def test_main_returns_2_when_void_no_twin_targets_nhl(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused/db")
    assert rs.main(["--sport", "nhl", "--void-no-twin", "--apply"]) == 2


def test_unknown_sport_is_refused(tmp_path):
    conn = FakeConn([], [])
    with pytest.raises(rs.UnsafeRequest, match="Unknown sport"):
        _run(conn, tmp_path, sports=["tenis"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stale_after_hours": -24},  # NOW() - interval '-24 hours' = NOW() + 24h
        {"stale_after_hours": 0},
        {"twin_window_hours": 0},
        {"twin_window_hours": 24 * 365},
    ],
)
def test_interval_knobs_are_range_checked(tmp_path, kwargs):
    """Both knobs feed SQL interval arithmetic. A negative stale window
    pulls FUTURE, in-play fixtures into the candidate set; an unbounded
    twin window reaches a rematch and settles from the wrong score."""
    conn = FakeConn([make_rec()], [make_match(TWIN)])
    with pytest.raises(rs.UnsafeRequest):
        _run(conn, tmp_path, **kwargs)
    assert conn.cur.updates == []


# ── audit-file durability ─────────────────────────────────────────────


def test_csv_is_write_once_and_never_clobbers_a_previous_audit(tmp_path):
    """A verification dry-run after the apply used to rewrite the audit
    file as a header-only CSV, destroying the only reversal record."""
    out = tmp_path / "repaired.csv"
    conn = FakeConn([make_rec()], [make_match(TWIN)])
    rs.run(conn, sports=["tennis"], apply_changes=True, out_csv=str(out))
    before = out.read_text(encoding="utf-8")
    assert len(_csv_rows(out)) == 1

    conn2 = FakeConn([], [])
    with pytest.raises(FileExistsError):
        rs.run(conn2, sports=["tennis"], apply_changes=False, out_csv=str(out))
    assert out.read_text(encoding="utf-8") == before


def test_default_out_csv_is_unique_per_run_and_scoped():
    a = rs.default_out_csv(["tennis"])
    b = rs.default_out_csv(["soccer", "tennis"])
    assert "tennis" in a and a != b
    assert a.startswith("backups/repaired_rec_ids_") and a.endswith(".csv")
    # A same-day second run must not resolve to the same path as the first.
    assert "T" in a and a.endswith(".csv")
