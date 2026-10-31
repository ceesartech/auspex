"""Integration test for scripts/repair_football_data_kickoffs.py and the
promote_raw lookup it unblocks, against a real Postgres with every migration
(the outcome trigger included).

The unit tests (tests/unit/test_repair_football_data_kickoffs.py,
test_promote_raw.py) plan in memory and fake the cursor; this runs the real
statements: the FK catalog read, the lock, the fold facts, the archive, the
odds/match_stats moves, the guarded fold flip (which fires
trg_match_outcome), the guarded copy delete, the guarded shift, --check, and
a full promote_raw run afterwards that must insert nothing.

Auto-marked `integration` by conftest; CI runs it against the service
Postgres:

    pytest -m integration services/data-ingestion/tests/e2e/test_repair_football_data_kickoffs.py
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg2
import pytest
from sqlalchemy import text

SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts"
UTC = timezone.utc


def _utc_dsn(url: str) -> str:
    """The test database URL with the session TimeZone pinned to UTC, prod's and CI's: the tool reads a legacy
    kickoff in the session zone (the UK wall clock promote_raw stored as if UTC), and _seed writes them in UTC. In a
    database left at, say, America/Denver the first check would otherwise see no legacy row at all."""
    return url + ("&" if "?" in url else "?") + "options=-c%20TimeZone%3DUTC"


DSN = _utc_dsn(os.environ["DATABASE_URL"])


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pr = _load("promote_raw")
rfk = _load("repair_football_data_kickoffs")

RAW_DDL = """
    CREATE TABLE raw_football_data (
        "Div" TEXT, "Date" TEXT, "Time" TEXT, "HomeTeam" TEXT, "AwayTeam" TEXT,
        "FTHG" TEXT, "FTAG" TEXT, "FTR" TEXT, "HTHG" TEXT, "HTAG" TEXT, "HTR" TEXT,
        league TEXT, season TEXT, loaded_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE ("Date", "HomeTeam", "AwayTeam")
    )
"""
TABLES = [
    "betting_recommendations",
    "predictions",
    "features_cache",
    "odds",
    "match_stats",
    "matches",
    "teams",
    "leagues",
]


@pytest.fixture
def seeded(engine):
    """An empty soccer world plus a raw_football_data this test owns (it
    refuses to touch one that already holds rows)."""
    with engine.begin() as conn:
        exists = conn.execute(text("SELECT to_regclass('public.raw_football_data')")).scalar_one()
        if exists is not None and conn.execute(text("SELECT count(*) FROM raw_football_data")).scalar_one():
            pytest.skip("raw_football_data holds rows in this database: not a disposable test database")
        if exists is None:
            conn.execute(text(RAW_DDL))
        conn.execute(text("TRUNCATE TABLE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE"))
    yield
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE"))
        if exists is None:
            conn.execute(text("DROP TABLE raw_football_data"))


def _id() -> str:
    return str(uuid.uuid4())


def _seed(engine) -> dict:
    """Four football-data fixtures in CN1 (promote_raw's league name, so its
    upserts land on these rows), each with its raw row:
      merge: the copy 1 h after its ESPN row, same score, with match_stats;
      fold:  the copy 1 h after a stale 'scheduled' row at its corrected
             kickoff; the stale row has pre-kickoff predictions and a void rec;
      held:  the same, but the stale row's prediction carries a pending rec;
      shift: a lone BST row at its legacy kickoff;
      plus a GMT row that is already right."""
    ids = {"league": _id()}
    teams = ["Shandong Taishan", "Yunnan Yukun", "Liaoning Tieren", "Shanghai Port", "Wuhan Three Towns", "Henan"]
    t = {name: _id() for name in teams}
    k_merge = datetime(2026, 7, 10, 11, 35, tzinfo=UTC)
    k_fold = datetime(2026, 5, 29, 11, 35, tzinfo=UTC)
    k_held = datetime(2026, 6, 5, 11, 35, tzinfo=UTC)
    k_shift = datetime(2026, 7, 17, 11, 0, tzinfo=UTC)
    k_gmt = datetime(2025, 12, 6, 8, 0, tzinfo=UTC)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO leagues (id, name, country, sport, external_ids) "
                "VALUES (:id, 'Chinese Super League', 'China', 'soccer', '{\"football_data\": \"CN1\"}')"
            ),
            {"id": ids["league"]},
        )
        for name, tid in t.items():
            conn.execute(
                text(
                    "INSERT INTO teams (id, name, normalized_name, league_id, sport) "
                    "VALUES (:id, :name, :norm, :lid, 'soccer')"
                ),
                {"id": tid, "name": name, "norm": pr.normalize_team_name(name), "lid": ids["league"]},
            )

        def match(key, home, away, when, status, score, metadata=None):
            ids[key] = _id()
            conn.execute(
                text(
                    "INSERT INTO matches (id, league_id, home_team_id, away_team_id, match_date, status, home_score, "
                    "away_score, season, metadata) VALUES (:id, :lid, :h, :a, :d, :s, :hs, :as_, '2026', "
                    "CAST(:md AS jsonb))"
                ),
                {
                    "id": ids[key],
                    "lid": ids["league"],
                    "h": t[home],
                    "a": t[away],
                    "d": when,
                    "s": status,
                    "hs": score[0],
                    "as_": score[1],
                    "md": json.dumps(metadata or {}),
                },
            )

        match(
            "espn",
            "Shandong Taishan",
            "Yunnan Yukun",
            k_merge,
            "finished",
            (4, 3),
            {"result_source": "espn_scoreboard"},
        )
        match("copy_merge", "Shandong Taishan", "Yunnan Yukun", k_merge + timedelta(hours=1), "finished", (4, 3))
        match("stale", "Liaoning Tieren", "Shanghai Port", k_fold, "scheduled", (None, None))
        match("copy_fold", "Liaoning Tieren", "Shanghai Port", k_fold + timedelta(hours=1), "finished", (2, 0))
        match("stale_held", "Wuhan Three Towns", "Henan", k_held, "scheduled", (None, None))
        match("copy_held", "Wuhan Three Towns", "Henan", k_held + timedelta(hours=1), "finished", (1, 1))
        match("lone", "Henan", "Shandong Taishan", k_shift + timedelta(hours=1), "finished", (0, 1))
        match("gmt", "Yunnan Yukun", "Henan", k_gmt, "finished", (3, 3))
        for team in ("Shandong Taishan", "Yunnan Yukun"):
            conn.execute(
                text("INSERT INTO match_stats (match_id, team_id, shots) VALUES (:m, :t, 9)"),
                {"m": ids["copy_merge"], "t": t[team]},
            )

        def prediction(key, match_key, ptype, outcome, rec_status=None):
            ids[key] = _id()
            conn.execute(
                text(
                    "INSERT INTO predictions (id, match_id, model_name, model_version, prediction_type, "
                    "predicted_outcome, confidence, probabilities, created_at) VALUES (:id, :m, 'ensemble_soccer', "
                    "'v1', :pt, :po, 0.6, '{}', :c)"
                ),
                {"id": ids[key], "m": ids[match_key], "pt": ptype, "po": outcome, "c": k_fold - timedelta(days=1)},
            )
            if rec_status:
                conn.execute(
                    text(
                        "INSERT INTO betting_recommendations (id, prediction_id, match_id, bet_type, selection, "
                        "odds_at_recommendation, bookmaker, confidence_rating, expected_value, kelly_stake, "
                        "recommended_stake, status, created_at) "
                        "VALUES (:id, :p, :m, :pt, :po, 2.1, 'Bet365', 'medium', 0.05, 0.01, 10, :st, :c)"
                    ),
                    {
                        "id": _id(),
                        "p": ids[key],
                        "m": ids[match_key],
                        "pt": ptype,
                        "po": outcome,
                        "st": rec_status,
                        "c": k_fold - timedelta(days=1),
                    },
                )

        prediction("p_result", "stale", "match_result", "home", rec_status="void")
        prediction("p_btts", "stale", "btts", "yes")
        prediction("p_dc", "stale", "double_chance", "home_draw")
        prediction("p_held", "stale_held", "match_result", "draw", rec_status="pending")

        for date_, time_, home, away, ft in (
            ("10/07/2026", "12:35", "Shandong Taishan", "Yunnan Yukun", ("4", "3")),
            ("29/05/2026", "12:35", "Liaoning Tieren", "Shanghai Port", ("2", "0")),
            ("05/06/2026", "12:35", "Wuhan Three Towns", "Henan", ("1", "1")),
            ("17/07/2026", "12:00", "Henan", "Shandong Taishan", ("0", "1")),
            ("06/12/2025", "08:00", "Yunnan Yukun", "Henan", ("3", "3")),
        ):
            conn.execute(
                text(
                    'INSERT INTO raw_football_data ("Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", league, '
                    "season) VALUES (:d, :t, :h, :a, :fh, :fa, 'CN1', '2026')"
                ),
                {"d": date_, "t": time_, "h": home, "a": away, "fh": ft[0], "fa": ft[1]},
            )
    ids["k_shift"] = k_shift
    return ids


def _row(engine, match_id):
    with engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT match_date, status, home_score, away_score, metadata->>'result_source' AS rs, "
                "metadata->>'result_adopted_from' AS src FROM matches WHERE id = :id"
            ),
            {"id": match_id},
        ).one_or_none()


def _count(engine, sql, **params):
    with engine.connect() as conn:
        return conn.execute(text(sql), params).scalar_one()


def test_merge_fold_shift_check_then_promote_inserts_nothing(engine, seeded, tmp_path):
    ids = _seed(engine)
    with psycopg2.connect(DSN) as conn:
        assert rfk.run_check(conn) == 1  # legacy rows and a duplicate

        # The pending rec on the held stale row's prediction keeps that copy where it is (pinned refusal).
        pin = {"merge": 1, "fold": 1, "refuse": 1}
        counts = rfk.run(
            conn, "dedupe", True, out_csv=str(tmp_path / "d.csv"), archive_dir=str(tmp_path), expect_plan=pin
        )
        assert (counts["copies_deleted"], counts["folded"], counts["match_stats_moved"], counts["plan_mismatch"]) == (
            2,
            1,
            2,
            0,
        )
        assert counts["trigger_graded_predictions"] == 2  # match_result and btts; double_chance is the grader's

        # The merge: copy gone, its stats on the ESPN row.
        assert _row(engine, ids["copy_merge"]) is None
        assert _count(engine, "SELECT count(*) FROM match_stats WHERE match_id = :m", m=ids["espn"]) == 2
        # The fold: the stale row took the copy's result; the trigger graded its pre-kickoff predictions and left
        # the void rec alone.
        assert _row(engine, ids["copy_fold"]) is None
        stale = _row(engine, ids["stale"])
        assert (stale.status, stale.home_score, stale.away_score, stale.rs, stale.src) == (
            "finished",
            2,
            0,
            "football_data",
            ids["copy_fold"],
        )
        with engine.connect() as c:
            graded = dict(
                c.execute(
                    text("SELECT prediction_type, is_correct FROM predictions WHERE match_id = :m"), {"m": ids["stale"]}
                ).all()
            )
        assert graded == {"match_result": True, "btts": False, "double_chance": None}
        assert _count(engine, "SELECT count(*) FROM betting_recommendations WHERE status = 'void'") == 1
        # The held one: untouched.
        assert _row(engine, ids["stale_held"]).status == "scheduled" and _row(engine, ids["copy_held"]) is not None
        archived = [json.loads(line) for f in tmp_path.glob("*.jsonl.gz") for line in gzip.open(f, "rt")]
        assert {(r["table"], r["role"]) for r in archived} >= {
            ("matches", "copy"),
            ("matches", "espn_row_before"),
            ("matches", "fold_row_before"),
            ("predictions", "fold_row_before"),
            ("betting_recommendations", "fold_row_before"),
            ("match_stats", "references_copy"),
        }

        # The shift: the lone row moves back an hour; the held copy is refused (a UNIQUE collision).
        counts = rfk.run(conn, "shift", True, out_csv=str(tmp_path / "s.csv"), expect_plan={"shift": 1, "refuse": 1})
        assert (counts["shifted"], counts["dedupe_pending"]) == (1, 0)
        assert _row(engine, ids["lone"]).match_date == ids["k_shift"]
        assert _row(engine, ids["gmt"]).match_date == datetime(2025, 12, 6, 8, 0, tzinfo=UTC)
        assert rfk.run_check(conn) == 1  # the held copy is still at its legacy kickoff

    # Settle the held rec, then the rest folds and the check passes.
    with engine.begin() as c:
        c.execute(text("UPDATE betting_recommendations SET status = 'void' WHERE status = 'pending'"))
    with psycopg2.connect(DSN) as conn:
        counts = rfk.run(conn, "dedupe", True, out_csv=str(tmp_path / "d2.csv"), archive_dir=str(tmp_path / "b"))
        assert (counts["folded"], counts["refused"]) == (1, 0)
        assert rfk.run_check(conn) == 0

    # A full promote_raw run now writes every fixture onto its existing row: nothing inserted, nothing refused.
    before = _count(engine, "SELECT count(*) FROM matches")
    stats = pr.promote(DSN)
    assert (stats.matches, stats.refused, stats.duplicates) == (5, 0, 0)
    assert _count(engine, "SELECT count(*) FROM matches") == before


def test_promote_raw_finds_the_espn_row_through_the_real_query(engine, seeded):
    ids = _seed(engine)
    with psycopg2.connect(DSN) as conn, conn.cursor(cursor_factory=pr.RealDictCursor) as cur:
        cur.execute("SELECT home_team_id, away_team_id FROM matches WHERE id = %s", (ids["espn"],))
        pair = cur.fetchone()
        when = pr.parse_match_date("10/07/2026", "12:35")
        rows = pr.find_existing_matches(cur, pair["home_team_id"], pair["away_team_id"], when, when.date())
        assert sorted(r["id"] for r in rows) == sorted([ids["espn"], ids["copy_merge"]])
        chosen, why = pr.choose_existing(rows)
        assert (chosen["id"], why, [d["id"] for d in pr.duplicates_beside(chosen, rows)]) == (
            ids["espn"],
            None,
            [ids["copy_merge"]],
        )
        # another team pair at the same kickoff is never a candidate
        cur.execute("SELECT home_team_id FROM matches WHERE id = %s", (ids["lone"],))
        assert (
            pr.find_existing_matches(cur, cur.fetchone()["home_team_id"], pair["away_team_id"], when, when.date()) == []
        )


def _seed_alias(engine) -> dict:
    """Monterrey v Santos Laguna as on prod 2026-10-31: ESPN's row filed under Santos (the Brazilian club's row) at
    01:05Z with a stale 'scheduled' row at 01:00Z, football-data's copy on Santos Laguna at its legacy 02:05Z (with
    a half-time and both clubs' match_stats), and the raw row."""
    ids = {"league": _id()}
    t = {name: _id() for name in ("Monterrey", "Santos", "Santos Laguna")}
    ids.update({f"team:{k}": v for k, v in t.items()})
    k = datetime(2026, 7, 19, 1, 5, tzinfo=UTC)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO leagues (id, name, country, sport, external_ids) "
                "VALUES (:id, 'Chinese Super League', 'China', 'soccer', '{\"football_data\": \"CN1\"}')"
            ),
            {"id": ids["league"]},
        )
        for name, tid in t.items():
            conn.execute(
                text(
                    "INSERT INTO teams (id, name, normalized_name, league_id, sport) "
                    "VALUES (:id, :name, :norm, :lid, 'soccer')"
                ),
                {"id": tid, "name": name, "norm": pr.normalize_team_name(name), "lid": ids["league"]},
            )
        for key, away, when, status, score, ht, md in (
            ("espn", "Santos", k, "finished", (3, 2), (None, None), {"result_source": "espn_scoreboard"}),
            ("stale", "Santos", k - timedelta(minutes=5), "scheduled", (None, None), (None, None), {}),
            ("copy", "Santos Laguna", k + timedelta(hours=1), "finished", (3, 2), (3, 1), {}),
        ):
            ids[key] = _id()
            conn.execute(
                text(
                    "INSERT INTO matches (id, league_id, home_team_id, away_team_id, match_date, status, home_score, "
                    "away_score, half_time_home, half_time_away, season, metadata) VALUES (:id, :lid, :h, :a, :d, "
                    ":s, :hs, :as_, :hh, :ha, '2026', CAST(:md AS jsonb))"
                ),
                {
                    "id": ids[key],
                    "lid": ids["league"],
                    "h": t["Monterrey"],
                    "a": t[away],
                    "d": when,
                    "s": status,
                    "hs": score[0],
                    "as_": score[1],
                    "hh": ht[0],
                    "ha": ht[1],
                    "md": json.dumps(md),
                },
            )
        for team in ("Monterrey", "Santos Laguna"):
            conn.execute(
                text("INSERT INTO match_stats (match_id, team_id, shots) VALUES (:m, :t, 11)"),
                {"m": ids["copy"], "t": t[team]},
            )
        conn.execute(
            text(
                'INSERT INTO raw_football_data ("Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", '
                "\"HTAG\", league, season) VALUES ('19/07/2026', '02:05', 'Monterrey', 'Santos Laguna', "
                "'3', '2', '3', '1', 'CN1', '2026')"
            )
        )
    ids["kickoff"] = k
    return ids


def test_alias_merge_then_promote_refuses_until_the_re_point(engine, seeded, tmp_path):
    """The re-point / kickoff-repair deadlock (review finding, 2026-10-31): repoint_team_matches refuses to move
    ESPN's row while football-data's copy sits on the target pair, and --shift refused to move the copy beside it.
    --dedupe now merges the copy into that row (the reviewed map says where it is about to go); promote_raw refuses
    to insert the fixture again until the row is re-pointed, then writes onto it."""
    ids = _seed_alias(engine)
    map_path = tmp_path / "repoint_team_matches.csv"
    map_path.write_text(
        "match_id,side,from_team,to_team,league,kickoff_utc,opponent,espn_event,evidence\n"
        f"{ids['espn']},away,Santos,Santos Laguna,Liga MX,2026-07-19 01:05,Monterrey,401877038,test\n"
        f"{ids['stale']},away,Santos,Santos Laguna,Liga MX,2026-07-19 01:00,Monterrey,401877038,test\n"
    )
    repoint_map = rfk.load_repoint_map(map_path)
    with psycopg2.connect(DSN) as conn:
        assert rfk.run_check(conn, repoint_map) == 1
        # Without the map the copy is not a dedupe candidate, and the shift refuses it (alias slot).
        counts = rfk.run(conn, "shift", False, out_csv=str(tmp_path / "s0.csv"))
        assert (counts["refused"], counts["dedupe_pending"]) == (1, 0)

        pin = {"merge": 0, "alias_merge": 1, "fold": 0, "refuse": 0}
        counts = rfk.run(
            conn,
            "dedupe",
            True,
            out_csv=str(tmp_path / "d.csv"),
            archive_dir=str(tmp_path),
            expect_plan=pin,
            repoint_map=repoint_map,
        )
        assert (
            counts["copies_deleted"],
            counts["match_stats_moved"],
            counts["ht_copied"],
            counts["plan_mismatch"],
        ) == (
            1,
            1,
            1,
            0,
        )
        assert _row(engine, ids["copy"]) is None
        with engine.connect() as c:
            kept = c.execute(
                text(
                    "SELECT away_team_id::text, half_time_home, half_time_away, "
                    "(SELECT array_agg(team_id::text) FROM match_stats WHERE match_id = m.id) AS stats "
                    "FROM matches m WHERE id = :id"
                ),
                {"id": ids["espn"]},
            ).one()
        # Still on Santos (repoint_team_matches moves it); only Monterrey's stats came across.
        assert (kept[0], kept[1], kept[2], kept[3]) == (ids["team:Santos"], 3, 1, [ids["team:Monterrey"]])
        archived = [json.loads(line) for f in tmp_path.glob("*.jsonl.gz") for line in gzip.open(f, "rt")]
        assert sum(1 for r in archived if r["table"] == "match_stats" and r["role"] == "references_copy") == 2

        counts = rfk.run(conn, "shift", True, out_csv=str(tmp_path / "s.csv"), repoint_map=repoint_map)
        assert (counts["planned"], counts["refused"], counts["dedupe_pending"]) == (0, 0, 0)
        assert rfk.run_check(conn, repoint_map) == 0

    before = _count(engine, "SELECT count(*) FROM matches")
    stats = pr.promote(DSN)
    assert (stats.matches, stats.refused) == (0, 1)  # the alias guard: nothing inserted beside ESPN's row
    assert _count(engine, "SELECT count(*) FROM matches") == before

    # repoint_team_matches moves both rows (the stale one with the ESPN one); then promote writes onto ESPN's row.
    with engine.begin() as c:
        c.execute(
            text("UPDATE matches SET away_team_id = :sl WHERE id IN (:a, :b)"),
            {"sl": ids["team:Santos Laguna"], "a": ids["espn"], "b": ids["stale"]},
        )
    stats = pr.promote(DSN)
    assert (stats.matches, stats.refused, stats.duplicates) == (1, 0, 0)
    assert _count(engine, "SELECT count(*) FROM matches") == before
    assert _count(engine, "SELECT count(*) FROM match_stats WHERE match_id = :m", m=ids["espn"]) == 2
