"""The 2026-10-31 results/settlement SQL against a real Postgres.

The rest of the suite runs these statements against stub cursors; here they
run on a throwaway cluster built from services/data-ingestion/db/migrations,
so the jsonb predicates and guards are Postgres's, not a stub's:

  * list_finished_matches lists a soccer match carrying a 90-minute score,
    never a fully graded NHL game carrying load_nhl_historical's key of the
    same name;
  * open recs on a cancelled match are voided after 72 h, guarded, except on
    a cancelled duplicate of a played fixture;
  * a money-settled rec is re-settled on a 90-minute score that arrives later;
  * ESPN's own cancel stays correctable (insert_finished_match,
    insert_scheduled_match, _release_redrawn_slot), a cancel recorded on
    purpose stays final on both upserts, and a
    cancelled tennis competition gives up its ESPN id;
  * regrade_predictions --match-id reads only those matches.

Skipped where no Postgres server binaries are on PATH (initdb, pg_ctl).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2")
from psycopg2.extras import RealDictCursor  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO / "scripts"
MIGRATIONS = REPO / "services" / "data-ingestion" / "db" / "migrations"

pytestmark = pytest.mark.skipif(
    not (shutil.which("initdb") and shutil.which("pg_ctl")), reason="no Postgres server binaries on PATH"
)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("grading_outcomes", "grading_outcomes.py")
gcm = _load("grade_completed_matches", "grade_completed_matches.py")
fu = _load("fetch_upcoming_pg", "fetch_upcoming.py")
rp = _load("regrade_predictions_pg", "regrade_predictions.py")


@pytest.fixture(scope="module")
def dsn(tmp_path_factory):
    base = tmp_path_factory.mktemp("pg")
    data = base / "data"
    # A short socket dir: a Unix socket path is limited to ~100 bytes.
    sock = Path(tempfile.mkdtemp(prefix="pgt", dir="/tmp"))
    try:
        subprocess.run(["initdb", "-D", str(data), "-U", "postgres", "-A", "trust"], check=True, capture_output=True)
        subprocess.run(
            ["pg_ctl", "-D", str(data), "-w", "-l", str(base / "log"), "-o", f"-k {sock} -c listen_addresses=''"]
            + ["start"],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:  # an environment that cannot host a cluster (root, no shm)
        shutil.rmtree(sock, ignore_errors=True)
        pytest.skip(f"cannot start a throwaway Postgres: {exc.stderr.decode(errors='replace')[-300:]}")
    try:
        admin = psycopg2.connect(host=str(sock), user="postgres", dbname="postgres")
        admin.autocommit = True
        admin.cursor().execute("CREATE DATABASE t")
        admin.close()
        conn = psycopg2.connect(host=str(sock), user="postgres", dbname="t")
        conn.autocommit = True
        for f in sorted(MIGRATIONS.glob("*.sql")):
            conn.cursor().execute(f.read_text())
        conn.close()
        yield {"host": str(sock), "user": "postgres", "dbname": "t"}
    finally:
        subprocess.run(["pg_ctl", "-D", str(data), "-m", "immediate", "stop"], capture_output=True)
        shutil.rmtree(sock, ignore_errors=True)


@pytest.fixture
def cur(dsn):
    conn = psycopg2.connect(**dsn)
    with conn.cursor(cursor_factory=RealDictCursor) as c:
        yield c
    conn.rollback()
    conn.close()


NOW = datetime.now(timezone.utc)


def _league(cur, sport: str) -> str:
    cur.execute(
        "INSERT INTO leagues (name, country, sport) VALUES (%s, 'X', %s) RETURNING id::text AS id",
        (f"{sport}-{uuid.uuid4()}", sport),
    )
    return cur.fetchone()["id"]


def _team(cur, sport: str) -> str:
    name = f"team-{uuid.uuid4()}"
    cur.execute(
        "INSERT INTO teams (name, normalized_name, sport) VALUES (%s, %s, %s) RETURNING id::text AS id",
        (name, name, sport),
    )
    return cur.fetchone()["id"]


def _match(cur, sport="soccer", status="finished", kickoff=None, scores=(3, 2), metadata=None, espn=None):
    league = _league(cur, sport)
    home, away = _team(cur, sport), _team(cur, sport)
    cur.execute(
        """
        INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, status, home_score, away_score,
                             season, metadata, external_ids)
        VALUES (%s, %s, %s, %s, %s, %s, %s, '2026', %s::jsonb, %s::jsonb)
        RETURNING id::text AS id
        """,
        (
            league,
            home,
            away,
            kickoff or NOW - timedelta(days=2),
            status,
            scores[0],
            scores[1],
            json.dumps(metadata or {}),
            json.dumps({"espn": espn} if espn else {}),
        ),
    )
    return {"id": cur.fetchone()["id"], "league": league, "home": home, "away": away}


def _prediction(cur, match_id, ptype="match_result", pick="home", actual="home", correct=True, model="ensemble"):
    cur.execute(
        """
        INSERT INTO predictions (match_id, model_name, model_version, prediction_type, predicted_outcome,
                                 confidence, probabilities, actual_outcome, is_correct, created_at)
        VALUES (%s, %s, 'v1', %s, %s, 0.5, '{}'::jsonb, %s, %s, NOW() - interval '30 days')
        RETURNING id::text AS id
        """,
        (match_id, model, ptype, pick, actual, correct),
    )
    return cur.fetchone()["id"]


def _rec(cur, match_id, prediction_id, status="pending", actual_result=None, profit_loss=None, selection="home"):
    cur.execute(
        """
        INSERT INTO betting_recommendations (prediction_id, match_id, bet_type, selection, odds_at_recommendation,
                                             bookmaker, recommended_stake, status, actual_result, profit_loss,
                                             created_at)
        VALUES (%s, %s, '1x2', %s, 2.0, 'b', 10, %s, %s, %s, NOW() - interval '30 days')
        RETURNING id::text AS id
        """,
        (prediction_id, match_id, selection, status, actual_result, profit_loss),
    )
    return cur.fetchone()["id"]


def _status(cur, table, row_id, *cols):
    cur.execute(f"SELECT {', '.join(cols)} FROM {table} WHERE id = %s", (row_id,))
    return tuple(cur.fetchone().values())


AET = {"result_detail": "AET", "regulation_home_score": 2, "regulation_away_score": 2, "went_extra": True}


class TestListFinishedMatches:
    def test_a_graded_nhl_game_with_the_nhl_key_is_not_listed(self, cur):
        nhl = _match(
            cur, "nhl", metadata={"regulation_winner": "tie", "regulation_home_score": 2, "regulation_away_score": 2}
        )
        _prediction(cur, nhl["id"], "moneyline", model="ensemble_nhl_ml")  # fully graded, no open rec
        soccer = _match(cur, "soccer", metadata=AET)
        _prediction(cur, soccer["id"])  # fully graded too: listed for its 90-minute score
        plain = _match(cur, "soccer", metadata={"result_detail": "FT"})
        _prediction(cur, plain["id"])
        listed = {m["match_id"] for m in gcm.list_finished_matches(cur, 14, None)}
        assert soccer["id"] in listed
        assert nhl["id"] not in listed and plain["id"] not in listed


class TestCancelledVoids:
    def test_voids_after_72h_guarded_and_spares_duplicates(self, cur):
        old = _match(cur, status="cancelled", kickoff=NOW - timedelta(days=4), scores=(None, None))
        dup = _match(
            cur, status="cancelled", kickoff=NOW - timedelta(days=4), scores=(None, None), metadata={"orphan_of": "x"}
        )
        fresh = _match(cur, status="cancelled", kickoff=NOW - timedelta(hours=10), scores=(None, None))
        r_old = _rec(cur, old["id"], _prediction(cur, old["id"], actual=None, correct=None))
        r_placed = _rec(cur, old["id"], _prediction(cur, old["id"], "btts", actual=None, correct=None), "placed")
        r_dup = _rec(cur, dup["id"], _prediction(cur, dup["id"], actual=None, correct=None))
        r_fresh = _rec(cur, fresh["id"], _prediction(cur, fresh["id"], actual=None, correct=None))
        ids = {r_old, r_placed, r_dup, r_fresh}
        recs = [r for r in gcm.list_recs_on_cancelled_matches(cur) if r["rec_id"] in ids]
        assert {r["rec_id"] for r in recs} == {r_old, r_placed, r_dup}

        voided = refused = 0
        for rec in recs:
            if rec["orphan_of"]:
                refused += 1
            else:
                voided += gcm.void_open_rec(cur, rec, "cancelled", gcm.CANCELLED_VOID_RESULT)
        assert (voided, refused) == (2, 1)
        for rid in (r_old, r_placed):
            assert _status(cur, "betting_recommendations", rid, "status", "actual_result", "profit_loss") == (
                "void",
                "void: match cancelled",
                Decimal("0.00"),
            )
        assert _status(cur, "betting_recommendations", r_dup, "status")[0] == "pending"
        assert _status(cur, "betting_recommendations", r_fresh, "status")[0] == "pending"

    def test_the_void_guard_holds_on_a_rescheduled_match(self, cur):
        m = _match(cur, status="cancelled", kickoff=NOW - timedelta(days=4), scores=(None, None))
        r = _rec(cur, m["id"], _prediction(cur, m["id"], actual=None, correct=None))
        cur.execute("UPDATE matches SET status = 'scheduled' WHERE id = %s", (m["id"],))
        assert (
            gcm.void_open_rec(cur, {"rec_id": r, "match_id": m["id"], "status": "pending"}, "cancelled", "x") is False
        )
        assert _status(cur, "betting_recommendations", r, "status")[0] == "pending"


class TestResettle:
    def test_a_rec_settled_on_the_final_score_is_resettled_on_90_minutes(self, cur):
        m = _match(cur, metadata=AET)  # 3-2 AET, 2-2 after 90
        p = _prediction(cur, m["id"], actual="draw", correct=False)
        r = _rec(cur, m["id"], p, status="won", actual_result="won", profit_loss=Decimal("10.00"))
        match = {**gcm.list_finished_matches(cur, 14, m["id"])[0]}
        assert gcm.resettle_recs_on_regulation(cur, match) == 1
        assert _status(cur, "betting_recommendations", r, "status", "actual_result", "profit_loss") == (
            "lost",
            "lost",
            Decimal("-10.00"),
        )
        assert gcm.resettle_recs_on_regulation(cur, match) == 0  # settled now: a no-op


class TestEspnCancels:
    SOCCER = fu.SPORT_CONFIGS["soccer"]
    TENNIS = fu.SPORT_CONFIGS["tennis"]
    ESPN = {"status_source": "espn_scoreboard", "espn_status": "STATUS_ABANDONED"}

    def _finish(self, cur, m, kickoff):
        return fu.insert_finished_match(
            cur, self.SOCCER, m["league"], m["home"], m["away"], kickoff, None, 2, 1, {"result_detail": "FT"}
        )

    @pytest.mark.parametrize(
        "metadata, written",
        [
            ({"status_source": "espn_scoreboard", "espn_status": "STATUS_ABANDONED"}, 1),
            ({"status_source": "espn_scoreboard", "espn_status": "STATUS_ABANDONED", "orphan_of": "t"}, 0),
            ({"status_source": "espn_scoreboard", "espn_status": "STATUS_POSTPONED"}, 0),
            ({}, 0),
        ],
    )
    def test_only_espns_own_cancel_takes_a_result(self, cur, metadata, written):
        kickoff = NOW - timedelta(days=1)
        m = _match(cur, status="cancelled", kickoff=kickoff, scores=(None, None), metadata=metadata)
        assert self._finish(cur, m, kickoff) == written
        expected = ("finished", 2, 1) if written else ("cancelled", None, None)
        assert _status(cur, "matches", m["id"], "status", "home_score", "away_score") == expected

    @pytest.mark.parametrize(
        "metadata, status_after",
        [
            ({"status_source": "espn_scoreboard", "espn_status": "STATUS_ABANDONED"}, "scheduled"),
            ({"status_source": "espn_scoreboard", "espn_status": "STATUS_ABANDONED", "orphan_of": "t"}, "cancelled"),
            (
                {"status_source": "espn_scoreboard", "espn_status": "STATUS_CANCELED", "redrawn_espn_id": "9"},
                "cancelled",
            ),
            ({"result_note": "operator"}, "cancelled"),  # fix_unplayed_soccer_rows: no status_source
            ({}, "cancelled"),
        ],
    )
    def test_the_fixtures_upsert_keeps_a_cancel_recorded_on_purpose(self, cur, metadata, status_after):
        kickoff = NOW + timedelta(days=1)
        m = _match(cur, status="cancelled", kickoff=kickoff, scores=(None, None), metadata=metadata)
        assert fu.insert_scheduled_match(cur, self.SOCCER, m["league"], m["home"], m["away"], kickoff, None) == 1
        assert _status(cur, "matches", m["id"], "status") == (status_after,)

    def test_the_fixtures_upsert_still_reschedules_a_postponed_row_and_keeps_a_finished_one(self, cur):
        for status, after in (("postponed", "scheduled"), ("finished", "finished")):
            kickoff = NOW + timedelta(days=2)
            m = _match(cur, status=status, kickoff=kickoff)
            fu.insert_scheduled_match(cur, self.SOCCER, m["league"], m["home"], m["away"], kickoff, None)
            assert _status(cur, "matches", m["id"], "status") == (after,)

    def test_an_espn_cancelled_slot_is_released_and_an_operators_is_not(self, cur):
        for metadata, released in ((self.ESPN, True), ({}, False)):
            m = _match(cur, "tennis", status="cancelled", scores=(None, None), metadata=metadata, espn="185138")
            row = {"id": m["id"], "status": "cancelled", "metadata": metadata}
            assert fu._release_redrawn_slot(cur, self.TENNIS, "185138", row) is released
            cur.execute("SELECT external_ids->>'espn' AS espn FROM matches WHERE id = %s", (m["id"],))
            assert (cur.fetchone()["espn"] is None) is released
            cur.execute("UPDATE matches SET external_ids = '{}'::jsonb WHERE id = %s", (m["id"],))

    def test_a_tennis_cancel_drops_the_espn_id_and_a_soccer_one_keeps_it(self, cur):
        for sport, drop in (("tennis", "185138"), ("soccer", None)):
            m = _match(cur, sport, status="scheduled", scores=(None, None), espn="185138")
            patch = json.dumps({"status_source": "espn_scoreboard", "espn_status": "STATUS_CANCELED"})
            cur.execute(fu.STATUS_UPDATE_SQL, ("cancelled", patch, drop, m["id"], "scheduled"))
            assert cur.fetchone() is not None
            cur.execute("SELECT status, external_ids->>'espn' AS espn FROM matches WHERE id = %s", (m["id"],))
            row = cur.fetchone()
            assert row["status"] == "cancelled" and (row["espn"] is None) == (sport == "tennis")
            cur.execute("UPDATE matches SET external_ids = '{}'::jsonb WHERE id = %s", (m["id"],))


class TestRegradeMatchIds:
    def test_the_read_is_limited_to_the_ids_and_ignores_the_window(self, cur):
        old = _match(cur, kickoff=NOW - timedelta(days=400))
        other = _match(cur)
        _prediction(cur, old["id"])
        _prediction(cur, other["id"])
        params = {"days": "30", "model_name": None, "types": None, "match_ids": [old["id"]]}
        cur.execute(rp.GRADED_PREDICTIONS_SQL, params)
        assert {r["match_id"] for r in cur.fetchall()} == {old["id"]}
        cur.execute(rp.GRADED_PREDICTIONS_SQL, {**params, "match_ids": None})
        got = {r["match_id"] for r in cur.fetchall()}
        assert other["id"] in got and old["id"] not in got
        cur.execute(rp.FINISHED_MATCHES_SQL, ([old["id"], str(uuid.uuid4())],))
        assert [r["id"] for r in cur.fetchall()] == [old["id"]]
