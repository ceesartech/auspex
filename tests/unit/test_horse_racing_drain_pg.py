"""Horse racing dormant: the drain and the stranded-rec close-out against a real
Postgres built from services/data-ingestion/db/migrations (each test gets its
own copy). The Racing API is stubbed at load_racing_api.fetch_race_result.

  * the drain asks only about races that carry a pending rec, are off, and are
    at most DRAIN_DAYS old; ingests a posted result exactly as the results pass
    does (race finished, positions, listed non-runners scratched) and the grader
    then settles the recs the way close_out's plan said it would;
  * nothing pending -> no API call and no credentials needed; an API failure or
    missing credentials with work to do fails the run; --dry-run writes nothing;
  * a result that misses an entrant a pending rec rides on is not ingested; a
    result that does not land on its race, or whose ingest raises, is rolled
    back and the races behind it still drain — each fails the run;
  * close_out_stranded_race_recs settles from a result, voids where the API has
    none, touches nothing inside the drain window or after --raced-before,
    refuses unmatched fields and pin drift, rolls everything back when the
    database does not end where the plan said, and a re-run plans nothing.

Skipped where no Postgres server binaries are on PATH (initdb, pg_ctl).
"""

from __future__ import annotations

import csv
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


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lra = _load("load_racing_api")
gcr = _load("grade_completed_races")
cos = _load("close_out_stranded_race_recs")

NOW = datetime.now(timezone.utc)
RACED_BEFORE = NOW + timedelta(days=1)


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    base = tmp_path_factory.mktemp("pg")
    data = base / "data"
    sock = Path(tempfile.mkdtemp(prefix="pgr", dir="/tmp"))  # a Unix socket path is limited to ~100 bytes
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
        admin.cursor().execute("CREATE DATABASE tmpl")
        admin.close()
        conn = psycopg2.connect(host=str(sock), user="postgres", dbname="tmpl")
        conn.autocommit = True
        for f in sorted(MIGRATIONS.glob("*.sql")):
            conn.cursor().execute(f.read_text())
        conn.close()
        yield str(sock)
    finally:
        subprocess.run(["pg_ctl", "-D", str(data), "-m", "immediate", "stop"], capture_output=True)
        shutil.rmtree(sock, ignore_errors=True)


@pytest.fixture
def dsn(cluster):
    """A fresh database per test: the drain and the close-out scan every pending rec."""
    name = f"t_{uuid.uuid4().hex[:12]}"
    admin = psycopg2.connect(host=cluster, user="postgres", dbname="postgres")
    admin.autocommit = True
    admin.cursor().execute(f"CREATE DATABASE {name} TEMPLATE tmpl")
    yield f"host={cluster} user=postgres dbname={name}"
    admin.cursor().execute(f"DROP DATABASE {name} WITH (FORCE)")
    admin.close()


@pytest.fixture
def db(dsn):
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        yield cur
    conn.close()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(lra.time, "sleep", lambda s: None)
    monkeypatch.setattr(cos.time, "sleep", lambda s: None)


def _race(cur, off: datetime, status: str = "scheduled", api_id: str | None = None) -> dict:
    api_id = api_id or f"rac_{uuid.uuid4().hex[:10]}"
    cur.execute(
        "INSERT INTO leagues (name, country, sport) VALUES (%s, 'United Kingdom', 'horse_racing') RETURNING id",
        (f"Testbury-{api_id}",),
    )
    league = cur.fetchone()["id"]
    cur.execute(
        """
        INSERT INTO races (league_id, external_ids, race_date, track_name, status)
        VALUES (%s, %s::jsonb, %s, 'Testbury', %s) RETURNING id::text AS id
        """,
        (league, json.dumps({"racing_api_race_id": api_id}), off, status),
    )
    return {"id": cur.fetchone()["id"], "api": api_id, "off": off}


def _entrant(cur, race: dict, horse: str, number: int, scratched: bool = False) -> str:
    cur.execute(
        """
        INSERT INTO horses (name, normalized_name) VALUES (%s, %s)
        ON CONFLICT (normalized_name) DO UPDATE SET name = EXCLUDED.name RETURNING id
        """,
        (horse, lra.normalize_name(horse)),
    )
    horse_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO race_entrants (race_id, horse_id, program_number, scratched) VALUES (%s, %s, %s, %s) "
        "RETURNING id::text AS id",
        (race["id"], horse_id, number, scratched),
    )
    return cur.fetchone()["id"]


def _rec(cur, race: dict, entrant: str, horse: str, bet_type="win", odds="5.00", stake="10.00", status="pending"):
    cur.execute(
        """
        INSERT INTO race_predictions (race_id, entrant_id, model_name, model_version, prediction_type, confidence)
        VALUES (%s, %s, 'market_consensus_v1', 'v1', 'win', 0.2)
        ON CONFLICT (race_id, entrant_id, model_name, model_version, prediction_type) DO UPDATE SET confidence = 0.2
        RETURNING id
        """,
        (race["id"], entrant),
    )
    pred = cur.fetchone()["id"]
    cur.execute(
        """
        INSERT INTO race_recommendations (race_prediction_id, race_id, entrant_id, bet_type, selection,
                                          odds_at_recommendation, recommended_stake, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id::text AS id
        """,
        (pred, race["id"], entrant, bet_type, horse, Decimal(odds), Decimal(stake), status),
    )
    return cur.fetchone()["id"]


def _payload(race: dict, runners: list[tuple[str, str]], non_runners: str = "") -> dict:
    return {
        "race_id": race["api"],
        "course": "Testbury",
        "off_dt": race["off"].isoformat(),
        "race_name": "Test Stakes",
        "runners": [
            {"horse": h, "position": p, "sp_dec": "5.00", "number": str(i + 1)} for i, (h, p) in enumerate(runners)
        ],
        "non_runners": non_runners,
    }


class FakeApi:
    """fetch_race_result stand-in: {api_race_id: payload | None | Exception}; records every call."""

    def __init__(self, answers: dict):
        self.answers = answers
        self.calls: list[str] = []

    def __call__(self, api_race_id, auth=None):
        self.calls.append(api_race_id)
        answer = self.answers[api_race_id]
        if isinstance(answer, Exception):
            raise answer
        return answer


def _rec_row(cur, rec_id):
    cur.execute("SELECT status, profit_loss, actual_result FROM race_recommendations WHERE id = %s", (rec_id,))
    r = cur.fetchone()
    return (r["status"], r["profit_loss"], r["actual_result"])


def _field(cur, race):
    cur.execute(cos.FIELD_SQL, ([race["id"]],))
    return [dict(r) for r in cur.fetchall()]


def _seed_full_field(cur, off):
    """Six declared, Foxtrot a non-runner: five run, so each-way pays two places."""
    race = _race(cur, off)
    names = ["Alpha (IRE)", "Bravo", "Charlie", "Delta (GB)", "Echo", "Foxtrot"]
    ent = {n: _entrant(cur, race, n, i + 1) for i, n in enumerate(names)}
    recs = {
        "win_alpha": _rec(cur, race, ent["Alpha (IRE)"], "Alpha (IRE)", odds="4.50", stake="10.00"),
        "win_charlie": _rec(cur, race, ent["Charlie"], "Charlie", odds="7.00", stake="3.33"),
        "place_bravo": _rec(cur, race, ent["Bravo"], "Bravo", bet_type="place", odds="2.10", stake="25.00"),
        "place_echo": _rec(cur, race, ent["Echo"], "Echo", bet_type="place", odds="3.00", stake="5.00"),
        "win_foxtrot": _rec(cur, race, ent["Foxtrot"], "Foxtrot", odds="9.00", stake="4.00"),
    }
    payload = _payload(
        race,
        [("Alpha (IRE)", "1"), ("Bravo", "2"), ("Charlie", "3"), ("Delta (GB)", "4"), ("Echo", "PU")],
        non_runners="Foxtrot (vet's certificate, lame)",
    )
    return race, recs, payload


# ── drain ─────────────────────────────────────────────────────────────────


def test_the_drain_ingests_a_posted_result_and_the_grader_settles_it_as_planned(dsn, db, monkeypatch):
    race, recs, payload = _seed_full_field(db, NOW - timedelta(hours=2))
    db.execute(
        """
        SELECT rec.id::text AS rec_id, h.normalized_name AS horse, rec.bet_type, rec.selection,
               rec.odds_at_recommendation AS odds, rec.recommended_stake AS stake
        FROM race_recommendations rec
        JOIN race_entrants e ON e.id = rec.entrant_id
        JOIN horses h ON h.id = e.horse_id
        """
    )
    planned = [dict(r) for r in db.fetchall()]
    verdicts = cos.plan_race_settlement(planned, payload, _field(db, race))
    plan = {rec["rec_id"]: v for rec, v in zip(planned, verdicts)}
    api = FakeApi({race["api"]: payload})
    monkeypatch.setattr(lra, "fetch_race_result", api)
    monkeypatch.setattr(lra, "_auth", lambda: object())

    totals = lra.run_drain(dsn)
    assert api.calls == [race["api"]]
    assert (totals["races"], totals["resolved"], totals["not_posted"], totals["errors"]) == (1, 1, 0, 0)
    db.execute("SELECT status FROM races WHERE id = %s", (race["id"],))
    assert db.fetchone()["status"] == "finished"
    db.execute(
        "SELECT h.normalized_name AS horse, e.finish_position, e.scratched FROM race_entrants e "
        "JOIN horses h ON h.id = e.horse_id WHERE e.race_id = %s",
        (race["id"],),
    )
    field = {r["horse"]: (r["finish_position"], r["scratched"]) for r in db.fetchall()}
    assert field == {
        "alpha": (1, False),
        "bravo": (2, False),
        "charlie": (3, False),
        "delta": (4, False),
        "echo": (None, False),  # pulled up: ran, unplaced
        "foxtrot": (None, True),  # the listed non-runner
    }

    gcr.run(dsn, 14, None)
    expected = {
        "win_alpha": ("won", Decimal("35.00"), "Alpha (IRE)"),
        "win_charlie": ("lost", Decimal("-3.33"), "Alpha (IRE)"),
        "place_bravo": ("won", Decimal("27.50"), "finished 2"),
        "place_echo": ("lost", Decimal("-5.00"), "unplaced"),
        "win_foxtrot": ("void", Decimal("0.00"), "Alpha (IRE)"),  # a non-runner is void, never lost
    }
    for key, rec_id in recs.items():
        assert _rec_row(db, rec_id) == expected[key], key
        # The close-out plan reaches the grader's verdict from the payload alone.
        assert (plan[rec_id]["new_status"], plan[rec_id]["profit_loss"], plan[rec_id]["actual_result"]) == expected[
            key
        ], key


def test_the_drain_asks_only_about_open_races_inside_its_window(dsn, db, monkeypatch, caplog):
    def race_with_rec(off, race_status="scheduled", rec_status="pending"):
        race = _race(db, off, status=race_status)
        _rec(db, race, _entrant(db, race, f"Horse {uuid.uuid4().hex[:6]}", 1), "h", status=rec_status)
        return race

    asked = race_with_rec(NOW - timedelta(days=1))
    not_off = race_with_rec(NOW - timedelta(minutes=lra.DRAIN_GRACE_MINUTES - 5))
    future = race_with_rec(NOW + timedelta(hours=5))
    stranded = race_with_rec(NOW - timedelta(days=lra.DRAIN_DAYS, hours=1))
    race_with_rec(NOW - timedelta(hours=3), race_status="finished")  # the grader's, not the drain's
    race_with_rec(NOW - timedelta(hours=3), rec_status="won")  # nothing pending
    api = FakeApi({asked["api"]: None})
    monkeypatch.setattr(lra, "fetch_race_result", api)
    monkeypatch.setattr(lra, "_auth", lambda: object())

    with caplog.at_level("WARNING", logger="load_racing_api"):
        totals = lra.run_drain(dsn)
    assert api.calls == [asked["api"]]
    assert totals["not_posted"] == 1 and totals["errors"] == 0 and totals["resolved"] == 0
    assert totals["upcoming_recs"] == 2 and totals["stranded_recs"] == 1
    # The stranded WARNING must not promise more than the close-out does: it covers races before its pin only.
    (warning,) = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warning.startswith("1 pending race rec(s)") and "before its reviewed RACED_BEFORE" in warning
    assert "need a new reviewed pin" in warning
    db.execute(
        "SELECT count(*) AS n FROM races WHERE status = 'scheduled' AND id = ANY(%s::uuid[])",
        ([asked["id"], not_off["id"], future["id"], stranded["id"]],),
    )
    assert db.fetchone()["n"] == 4


def test_nothing_pending_makes_no_call_and_needs_no_credentials(dsn, db, monkeypatch):
    race = _race(db, NOW - timedelta(hours=3), status="finished")
    _rec(db, race, _entrant(db, race, "Settled", 1), "Settled", status="lost")

    def boom(*a, **k):
        raise AssertionError("no API call may happen with nothing pending")

    monkeypatch.setattr(lra, "fetch_race_result", boom)
    monkeypatch.setattr(lra, "_auth", lambda: None)
    assert lra.main(["--drain", "--database-url", dsn]) == 0


def test_work_without_credentials_fails_the_run(dsn, db, monkeypatch):
    race = _race(db, NOW - timedelta(hours=3))
    _rec(db, race, _entrant(db, race, "Waiting", 1), "Waiting")
    monkeypatch.setattr(lra, "fetch_race_result", FakeApi({}))
    monkeypatch.setattr(lra, "_auth", lambda: None)
    assert lra.main(["--drain", "--database-url", dsn]) == 1


def test_an_api_failure_is_an_error_not_no_result(dsn, db, monkeypatch):
    race = _race(db, NOW - timedelta(hours=3))
    _rec(db, race, _entrant(db, race, "Waiting", 1), "Waiting")
    monkeypatch.setattr(lra, "fetch_race_result", FakeApi({race["api"]: lra.RacingApiError("HTTP 401")}))
    monkeypatch.setattr(lra, "_auth", lambda: object())
    assert lra.main(["--drain", "--database-url", dsn]) == 1
    db.execute("SELECT status FROM races WHERE id = %s", (race["id"],))
    assert db.fetchone()["status"] == "scheduled"


def test_a_dry_run_writes_nothing(dsn, db, monkeypatch):
    race, recs, payload = _seed_full_field(db, NOW - timedelta(hours=2))
    monkeypatch.setattr(lra, "fetch_race_result", FakeApi({race["api"]: payload}))
    monkeypatch.setattr(lra, "_auth", lambda: object())
    assert lra.main(["--drain", "--dry-run", "--database-url", dsn]) == 0
    db.execute("SELECT status FROM races WHERE id = %s", (race["id"],))
    assert db.fetchone()["status"] == "scheduled"
    db.execute(
        "SELECT count(*) AS n FROM race_entrants WHERE race_id = %s AND finish_position IS NOT NULL", (race["id"],)
    )
    assert db.fetchone()["n"] == 0


def test_a_result_that_does_not_land_on_its_race_is_rolled_back(dsn, db, monkeypatch):
    race, recs, payload = _seed_full_field(db, NOW - timedelta(hours=2))
    later, later_recs, later_payload = _seed_full_field(db, NOW - timedelta(hours=1))
    del payload["off_dt"]  # upsert_race skips it: ingest_race returns no race id
    db.execute("SELECT count(*) AS n FROM leagues")
    leagues_before = db.fetchone()["n"]
    monkeypatch.setattr(lra, "fetch_race_result", FakeApi({race["api"]: payload, later["api"]: later_payload}))
    monkeypatch.setattr(lra, "_auth", lambda: object())

    assert lra.main(["--drain", "--database-url", dsn]) == 1
    db.execute("SELECT id::text AS id, status FROM races WHERE id = ANY(%s::uuid[])", ([race["id"], later["id"]],))
    assert {r["id"]: r["status"] for r in db.fetchall()} == {race["id"]: "scheduled", later["id"]: "finished"}
    db.execute("SELECT count(*) AS n FROM leagues")
    assert db.fetchone()["n"] == leagues_before + 1  # the later race's track; the skipped one's was rolled back
    db.execute(
        "SELECT count(*) AS n FROM race_entrants WHERE race_id = %s AND (finish_position IS NOT NULL OR scratched)",
        (race["id"],),
    )
    assert db.fetchone()["n"] == 0


def test_one_ingest_that_raises_does_not_block_the_races_behind_it(dsn, db, monkeypatch):
    first, _, first_payload = _seed_full_field(db, NOW - timedelta(hours=3))
    second, second_recs, second_payload = _seed_full_field(db, NOW - timedelta(hours=1))
    real_ingest = lra.ingest_race

    def ingest(cur, payload, finished):
        if payload["race_id"] == first["api"]:
            real_ingest(cur, payload, finished)  # writes, then fails: the rollback must take it all back
            raise psycopg2.errors.NumericValueOutOfRange("numeric field overflow")
        return real_ingest(cur, payload, finished)

    monkeypatch.setattr(lra, "ingest_race", ingest)
    monkeypatch.setattr(lra, "fetch_race_result", FakeApi({first["api"]: first_payload, second["api"]: second_payload}))
    monkeypatch.setattr(lra, "_auth", lambda: object())

    totals = lra.run_drain(dsn)
    assert (totals["races"], totals["resolved"], totals["errors"]) == (2, 1, 1)
    assert lra.main(["--drain", "--database-url", dsn]) == 1  # the next tick asks again and fails again
    db.execute("SELECT id::text AS id, status FROM races WHERE id = ANY(%s::uuid[])", ([first["id"], second["id"]],))
    assert {r["id"]: r["status"] for r in db.fetchall()} == {first["id"]: "scheduled", second["id"]: "finished"}
    db.execute(
        "SELECT count(*) AS n FROM race_entrants WHERE race_id = %s AND finish_position IS NOT NULL", (first["id"],)
    )
    assert db.fetchone()["n"] == 0


def _misspelt_race(db, off, place_rec_on_alpha=False, rec_on_misspelt=True):
    """The card spells a runner 'Jo Arengeot Bey Aa', the result 'Jo Arengeot Bey (FR)' (prod, Ayr 31 Oct)."""
    race = _race(db, off)
    names = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Jo Arengeot Bey Aa"]
    ent = {n: _entrant(db, race, n, i + 1) for i, n in enumerate(names)}
    recs = [_rec(db, race, ent["Alpha"], "Alpha", bet_type="place" if place_rec_on_alpha else "win")]
    if rec_on_misspelt:
        recs.append(_rec(db, race, ent["Jo Arengeot Bey Aa"], "Jo Arengeot Bey Aa", odds="9.00"))
    payload = _payload(
        race,
        [
            ("Jo Arengeot Bey (FR)", "1"),
            ("Alpha", "2"),
            ("Bravo", "3"),
            ("Charlie", "4"),
            ("Delta", "5"),
            ("Echo", "6"),
        ],
    )
    return race, recs, payload


@pytest.mark.parametrize(
    "place_rec_on_alpha, rec_on_misspelt, refused",
    [
        (False, True, True),  # the rec on the misspelt horse would be graded lost — it won
        (True, False, True),  # a place rec: the field the grader counts would carry the twin
        (False, False, False),  # win recs only, none on it: the winner is read from positions — ingested
    ],
)
def test_a_result_that_misses_an_entrant_a_rec_rides_on_is_not_ingested(
    dsn, db, monkeypatch, caplog, place_rec_on_alpha, rec_on_misspelt, refused
):
    race, recs, payload = _misspelt_race(db, NOW - timedelta(hours=2), place_rec_on_alpha, rec_on_misspelt)
    monkeypatch.setattr(lra, "fetch_race_result", FakeApi({race["api"]: payload}))
    monkeypatch.setattr(lra, "_auth", lambda: object())

    assert lra.main(["--drain", "--dry-run", "--database-url", dsn]) == (1 if refused else 0)
    with caplog.at_level("WARNING", logger="load_racing_api"):
        assert lra.main(["--drain", "--database-url", dsn]) == (1 if refused else 0)
    db.execute("SELECT status FROM races WHERE id = %s", (race["id"],))
    assert db.fetchone()["status"] == ("scheduled" if refused else "finished")
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "jo arengeot bey aa" in messages
    if refused:
        assert "NOT ingested" in messages and "result runners not on our card: jo arengeot bey" in messages
    else:
        gcr.run(dsn, 14, None)
        assert _rec_row(db, recs[0]) == ("lost", Decimal("-10.00"), "Jo Arengeot Bey (FR)")


# ── close-out ─────────────────────────────────────────────────────────────


def _closeout_world(db):
    settle, settle_recs, payload = _seed_full_field(db, NOW - timedelta(days=10))
    void = _race(db, NOW - timedelta(days=20))
    void_recs = [
        _rec(db, void, _entrant(db, void, "Abandoned One", 1), "Abandoned One"),
        _rec(db, void, _entrant(db, void, "Abandoned Two", 2), "Abandoned Two", bet_type="place"),
    ]
    in_window = _race(db, NOW - timedelta(days=1))  # the drain's
    window_rec = _rec(db, in_window, _entrant(db, in_window, "Drained", 1), "Drained")
    api = FakeApi({settle["api"]: payload, void["api"]: None})
    return settle, settle_recs, void, void_recs, window_rec, api


def _conn(dsn):
    return psycopg2.connect(dsn)


def test_close_out_settles_from_a_result_voids_without_one_and_a_rerun_plans_nothing(dsn, db, tmp_path):
    settle, settle_recs, void, void_recs, window_rec, api = _closeout_world(db)
    conn = _conn(dsn)
    dry = cos.run(conn, RACED_BEFORE, api, out_csv=str(tmp_path / "dry.csv"))
    assert dry["plan"] == {"candidates": 7, "races": 2, "settle": 5, "void": 2, "unmatched": 0, "errors": 0}
    assert not dry["applied"] and _rec_row(db, void_recs[0])[0] == "pending"

    applied = cos.run(
        conn, RACED_BEFORE, api, apply_changes=True, out_csv=str(tmp_path / "apply.csv"), expect_plan=dry["plan"]
    )
    assert applied["applied"] and not applied["plan_mismatch"] and not applied["refused"]
    assert _rec_row(db, settle_recs["win_alpha"]) == ("won", Decimal("35.00"), "Alpha (IRE)")
    assert _rec_row(db, settle_recs["win_foxtrot"]) == ("void", Decimal("0.00"), "Alpha (IRE)")
    for rec_id in void_recs:
        assert _rec_row(db, rec_id) == ("void", Decimal("0.00"), cos.VOID_LABEL)
    assert _rec_row(db, window_rec)[0] == "pending"  # inside the drain window: never a candidate
    db.execute("SELECT id::text AS id, status FROM races WHERE id = ANY(%s::uuid[])", ([settle["id"], void["id"]],))
    assert {r["id"]: r["status"] for r in db.fetchall()} == {settle["id"]: "finished", void["id"]: "scheduled"}

    rows = list(csv.DictReader(open(tmp_path / "apply.csv")))
    assert {r["rec_id"]: r["action"] for r in rows} == {
        **{rec_id: "settle" for rec_id in settle_recs.values()},
        **{rec_id: "void" for rec_id in void_recs},
    }
    archive = [json.loads(line) for line in open(tmp_path / "apply.reversal.jsonl")]
    by_table = {}
    for a in archive:
        by_table.setdefault(a["table"], []).append(a["row"])
    assert {r["status"] for r in by_table["race_recommendations"]} == {"pending"}  # the pre-images
    assert {r["id"] for r in by_table["races"]} == {settle["id"], void["id"]}
    assert len(by_table["race_entrants"]) == 8 and all(r["finish_position"] is None for r in by_table["race_entrants"])

    again = cos.run(conn, RACED_BEFORE, api, out_csv=str(tmp_path / "verify.csv"))
    assert again["plan"]["candidates"] == 0
    with pytest.raises(FileExistsError):  # the audit CSV is write-once
        cos.run(conn, RACED_BEFORE, api, out_csv=str(tmp_path / "apply.csv"))
    conn.close()


def test_raced_before_keeps_later_races_out(dsn, db, tmp_path):
    settle, settle_recs, void, void_recs, window_rec, api = _closeout_world(db)
    conn = _conn(dsn)
    plan = cos.run(conn, NOW - timedelta(days=15), api, out_csv=str(tmp_path / "dry.csv"))["plan"]
    assert (plan["candidates"], plan["races"], plan["void"], plan["settle"]) == (2, 1, 2, 0)
    assert api.calls == [void["api"]]
    conn.close()


def test_a_plan_that_differs_from_the_pin_writes_nothing(dsn, db, tmp_path):
    settle, settle_recs, void, void_recs, window_rec, api = _closeout_world(db)
    conn = _conn(dsn)
    pin = {"candidates": 7, "races": 2, "settle": 4, "void": 3, "unmatched": 0, "errors": 0}
    out = cos.run(conn, RACED_BEFORE, api, apply_changes=True, out_csv=str(tmp_path / "a.csv"), expect_plan=pin)
    assert out["plan_mismatch"] and not out["applied"]
    assert {_rec_row(db, r)[0] for r in [*settle_recs.values(), *void_recs]} == {"pending"}
    assert not (tmp_path / "a.reversal.jsonl").exists()
    conn.close()


def test_an_entrant_the_result_does_not_account_for_refuses_the_apply(dsn, db, tmp_path):
    race = _race(db, NOW - timedelta(days=10))
    a = _entrant(db, race, "Known", 1)
    _entrant(db, race, "Ghost", 2)  # neither ran nor is listed as a non-runner: a name mismatch?
    rec = _rec(db, race, a, "Known")
    api = FakeApi({race["api"]: _payload(race, [("Known", "1"), ("Stranger", "2")])})
    conn = _conn(dsn)
    out = cos.run(conn, RACED_BEFORE, api, out_csv=str(tmp_path / "dry.csv"))
    assert out["plan"]["unmatched"] == 1 and out["refused"]
    pin = dict(out["plan"])
    out = cos.run(conn, RACED_BEFORE, api, apply_changes=True, out_csv=str(tmp_path / "a.csv"), expect_plan=pin)
    assert out["refused"] and not out["applied"] and _rec_row(db, rec)[0] == "pending"
    conn.close()


def test_a_grader_that_does_not_match_the_plan_rolls_everything_back(dsn, db, tmp_path, monkeypatch):
    settle, settle_recs, void, void_recs, window_rec, api = _closeout_world(db)
    conn = _conn(dsn)
    pin = cos.run(conn, RACED_BEFORE, api, out_csv=str(tmp_path / "dry.csv"))["plan"]

    def wrong_grade(cur, race_id):
        cur.execute("UPDATE race_recommendations SET status = 'lost', profit_loss = 0 WHERE race_id = %s", (race_id,))

    monkeypatch.setattr(cos.gcr, "grade_race", wrong_grade)
    with pytest.raises(cos.ApplyMismatch):
        cos.run(conn, RACED_BEFORE, api, apply_changes=True, out_csv=str(tmp_path / "a.csv"), expect_plan=pin)
    assert {_rec_row(db, r)[0] for r in [*settle_recs.values(), *void_recs]} == {"pending"}
    db.execute("SELECT status FROM races WHERE id = %s", (settle["id"],))
    assert db.fetchone()["status"] == "scheduled"
    conn.close()


def test_a_rec_settled_elsewhere_after_the_plan_rolls_the_apply_back(dsn, db, tmp_path):
    settle, settle_recs, void, void_recs, window_rec, api = _closeout_world(db)
    conn = _conn(dsn)
    pin = cos.run(conn, RACED_BEFORE, api, out_csv=str(tmp_path / "dry.csv"))["plan"]

    class Racing(FakeApi):
        def __call__(self, api_race_id, auth=None):
            out = super().__call__(api_race_id, auth)
            if len(self.calls) == 2:  # between the apply's plan and its write
                db.execute("UPDATE race_recommendations SET status = 'void' WHERE id = %s", (void_recs[0],))
            return out

    with pytest.raises(cos.ApplyMismatch):
        cos.run(
            conn,
            RACED_BEFORE,
            Racing(api.answers),
            apply_changes=True,
            out_csv=str(tmp_path / "a.csv"),
            expect_plan=pin,
        )
    assert {_rec_row(db, r)[0] for r in settle_recs.values()} == {"pending"}
    assert _rec_row(db, void_recs[1])[0] == "pending"
    conn.close()
