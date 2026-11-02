"""scripts/mma_identity_cleanup.py against a real Postgres (a throwaway cluster
built from services/data-ingestion/db/migrations, one database per test):

  * a dry run writes nothing; a pinned apply commits the whole plan in one
    transaction: the split row's other athlete's bouts moved to a new row
    (stamped, aliased, split_from), the split row stamped (split_into), the
    alias added, the orphan deleted, every listed row stamped; the archive
    holds every pre-image; --check-after holds and a re-plan is 'done';
  * the deployed resolver then resolves both split fighters to their own
    rows by athlete id and refuses a third one;
  * a row a sweep stamps between the dry run and the apply does not stop it;
  * an apply whose plan changed since the dry run refuses with nothing
    written; so does a rec on a bout to move; an in-transaction after-state
    that fails rolls everything back; an archive that already exists stops
    it before any write;
  * the SQL guards: a bout, an orphan or a team row that changed after the
    plan stops the write (GuardError);
  * --check-after counts a row that lost its athlete id.

Skipped where no Postgres server binaries are on PATH (initdb, pg_ctl).
"""

from __future__ import annotations

import dataclasses
import gzip
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
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

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fu = sys.modules.get("fetch_upcoming") or _load("fetch_upcoming", "fetch_upcoming.py")
mic = _load("mma_identity_cleanup_pg", "mma_identity_cleanup.py")

MMA = fu.SPORT_CONFIGS["mma"]
CARD = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(days=200)
KEEP, MOVE = "3895544", "4333158"
NEW_ID = "d31648b1-b3f3-4c55-9d63-e8b112a498ed"
NEW_NAME = "Bruno Silva (ESPN 4333158)"


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    base = tmp_path_factory.mktemp("pg")
    data = base / "data"
    sock = Path(tempfile.mkdtemp(prefix="pgi", dir="/tmp"))  # a Unix socket path is limited to ~100 bytes
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
        admin.cursor().execute("CREATE DATABASE tpl")
        conn = psycopg2.connect(host=str(sock), user="postgres", dbname="tpl")
        conn.autocommit = True
        for f in sorted(MIGRATIONS.glob("*.sql")):
            conn.cursor().execute(f.read_text())
        conn.close()
        yield {"host": str(sock), "admin": admin}
        admin.close()
    finally:
        subprocess.run(["pg_ctl", "-D", str(data), "-m", "immediate", "stop"], capture_output=True)
        shutil.rmtree(sock, ignore_errors=True)


@pytest.fixture
def dsn(cluster):
    name = f"t{uuid.uuid4().hex[:12]}"
    cluster["admin"].cursor().execute(f"CREATE DATABASE {name} TEMPLATE tpl")
    yield {"host": cluster["host"], "user": "postgres", "dbname": name}
    cluster["admin"].cursor().execute(f"DROP DATABASE {name} WITH (FORCE)")


@pytest.fixture
def conn(dsn):
    c = psycopg2.connect(**dsn)
    yield c
    c.rollback()
    c.close()


@pytest.fixture(autouse=True)
def identity_stats(monkeypatch):
    for key in list(fu.IDENTITY_STATS):
        monkeypatch.setitem(fu.IDENTITY_STATS, key, 0)


def _cur(conn):
    return conn.cursor(cursor_factory=RealDictCursor)


def _fighter(cur, name, espn=None, aliases=None):
    cur.execute(
        "INSERT INTO teams (name, normalized_name, sport, external_ids, metadata) "
        "VALUES (%s, %s, 'mma', %s::jsonb, %s::jsonb) RETURNING id::text AS id",
        (
            name,
            fu.normalize_team(name),
            json.dumps({"espn": espn} if espn else {}),
            json.dumps({"aliases": aliases} if aliases else {}),
        ),
    )
    return cur.fetchone()["id"]


def _bout(cur, league, home, away, kickoff, status="finished"):
    cur.execute(
        """INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, status, season,
                               home_score, away_score)
           VALUES (%s, %s, %s, %s, %s, '2024', 1, 0) RETURNING id::text AS id""",
        (league, home, away, kickoff, status),
    )
    return cur.fetchone()["id"]


def _pred(cur, match_id):
    cur.execute(
        """INSERT INTO predictions (match_id, model_name, model_version, prediction_type, predicted_outcome,
                                    confidence, probabilities)
           VALUES (%s, 'ensemble_mma_ml', 'v1', 'moneyline', 'home', 0.6, '{"home": 0.6, "away": 0.4}'::jsonb)
           RETURNING id::text AS id""",
        (match_id,),
    )
    return cur.fetchone()["id"]


def _rec(cur, match_id):
    pid = _pred(cur, match_id)
    cur.execute(
        """INSERT INTO betting_recommendations (prediction_id, match_id, bet_type, selection, odds_at_recommendation,
                                                bookmaker, recommended_stake, status)
           VALUES (%s, %s, 'moneyline', 'home', 2.5, 'b', 10, 'won')""",
        (pid, match_id),
    )


def _world(conn) -> dict:
    with _cur(conn) as cur:
        league = str(fu.ensure_league(cur, MMA, "UFC", "UFC", "USA"))
        w = {"league": league}
        w["bruno"] = _fighter(cur, "Bruno Silva")
        for key in ("weidman", "durden", "naurdiev", "aslan", "opp"):
            w[key] = _fighter(cur, key.title())
        w["raf"] = _fighter(cur, "Rafael Cerqueira")
        w["jose"] = _fighter(cur, "José Luiz")
        w["zach"] = _fighter(cur, "Zach Reese", espn="5143223", aliases=["Zachary Reese"])
        w["garry"] = _fighter(cur, "Ian Machado Garry")
        w["m1"] = _bout(cur, league, w["weidman"], w["bruno"], CARD)  # Blindado: moves
        w["m2"] = _bout(cur, league, w["bruno"], w["durden"], CARD + timedelta(days=100))  # Bulldog: stays
        w["m3"] = _bout(cur, league, w["bruno"], w["naurdiev"], CARD + timedelta(days=150))  # Blindado: moves
        w["m4"] = _bout(cur, league, w["aslan"], w["raf"], CARD + timedelta(days=150))
        w["m5"] = _bout(cur, league, w["zach"], w["garry"], CARD + timedelta(days=170))
        w["pred1"] = _pred(cur, w["m1"])
    conn.commit()
    bouts = (
        mic.SplitBout(w["m1"], CARD, "finished", "away", "Weidman", "1", MOVE, "move"),
        mic.SplitBout(w["m2"], CARD + timedelta(days=100), "finished", "home", "Durden", "2", KEEP, "keep"),
        mic.SplitBout(w["m3"], CARD + timedelta(days=150), "finished", "home", "Naurdiev", "3", MOVE, "move"),
    )
    w["lists"] = mic.Lists(
        mic.SplitList(w["bruno"], "Bruno Silva", NEW_ID, NEW_NAME, KEEP, MOVE, bouts),
        (mic.AliasRow(w["raf"], "Rafael Cerqueira", "Raffael Cerqueira", "5212598"),),
        (mic.OrphanRow(w["jose"], "José Luiz"),),
        (
            mic.IdRow(w["raf"], "Rafael Cerqueira", "5212598"),
            mic.IdRow(w["zach"], "Zach Reese", "5143223"),
            mic.IdRow(w["garry"], "Ian Machado Garry", "4245133"),
            mic.IdRow(w["aslan"], "Aslan", "5100001"),
        ),
    )
    return w


def _run(conn, w, tmp_path, apply=False, expect=None, resolvers=None, tag="x", archive=None):
    res, why = mic.load_resolvers()
    assert why is None, why
    plan_out = tmp_path / f"{tag}.plan.json"
    rc = mic.run(
        conn,
        apply,
        w["lists"],
        tmp_path / f"{tag}.csv",
        archive or tmp_path / f"{tag}.jsonl.gz",
        resolvers or res,
        plan_out=plan_out,
        expect_plan=expect,
    )
    return rc, json.loads(plan_out.read_text())


def _one(conn, sql, *args):
    with _cur(conn) as cur:
        cur.execute(sql, args)
        row = cur.fetchone()
    conn.rollback()
    return row


def _fingerprint(conn):
    with _cur(conn) as cur:
        cur.execute("SELECT id::text, name, external_ids, metadata FROM teams ORDER BY id")
        teams = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT id::text, home_team_id::text, away_team_id::text, status FROM matches ORDER BY id")
        matches = [dict(r) for r in cur.fetchall()]
    conn.rollback()
    return json.dumps([teams, matches], sort_keys=True, default=str)


def test_a_pinned_apply_splits_aliases_deletes_stamps_and_resolves(conn, tmp_path):
    w = _world(conn)
    before = _fingerprint(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    assert rc == 0, dry
    assert _fingerprint(conn) == before  # the dry run wrote nothing
    assert mic.plan_line(dry["summary"]) == (
        "PLAN split=1 split_done=0 split_refuse=0 bouts_moved=2 predictions_on_moved=1 recs_on_moved=0 "
        "alias_add=1 alias_done=0 alias_refuse=0 orphan_delete=1 orphan_done=0 orphan_refuse=0 "
        "ids_stamp=3 ids_done=1 ids_refuse=0"
    )
    rc, applied = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 0 and applied["digest"] == dry["digest"]

    n = _one(conn, "SELECT name, external_ids, metadata FROM teams WHERE id = %s", NEW_ID)
    assert n["name"] == NEW_NAME and n["external_ids"] == {"espn": MOVE}
    assert n["metadata"]["aliases"] == ["Bruno Silva"] and n["metadata"]["split_from"]["team_id"] == w["bruno"]
    t = _one(conn, "SELECT external_ids, metadata FROM teams WHERE id = %s", w["bruno"])
    assert t["external_ids"] == {"espn": KEEP} and t["metadata"]["split_into"][0]["team_id"] == NEW_ID
    assert _one(conn, "SELECT away_team_id::text AS a FROM matches WHERE id = %s", w["m1"])["a"] == NEW_ID
    assert _one(conn, "SELECT home_team_id::text AS h FROM matches WHERE id = %s", w["m3"])["h"] == NEW_ID
    assert _one(conn, "SELECT home_team_id::text AS h FROM matches WHERE id = %s", w["m2"])["h"] == w["bruno"]
    assert _one(conn, "SELECT match_id::text AS m FROM predictions WHERE id = %s", w["pred1"])["m"] == w["m1"]
    raf = _one(conn, "SELECT external_ids, metadata FROM teams WHERE id = %s", w["raf"])
    assert raf["external_ids"] == {"espn": "5212598"} and raf["metadata"]["aliases"] == ["Raffael Cerqueira"]
    assert _one(conn, "SELECT count(*) AS n FROM teams WHERE id = %s", w["jose"])["n"] == 0
    assert _one(conn, "SELECT external_ids FROM teams WHERE id = %s", w["garry"])["external_ids"] == {"espn": "4245133"}

    # the archive: every pre-image (split row, aliased/orphan/stamped rows, the moved bouts) and the inserted row
    with gzip.open(tmp_path / "apply.jsonl.gz", "rt") as fh:
        recs = [json.loads(line) for line in fh]
    pre_teams = {r["row"]["id"] for r in recs if r["table"] == "teams" and r["role"] == "pre_image"}
    assert pre_teams == {w["bruno"], w["raf"], w["jose"], w["garry"], w["aslan"]}
    assert {r["row"]["id"] for r in recs if r["table"] == "matches"} == {w["m1"], w["m3"]}
    assert [r["row"]["id"] for r in recs if r["role"] == "inserted"] == [NEW_ID]
    jose = next(r["row"] for r in recs if r["row"]["id"] == w["jose"])
    assert jose["name"] == "José Luiz" and jose["external_ids"] == {}

    res, _ = mic.load_resolvers()
    with _cur(conn) as cur:
        assert mic.after_state_ok(mic.check_after(cur, dry, res))
    conn.rollback()
    rc, again = _run(conn, w, tmp_path, tag="again")
    assert rc == 0 and mic.plan_line(again["summary"]) == (
        "PLAN split=0 split_done=1 split_refuse=0 bouts_moved=0 predictions_on_moved=0 recs_on_moved=0 "
        "alias_add=0 alias_done=1 alias_refuse=0 orphan_delete=0 orphan_done=1 orphan_refuse=0 "
        "ids_stamp=0 ids_done=4 ids_refuse=0"
    )

    # the deployed resolver: both Bruno Silvas by athlete id, a third one refused, the alias and the stamps
    with _cur(conn) as cur:
        assert fu.ensure_team(cur, MMA, "Bruno Silva", w["league"], espn_id=KEEP) == w["bruno"]
        assert fu.ensure_team(cur, MMA, "Bruno Silva", w["league"], espn_id=MOVE) == NEW_ID
        assert fu.ensure_team(cur, MMA, "Bruno Silva", w["league"], espn_id="7777777") is None
        assert fu.IDENTITY_STATS["team_conflicts"] == 1
        assert fu.ensure_team(cur, MMA, "Raffael Cerqueira", w["league"], espn_id="5212598") == w["raf"]
        assert fu.ensure_team(cur, MMA, "Ian Garry", w["league"], espn_id="4245133") == w["garry"]
        assert fu.ensure_team(cur, MMA, "José Luiz", w["league"], espn_id="3097880") not in (None, w["jose"])
    conn.rollback()


def test_a_row_a_sweep_stamped_after_the_dry_run_does_not_stop_the_apply(conn, tmp_path):
    w = _world(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    assert rc == 0
    with _cur(conn) as cur:  # the 15-minute sweep finds Garry by name and stamps him
        assert fu.ensure_team(cur, MMA, "Ian Machado Garry", w["league"], espn_id="4245133") == w["garry"]
    conn.commit()
    rc, applied = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 0 and applied["summary"]["ids_stamp"] == 2 and applied["summary"]["ids_done"] == 2


def test_an_apply_whose_plan_changed_refuses_with_nothing_written(conn, tmp_path):
    w = _world(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    with _cur(conn) as cur:
        cur.execute("UPDATE matches SET status = 'cancelled' WHERE id = %s", (w["m3"],))
    conn.commit()
    before = _fingerprint(conn)
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 2 and _fingerprint(conn) == before
    assert not (tmp_path / "apply.jsonl.gz").exists()


def test_a_rec_on_a_bout_to_move_refuses_the_split(conn, tmp_path):
    w = _world(conn)
    with _cur(conn) as cur:
        _rec(cur, w["m3"])
    conn.commit()
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    assert rc == 1 and dry["split"]["action"] == "refuse" and dry["summary"]["recs_on_moved"] == 1
    before = _fingerprint(conn)
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 2 and _fingerprint(conn) == before


def test_an_in_transaction_resolver_miss_rolls_everything_back(conn, tmp_path):
    w = _world(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    res, _ = mic.load_resolvers()
    real = res.find_team

    def blind_to_the_new_row(cur, name, espn):  # the resolver misses only the moved athlete
        return None if espn == MOVE else real(cur, name, espn)

    before = _fingerprint(conn)
    rc, _ = _run(
        conn,
        w,
        tmp_path,
        apply=True,
        expect=dry,
        tag="apply",
        resolvers=dataclasses.replace(res, find_team=blind_to_the_new_row),
    )
    assert rc == 1 and _fingerprint(conn) == before


def test_an_existing_archive_stops_the_apply_before_any_write(conn, tmp_path):
    w = _world(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    archive = tmp_path / "taken.jsonl.gz"
    archive.write_bytes(b"x")
    before = _fingerprint(conn)
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply", archive=archive)
    assert rc == 1 and _fingerprint(conn) == before and archive.read_bytes() == b"x"


@pytest.mark.parametrize("change", ["bout", "orphan", "stamped", "alias"])
def test_the_sql_guards_stop_a_write_on_a_changed_row(conn, change):
    w = _world(conn)
    res, _ = mic.load_resolvers()
    with _cur(conn) as cur:
        state = mic.load_state(cur, w["lists"])
        plan = mic.plan_run(w["lists"], state, res)
        if change == "bout":
            cur.execute("UPDATE matches SET status = 'cancelled' WHERE id = %s", (w["m3"],))
        elif change == "orphan":
            _bout(cur, w["league"], w["jose"], w["opp"], CARD + timedelta(days=400), "scheduled")
        elif change == "stamped":
            cur.execute('UPDATE teams SET external_ids = \'{"espn": "1"}\'::jsonb WHERE id = %s', (w["garry"],))
        elif change == "alias":
            cur.execute(
                'UPDATE teams SET metadata = \'{"aliases": ["R. Cerqueira"]}\'::jsonb WHERE id = %s', (w["raf"],)
            )
        with pytest.raises(mic.GuardError, match="touched 0 row"):
            mic.apply_writes(cur, plan, w["lists"], state, res)
    conn.rollback()


def test_check_after_counts_a_row_that_lost_its_athlete_id(conn, tmp_path):
    w = _world(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 0
    with _cur(conn) as cur:
        cur.execute("UPDATE teams SET external_ids = '{}'::jsonb WHERE id = %s", (w["garry"],))
    conn.commit()
    res, _ = mic.load_resolvers()
    with _cur(conn) as cur:
        out = mic.check_after(cur, dry, res)
    conn.rollback()
    assert out["ids_missing"] == 1 and not mic.after_state_ok(out)
    assert mic.run_check_after(conn, tmp_path / "dry.plan.json", res, None) == 1


def test_a_team_write_in_flight_fails_the_apply_with_nothing_written(conn, dsn, tmp_path, monkeypatch):
    """The apply takes SHARE ROW EXCLUSIVE on teams before it reads: a sweep holding a team row (an uncommitted
    stamp) makes it give up after lock_timeout with nothing written, instead of planning around a write it cannot
    see."""
    w = _world(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    monkeypatch.setattr(mic, "LOCK_TIMEOUT", "300ms")
    res, _ = mic.load_resolvers()
    sweep = psycopg2.connect(**dsn)
    try:
        with sweep.cursor() as cur:
            cur.execute("UPDATE teams SET updated_at = now() WHERE id = %s", (w["garry"],))
        before = _fingerprint(conn)
        rc = mic.run(
            conn,
            True,
            w["lists"],
            tmp_path / "a.csv",
            tmp_path / "a.jsonl.gz",
            res,
            plan_out=tmp_path / "a.json",
            expect_plan=dry,
        )
    finally:
        sweep.rollback()
        sweep.close()
    assert rc == 2 and _fingerprint(conn) == before
    assert not (tmp_path / "a.jsonl.gz").exists() and not (tmp_path / "a.json").exists()
