"""scripts/mma_orphan_cleanup.py against a real Postgres (a throwaway cluster
built from services/data-ingestion/db/migrations, one database per test):

  * a dry run writes nothing and reads in a READ ONLY transaction; a pinned
    apply deletes exactly the listed rows in one transaction, archives their
    full pre-images first, leaves every other row (an unlisted bout-less row
    included, which it reports), and --check-after holds; a re-plan is 'done';
  * a listed row that gained a bout, an ESPN id or metadata after the dry run
    stops the apply with nothing written; a reference of any kind (a foreign
    key, another team row's or an MMA match's JSON, a prediction's, feature
    row's, odds row's or rec's JSON, an alias claim) refuses the row;
  * the DELETE's guards: a row changed in any observed column, or referenced
    under any foreign key, after the plan is never deleted (GuardError);
  * a cascade or SET NULL that slipped past the plan and the guards rolls
    everything back (the in-transaction after-state), and so does each other
    gate of it on its own: a listed row still there although teams fell by the
    planned count, and teams falling by more than the planned deletes; an
    archive that already exists, or one that holds fewer rows than the plan
    deletes, stops it before any write;
  * the teams lock: EXCLUSIVE blocks a bout insert (its foreign-key check),
    SHARE ROW EXCLUSIVE would not; the apply waits for an in-flight bout insert
    and refuses on its lock timeout with nothing written;
  * --check-after counts a deleted row that came back.

Skipped where no Postgres server binaries are on PATH (initdb, pg_ctl).
"""

from __future__ import annotations

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
from psycopg2 import errors as pg_errors  # noqa: E402
from psycopg2.extras import RealDictCursor  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO / "scripts"
MIGRATIONS = REPO / "services" / "data-ingestion" / "db" / "migrations"

pytestmark = pytest.mark.skipif(
    not (shutil.which("initdb") and shutil.which("pg_ctl")), reason="no Postgres server binaries on PATH"
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

_spec = importlib.util.spec_from_file_location("mma_orphan_cleanup_pg", SCRIPTS_DIR / "mma_orphan_cleanup.py")
moc = importlib.util.module_from_spec(_spec)
sys.modules["mma_orphan_cleanup_pg"] = moc
_spec.loader.exec_module(moc)

KICKOFF = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=30)


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    base = tmp_path_factory.mktemp("pg")
    data = base / "data"
    sock = Path(tempfile.mkdtemp(prefix="pgo", dir="/tmp"))  # a Unix socket path is limited to ~100 bytes
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


def _cur(conn):
    return conn.cursor(cursor_factory=RealDictCursor)


def _team(cur, name, league=None, sport="mma", espn=None, metadata=None):
    cur.execute(
        "INSERT INTO teams (name, normalized_name, league_id, sport, external_ids, metadata) "
        "VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb) RETURNING id::text AS id",
        (name, name.lower(), league, sport, json.dumps({"espn": espn} if espn else {}), json.dumps(metadata or {})),
    )
    return cur.fetchone()["id"]


def _bout(cur, league, home, away, kickoff=KICKOFF, metadata=None):
    cur.execute(
        """INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, status, season, metadata)
           VALUES (%s, %s, %s, %s, 'finished', '2026', %s::jsonb) RETURNING id::text AS id""",
        (league, home, away, kickoff, json.dumps(metadata or {})),
    )
    return cur.fetchone()["id"]


def _world(conn) -> dict:
    with _cur(conn) as cur:
        cur.execute("INSERT INTO leagues (name, country, sport) VALUES ('UFC', 'USA', 'mma') RETURNING id::text AS id")
        lg = cur.fetchone()["id"]
        w = {"league": lg}
        w["vera"] = _team(cur, "Rodrigo Vera", lg)
        w["tba"] = _team(cur, "TBA", lg)
        w["opp"] = _team(cur, "Opponent TBA", lg)
        w["leka"] = _team(cur, "Jovan Leka", lg, espn="5000001")
        w["poppeck"] = _team(cur, "Alexander Poppeck", lg, espn="5000002")
        w["loner"] = _team(cur, "Unlisted Loner", lg)  # bout-less but not approved: reported, never touched
        w["bout"] = _bout(cur, lg, w["leka"], w["poppeck"])
    conn.commit()
    w["orphans"] = (
        moc.Orphan(w["vera"], "Rodrigo Vera", ""),
        moc.Orphan(w["tba"], "TBA", ""),
        moc.Orphan(w["opp"], "Opponent TBA", ""),
    )
    return w


def _run(conn, w, tmp_path, apply=False, expect=None, tag="x", archive=None):
    plan_out = tmp_path / f"{tag}.plan.json"
    rc = moc.run(
        conn,
        apply,
        w["orphans"],
        tmp_path / f"{tag}.csv",
        archive or tmp_path / f"{tag}.jsonl.gz",
        plan_out=plan_out,
        expect_plan=expect,
    )
    return rc, (json.loads(plan_out.read_text()) if plan_out.exists() else None)


def _exec(conn, sql, *args):
    with _cur(conn) as cur:
        cur.execute(sql, args)
    conn.commit()


def _one(conn, sql, *args):
    with _cur(conn) as cur:
        cur.execute(sql, args)
        row = cur.fetchone()
    conn.rollback()
    return row


def _fingerprint(conn):
    with _cur(conn) as cur:
        cur.execute("SELECT id::text, name, external_ids, metadata, updated_at FROM teams ORDER BY id")
        teams = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT id::text, home_team_id::text, away_team_id::text, status FROM matches ORDER BY id")
        matches = [dict(r) for r in cur.fetchall()]
    conn.rollback()
    return json.dumps([teams, matches], sort_keys=True, default=str)


def test_a_pinned_apply_deletes_exactly_the_listed_rows(conn, tmp_path):
    w = _world(conn)
    before = _fingerprint(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    assert rc == 0 and moc.plan_line(dry["summary"]) == "PLAN listed=3 delete=3 done=0 refuse=0"
    assert _fingerprint(conn) == before  # the dry run wrote nothing
    assert dry["other_boutless"] == [{"id": w["loner"], "name": "Unlisted Loner"}]
    rc, applied = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 0 and applied["digest"] == dry["digest"]
    left = _one(conn, "SELECT count(*) AS n FROM teams WHERE id = ANY(%s::uuid[])", [w["vera"], w["tba"], w["opp"]])
    assert left["n"] == 0
    for key in ("leka", "poppeck", "loner"):
        assert _one(conn, "SELECT count(*) AS n FROM teams WHERE id = %s", w[key])["n"] == 1
    assert _one(conn, "SELECT count(*) AS n FROM matches WHERE id = %s", w["bout"])["n"] == 1

    with gzip.open(tmp_path / "apply.jsonl.gz", "rt") as fh:
        recs = [json.loads(line) for line in fh]
    assert {r["row"]["id"] for r in recs} == {w["vera"], w["tba"], w["opp"]}
    assert all(r["table"] == "teams" and r["role"] == "pre_image" for r in recs)
    vera = next(r["row"] for r in recs if r["row"]["id"] == w["vera"])
    assert vera["name"] == "Rodrigo Vera" and vera["league_id"] == w["league"] and "created_at" in vera

    assert moc.run_check_after(conn, tmp_path / "dry.plan.json") == 0
    rc, again = _run(conn, w, tmp_path, tag="again")
    assert rc == 0 and moc.plan_line(again["summary"]) == "PLAN listed=3 delete=0 done=3 refuse=0"


def test_the_dry_run_reads_in_a_read_only_transaction(conn, tmp_path, monkeypatch):
    w = _world(conn)
    seen = []
    real = moc.load_state

    def spy(cur, orphans):
        cur.execute("SHOW transaction_read_only")
        seen.append(cur.fetchone()["transaction_read_only"])
        return real(cur, orphans)

    monkeypatch.setattr(moc, "load_state", spy)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    assert rc == 0 and seen == ["on"]
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 0 and seen == ["on", "off"]


@pytest.mark.parametrize(
    "change",
    [
        "INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, status, season) "
        "SELECT league_id, id, '{leka}'::uuid, now() + interval '5 days', 'scheduled', '2026' FROM teams "
        "WHERE id = '{vera}'",
        "UPDATE teams SET external_ids = '{{\"espn\": \"5099999\"}}' WHERE id = '{vera}'",
        "UPDATE teams SET metadata = '{{\"aliases\": [\"R. Vera\"]}}' WHERE id = '{vera}'",
        "UPDATE teams SET is_active = false WHERE id = '{vera}'",  # only updated_at and is_active change
    ],
)
def test_a_listed_row_that_changed_after_the_dry_run_stops_the_apply(conn, tmp_path, change):
    w = _world(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    assert rc == 0
    _exec(conn, change.format(**w))
    before = _fingerprint(conn)
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 2 and _fingerprint(conn) == before
    assert not (tmp_path / "apply.jsonl.gz").exists()


@pytest.mark.parametrize(
    "setup, why",
    [
        (
            'UPDATE teams SET metadata = \'{{"merged_from": [{{"team_id": "{vera}"}}]}}\' WHERE id = \'{leka}\'',
            "teams.json",
        ),
        ("UPDATE matches SET metadata = '{{\"replaced\": \"{vera}\"}}' WHERE id = '{bout}'", "matches.json"),
        (
            "INSERT INTO predictions (match_id, model_name, model_version, prediction_type, predicted_outcome, "
            "confidence, probabilities, metadata) VALUES ('{bout}', 'm', 'v', 'moneyline', 'home', 0.5, '{{}}', "
            '\'{{"opponent": "{vera}"}}\')',
            "predictions.json",
        ),
        (
            "INSERT INTO features_cache (match_id, feature_set, features, feature_version) VALUES ('{bout}', 'mma', "
            "'{{\"away_id\": \"{tba}\"}}', 'v1')",
            "features_cache.json",
        ),
        (
            "INSERT INTO odds (match_id, bookmaker, market_type, selection, odds_decimal, metadata) VALUES ('{bout}', "
            "'b', 'moneyline', 'home', 1.9, '{{\"team\": \"{opp}\"}}')",
            "odds.json",
        ),
        (
            "INSERT INTO betting_recommendations (match_id, bet_type, selection, odds_at_recommendation, bookmaker, "
            "risk_factors) VALUES ('{bout}', 'moneyline', 'home', 1.9, 'b', '[\"opponent {vera}\"]')",
            "betting_recommendations.json",
        ),
        ("INSERT INTO players (name, normalized_name, team_id) VALUES ('P', 'p', '{vera}')", "players.team_id"),
        ("INSERT INTO match_stats (match_id, team_id) VALUES ('{bout}', '{tba}')", "match_stats.team_id"),
        ("UPDATE teams SET metadata = '{{\"aliases\": [\"Rodrigo  VÉRA\"]}}' WHERE id = '{poppeck}'", "as an alias"),
    ],
)
def test_a_reference_of_any_kind_refuses_the_row(conn, tmp_path, setup, why):
    w = _world(conn)
    _exec(conn, setup.format(**w))
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    assert rc == 1 and dry["summary"]["refuse"] == 1 and dry["summary"]["delete"] == 2
    (refused,) = [r for r in dry["rows"] if r["action"] == "refuse"]
    assert why in refused["reason"], refused["reason"]
    before = _fingerprint(conn)
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 2 and _fingerprint(conn) == before  # a plan that refuses a row never applies


@pytest.mark.parametrize(
    "change",
    [
        "UPDATE teams SET name = 'Rodrigo Vera ' WHERE id = '{vera}'",
        "UPDATE teams SET normalized_name = 'rodrigo  vera' WHERE id = '{vera}'",
        "UPDATE teams SET league_id = NULL WHERE id = '{vera}'",
        "UPDATE teams SET country = 'MX' WHERE id = '{vera}'",
        "UPDATE teams SET is_active = false WHERE id = '{vera}'",
        "UPDATE teams SET external_ids = '{{\"espn\": \"5099999\"}}' WHERE id = '{vera}'",
        "UPDATE teams SET metadata = '{{\"x\": 1}}' WHERE id = '{vera}'",
        "UPDATE teams SET created_at = created_at - interval '1 second' WHERE id = '{vera}'",
        "INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, status, season) "
        "VALUES ('{league}', '{vera}', '{leka}', now(), 'scheduled', '2026')",
        "INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, status, season) "
        "VALUES ('{league}', '{leka}', '{vera}', now(), 'scheduled', '2026')",
        "INSERT INTO players (name, normalized_name, team_id) VALUES ('P', 'p', '{vera}')",
    ],
)
def test_the_delete_guard_never_deletes_a_row_changed_after_the_plan(conn, change):
    w = _world(conn)
    with _cur(conn) as cur:
        cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        state = moc.load_state(cur, w["orphans"])
    conn.rollback()
    (p,) = [x for x in moc.plan_run(w["orphans"], state) if x.team_id == w["vera"]]
    assert p.action == "delete"
    _exec(conn, change.format(**w))
    with _cur(conn) as cur:
        with pytest.raises(moc.GuardError, match="touched 0 row"):
            moc.mic._expect(cur, moc.delete_sql(state.team_fks), moc.delete_params(p), "delete")
    conn.rollback()
    assert _one(conn, "SELECT count(*) AS n FROM teams WHERE id = %s", w["vera"])["n"] == 1


def test_the_unchanged_row_passes_the_delete_guard(conn):
    w = _world(conn)
    with _cur(conn) as cur:
        state = moc.load_state(cur, w["orphans"])
        (p,) = [x for x in moc.plan_run(w["orphans"], state) if x.team_id == w["vera"]]
        moc.mic._expect(cur, moc.delete_sql(state.team_fks), moc.delete_params(p), "delete")
    conn.rollback()


def test_a_cascade_past_the_plan_and_the_guards_rolls_everything_back(conn, tmp_path, monkeypatch):
    """A reference the plan missed and a DELETE without its foreign-key guards (both simulated): the
    in-transaction after-state sees the bout the cascade took and rolls the whole run back."""
    w = _world(conn)
    _exec(conn, f"UPDATE matches SET away_team_id = '{w['vera']}' WHERE id = '{w['bout']}'")
    real_state, real_sql = moc.load_state, moc.delete_sql

    def blind_state(cur, orphans):
        st = real_state(cur, orphans)
        st.refs = {k: {c: 0 for c in v} for k, v in st.refs.items()}
        return st

    monkeypatch.setattr(moc, "load_state", blind_state)
    monkeypatch.setattr(moc, "delete_sql", lambda fks: real_sql(()))
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    assert rc == 0 and dry["summary"]["delete"] == 3
    before = _fingerprint(conn)
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 1 and _fingerprint(conn) == before
    assert _one(conn, "SELECT count(*) AS n FROM matches WHERE id = %s", w["bout"])["n"] == 1


def test_a_set_null_past_the_plan_and_the_guards_rolls_everything_back(conn, tmp_path, monkeypatch):
    w = _world(conn)
    _exec(conn, f"INSERT INTO players (name, normalized_name, team_id) VALUES ('P', 'p', '{w['tba']}')")
    real_state, real_sql = moc.load_state, moc.delete_sql

    def blind_state(cur, orphans):
        st = real_state(cur, orphans)
        st.refs = {k: {c: 0 for c in v} for k, v in st.refs.items()}
        return st

    monkeypatch.setattr(moc, "load_state", blind_state)
    monkeypatch.setattr(moc, "delete_sql", lambda fks: real_sql(()))
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 1
    assert _one(conn, "SELECT count(*) AS n FROM players WHERE team_id = %s", w["tba"])["n"] == 1


def test_an_existing_archive_stops_it_before_any_write(conn, tmp_path):
    w = _world(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    archive = tmp_path / "taken.jsonl.gz"
    archive.write_text("someone else's")
    before = _fingerprint(conn)
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply", archive=archive)
    assert rc == 1 and _fingerprint(conn) == before and archive.read_text() == "someone else's"


def test_apply_needs_the_dry_runs_plan(conn, tmp_path):
    w = _world(conn)
    before = _fingerprint(conn)
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=None, tag="a1")
    assert rc == 2
    rc, _ = _run(conn, w, tmp_path, apply=True, expect={"tool": "mma_identity_cleanup", "digest": "x"}, tag="a2")
    assert rc == 2 and _fingerprint(conn) == before


def test_a_listed_row_already_gone_is_done(conn, tmp_path):
    w = _world(conn)
    _exec(conn, f"DELETE FROM teams WHERE id = '{w['opp']}'")
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    assert rc == 0 and moc.plan_line(dry["summary"]) == "PLAN listed=3 delete=2 done=1 refuse=0"
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 0 and moc.run_check_after(conn, tmp_path / "dry.plan.json") == 0


def _insert_bout_uncommitted(dsn, home, away, league):
    """Another session's bout insert, left open: its foreign-key check holds FOR KEY SHARE on both teams rows (and
    ROW SHARE on teams) until it ends."""
    other = psycopg2.connect(**dsn)
    with other.cursor() as cur:
        cur.execute(
            "INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, status, season) "
            "VALUES (%s, %s, %s, now() + interval '3 days', 'scheduled', '2026')",
            (league, home, away),
        )
    return other


def test_exclusive_blocks_a_bout_insert_and_share_row_exclusive_would_not(conn, dsn):
    w = _world(conn)
    for mode, blocks in (("EXCLUSIVE", True), ("SHARE ROW EXCLUSIVE", False)):
        with conn.cursor() as cur:
            cur.execute(f"LOCK TABLE teams IN {mode} MODE")
        other = psycopg2.connect(**dsn)
        try:
            with other.cursor() as cur:
                cur.execute("SET lock_timeout = '300ms'")
                if blocks:
                    with pytest.raises(pg_errors.LockNotAvailable):
                        cur.execute(
                            "INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, status, season)"
                            " VALUES (%s, %s, %s, now(), 'scheduled', '2026')",
                            (w["league"], w["leka"], w["poppeck"]),
                        )
                else:
                    cur.execute(
                        "INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, status, season)"
                        " VALUES (%s, %s, %s, now(), 'scheduled', '2026')",
                        (w["league"], w["leka"], w["poppeck"]),
                    )
            other.rollback()
        finally:
            other.close()
        conn.rollback()


def test_the_apply_waits_for_an_in_flight_bout_insert_and_refuses_on_its_lock_timeout(conn, dsn, tmp_path, monkeypatch):
    w = _world(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    monkeypatch.setattr(moc, "LOCK_TIMEOUT", "300ms")
    other = _insert_bout_uncommitted(dsn, w["vera"], w["leka"], w["league"])  # a sweep mid-insert of Vera's bout
    try:
        rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
        assert rc == 2 and not (tmp_path / "apply.jsonl.gz").exists()
    finally:
        other.rollback()
        other.close()
    assert _one(conn, "SELECT count(*) AS n FROM teams WHERE id = %s", w["vera"])["n"] == 1
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply2")  # the sweep rolled back: it applies
    assert rc == 0


def test_check_after_counts_a_deleted_row_that_came_back(conn, tmp_path):
    w = _world(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 0 and moc.run_check_after(conn, tmp_path / "dry.plan.json") == 0
    with gzip.open(tmp_path / "apply.jsonl.gz", "rt") as fh:
        row = next(json.loads(line)["row"] for line in fh if w["tba"] in line)
    _exec(
        conn,
        "INSERT INTO teams (id, name, normalized_name, league_id, sport, external_ids, metadata) "
        "VALUES (%s, %s, %s, %s, 'mma', '{}', '{}')",
        row["id"],
        row["name"],
        row["normalized_name"],
        row["league_id"],
    )
    assert moc.run_check_after(conn, tmp_path / "dry.plan.json") == 1


def test_a_delete_whose_guard_no_longer_matches_stops_at_that_delete(conn, tmp_path, monkeypatch, caplog):
    """Each DELETE must touch exactly one row: one that touches none (the row differs from what the plan saw)
    stops the run right there, and nothing is committed."""
    w = _world(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    real = moc.delete_params

    def stale(p):
        params = real(p)
        if p.team_id == w["tba"]:
            params["updated_at"] = params["updated_at"] - timedelta(seconds=1)
        return params

    monkeypatch.setattr(moc, "delete_params", stale)
    before = _fingerprint(conn)
    with caplog.at_level("ERROR"):
        rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 1 and _fingerprint(conn) == before
    assert "touched 0 row(s), expected 1" in caplog.text and "delete 'TBA'" in caplog.text


def _delete_by_id_only():
    return moc.sql.SQL("DELETE FROM teams t WHERE t.id = %(id)s::uuid RETURNING t.id")


def test_a_listed_row_left_behind_rolls_everything_back(conn, tmp_path, monkeypatch, caplog):
    """The orphans_left gate on its own: Vera's DELETE (simulated faulty) removes the unlisted, unreferenced loner
    instead, so teams falls by exactly the 3 planned and nothing cascades — only 'a listed row is still there'
    catches it."""
    w = _world(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    real = moc.delete_params

    def swapped(p):
        params = real(p)
        if p.team_id == w["vera"]:
            params["id"] = w["loner"]
        return params

    monkeypatch.setattr(moc, "delete_sql", lambda fks: _delete_by_id_only())
    monkeypatch.setattr(moc, "delete_params", swapped)
    before = _fingerprint(conn)
    with caplog.at_level("INFO"):
        rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 1 and _fingerprint(conn) == before
    assert "orphans_left=1 teams_removed=3 cascaded=0" in caplog.text, caplog.text
    assert "1 listed row(s) left" in caplog.text and "WRITTEN" not in caplog.text


def test_teams_falling_by_more_than_the_plan_rolls_everything_back(conn, tmp_path, monkeypatch, caplog):
    """The teams_removed gate on its own: every listed row goes and nothing cascades, but an extra (simulated)
    delete of the unlisted loner inside the transaction makes teams fall by 4, not the 3 planned."""
    w = _world(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    real = moc.check_after

    def with_extra_delete(cur, doc):
        cur.execute("DELETE FROM teams WHERE id = %s", (w["loner"],))
        return real(cur, doc)

    monkeypatch.setattr(moc, "check_after", with_extra_delete)
    before = _fingerprint(conn)
    with caplog.at_level("INFO"):
        rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 1 and _fingerprint(conn) == before
    assert "orphans_left=0 teams_removed=4 cascaded=0" in caplog.text, caplog.text
    assert "teams fell by 4 (planned 3)" in caplog.text and "WRITTEN" not in caplog.text


def test_an_archive_short_of_the_plan_stops_it_before_any_delete(conn, tmp_path, monkeypatch, caplog):
    """A planned delete whose row the archive cannot read (simulated: a phantom row in both the dry run's and the
    apply's plan, so the digests agree) stops the run at the archive, before the first DELETE."""
    w = _world(conn)
    ghost = "00000000-0000-4000-8000-0000000000aa"
    real_plan, real_expect = moc.plan_run, moc.mic._expect
    deletes = []

    def with_ghost(orphans, state):
        plan = real_plan(orphans, state)
        observed = dict(next(p.observed for p in plan if p.action == "delete"), name="Ghost")
        return plan + [moc.RowPlan(ghost, "Ghost", "delete", "phantom", observed)]

    def counting(cur, query, params, what):
        deletes.append(what)
        return real_expect(cur, query, params, what)

    monkeypatch.setattr(moc, "plan_run", with_ghost)
    monkeypatch.setattr(moc.mic, "_expect", counting)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    assert rc == 0 and dry["summary"]["delete"] == 4
    before = _fingerprint(conn)
    with caplog.at_level("ERROR"):
        rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 1 and _fingerprint(conn) == before
    assert "archived 3 row(s), the plan deletes 4" in caplog.text, caplog.text
    assert deletes == []  # no DELETE ran
