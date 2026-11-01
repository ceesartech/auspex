"""The PENDING query of scripts/ops/backfill_soccer_never_resulted.sh (runbook step 12) against a real Postgres: a
throwaway cluster built from services/data-ingestion/db/migrations, the SQL exactly as the wrapper writes it.

It lists the soccer rows still 'scheduled' 6 h or more after kickoff and short of the never_resulted class (96 h:
repair_fixture_orphans takes a row once its kickoff is strictly more than 96 h before the as-of instant, so a row at
exactly 96 h is still listed here) — the rows no step reaches yet (e.g. Santos v Necaxa if runbook step 1 runs after
the results sweep's look-back has passed it and step 12 before its 96 h). Never a finished row, another sport's, one
inside the 6 h, or a never_resulted one. The query runs read only and ends with PENDING_OK.

Skipped where no Postgres server binaries are on PATH (initdb, pg_ctl).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2")

REPO = Path(__file__).resolve().parents[2]
WRAPPER = REPO / "scripts" / "ops" / "backfill_soccer_never_resulted.sh"
MIGRATIONS = REPO / "services" / "data-ingestion" / "db" / "migrations"

pytestmark = pytest.mark.skipif(
    not (shutil.which("initdb") and shutil.which("pg_ctl")), reason="no Postgres server binaries on PATH"
)

AS_OF = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)


def _pending_sql(as_of: datetime) -> str:
    """The wrapper's heredoc with its two shell variables filled in, as bash would."""
    text = WRAPPER.read_text()
    body = re.search(r'cat > "\$W/never_resulted_pending\.sql" <<SQLEOF\n(.*?)\nSQLEOF\n', text, re.S)
    hours = re.search(r"^PENDING_AFTER_HOURS=(\d+)$", text, re.M)
    assert body and hours
    sql = body.group(1).replace("$PENDING_AFTER_HOURS", hours.group(1)).replace("$at", as_of.strftime("%FT%TZ"))
    assert "$" not in sql
    return sql


@pytest.fixture(scope="module")
def dsn(tmp_path_factory):
    base = tmp_path_factory.mktemp("pg")
    data = base / "data"
    sock = Path(tempfile.mkdtemp(prefix="pgt", dir="/tmp"))  # a Unix socket path is limited to ~100 bytes
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


def _row(cur, sport: str, home: str, away: str, kickoff: datetime, status: str = "scheduled") -> str:
    cur.execute(
        "INSERT INTO leagues (name, country, sport) VALUES (%s, 'X', %s) RETURNING id",
        (f"{sport} league {uuid.uuid4().hex[:6]}", sport),
    )
    league = cur.fetchone()[0]
    ids = []
    for name in (home, away):
        cur.execute(
            "INSERT INTO teams (name, normalized_name, sport) VALUES (%s, %s, %s) RETURNING id",
            (name, f"{name.lower()} {uuid.uuid4().hex[:6]}", sport),
        )
        ids.append(cur.fetchone()[0])
    cur.execute(
        "INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, status, season)"
        " VALUES (%s, %s, %s, %s, %s, '2026') RETURNING id::text",
        (league, ids[0], ids[1], kickoff, status),
    )
    return cur.fetchone()[0]


def _run_as_psql(dsn: dict, sql: str) -> list[str]:
    """Each statement in turn on an autocommit connection (BEGIN READ ONLY ... ROLLBACK as psql runs them); every
    row a SELECT returns, one value per line, as psql -At prints them."""
    out: list[str] = []
    conn = psycopg2.connect(**dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            for stmt in (s.strip() for s in sql.split(";\n")):
                if not stmt:
                    continue
                cur.execute(stmt.rstrip(";"))
                if cur.description:
                    out.extend(str(r[0]) for r in cur.fetchall())
    finally:
        conn.close()
    return out


def test_the_pending_query_lists_only_scheduled_soccer_rows_6_to_96_h_after_kickoff(dsn):
    conn = psycopg2.connect(**dsn)
    with conn, conn.cursor() as cur:
        listed_7h = _row(cur, "soccer", "Santos Laguna", "Necaxa", AS_OF - timedelta(hours=7))
        listed_96h = _row(cur, "soccer", "Guadalajara Chivas", "Atlante", AS_OF - timedelta(hours=96))
        _row(cur, "soccer", "Too", "Recent", AS_OF - timedelta(hours=5, minutes=59))
        _row(cur, "soccer", "Never", "Resulted", AS_OF - timedelta(hours=96, minutes=1))
        _row(cur, "soccer", "Already", "Finished", AS_OF - timedelta(hours=10), status="finished")
        _row(cur, "soccer", "Called", "Off", AS_OF - timedelta(hours=10), status="cancelled")
        _row(cur, "nhl", "Other", "Sport", AS_OF - timedelta(hours=10))
        _row(cur, "soccer", "Not", "Yet", AS_OF + timedelta(hours=2))
    conn.close()

    lines = _run_as_psql(dsn, _pending_sql(AS_OF))
    assert lines[-1] == "PENDING_OK"
    pending = [line for line in lines if line.startswith("PENDING ")]
    assert len(pending) == 2, lines  # ordered by kickoff
    assert re.fullmatch(
        rf"PENDING {listed_96h} kickoff 2026-11-01T12:00Z Guadalajara Chivas v Atlante \(soccer league \w+\)"
        r" - joins the never_resulted class at 2026-11-05T12:00Z",
        pending[0],
    ), pending[0]
    assert re.fullmatch(
        rf"PENDING {listed_7h} kickoff 2026-11-05T05:00Z Santos Laguna v Necaxa \(soccer league \w+\)"
        r" - joins the never_resulted class at 2026-11-09T05:00Z",
        pending[1],
    ), pending[1]


def test_the_pending_query_is_read_only(dsn):
    sql = _pending_sql(AS_OF)
    statements = [s.strip() for s in sql.split(";\n") if s.strip()]
    assert statements[0] == "BEGIN READ ONLY" and statements[-1].rstrip(";") == "ROLLBACK"
    assert all(s.upper().startswith("SELECT") for s in statements[1:-1])
