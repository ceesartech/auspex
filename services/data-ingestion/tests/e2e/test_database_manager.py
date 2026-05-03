"""Integration: DatabaseManager round-trips against real Postgres."""

import uuid
from datetime import datetime, timezone


def test_execute_query_insert_and_fetch(db_manager):
    """A plain INSERT followed by SELECT returns the row we just wrote."""
    name = f"League-{uuid.uuid4()}"
    db_manager.execute_query(
        "INSERT INTO leagues (name, country, sport) VALUES (%s, %s, %s)",
        (name, "Testland", "soccer"),
    )

    rows = db_manager.execute_query(
        "SELECT name, country, sport FROM leagues WHERE name = %s",
        (name,),
        fetch=True,
    )
    assert len(rows) == 1
    assert rows[0]["name"] == name
    assert rows[0]["country"] == "Testland"


def test_upsert_inserts_new_row(db_manager):
    name = f"League-{uuid.uuid4()}"
    rows = db_manager.upsert(
        "leagues",
        {"name": name, "country": "Spain", "sport": "soccer"},
        conflict_columns=["name", "country", "sport"],
        update_columns=["country"],
    )
    assert len(rows) == 1
    assert rows[0]["country"] == "Spain"


def test_upsert_updates_on_conflict(db_manager):
    name = f"League-{uuid.uuid4()}"
    db_manager.execute_query(
        "INSERT INTO leagues (name, country, sport, tier) VALUES (%s, %s, %s, %s)",
        (name, "Italy", "soccer", 1),
    )

    rows = db_manager.upsert(
        "leagues",
        {"name": name, "country": "Italy", "sport": "soccer", "tier": 2},
        conflict_columns=["name", "country", "sport"],
        update_columns=["tier"],
    )
    assert len(rows) == 1
    assert rows[0]["tier"] == 2

    fresh = db_manager.execute_query(
        "SELECT tier FROM leagues WHERE name = %s",
        (name,),
        fetch=True,
    )
    assert fresh[0]["tier"] == 2


def test_transaction_rollback_on_error(db_manager):
    """A failing query inside get_connection() rolls back, leaving no rows."""
    import psycopg2

    name = f"League-{uuid.uuid4()}"
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO leagues (name, country, sport) VALUES (%s, %s, %s)",
                    (name, "Germany", "soccer"),
                )
                # Force an error: insert into a nonexistent table.
                cursor.execute("INSERT INTO no_such_table VALUES (1)")
    except psycopg2.errors.UndefinedTable:
        pass

    rows = db_manager.execute_query(
        "SELECT 1 FROM leagues WHERE name = %s",
        (name,),
        fetch=True,
    )
    assert rows == []


def test_insert_many_bulk_writes(db_manager):
    base = uuid.uuid4().hex[:8]
    rows = [(f"BulkLeague-{base}-{i}", "Bulkland", "soccer") for i in range(5)]
    inserted = db_manager.insert_many("leagues", ["name", "country", "sport"], rows)
    assert inserted == 5

    fetched = db_manager.execute_query(
        "SELECT COUNT(*) AS n FROM leagues WHERE country = 'Bulkland'",
        fetch=True,
    )
    assert fetched[0]["n"] == 5


def test_match_insertion_with_fk_chain(db_manager):
    """Insert league → teams → match exercises the real FK constraints."""
    league_id = db_manager.execute_query(
        "INSERT INTO leagues (name, country, sport) VALUES (%s, %s, %s) RETURNING id",
        (f"FK-{uuid.uuid4()}", "Wales", "soccer"),
        fetch=True,
    )[0]["id"]

    home = db_manager.execute_query(
        "INSERT INTO teams (name, normalized_name, league_id, country, sport) "
        "VALUES (%s, %s, %s, %s, %s) RETURNING id",
        (f"Home-{uuid.uuid4()}", f"home-{uuid.uuid4()}", league_id, "Wales", "soccer"),
        fetch=True,
    )[0]["id"]

    away = db_manager.execute_query(
        "INSERT INTO teams (name, normalized_name, league_id, country, sport) "
        "VALUES (%s, %s, %s, %s, %s) RETURNING id",
        (f"Away-{uuid.uuid4()}", f"away-{uuid.uuid4()}", league_id, "Wales", "soccer"),
        fetch=True,
    )[0]["id"]

    match_id = db_manager.execute_query(
        "INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, season, status) "
        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
        (league_id, home, away, datetime.now(timezone.utc), "2025-2026", "scheduled"),
        fetch=True,
    )[0]["id"]

    assert match_id is not None
