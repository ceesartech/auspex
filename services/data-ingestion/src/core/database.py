"""Database connection and utilities"""

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from psycopg2 import pool
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import logging
from .config import ScraperConfig

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manage database connections and operations"""

    def __init__(self, config: ScraperConfig):
        self.config = config
        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=config.database_url
        )
        logger.info("Database connection pool created")

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        conn = self.connection_pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            self.connection_pool.putconn(conn)

    def execute_query(self, query: str, params: tuple = None, fetch: bool = False):
        """Execute a query and optionally fetch results"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                if fetch:
                    return cursor.fetchall()
                return cursor.rowcount

    def insert_many(self, table: str, columns: List[str], values: List[tuple]):
        """Bulk insert with execute_values for performance"""
        if not values:
            return 0

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                query = f"""
                    INSERT INTO {table} ({', '.join(columns)})
                    VALUES %s
                    ON CONFLICT DO NOTHING
                """
                execute_values(cursor, query, values)
                return cursor.rowcount

    def upsert(
        self,
        table: str,
        data: Dict[str, Any],
        conflict_columns: List[str],
        update_columns: Optional[List[str]] = None
    ):
        """Insert or update on conflict"""
        columns = list(data.keys())
        values = [data[col] for col in columns]

        if update_columns is None:
            update_columns = [col for col in columns if col not in conflict_columns]

        update_clause = ', '.join([f"{col} = EXCLUDED.{col}" for col in update_columns])

        query = f"""
            INSERT INTO {table} ({', '.join(columns)})
            VALUES ({', '.join(['%s'] * len(columns))})
            ON CONFLICT ({', '.join(conflict_columns)})
            DO UPDATE SET {update_clause}
            RETURNING *
        """

        return self.execute_query(query, tuple(values), fetch=True)

    def get_or_create_team(self, name: str, league_id: int, external_id: str = None) -> int:
        """Get team_id or create team if doesn't exist"""
        # Try to find existing team
        query = """
            SELECT team_id FROM teams
            WHERE name = %s AND league_id = %s
        """
        result = self.execute_query(query, (name, league_id), fetch=True)

        if result:
            return result[0]['team_id']

        # Create new team
        insert_query = """
            INSERT INTO teams (name, league_id, external_id)
            VALUES (%s, %s, %s)
            RETURNING team_id
        """
        result = self.execute_query(
            insert_query,
            (name, league_id, external_id),
            fetch=True
        )
        return result[0]['team_id']

    def get_or_create_league(self, name: str, country: str, sport: str, season: str) -> int:
        """Get league_id or create league if doesn't exist"""
        query = """
            SELECT league_id FROM leagues
            WHERE name = %s AND season = %s AND sport = %s
        """
        result = self.execute_query(query, (name, season, sport), fetch=True)

        if result:
            return result[0]['league_id']

        insert_query = """
            INSERT INTO leagues (name, country, sport, season)
            VALUES (%s, %s, %s, %s)
            RETURNING league_id
        """
        result = self.execute_query(
            insert_query,
            (name, country, sport, season),
            fetch=True
        )
        return result[0]['league_id']
