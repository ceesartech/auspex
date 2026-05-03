"""Database manager for feature engineering service."""

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from .config import FeatureConfig

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage database connections and operations."""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=config.db_min_connections,
            maxconn=config.db_max_connections,
            dsn=config.database_url,
        )
        logger.info("Database connection pool created")

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
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

    def execute_query(self, query: str, params: tuple = None, fetch: bool = False) -> Any:
        """Execute a query and optionally fetch results."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                if fetch:
                    return cursor.fetchall()
                return cursor.rowcount

    def call_function(self, func_name: str, params: tuple) -> List[Dict]:
        """Call a PostgreSQL function and return results."""
        placeholders = ", ".join(["%s"] * len(params))
        query = f"SELECT * FROM {func_name}({placeholders})"
        return self.execute_query(query, params, fetch=True)

    def insert_many(self, table: str, columns: List[str], values: List[tuple]) -> int:
        """Bulk insert with execute_values for performance."""
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
        update_columns: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Insert or update on conflict."""
        columns = list(data.keys())
        values = [data[col] for col in columns]

        if update_columns is None:
            update_columns = [col for col in columns if col not in conflict_columns]

        update_clause = ", ".join([f"{col} = EXCLUDED.{col}" for col in update_columns])

        query = f"""
            INSERT INTO {table} ({', '.join(columns)})
            VALUES ({', '.join(['%s'] * len(columns))})
            ON CONFLICT ({', '.join(conflict_columns)})
            DO UPDATE SET {update_clause}
            RETURNING *
        """

        return self.execute_query(query, tuple(values), fetch=True)

    def close(self):
        """Close all connections in the pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Database connection pool closed")
