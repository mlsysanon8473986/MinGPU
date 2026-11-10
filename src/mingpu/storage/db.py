"""Database connection, schema migrations and transactions."""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb


def _default_migrations_dir() -> Path:
    """``$MINGPU_DB_MIGRATIONS``, or the ``migrations/`` directory at the root of the repository."""
    env = os.environ.get("MINGPU_DB_MIGRATIONS")
    return Path(env) if env else Path(__file__).resolve().parents[3] / "migrations"


MIGRATIONS_DIR = _default_migrations_dir()

#: Default database location; override with ``$MINGPU_DB``.
DEFAULT_DB_PATH = Path(os.environ.get("MINGPU_DB", Path.cwd() / "mingpu.duckdb"))


def connect(db_path: str | os.PathLike[str] | None = None) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the DuckDB database at ``db_path``."""
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def _statements(sql: str) -> list[str]:
    """Split a migration script into individual statements.

    Our migrations contain no string literals or comments with semicolons, so a
    simple split is safe and avoids depending on multi-statement ``execute``.
    """
    lines = [line for line in sql.splitlines() if not line.strip().startswith("--")]
    return [stmt.strip() for stmt in "\n".join(lines).split(";") if stmt.strip()]


def migrate(conn: duckdb.DuckDBPyConnection) -> int:
    """Apply pending migrations, returning the resulting schema version."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
          version    INTEGER PRIMARY KEY,
          applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_version").fetchall()}

    version = 0
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        version = int(path.name.split("_", 1)[0])
        if version in applied:
            continue
        for statement in _statements(path.read_text()):
            conn.execute(statement)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", [version])
    return max(applied | {version}, default=0)


@contextmanager
def open_db(
    conn: duckdb.DuckDBPyConnection | None = None,
    db_path: str | os.PathLike[str] | None = None,
) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """A migrated connection: ``conn``, which its caller keeps, or a new one closed on exit."""
    owned = conn is None
    db = connect(db_path) if owned else conn
    try:
        migrate(db)
        yield db
    finally:
        if owned:
            db.close()


@contextmanager
def transaction(conn: duckdb.DuckDBPyConnection) -> Generator[None, None, None]:
    """One transaction: committed at the end, rolled back on any exception, Ctrl+C included."""
    conn.begin()
    try:
        yield
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def fetch_dicts(cursor: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """The rows of ``cursor`` as ``{column: value}``."""
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def load_json(value: Any) -> Any:
    """Normalize a JSON column value (DuckDB may hand back str) into Python."""
    if value is None or isinstance(value, (dict, list, bool, int, float)):
        return value
    return json.loads(value)
