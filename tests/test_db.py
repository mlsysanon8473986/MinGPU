"""Connection ownership: open_db closes only the connections it opened."""

import duckdb
import pytest

from mingpu.storage import db


def test_open_db_closes_only_its_own_connection(conn, tmp_path):
    with db.open_db(conn) as opened:
        assert opened is conn
    assert conn.execute("SELECT 1").fetchone() == (1,)

    with db.open_db(db_path=tmp_path / "x.duckdb") as opened:
        assert opened.execute("SELECT count(*) FROM search").fetchone() == (0,)  # migrated
    with pytest.raises(duckdb.Error):
        opened.execute("SELECT 1")
