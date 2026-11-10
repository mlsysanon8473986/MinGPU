"""A search and its candidate models in the database.

These functions never commit or roll back: the caller owns the transaction.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import duckdb

from .db import fetch_dicts, load_json

#: Allowed values of ``search.status``.
SEARCH_STATUSES: tuple[str, ...] = ("created", "running", "done", "error")


def create_search(
    conn: duckdb.DuckDBPyConnection,
    *,
    space: dict[str, Any],
    system: dict[str, Any],
    f_peak: float,
    restrictions: dict[str, Any],
    request: dict[str, Any] | None = None,
    mingpu_version: str | None = None,
    calculon_commit: str | None = None,
) -> str:
    """Insert a search from its space (``SearchSpace.to_json``); return its id.

    The column ``labels`` keeps its default, an empty list: labels are no longer set.
    """
    search_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO search (id, target_loss, law, seq_len, b_max, vocab, datatype,
                            system, f_peak, restrictions, top_k, layers, n_gpus, space,
                            request, mingpu_version, calculon_commit)
        VALUES (CAST(? AS UUID), ?, CAST(? AS JSON), ?, ?, ?, ?,
                CAST(? AS JSON), ?, CAST(? AS JSON), ?, CAST(? AS INTEGER[]),
                CAST(? AS INTEGER[]), CAST(? AS JSON), CAST(? AS JSON), ?, ?)
        """,
        [
            search_id,
            space["target_loss"],
            json.dumps(space["law"]),
            space["seq_len"],
            space["b_max"],
            space["vocab"],
            space["datatype"],
            json.dumps(system),
            f_peak,
            json.dumps(restrictions),
            space["top_k"],
            list(space["layers"]),
            list(space["n_gpus"]),
            json.dumps(space),
            None if request is None else json.dumps(request),
            mingpu_version,
            calculon_commit,
        ],
    )
    return search_id


def set_status(
    conn: duckdb.DuckDBPyConnection, search_id: str, status: str, error: str | None = None
) -> None:
    if status not in SEARCH_STATUSES:
        raise ValueError(f"unknown search status {status!r}")
    conn.execute(
        "UPDATE search SET status = ?, error = ?, updated_at = ? WHERE id = CAST(? AS UUID)",
        [status, error, datetime.now(UTC), search_id],
    )


def add_model(
    conn: duckdb.DuckDBPyConnection,
    search_id: str,
    *,
    layers: int,
    application: dict[str, Any],
    n_params: int,
    d_req: float | None,
    c_req: float | None,
) -> None:
    conn.execute(
        "INSERT INTO candidate_models VALUES (CAST(? AS UUID), ?, CAST(? AS JSON), ?, ?, ?)",
        [search_id, layers, json.dumps(application), n_params, d_req, c_req],
    )


def get_search(conn: duckdb.DuckDBPyConnection, search_id: str) -> dict[str, Any] | None:
    """The search row, with its JSON columns decoded, or None."""
    rows = fetch_dicts(conn.execute("SELECT * FROM search WHERE id = CAST(? AS UUID)", [search_id]))
    if not rows:
        return None
    row = rows[0]
    for key in ("law", "system", "restrictions", "space", "request"):
        row[key] = load_json(row[key])
    row["id"] = str(row["id"])
    return row


def get_models(conn: duckdb.DuckDBPyConnection, search_id: str) -> list[dict[str, Any]]:
    """The candidate models of the search, by layers, with their application decoded."""
    rows = fetch_dicts(
        conn.execute(
            "SELECT * FROM candidate_models WHERE search_id = CAST(? AS UUID) ORDER BY layers",
            [search_id],
        )
    )
    for row in rows:
        row["application"] = load_json(row["application"])
    return rows


def list_searches(conn: duckdb.DuckDBPyConnection, limit: int = 20) -> list[dict[str, Any]]:
    """The latest searches with their progress and the objective they were made for."""
    rows = fetch_dicts(
        conn.execute(
            """
            SELECT e.*, s.request->>'objective' AS objective
            FROM search_progress e JOIN search s ON s.id = e.id
            ORDER BY e.created_at DESC LIMIT ?
            """,
            [limit],
        )
    )
    for row in rows:
        row["id"] = str(row["id"])
    return rows


def best_per_pair(conn: duckdb.DuckDBPyConnection, search_id: str) -> list[dict[str, Any]]:
    """The best parallel plan of every pair of the search (view ``best_per_pair``)."""
    return fetch_dicts(
        conn.execute(
            "SELECT * FROM best_per_pair WHERE search_id = CAST(? AS UUID) ORDER BY layers, n_gpu",
            [search_id],
        )
    )
