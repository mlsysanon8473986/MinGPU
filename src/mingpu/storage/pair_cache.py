"""The cache of pairs: each pair is searched once and stored by content.

A pair (application, system, GPU count, batch, datatype, restriction set,
Calculon commit) is stored under the SHA-256 of those inputs (:func:`pair_key`). A search
links to the pairs it uses, so a later search, or the same one run again after an interruption,
reuses them.

These functions never commit or roll back: the caller owns the transaction
(:mod:`mingpu.planner.execute_search.outer_search` writes each pair in one).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import duckdb
from calculon.llm import Llm

from .db import fetch_dicts

if TYPE_CHECKING:
    from ..planner.execute_search.inner_search import InnerSearchResult

#: Calculon's execution fields, stored as columns of ``inner_search_pair``.
EXECUTION_FIELDS: tuple[str, ...] = tuple(Llm.Execution.fields())
#: Calculon statistics stored as columns of ``inner_search_pair`` (and all of them in ``stats``).
EXECUTION_METRICS: tuple[str, ...] = (
    "sample_rate",
    "total_time",
    "total_efficiency",
    "compute_efficiency",
    "system_efficiency",
    "bubble_time",
    "fw_time",
    "bw_time",
    "optim_step_time",
    "recompute_time",
    "tp_comm_exposed_time",
    "pp_comm_exposed_time",
    "dp_comm_exposed_time",
    "weight_space",
    "act_space",
    "proc_mem_tier1_cap_req",
    "useful_flops",
)


def canonical_json(payload: Any) -> str:
    """Deterministic JSON encoding used for hashing and storage."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sha256_of(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def pair_key(
    application: dict[str, Any],
    system: dict[str, Any],
    n_gpu: int,
    b_max: int,
    datatype: str,
    restrictions: dict[str, Any],
    calculon_commit: str | None,
) -> str:
    """Content key of a pair: the SHA-256 of every input its result depends on."""
    return sha256_of(
        {
            "application": application,
            "system": system,
            "n_gpu": n_gpu,
            "b_max": b_max,
            "datatype": datatype,
            "restrictions": restrictions,
            "calculon_commit": calculon_commit,
        }
    )


def find_pair(conn: duckdb.DuckDBPyConnection, key: str) -> dict[str, Any] | None:
    """The cached pair with this key, or None."""
    rows = fetch_dicts(conn.execute("SELECT id, top_k FROM cache_pairs WHERE sha256 = ?", [key]))
    return rows[0] if rows else None


def covers(stored_top_k: int | None, wanted_top_k: int | None) -> bool:
    """Whether a pair stored with ``stored_top_k`` executions serves ``wanted_top_k``."""
    if stored_top_k is None:
        return True
    return wanted_top_k is not None and stored_top_k >= wanted_top_k


def save_pair(
    conn: duckdb.DuckDBPyConnection,
    key: str,
    result: InnerSearchResult,
    *,
    b_max: int,
    datatype: str,
    restrictions: dict[str, Any],
    calculon_commit: str | None,
    top_k: int | None,
) -> int:
    """Store a searched pair under ``key``, replacing the executions of a stored one.

    A pair searched again has the same inputs, hence the same counts; only its
    executions, ``top_k``, ``valid`` and time change. (DuckDB cannot update the
    list columns of a row that other tables reference, so they are checked, not
    rewritten.)
    """
    count = result.count
    old = find_pair(conn, key)
    values = [
        count.generated,
        list(count.steps),
        list(count.left_after),
        count.kept,
        result.evaluated,
        result.valid,
        top_k,
        result.seconds,
    ]
    if old is None:
        pair_id = conn.execute(
            """
            INSERT INTO cache_pairs (sha256, n_gpu, b_max, datatype,
                              restrictions, calculon_commit, generated, steps, left_after,
                              kept, evaluated, valid, top_k, seconds)
            VALUES (?, ?, ?, ?, CAST(? AS JSON), ?, ?, CAST(? AS INTEGER[]),
                    CAST(? AS BIGINT[]), ?, ?, ?, ?, ?)
            RETURNING id
            """,
            [
                key,
                count.n_gpu,
                b_max,
                datatype,
                canonical_json(restrictions),
                calculon_commit,
                *values,
            ],
        ).fetchone()[0]
    else:
        pair_id = old["id"]
        stored = conn.execute(
            "SELECT generated, steps, left_after, kept FROM cache_pairs WHERE id = ?", [pair_id]
        ).fetchone()
        if list(stored) != values[:4]:
            raise ValueError(f"pair {pair_id} searched again gives other counts: {stored}")
        conn.execute("DELETE FROM inner_search_pair WHERE pair_id = ?", [pair_id])
        conn.execute(
            "UPDATE cache_pairs SET evaluated = ?, valid = ?, top_k = ?, seconds = ?, "
            "updated_at = ? WHERE id = ?",
            [*values[4:], datetime.now(UTC), pair_id],
        )
    _insert_executions(conn, pair_id, result.executions)
    return pair_id


def _insert_executions(
    conn: duckdb.DuckDBPyConnection, pair_id: int, executions: list[dict[str, Any]]
) -> None:
    if not executions:
        return
    columns = ["pair_id", "rank", *EXECUTION_FIELDS, *EXECUTION_METRICS, "stats"]
    marks = ", ".join(["?"] * (len(columns) - 1) + ["CAST(? AS JSON)"])
    rows = []
    for rank, e in enumerate(executions, start=1):
        stats = {k: v for k, v in e.items() if k not in EXECUTION_FIELDS}
        rows.append(
            [
                pair_id,
                rank,
                *(e.get(f) for f in EXECUTION_FIELDS),
                *(e.get(m) for m in EXECUTION_METRICS),
                json.dumps(stats),
            ]
        )
    conn.executemany(
        f"INSERT INTO inner_search_pair ({', '.join(columns)}) VALUES ({marks})",
        rows,
    )


def link_pair(
    conn: duckdb.DuckDBPyConnection,
    search_id: str,
    layers: int,
    n_gpu: int,
    pair_id: int,
    *,
    reused: bool,
) -> None:
    conn.execute(
        "INSERT INTO search_pair (search_id, layers, n_gpu, pair_id, reused) "
        "VALUES (CAST(? AS UUID), ?, ?, ?, ?)",
        [search_id, layers, n_gpu, pair_id, reused],
    )
