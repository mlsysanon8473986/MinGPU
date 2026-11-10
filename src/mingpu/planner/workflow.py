"""The MinGPU workflow: from a request to the answer of its objective.

:func:`run_request` is the top of the method (the workflow figure of the paper): it runs the search
a :class:`~mingpu.planner.define_search_space.request.SearchRequest` defines and applies its
objective.
"""

from __future__ import annotations

import duckdb

from ..storage.db import open_db
from .define_search_space.request import SearchRequest
from .execute_search.outer_search import ProgressCallback, run_outer_search
from .objectives.result import ObjectiveResult, run_objective


def run_request(
    request: SearchRequest,
    *,
    conn: duckdb.DuckDBPyConnection | None = None,
    db_path: str | None = None,
    workers: int = 1,
    progress: ProgressCallback | None = None,
) -> ObjectiveResult:
    """Search what ``request`` defines and apply its objective."""
    with open_db(conn, db_path) as db:
        search_id = run_outer_search(
            request.to_search_space(),
            conn=db,
            workers=workers,
            request=request.to_json(),
            progress=progress,
        )
        return run_objective(db, search_id, **request.objective_settings())
