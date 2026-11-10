"""Outer search: the search over the pairs (candidate model, candidate GPU count).

A search has two levels. The outer search varies the pair: it goes through every pair of
the candidate models and candidate GPU counts of a :class:`SearchSpace`. For each pair, the
inner search (:func:`mingpu.planner.execute_search.inner_search.inner_search`) fixes the
candidate model and the candidate GPU count and searches their parallel plans for the best
one.

:func:`run_outer_search` records the search space and the ``N``, ``D_req`` and ``C_req`` of each
candidate model, then runs the outer search over the models that reach the target loss. Each
pair is written in its own transaction as soon as it is done. A pair already in the cache
(same inputs, enough candidate parallel plans stored) is linked instead of searched again, so
running an interrupted search again continues where it stopped.

A search covers every triple (candidate model, candidate GPU count, candidate parallel plan),
equivalently every pair with its best parallel plan.

The objectives are queries on the stored pairs (``fewest_gpus`` and
``shortest_training`` in ``migrations/0001_schema.sql``), so changing the
window, the GPU range or the MFU tolerance needs no new search.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import duckdb

from ... import __version__
from ...storage.db import open_db, transaction
from ...storage.pair_cache import covers, find_pair, link_pair, pair_key, save_pair
from ...storage.searches import add_model, create_search, get_models, set_status
from ..define_search_space.candidate_models import define_candidate_models
from ..define_search_space.gpu_systems import load_system, peak_flops
from ..define_search_space.restrictions import restriction_settings
from ..define_search_space.search_space import SearchSpace
from .inner_search import InnerSearchResult, WorkerPool, calculon_commit, inner_search


@dataclass
class PairProgress:
    """Reported after each pair: position, the pair, and its result when it was searched."""

    done: int
    total: int
    layers: int
    n_gpu: int
    reused: bool
    result: InnerSearchResult | None


ProgressCallback = Callable[[PairProgress], None]


@dataclass(frozen=True)
class _InputsSharedByEveryPair:
    """What every pair of a search shares."""

    search_id: str
    space: SearchSpace
    system: dict[str, Any]
    restrictions: dict[str, Any]
    commit: str | None


def _record_search_and_candidate_models(
    conn: duckdb.DuckDBPyConnection,
    space: SearchSpace,
    *,
    request: dict[str, Any] | None = None,
) -> _InputsSharedByEveryPair:
    """Record ``space`` and its models; return what its pairs share. Nothing is searched yet."""
    system = load_system(space.gpu)
    commit = calculon_commit()
    candidates = define_candidate_models(space.layers, space.target_loss, space.law, space.vocab)
    restrictions = restriction_settings(system, space.no_tp_overlap, space.tensor_par)
    with transaction(conn):
        search_id = create_search(
            conn,
            space=space.to_json(),
            system=system,
            f_peak=peak_flops(system, space.datatype),
            restrictions=restrictions,
            request=request,
            mingpu_version=__version__,
            calculon_commit=commit,
        )
        for candidate in candidates:
            model = candidate.model
            add_model(
                conn,
                search_id,
                layers=model.layers,
                application=model.application(space.seq_len),
                n_params=model.n_params,
                d_req=candidate.d_req,
                c_req=candidate.c_req,
            )
    return _InputsSharedByEveryPair(search_id, space, system, restrictions, commit)


def _pairs_to_search(
    conn: duckdb.DuckDBPyConnection, ctx: _InputsSharedByEveryPair
) -> list[tuple[dict[str, Any], int]]:
    """Every pair (candidate model, candidate GPU count) of the search.

    A model that cannot reach the target loss has no pairs.
    """
    models = [m for m in get_models(conn, ctx.search_id) if m["d_req"] is not None]
    return [(m, n) for m in models for n in ctx.space.n_gpus]


def _from_cache_or_inner_search(
    conn: duckdb.DuckDBPyConnection,
    ctx: _InputsSharedByEveryPair,
    model: dict[str, Any],
    n_gpu: int,
    pool: WorkerPool | None,
) -> tuple[str, int | None, InnerSearchResult | None]:
    """The key of a pair, and either the id of its cached row or the result of searching it."""
    space = ctx.space
    application = model["application"]
    key = pair_key(
        application, ctx.system, n_gpu, space.b_max, space.datatype, ctx.restrictions, ctx.commit
    )
    cached = find_pair(conn, key)
    if cached is not None and covers(cached["top_k"], space.top_k):
        return key, cached["id"], None
    result = inner_search(
        application,
        ctx.system,
        n_gpu,
        space.b_max,
        datatype=space.datatype,
        no_tp_overlap=space.no_tp_overlap,
        tensor_par=space.tensor_par,
        top_k=space.top_k,
        pool=pool,
    )
    return key, None, result


def _save_and_link_pair(
    conn: duckdb.DuckDBPyConnection,
    ctx: _InputsSharedByEveryPair,
    model: dict[str, Any],
    n_gpu: int,
    key: str,
    cached_id: int | None,
    result: InnerSearchResult | None,
) -> None:
    """Link a pair to the search, saving it first if it was searched, in one transaction."""
    with transaction(conn):
        pair_id = cached_id
        if result is not None:
            pair_id = save_pair(
                conn,
                key,
                result,
                b_max=ctx.space.b_max,
                datatype=ctx.space.datatype,
                restrictions=ctx.restrictions,
                calculon_commit=ctx.commit,
                top_k=ctx.space.top_k,
            )
        link_pair(conn, ctx.search_id, model["layers"], n_gpu, pair_id, reused=result is None)


def _evaluate_every_pair_and_set_status(
    conn: duckdb.DuckDBPyConnection,
    ctx: _InputsSharedByEveryPair,
    *,
    workers: int = 1,
    progress: ProgressCallback | None = None,
) -> None:
    """Search every pair of the search, or take it from the cache; models that cannot reach
    the target loss are skipped. Marks the search ``done``, or ``error`` and re-raises."""
    search_id = ctx.search_id
    pairs = _pairs_to_search(conn, ctx)
    set_status(conn, search_id, "running")
    pool = WorkerPool(workers) if workers > 1 else None
    try:
        for position, (model, n_gpu) in enumerate(pairs, start=1):
            key, cached_id, result = _from_cache_or_inner_search(conn, ctx, model, n_gpu, pool)
            _save_and_link_pair(conn, ctx, model, n_gpu, key, cached_id, result)
            if progress is not None:
                progress(
                    PairProgress(
                        position, len(pairs), model["layers"], n_gpu, result is None, result
                    )
                )
        set_status(conn, search_id, "done")
    except BaseException as exc:
        set_status(conn, search_id, "error", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if pool is not None:
            pool.close()


def run_outer_search(
    space: SearchSpace,
    *,
    conn: duckdb.DuckDBPyConnection | None = None,
    db_path: str | None = None,
    workers: int = 1,
    request: dict[str, Any] | None = None,
    progress: ProgressCallback | None = None,
) -> str:
    """Record ``space`` and search all its pairs; return the search id.

    When ``conn`` is given the caller keeps it; otherwise the database at
    ``db_path`` (or the default) is opened and closed here.
    """
    with open_db(conn, db_path) as db:
        ctx = _record_search_and_candidate_models(db, space, request=request)
        _evaluate_every_pair_and_set_status(db, ctx, workers=workers, progress=progress)
        return ctx.search_id
