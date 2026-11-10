"""Inner search: the best parallel plan of one pair (paper, Sec. 4.2).

The inner search fixes the pair, a candidate model and a candidate GPU count, and searches its
parallel plans; the outer search (:mod:`mingpu.planner.execute_search.outer_search`) varies the
pair. A parallel plan is an execution of Calculon; those that pass the filter are the
candidate parallel plans of the pair. For a pair:

1. Calculon's own generator, ``AllExecutions.all_executions``
   (``all_executions.py:88-131``), at the batch ``B_max`` and the datatype;
2. MinGPU's filter (:mod:`mingpu.planner.define_search_space.restrictions`), which counts the plans
   left after each step;
3. Calculon's own evaluator, ``AllExecutions.search``
   (``all_executions.py:190-202``), on the candidate parallel plans, in ordered chunks
   over a ``multiprocessing`` pool, or in this process without one;
4. the feasible candidate parallel plans (those that fit in memory) sorted by sample
   rate, highest first, of which the ``top_k`` are kept; the first is the best parallel plan.

Calculon is not modified. Only the calling process should write the results to
the database; the workers only evaluate.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import calculon
from calculon.llm import AllExecutions, Llm

from ..define_search_space.restrictions import Restrictions

#: Columns of an evaluated execution: Calculon's execution fields, then its statistics.
RESULT_FIELDS: tuple[str, ...] = tuple(Llm.Execution.fields()) + tuple(Llm.get_stats_fields())
N_EXECUTION_FIELDS = len(Llm.Execution.fields())
SAMPLE_RATE = RESULT_FIELDS.index("sample_rate")


def calculon_commit() -> str | None:
    """The commit of the Calculon in use, as 8 characters, or None when it cannot be known.

    The submodule has its own ``.git`` and reports its HEAD. A Calculon without ``.git`` is not
    the submodule, and asking git there would reach a repository above it.
    """
    repo = Path(calculon.__file__).resolve().parents[1]
    if not (repo / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return None
    return out.stdout.strip() or None


@dataclass
class PlanCounts:
    """Parallel plans of one pair: generated, then left after each restriction step."""

    n_gpu: int
    generated: int
    steps: list[int]
    left_after: list[int]  # left after each step of ``steps``, cumulative (column left_after)

    @property
    def kept(self) -> int:
        return self.left_after[-1]


@dataclass
class InnerSearchResult:
    """Outcome of the search of one pair: its counts and its best candidate parallel plans."""

    count: PlanCounts
    evaluated: int
    valid: int
    seconds: float
    executions: list[dict[str, Any]] = field(default_factory=list)


def _calculon_inputs(application: dict[str, Any], system: dict[str, Any]):
    return Llm.Application(application), calculon.System(system)


def _generate_and_filter(
    app: Llm.Application,
    syst: calculon.System,
    n_gpu: int,
    b_max: int,
    datatype: str,
    restrictions: Restrictions,
    *,
    keep: bool = True,
) -> tuple[list[tuple], int, list[int]]:
    """One pass over Calculon's generator through the filter.

    Returns the kept executions (empty when ``keep`` is false), the number
    generated, and the number left after each enabled step.
    """
    steps = restrictions.steps
    left = [0] * len(steps)
    kept: list[tuple] = []
    generated = 0
    for e in AllExecutions.all_executions(app, syst, n_gpu, b_max, datatype, [True]):
        generated += 1
        failed = restrictions.first_failed(e)
        passed = len(steps) if failed is None else steps.index(failed)
        for i in range(passed):
            left[i] += 1
        if failed is None and keep:
            kept.append(e)
    return kept, generated, left


def _lower_priority() -> None:
    import psutil

    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 10)


class WorkerPool:
    """A pool of ``workers`` processes, below normal priority.

    Use it as a context manager and pass it to :func:`inner_search` to evaluate
    many pairs without starting new processes for each.
    """

    def __init__(self, workers: int) -> None:
        if workers < 1:
            raise ValueError(f"workers must be positive: {workers}")
        self.workers = workers
        self._pool = mp.Pool(workers, initializer=_lower_priority)

    def starmap(self, func, iterable):
        return self._pool.starmap(func, iterable)

    def close(self) -> None:
        self._pool.close()
        self._pool.join()

    def __enter__(self) -> WorkerPool:
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self._pool.terminate()
            self._pool.join()


def _evaluate_chunk(app: Llm.Application, syst: calculon.System, chunk: list[tuple]):
    return len(chunk), AllExecutions.search(app, syst, chunk)


def _evaluate_with_calculon(
    app: Llm.Application,
    syst: calculon.System,
    executions: list[tuple],
    *,
    pool: WorkerPool | None = None,
    chunks_per_worker: int = 4,
) -> tuple[int, list[tuple]]:
    """Evaluate ``executions``; return how many were evaluated and the valid rows.

    The rows are execution fields followed by Calculon's statistics, sorted by
    sample rate, highest first (ties keep the generator's order).
    """
    if not executions:
        return 0, []
    if pool is None:
        results = [_evaluate_chunk(app, syst, executions)]
    else:
        n_chunks = min(len(executions), pool.workers * chunks_per_worker)
        size = math.ceil(len(executions) / n_chunks)
        chunks = [executions[i : i + size] for i in range(0, len(executions), size)]
        results = pool.starmap(_evaluate_chunk, [(app, syst, chunk) for chunk in chunks])
    evaluated = sum(n for n, _ in results)
    valid = [row for _, rows in results for row in rows]
    valid.sort(key=lambda row: row[SAMPLE_RATE], reverse=True)
    return evaluated, valid


def count_plans(
    application: dict[str, Any],
    system: dict[str, Any],
    n_gpu: int,
    b_max: int,
    *,
    datatype: str,
    no_tp_overlap: bool = False,
    tensor_par: int | None = None,
) -> PlanCounts:
    """Size of the search space of one pair after each step; nothing is evaluated."""
    app, syst = _calculon_inputs(application, system)
    restrictions = Restrictions(
        app, syst, datatype, no_tp_overlap=no_tp_overlap, tensor_par=tensor_par
    )
    _, generated, left = _generate_and_filter(
        app, syst, n_gpu, b_max, datatype, restrictions, keep=False
    )
    return PlanCounts(n_gpu, generated, restrictions.steps, left)


def inner_search(
    application: dict[str, Any],
    system: dict[str, Any],
    n_gpu: int,
    b_max: int,
    *,
    datatype: str,
    top_k: int | None,
    no_tp_overlap: bool = False,
    tensor_par: int | None = None,
    pool: WorkerPool | None = None,
) -> InnerSearchResult:
    """The search of one pair; ``top_k=None`` keeps every feasible candidate parallel plan.

    ``datatype`` and ``top_k`` have no default here: those of the paper are in
    :class:`~mingpu.planner.define_search_space.request.SearchRequest`.

    Raises ``RuntimeError`` if Calculon did not evaluate exactly the candidate
    parallel plans, or if a returned plan fails the restrictions.
    """
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be positive or None: {top_k}")
    app, syst = _calculon_inputs(application, system)
    restrictions = Restrictions(
        app, syst, datatype, no_tp_overlap=no_tp_overlap, tensor_par=tensor_par
    )
    start = time.perf_counter()
    kept, generated, left = _generate_and_filter(app, syst, n_gpu, b_max, datatype, restrictions)
    count = PlanCounts(n_gpu, generated, restrictions.steps, left)
    evaluated, valid = _evaluate_with_calculon(app, syst, kept, pool=pool)
    if evaluated != len(kept):
        raise RuntimeError(f"Calculon evaluated {evaluated} of {len(kept)} kept executions")
    top = valid if top_k is None else valid[:top_k]
    bad = [row for row in top if not restrictions.keep(row[:N_EXECUTION_FIELDS])]
    if bad:
        raise RuntimeError(f"{len(bad)} returned executions fail the restrictions")
    return InnerSearchResult(
        count=count,
        evaluated=evaluated,
        valid=len(valid),
        seconds=time.perf_counter() - start,
        executions=[_as_dict(row) for row in top],
    )


def _as_dict(row: Iterable[Any]) -> dict[str, Any]:
    """An evaluated row as ``{field: value}`` over :data:`RESULT_FIELDS`."""
    return dict(zip(RESULT_FIELDS, row, strict=True))
