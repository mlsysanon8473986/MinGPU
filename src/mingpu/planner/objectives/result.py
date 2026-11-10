"""Apply an objective to a stored search.

:func:`run_objective` reads the stored pairs and applies an objective, so another window, GPU
range or tolerance needs no new search. The rules of the objectives are in
:mod:`mingpu.planner.objectives.rules`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import duckdb

from ...storage.searches import best_per_pair, get_models
from .rules import (
    DEFAULT_DELTA,
    Answer,
    Objective,
    fewest_gpus,
    fewest_gpus_per_model,
    shortest_training,
    shortest_training_per_model,
)


@dataclass
class ObjectiveResult:
    """The answer over all candidate models, and that of each one alone (None: no feasible plan)."""

    search_id: str
    objective: Objective
    settings: dict[str, Any]
    answer: Answer | None
    per_model: dict[int, Answer | None] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "search_id": self.search_id,
            "objective": self.objective,
            "settings": self.settings,
            "answer": None if self.answer is None else self.answer.to_json(),
            "per_model": {
                str(layers): None if a is None else a.to_json()
                for layers, a in self.per_model.items()
            },
        }


def run_objective(
    conn: duckdb.DuckDBPyConnection,
    search_id: str,
    *,
    objective: Objective,
    window_days: float | None = None,
    n_gpu_lo: float = 0,
    n_gpu_hi: float = math.inf,
    n_gpu_min: float | None = None,
    n_gpu_max: float | None = None,
    delta: float = DEFAULT_DELTA,
) -> ObjectiveResult:
    """Apply one objective to the stored search ``search_id``.

    ``objective="fewest_gpus"`` needs ``window_days`` and uses the GPU counts in
    ``[n_gpu_lo, n_gpu_hi]``; ``objective="shortest_training"`` needs ``n_gpu_min`` and
    ``n_gpu_max``.
    """
    rows = best_per_pair(conn, search_id)
    layers = [m["layers"] for m in get_models(conn, search_id)]
    if objective == "fewest_gpus":
        if window_days is None:
            raise ValueError("objective 'fewest_gpus' needs window_days")
        settings = {"window_days": window_days, "n_gpu_lo": n_gpu_lo, "n_gpu_hi": n_gpu_hi}
        per = fewest_gpus_per_model(rows, window_days, n_gpu_lo, n_gpu_hi)
        best = fewest_gpus(rows, window_days, n_gpu_lo, n_gpu_hi)
    elif objective == "shortest_training":
        if n_gpu_min is None or n_gpu_max is None:
            raise ValueError("objective 'shortest_training' needs n_gpu_min and n_gpu_max")
        settings = {"n_gpu_min": n_gpu_min, "n_gpu_max": n_gpu_max, "delta": delta}
        per = shortest_training_per_model(rows, n_gpu_min, n_gpu_max, delta)
        best = shortest_training(rows, n_gpu_min, n_gpu_max, delta)
    else:
        raise ValueError(f"unknown objective {objective!r}")
    return ObjectiveResult(search_id, objective, settings, best, {k: per.get(k) for k in layers})
