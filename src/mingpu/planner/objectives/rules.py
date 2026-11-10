"""The two objectives of the paper (Sec. 3, Sec. 4.3, Appendix B), on stored pairs.

Both take the rows of the view ``best_per_pair`` (the best parallel plan of every
pair of a search) and follow the same rules as the SQL macros of
``migrations/0001_schema.sql``:

Objective 1, fewest GPUs on a window ``T``
    the smallest ``N_gpu`` in ``[n_gpu_lo, n_gpu_hi]`` on which a candidate model trains within
    ``T``; ties go to the shorter training, then to the smaller model.
Objective 2, shortest training on a GPU budget
    ``U*`` is the highest MFU on ``[n_gpu_min, n_gpu_max]`` GPUs; among the best parallel
    plans with an MFU of at least ``(1 - delta) U*``, the shortest training; ties go to
    the higher MFU, then to fewer GPUs, then to the smaller model.

The ``*_per_model`` functions give the answer of each model alone, with the
model's own ``U*`` for Objective 2, as the rows of the paper's Table 2.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Literal

#: Plan columns of ``best_per_pair`` copied into a :class:`Answer`.
PLAN_FIELDS: tuple[str, ...] = (
    "tensor_par",
    "pipeline_par",
    "data_par",
    "pipeline_interleaving",
    "microbatch_size",
    "activation_recompute",
    "optimizer_sharding",
    "tensor_par_overlap",
    "data_par_overlap",
    "batch_size",
)
DEFAULT_DELTA = 0.15
#: The two objectives of the paper.
Objective = Literal["fewest_gpus", "shortest_training"]


@dataclass(frozen=True)
class Answer:
    """The triple an objective chooses: a model, its tokens, a GPU count and their best plan."""

    layers: int
    n_params: int
    d_req: float
    c_req: float
    n_gpu: int
    plan: dict[str, Any]
    mfu: float
    mfu_6n: float
    t_iter: float
    t_train_days: float
    gpu_hours: float
    u_star: float | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any], u_star: float | None = None) -> Answer:
        return cls(
            layers=row["layers"],
            n_params=row["n_params"],
            d_req=row["d_req"],
            c_req=row["c_req"],
            n_gpu=row["n_gpu"],
            plan={f: row[f] for f in PLAN_FIELDS},
            mfu=row["mfu"],
            mfu_6n=row["mfu_6n"],
            t_iter=row["total_time"],
            t_train_days=row["t_train_days"],
            gpu_hours=row["gpu_hours"],
            u_star=u_star,
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _with_plan(rows: Iterable[dict[str, Any]], lo: float, hi: float) -> list[dict[str, Any]]:
    return [r for r in rows if r["mfu"] is not None and lo <= r["n_gpu"] <= hi]


def _by_model(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    models: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        models.setdefault(r["layers"], []).append(r)
    return models


def fewest_gpus_per_model(
    rows: Iterable[dict[str, Any]],
    window_days: float,
    n_gpu_lo: float = 0,
    n_gpu_hi: float = math.inf,
) -> dict[int, Answer]:
    """Per model, the fewest GPUs in ``[n_gpu_lo, n_gpu_hi]`` that train it in ``window_days``."""
    feasible = [r for r in _with_plan(rows, n_gpu_lo, n_gpu_hi) if r["t_train_days"] <= window_days]
    return {
        layers: Answer.from_row(min(mine, key=lambda r: r["n_gpu"]))
        for layers, mine in sorted(_by_model(feasible).items())
    }


def fewest_gpus(
    rows: Iterable[dict[str, Any]],
    window_days: float,
    n_gpu_lo: float = 0,
    n_gpu_hi: float = math.inf,
) -> Answer | None:
    """Objective 1 over all models; None when no model trains within the window."""
    per_model = fewest_gpus_per_model(rows, window_days, n_gpu_lo, n_gpu_hi).values()
    return min(per_model, key=lambda c: (c.n_gpu, c.t_train_days, c.layers), default=None)


def _fastest_in_band(rows: list[dict[str, Any]], u_star: float, delta: float) -> dict[str, Any]:
    band = [r for r in rows if r["mfu"] >= (1 - delta) * u_star]
    return min(band, key=lambda r: (r["t_train_days"], -r["mfu"], r["n_gpu"], r["layers"]))


def shortest_training_per_model(
    rows: Iterable[dict[str, Any]],
    n_gpu_min: float,
    n_gpu_max: float,
    delta: float = DEFAULT_DELTA,
) -> dict[int, Answer]:
    """Per model, the shortest training within ``delta`` of the model's own highest MFU."""
    out = {}
    for layers, mine in sorted(_by_model(_with_plan(rows, n_gpu_min, n_gpu_max)).items()):
        u_star = max(r["mfu"] for r in mine)
        out[layers] = Answer.from_row(_fastest_in_band(mine, u_star, delta), u_star)
    return out


def shortest_training(
    rows: Iterable[dict[str, Any]],
    n_gpu_min: float,
    n_gpu_max: float,
    delta: float = DEFAULT_DELTA,
) -> Answer | None:
    """Objective 2 over all models, ``U*`` over all of them; None without a valid plan."""
    candidates = _with_plan(rows, n_gpu_min, n_gpu_max)
    if not candidates:
        return None
    u_star = max(r["mfu"] for r in candidates)
    return Answer.from_row(_fastest_in_band(candidates, u_star, delta), u_star)
