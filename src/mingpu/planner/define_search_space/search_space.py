"""The search space: the contract between defining a search and running it.

:class:`~mingpu.planner.define_search_space.request.SearchRequest` produces a :class:`SearchSpace`,
and :mod:`mingpu.planner.execute_search.outer_search` runs one without knowing about objectives,
windows or budgets. It can also be written by hand, with explicit models and GPU counts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .candidate_models import LAYER_STEP
from .scaling_law import ScalingLaw


@dataclass(frozen=True)
class SearchSpace:
    """The search space of a request. ``top_k=None`` stores every feasible candidate parallel plan.

    Every setting is explicit: the defaults of the paper are those of ``SearchRequest``, and only
    the two options that are off by default have a value here.
    """

    target_loss: float
    layers: tuple[int, ...]
    n_gpus: tuple[int, ...]
    seq_len: int
    b_max: int
    vocab: int
    gpu: str
    datatype: str
    law: ScalingLaw
    top_k: int | None
    no_tp_overlap: bool = False
    tensor_par: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "n_gpus", tuple(self.n_gpus))
        if not self.layers or not self.n_gpus:
            raise ValueError("a search needs at least one model and one GPU count")
        if any(layers <= 0 or layers % LAYER_STEP for layers in self.layers):
            raise ValueError(f"layer counts must be positive multiples of {LAYER_STEP}")
        if any(n <= 0 for n in self.n_gpus):
            raise ValueError("GPU counts must be positive")
        if len(set(self.layers)) != len(self.layers) or len(set(self.n_gpus)) != len(self.n_gpus):
            raise ValueError("layer counts and GPU counts must not repeat")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be positive or None: {self.top_k}")

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["layers"], data["n_gpus"] = list(self.layers), list(self.n_gpus)
        return data
