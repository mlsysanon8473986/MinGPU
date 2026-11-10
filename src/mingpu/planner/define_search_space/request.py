"""The user's request (paper, Sec. 3 and Fig. 2) and the search it defines (Sec. 4.1).

A request gives the target loss ``L*``, the GPU type and one budget:

``objective="fewest_gpus"`` (Objective 1)
    a training window ``window_days``; the answer is the fewest GPUs.
``objective="shortest_training"`` (Objective 2)
    a GPU budget ``n_gpu_max`` (and ``n_gpu_min``, ``N_gpu_max / 3`` by default); the
    answer is the shortest training at an MFU within ``delta`` of the highest.

:meth:`SearchRequest.to_search_space` turns it into a
:class:`~mingpu.planner.define_search_space.search_space.SearchSpace`:
the compute-optimal model ``l_C`` and the candidate models around it (unless
``layers`` is given), and the candidate GPU counts of the budget (unless ``n_gpus`` is
given). The batch ``b_max``, the sequence length and the vocabulary default to
those of Llama 3.1 405B.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, get_args

from ..objectives.rules import DEFAULT_DELTA, Objective
from .candidate_gpu_counts import (
    DEFAULT_GPU_STEP,
    DEFAULT_MFU_RANGE,
    budget_gpu_counts,
    budget_gpu_range,
    window_gpu_counts,
    window_gpu_range,
)
from .candidate_models import (
    DEFAULT_SPAN,
    DEFAULT_VOCAB,
    CandidateModel,
    candidate_layers,
    compute_optimal_model,
)
from .scaling_law import BESIROGLU, ScalingLaw
from .search_space import SearchSpace

OBJECTIVES: tuple[str, ...] = get_args(Objective)


@dataclass(frozen=True)
class SearchRequest:
    """What the user asks MinGPU (see the module docstring)."""

    target_loss: float
    objective: Objective
    window_days: float | None = None
    n_gpu_max: int | None = None
    n_gpu_min: int | None = None
    delta: float = DEFAULT_DELTA
    gpu: str = "h100"
    seq_len: int = 8192
    b_max: int = 2048
    vocab: int = DEFAULT_VOCAB
    datatype: str = "bfloat16"
    law: ScalingLaw = BESIROGLU
    layers: tuple[int, ...] | None = None
    span: int = DEFAULT_SPAN
    n_gpus: tuple[int, ...] | None = None
    n_gpu_step: int = DEFAULT_GPU_STEP
    mfu_range: tuple[float, float] = DEFAULT_MFU_RANGE
    no_tp_overlap: bool = False
    tensor_par: int | None = None
    top_k: int | None = 10

    def __post_init__(self) -> None:
        for name in ("layers", "n_gpus", "mfu_range"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}: {self.objective!r}")
        if self.objective == "fewest_gpus" and not (self.window_days and self.window_days > 0):
            raise ValueError("objective 'fewest_gpus' needs a positive window_days")
        if self.objective == "shortest_training":
            if not (self.n_gpu_max and self.n_gpu_max > 0):
                raise ValueError("objective 'shortest_training' needs a positive n_gpu_max")
            if self.n_gpu_min is not None and not 0 < self.n_gpu_min <= self.n_gpu_max:
                raise ValueError(
                    f"need 0 < n_gpu_min <= n_gpu_max: {self.n_gpu_min}, {self.n_gpu_max}"
                )
        if self.n_gpu_step <= 0:
            raise ValueError(f"n_gpu_step must be positive: {self.n_gpu_step}")
        if not 0 <= self.delta < 1:
            raise ValueError(f"delta must be in [0, 1): {self.delta}")

    # -- the candidate models, the candidate GPU counts and the GPU range of the answer ---------

    def compute_optimal_model(self) -> CandidateModel:
        """``l_C`` of this request, with its ``D_req`` and ``C_req``."""
        return compute_optimal_model(self.target_loss, self.law, self.vocab)

    def candidate_layers(self) -> list[int]:
        """The candidate models searched by default: ``l_C - span`` to ``l_C + span``."""
        return candidate_layers(self.compute_optimal_model().model.layers, self.span)

    def candidate_gpu_counts(self) -> list[int]:
        """The candidate GPU counts searched by default, from the budget and the step."""
        if self.objective == "shortest_training":
            return budget_gpu_counts(self.n_gpu_max, self.n_gpu_min, self.n_gpu_step)
        return window_gpu_counts(*self.gpu_range(), self.n_gpu_step)

    def gpu_range(self) -> tuple[float, float]:
        """The GPU counts the answer may use.

        Objective 1: the counts on which ``l_C`` meets the window at an MFU within
        ``mfu_range``. Objective 2: ``[n_gpu_min, n_gpu_max]``.
        """
        if self.objective == "shortest_training":
            return budget_gpu_range(self.n_gpu_max, self.n_gpu_min)
        return window_gpu_range(
            self.target_loss,
            self.window_days,
            self.law,
            self.vocab,
            self.gpu,
            self.datatype,
            self.mfu_range,
        )

    def to_search_space(self) -> SearchSpace:
        """The search this request defines; explicit ``layers`` and ``n_gpus`` win."""
        return SearchSpace(
            target_loss=self.target_loss,
            layers=self.layers or tuple(self.candidate_layers()),
            n_gpus=self.n_gpus or tuple(self.candidate_gpu_counts()),
            seq_len=self.seq_len,
            b_max=self.b_max,
            vocab=self.vocab,
            gpu=self.gpu,
            datatype=self.datatype,
            law=self.law,
            no_tp_overlap=self.no_tp_overlap,
            tensor_par=self.tensor_par,
            top_k=self.top_k,
        )

    def objective_settings(self) -> dict[str, Any]:
        """The objective and its settings, as the arguments of ``run_objective``."""
        lo, hi = self.gpu_range()
        if self.objective == "fewest_gpus":
            return {
                "objective": "fewest_gpus",
                "window_days": self.window_days,
                "n_gpu_lo": lo,
                "n_gpu_hi": hi,
            }
        return {
            "objective": "shortest_training",
            "n_gpu_min": lo,
            "n_gpu_max": hi,
            "delta": self.delta,
        }

    # -- JSON ------------------------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        for name in ("layers", "n_gpus", "mfu_range"):
            if data[name] is not None:
                data[name] = list(data[name])
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SearchRequest:
        return cls(**{**data, "law": ScalingLaw(**data["law"])})
