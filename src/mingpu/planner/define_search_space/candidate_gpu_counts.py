"""Candidate GPU counts: the range the answer may use and the counts that are searched.

Both are the same for every candidate model and follow from the user's budget (paper, Sec. 4.1):

* on a GPU budget (Objective 2), the range is ``[N_gpu_min, N_gpu_max]``, with
  ``N_gpu_min = N_gpu_max / 3`` by default (:func:`budget_gpu_range`); the counts are the multiples
  of the step in it and ``N_gpu_max`` itself (:func:`budget_gpu_counts`);
* on a window ``T`` (Objective 1), the range holds the counts on which the compute-optimal
  model ``l_C`` would meet ``T`` at an MFU between 30% and 70%,
  ``C_req(l_C) / (0.7 F_peak T) <= N_gpu <= C_req(l_C) / (0.3 F_peak T)``
  (:func:`window_gpu_range`); the counts are the multiples of the step in it
  (:func:`window_gpu_counts`).

An explicit list can be used instead of either.
"""

from __future__ import annotations

import math

from .candidate_models import compute_optimal_model
from .gpu_systems import load_system, peak_flops
from .scaling_law import ScalingLaw, UnreachableLoss

#: MFU range ``(U_min, U_max)`` of the window range.
DEFAULT_MFU_RANGE: tuple[float, float] = (0.3, 0.7)
#: Default step of the candidate GPU counts, for both objectives.
DEFAULT_GPU_STEP = 256
#: The default lower limit of a GPU budget is the budget divided by this number.
BUDGET_DIVISOR = 3
SECONDS_PER_DAY = 86_400


def budget_gpu_range(n_gpu_max: int, n_gpu_min: float | None = None) -> tuple[float, float]:
    """``(n_gpu_min, n_gpu_max)``, with ``n_gpu_min = n_gpu_max / 3`` by default."""
    return (n_gpu_max / BUDGET_DIVISOR if n_gpu_min is None else n_gpu_min), n_gpu_max


def _gpu_range_for_compute(
    c_req: float,
    f_peak: float,
    window_days: float,
    mfu_range: tuple[float, float] = DEFAULT_MFU_RANGE,
) -> tuple[float, float]:
    """GPUs on which ``c_req`` FLOPs fit in ``window_days`` at an MFU within ``mfu_range``.

    Returns ``(lo, hi)``: ``lo`` at the highest MFU, ``hi`` at the lowest.
    """
    u_min, u_max = mfu_range
    if not 0 < u_min < u_max <= 1:
        raise ValueError(f"mfu_range must satisfy 0 < U_min < U_max <= 1: {mfu_range}")
    seconds = window_days * SECONDS_PER_DAY
    return c_req / (u_max * f_peak * seconds), c_req / (u_min * f_peak * seconds)


def window_gpu_range(
    target_loss: float,
    window_days: float,
    law: ScalingLaw,
    vocab: int,
    gpu: str,
    datatype: str,
    mfu_range: tuple[float, float] = DEFAULT_MFU_RANGE,
) -> tuple[float, float]:
    """The GPU counts on which ``l_C`` meets the window at an MFU within ``mfu_range``."""
    l_c = compute_optimal_model(target_loss, law, vocab)
    if l_c.c_req is None:
        raise UnreachableLoss(
            f"the compute-optimal model (l = {l_c.model.layers}) cannot reach loss {target_loss}"
        )
    f_peak = peak_flops(load_system(gpu), datatype)
    return _gpu_range_for_compute(l_c.c_req, f_peak, window_days, mfu_range)


def budget_gpu_counts(
    n_gpu_max: int, n_gpu_min: float | None = None, step: int = DEFAULT_GPU_STEP
) -> list[int]:
    """GPU counts on a budget of ``n_gpu_max`` GPUs: multiples of ``step`` and the budget.

    The multiples of ``step`` in the range of :func:`budget_gpu_range`, then ``n_gpu_max``
    itself if it is not one of them, so the budget is always searched.
    """
    lo, _ = budget_gpu_range(n_gpu_max, n_gpu_min)
    if not 0 < lo <= n_gpu_max:
        raise ValueError(f"need 0 < n_gpu_min <= n_gpu_max: n_gpu_min {lo}, n_gpu_max {n_gpu_max}")
    first = math.ceil(lo / step) * step
    counts = list(range(first, n_gpu_max + 1, step))
    if not counts or counts[-1] != n_gpu_max:
        counts.append(n_gpu_max)
    return counts


def window_gpu_counts(lo: float, hi: float, step: int = DEFAULT_GPU_STEP) -> list[int]:
    """Every multiple of ``step`` in ``[lo, hi]``."""
    first = math.ceil(lo / step) * step
    last = math.floor(hi / step) * step
    if last < first or first <= 0:
        raise ValueError(f"no positive multiple of {step} between {lo} and {hi}")
    return list(range(first, last + 1, step))
