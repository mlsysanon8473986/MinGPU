import pytest

from mingpu.planner.define_search_space.candidate_gpu_counts import (
    budget_gpu_counts,
    window_gpu_counts,
)


def test_budget_gpu_counts():
    """Multiples of the step from a third of the budget (or n_gpu_min), plus the budget itself."""
    counts = budget_gpu_counts(10000)
    assert counts[0] == 3584 and counts[-2:] == [9984, 10000]
    assert all(n % 256 == 0 for n in counts[:-1])
    assert budget_gpu_counts(7680, n_gpu_min=4000)[0] == 4096
    assert budget_gpu_counts(512, n_gpu_min=512) == [512]
    assert budget_gpu_counts(300) == [256, 300]
    assert budget_gpu_counts(100) == [100]
    with pytest.raises(ValueError):
        budget_gpu_counts(7680, n_gpu_min=8000)


def test_window_gpu_counts():
    """Every multiple of the step in the range; a range with none is an error."""
    assert window_gpu_counts(8, 8, 8) == [8]
    assert window_gpu_counts(1, 40, 8) == [8, 16, 24, 32, 40]
    with pytest.raises(ValueError):
        window_gpu_counts(9, 15, 8)
