import pytest

from mingpu.planner.define_search_space.request import SearchRequest


def test_validation():
    with pytest.raises(ValueError, match="objective must be"):
        SearchRequest(target_loss=2.0, objective="cost")
    with pytest.raises(ValueError, match="window_days"):
        SearchRequest(target_loss=2.0, objective="fewest_gpus")
    with pytest.raises(ValueError, match="n_gpu_max"):
        SearchRequest(target_loss=2.0, objective="shortest_training")
    with pytest.raises(ValueError, match="n_gpu_min"):
        SearchRequest(target_loss=2.0, objective="shortest_training", n_gpu_max=100, n_gpu_min=200)
    with pytest.raises(ValueError, match="delta"):
        SearchRequest(target_loss=2.0, objective="shortest_training", n_gpu_max=100, delta=1.0)
    with pytest.raises(ValueError, match="n_gpu_step"):
        SearchRequest(target_loss=2.0, objective="shortest_training", n_gpu_max=100, n_gpu_step=0)
