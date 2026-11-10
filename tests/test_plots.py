"""The figure of a stored search (mingpu plot-search), on a small real search."""

import pytest

pytest.importorskip("matplotlib")

from mingpu import connect, run_objective  # noqa: E402
from mingpu.plotting import draw, plot_search  # noqa: E402

TIME = dict(objective="shortest_training", n_gpu_min=8, n_gpu_max=16, delta=0.15)
GPUS = dict(objective="fewest_gpus", window_days=1)


@pytest.fixture(scope="module")
def search(small_search_template, tmp_path_factory):
    s = small_search_template.copy_to(tmp_path_factory.mktemp("plots"))
    conn = connect(s.path)
    yield conn, s.search_id
    conn.close()


def test_both_objectives_write_a_png(search, tmp_path):
    conn, search_id = search
    for args in (GPUS, TIME):
        a = run_objective(conn, search_id, **args)
        paths = plot_search(conn, a, tmp_path, f"fig_{args['objective']}")
        assert [p.suffix for p in paths] == [".png"]
        assert all(p.stat().st_size > 5_000 for p in paths)


def test_no_plan_in_range_is_an_error(search):
    conn, search_id = search
    a = run_objective(
        conn, search_id, objective="shortest_training", n_gpu_min=1, n_gpu_max=2, delta=0.15
    )
    with pytest.raises(ValueError, match="nothing to draw"):
        draw(conn, a)
