"""The objective macros on a synthetic search (no Calculon).

Two models, 8 and 16 layers, with D_req chosen so that t_train (days) is
1 / sample_rate for l = 8 and 2 / sample_rate for l = 16 at s = 1,000; the
sample rates are powers of two so the times are exact.
"""

import math

import pytest

from mingpu.planner.define_search_space.candidate_models import Model
from mingpu.planner.define_search_space.gpu_systems import load_system
from mingpu.planner.execute_search.inner_search import InnerSearchResult, PlanCounts
from mingpu.storage.pair_cache import link_pair, save_pair
from mingpu.storage.searches import add_model, create_search

SEQ = 1000
DAY = 86400 * SEQ  # D_req of l = 8, so that t_train in days is 1 / sample_rate
#: (layers, n_gpu) -> (t_train days, MFU); None: no valid plan
PAIRS = {
    (8, 100): (64, 0.40),
    (8, 200): (32, 0.45),
    (8, 300): (16, 0.30),
    (8, 400): (8, 0.36),
    (8, 500): None,
    (16, 100): (128, 0.50),
    (16, 200): (16, 0.48),
    (16, 300): (8, 0.37),
    (16, 500): None,
}


@pytest.fixture
def search_id(conn):
    space = dict(
        target_loss=2.6, law={"name": "test"}, seq_len=SEQ, b_max=32, vocab=128_256,
        datatype="bfloat16", top_k=1, layers=[8, 16], n_gpus=[100, 200, 300, 400, 500],
    )  # fmt: skip
    sid = create_search(
        conn, space=space, system=load_system("h100"), f_peak=989e12, restrictions={"steps": [1]}
    )
    for layers in (8, 16):
        model = Model(layers)
        d_req = DAY * layers / 8
        add_model(
            conn, sid, layers=layers, application=model.application(SEQ),
            n_params=model.n_params, d_req=d_req, c_req=6 * model.n_params * d_req,
        )  # fmt: skip
        for (pl, n_gpu), value in PAIRS.items():
            if pl != layers:
                continue
            executions = []
            if value is not None:
                days, mfu = value
                executions = [
                    {"sample_rate": (layers / 8) / days, "total_efficiency": mfu, "tensor_par": 8}
                ]
            result = InnerSearchResult(
                count=PlanCounts(n_gpu, 1, [1], [1]),
                evaluated=1,
                valid=len(executions),
                seconds=0.0,
                executions=executions,
            )
            pair_id = save_pair(
                conn, f"fake-{layers}-{n_gpu}", result, b_max=32, datatype="bfloat16",
                restrictions={"steps": [1]},
                calculon_commit=None, top_k=1,
            )  # fmt: skip
            link_pair(conn, sid, layers, n_gpu, pair_id, reused=False)
    return sid


def query(conn, sql, *args):
    cursor = conn.execute(sql, list(args))
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def picks(rows):
    return [(r["layers"], r["n_gpu"]) for r in rows]


def test_fewest_gpus(conn, search_id):
    per_model = query(conn, "SELECT * FROM fewest_gpus_per_model(?, 40) ORDER BY layers", search_id)
    assert picks(per_model) == [(8, 200), (16, 200)]
    # both models need 200 GPUs: the shorter training wins
    assert picks(query(conn, "SELECT * FROM fewest_gpus(?, 40)", search_id)) == [(16, 200)]
    assert picks(query(conn, "SELECT * FROM fewest_gpus(?, 100)", search_id)) == [(8, 100)]
    assert query(conn, "SELECT * FROM fewest_gpus(?, 4)", search_id) == []
    # within a GPU range
    rows = query(conn, "SELECT * FROM fewest_gpus(?, 40, n_gpu_lo := 250)", search_id)
    assert picks(rows) == [(16, 300)]  # tie at 300 GPUs, 8 days against 16
    rows = query(conn, "SELECT * FROM fewest_gpus_per_model(?, 100, n_gpu_hi := 150)", search_id)
    assert picks(rows) == [(8, 100)]


def test_shortest_training(conn, search_id):
    # U* is the highest MFU of all models
    row = query(conn, "SELECT * FROM shortest_training(?, 100, 400)", search_id)[0]
    assert (row["layers"], row["n_gpu"], row["u_star"]) == (16, 200, 0.50)
    # band of 30%: 8 @ 400 and 16 @ 300 both take 8 days; the higher MFU wins
    row = query(conn, "SELECT * FROM shortest_training(?, 100, 400, delta := 0.3)", search_id)[0]
    assert (row["layers"], row["n_gpu"]) == (16, 300)
    # only the GPU counts of the budget
    row = query(conn, "SELECT * FROM shortest_training(?, 100, 250, delta := 0.5)", search_id)[0]
    assert (row["layers"], row["n_gpu"]) == (16, 200)


def test_shortest_training_per_model_uses_each_u_star(conn, search_id):
    rows = query(
        conn, "SELECT * FROM shortest_training_per_model(?, 100, 400) ORDER BY layers", search_id
    )
    assert picks(rows) == [(8, 200), (16, 200)]
    assert [r["u_star"] for r in rows] == [0.45, 0.50]


def test_run_objective(conn, search_id):
    """Any objective on a stored search, with no new search; a missing budget is an error."""
    from mingpu.planner.objectives.result import run_objective

    a = run_objective(conn, search_id, objective="fewest_gpus", window_days=40)
    assert (a.answer.layers, a.answer.n_gpu) == (16, 200)
    assert math.isinf(a.settings["n_gpu_hi"])
    none = run_objective(conn, search_id, objective="fewest_gpus", window_days=4)
    assert none.answer is None and all(v is None for v in none.per_model.values())
    for bad in (
        dict(objective="fewest_gpus"),
        dict(objective="shortest_training", n_gpu_max=400),
        dict(objective="cost"),
    ):
        with pytest.raises(ValueError):
            run_objective(conn, search_id, **bad)


def test_python_rule_agrees_with_sql_on_objective_1(conn, search_id):
    from mingpu.planner.objectives.rules import fewest_gpus, fewest_gpus_per_model
    from mingpu.storage.searches import best_per_pair

    rows = best_per_pair(conn, search_id)
    for window, n_gpu_lo, n_gpu_hi in [(40, 0, 10**6), (100, 0, 10**6), (4, 0, 10**6),
                                        (40, 250, 10**6)]:  # fmt: skip
        args = (search_id, window, n_gpu_lo, n_gpu_hi)
        sql = query(conn, "SELECT * FROM fewest_gpus(?, ?, n_gpu_lo := ?, n_gpu_hi := ?)", *args)
        choice = fewest_gpus(rows, window, n_gpu_lo, n_gpu_hi)
        assert picks(sql) == ([] if choice is None else [(choice.layers, choice.n_gpu)])
        sql_per = query(
            conn,
            "SELECT * FROM fewest_gpus_per_model(?, ?, n_gpu_lo := ?, n_gpu_hi := ?) "
            "ORDER BY layers",
            *args,
        )  # fmt: skip
        per = fewest_gpus_per_model(rows, window, n_gpu_lo, n_gpu_hi)
        assert picks(sql_per) == [(c.layers, c.n_gpu) for c in per.values()]


def test_python_rule_agrees_with_sql_on_objective_2(conn, search_id):
    from mingpu.planner.objectives.rules import shortest_training, shortest_training_per_model
    from mingpu.storage.searches import best_per_pair

    rows = best_per_pair(conn, search_id)
    for n_gpu_min, n_gpu_max, delta in [(100, 400, 0.15), (100, 400, 0.3), (100, 250, 0.5),
                                        (300, 500, 0.15)]:  # fmt: skip
        args = (search_id, n_gpu_min, n_gpu_max, delta)
        sql = query(conn, "SELECT * FROM shortest_training(?, ?, ?, delta := ?)", *args)
        choice = shortest_training(rows, n_gpu_min, n_gpu_max, delta)
        assert picks(sql) == [(choice.layers, choice.n_gpu)]
        assert sql[0]["u_star"] == choice.u_star
        sql_per = query(
            conn, "SELECT * FROM shortest_training_per_model(?, ?, ?, delta := ?) ORDER BY layers",
            *args,
        )  # fmt: skip
        per = shortest_training_per_model(rows, n_gpu_min, n_gpu_max, delta)
        assert [(r["layers"], r["n_gpu"], r["u_star"]) for r in sql_per] == [
            (c.layers, c.n_gpu, c.u_star) for c in per.values()
        ]
