"""Inner search on a tiny pair: l = 8 on 16 H100 GPUs, B_max = 32, t = 8, no TP overlap.

Calculon's generator yields 54,208 executions for it and the filter keeps 114,
which Calculon evaluates in about 2 s.
"""

import json

import pytest

from mingpu.planner.define_search_space.candidate_models import Model
from mingpu.planner.define_search_space.gpu_systems import load_system
from mingpu.planner.execute_search.inner_search import (
    RESULT_FIELDS,
    WorkerPool,
    calculon_commit,
    count_plans,
    inner_search,
)
from mingpu.storage.pair_cache import EXECUTION_FIELDS, save_pair

PAIR = dict(n_gpu=16, b_max=32, datatype="bfloat16", no_tp_overlap=True, tensor_par=8)


@pytest.fixture(scope="module")
def inputs():
    return Model(8).application(2048), load_system("h100")


@pytest.fixture(scope="module")
def result(inputs):
    return inner_search(*inputs, top_k=None, **PAIR)


def test_count(inputs):
    count = count_plans(*inputs, **PAIR)
    assert count.generated == 54_208
    assert count.steps == [1, 2, 3, 4, 5, 6, 7, 8]
    assert count.kept == 114
    assert all(
        a >= b for a, b in zip([count.generated] + count.left_after, count.left_after, strict=False)
    )


def test_search(inputs, result):
    """The search evaluates what the count kept and returns the valid plans, best first."""
    count = count_plans(*inputs, **PAIR)
    assert (result.count.generated, result.count.left_after) == (count.generated, count.left_after)
    assert result.evaluated == count.kept
    assert 0 < result.valid <= result.evaluated and len(result.executions) == result.valid
    rates = [e["sample_rate"] for e in result.executions]
    assert rates == sorted(rates, reverse=True)
    for e in result.executions:
        assert set(e) == set(RESULT_FIELDS)
        assert e["tensor_par"] == 8 and e["tensor_par_overlap"] == "none"
        assert e["tensor_par_comm_type"] == "rs_ag"
        assert e["seq_par_ag_redo"] == (e["activation_recompute"] != "full")
        assert e["batch_size"] <= 32 and e["num_procs"] == 16


def test_pool_gives_the_same_rows(inputs, result):
    with WorkerPool(2) as pool:
        pooled = inner_search(*inputs, top_k=None, pool=pool, **PAIR)
    assert pooled.executions == result.executions
    with pytest.raises(ValueError):
        WorkerPool(0)


def test_stored_executions_round_trip(conn, result):
    """Every field and statistic of every plan comes back from the database as it was."""
    pair_id = save_pair(
        conn, "key", result, b_max=32, datatype="bfloat16", restrictions={},
        calculon_commit=None, top_k=None,
    )  # fmt: skip
    columns = ", ".join(EXECUTION_FIELDS)
    cursor = conn.execute(
        f"SELECT {columns}, stats FROM inner_search_pair WHERE pair_id = ? ORDER BY rank", [pair_id]
    )
    names = [d[0] for d in cursor.description]
    stored = []
    for values in cursor.fetchall():
        row = dict(zip(names, values, strict=True))
        stats = json.loads(row.pop("stats"))
        stored.append({**row, **stats})
    assert stored == result.executions


def test_calculon_commit_reads_the_submodule(tmp_path, monkeypatch):
    import calculon

    commit = calculon_commit()
    assert commit is not None and len(commit) == 8
    # a Calculon without its own .git is unknown, not the commit of the repository above
    fake = tmp_path / "pkg" / "calculon"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("")
    monkeypatch.setattr(calculon, "__file__", str(fake / "__init__.py"))
    assert calculon_commit() is None
