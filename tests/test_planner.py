"""Planner searches end to end on tiny pairs (l = 8 and 16 on 8 and 16 H100, t = 8).

Most tests read a copy of the small search of conftest.py, run once per session.
"""

import pytest

from mingpu import BESIROGLU, SearchSpace, connect, run_outer_search
from mingpu.storage.searches import best_per_pair, get_models, get_search, list_searches

SPEC = dict(
    target_loss=2.6, seq_len=2048, b_max=32, vocab=128_256, gpu="h100", datatype="bfloat16",
    law=BESIROGLU, top_k=1, no_tp_overlap=True, tensor_par=8,
)  # fmt: skip


@pytest.fixture(scope="module")
def searched(small_search_template, tmp_path_factory):
    """One copy for the module: the tests below add searches to it in order."""
    s = small_search_template.copy_to(tmp_path_factory.mktemp("planner"))
    return s.path, s.space, s.search_id, s.events


def test_search_is_stored(searched):
    db, space, search_id, events = searched
    conn = connect(db)
    record = get_search(conn, search_id)
    assert record["status"] == "done" and record["error"] is None
    assert record["space"] == space.to_json()
    assert record["restrictions"] == {
        "steps": [1, 2, 3, 4, 5, 6, 7, 8],
        "t_max": 8,
        "tensor_par": 8,
    }
    assert record["f_peak"] == 989e12
    models = get_models(conn, search_id)
    assert [m["layers"] for m in models] == [8, 16]
    for m in models:
        assert m["d_req"] == pytest.approx(space.law.d_req(m["n_params"], 2.6))
        assert m["c_req"] == pytest.approx(6 * m["n_params"] * m["d_req"])
    assert [(e.done, e.total, e.reused) for e in events] == [(i, 4, False) for i in (1, 2, 3, 4)]
    conn.close()


def test_best_per_pair(searched):
    db, space, search_id, _ = searched
    conn = connect(db)
    rows = best_per_pair(conn, search_id)
    conn.close()
    assert [(r["layers"], r["n_gpu"]) for r in rows] == [(8, 8), (8, 16), (16, 8), (16, 16)]
    for r in rows:
        assert r["tensor_par"] == 8 and r["reused"] is False
        assert r["t_train_s"] == pytest.approx(r["d_req"] / (r["sample_rate"] * space.seq_len))
        assert r["t_train_days"] == pytest.approx(r["t_train_s"] / 86400)
        assert r["mfu_6n"] == pytest.approx(
            6 * r["n_params"] * r["sample_rate"] * space.seq_len / (r["n_gpu"] * 989e12)
        )
        assert r["gpu_hours"] == pytest.approx(r["n_gpu"] * r["t_train_s"] / 3600)


def test_second_search_reuses_the_pairs(searched):
    db, space, _, _ = searched
    events = []
    again = run_outer_search(space, db_path=str(db), progress=events.append)
    conn = connect(db)
    assert all(e.reused and e.result is None for e in events) and len(events) == 4
    assert all(r["reused"] for r in best_per_pair(conn, again))
    assert conn.execute("SELECT count(*) FROM cache_pairs").fetchone()[0] == 4
    summary = [s for s in list_searches(conn) if s["id"] == again][0]
    assert (summary["pairs"], summary["pairs_done"], summary["pairs_reused"]) == (4, 4, 4)
    conn.close()


def test_more_executions_search_the_pair_again(searched):
    db, _, _, _ = searched
    space = SearchSpace(layers=(8,), n_gpus=(8,), **{**SPEC, "top_k": None})
    events = []
    run_outer_search(space, db_path=str(db), progress=events.append)
    assert [e.reused for e in events] == [False]
    conn = connect(db)
    top_k, valid = conn.execute(
        "SELECT p.top_k, p.valid FROM cache_pairs p JOIN search_pair s ON s.pair_id = p.id "
        "WHERE s.layers = 8 AND s.n_gpu = 8 ORDER BY s.created_at DESC LIMIT 1"
    ).fetchone()
    assert top_k is None
    pair_id = conn.execute(
        "SELECT pair_id FROM search_pair WHERE layers = 8 AND n_gpu = 8 LIMIT 1"
    ).fetchone()[0]
    stored = conn.execute(
        "SELECT count(*) FROM inner_search_pair WHERE pair_id = ?", [pair_id]
    ).fetchone()[0]
    assert stored == valid == 18
    assert (
        conn.execute("SELECT count(*) FROM cache_pairs").fetchone()[0] == 4
    )  # replaced, not added
    conn.close()


def test_interrupted_search_continues_when_run_again(tmp_path):
    db = str(tmp_path / "again.duckdb")
    space = SearchSpace(layers=(8, 16), n_gpus=(8,), **SPEC)

    def stop_after_first(event):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_outer_search(space, db_path=db, progress=stop_after_first)
    conn = connect(db)
    search_id = conn.execute("SELECT id FROM search").fetchone()[0]
    record = get_search(conn, str(search_id))
    assert record["status"] == "error" and "KeyboardInterrupt" in record["error"]
    assert len(best_per_pair(conn, str(search_id))) == 1
    conn.close()

    events = []
    again = run_outer_search(space, db_path=db, progress=events.append)
    assert [(e.layers, e.n_gpu, e.reused) for e in events] == [(8, 8, True), (16, 8, False)]
    conn = connect(db)
    assert get_search(conn, again)["status"] == "done"
    assert len(best_per_pair(conn, again)) == 2
    assert get_search(conn, str(search_id))["status"] == "error"  # the first stays as it stopped
    conn.close()


def test_model_that_cannot_reach_the_loss_is_skipped(tmp_path):
    db = str(tmp_path / "unreachable.duckdb")
    space = SearchSpace(layers=(8, 16), n_gpus=(8,), **{**SPEC, "target_loss": 2.2})
    search_id = run_outer_search(space, db_path=db)
    conn = connect(db)
    models = {m["layers"]: m for m in get_models(conn, search_id)}
    assert models[8]["d_req"] is None and models[8]["c_req"] is None
    assert models[16]["d_req"] > 0
    assert [r["layers"] for r in best_per_pair(conn, search_id)] == [16]
    conn.close()


def test_spec_validation_and_json():
    space = SearchSpace(layers=[8, 16], n_gpus=[8], **SPEC)
    assert space.layers == (8, 16)
    assert space.to_json()["layers"] == [8, 16]
    for bad in (
        dict(layers=(), n_gpus=(8,)),
        dict(layers=(12,), n_gpus=(8,)),
        dict(layers=(8,), n_gpus=(0,)),
        dict(layers=(8, 8), n_gpus=(8,)),
        dict(layers=(8,), n_gpus=(8,), top_k=0),
    ):
        with pytest.raises(ValueError):
            SearchSpace(**{**SPEC, **bad})
