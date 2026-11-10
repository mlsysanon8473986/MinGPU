"""The commands, on the four pairs of the shared small search."""

import json
from functools import partial

import pytest
from typer.testing import CliRunner

from mingpu import connect
from mingpu.cli.commands import search
from mingpu.cli.main import app
from mingpu.cli.utils import _objective_result
from mingpu.planner.define_search_space.candidate_models import Model
from mingpu.planner.define_search_space.gpu_systems import load_system
from mingpu.planner.define_search_space.request import SearchRequest
from mingpu.planner.execute_search.inner_search import count_plans
from mingpu.storage.searches import get_search

RUNNER = CliRunner()
WIDE = {"COLUMNS": "200"}
SMALL = ("--seq-size", 2048, "--batch-size", 32, "--no-tp-overlap", "--workers", 1)


def run(*args):
    return RUNNER.invoke(app, [str(a) for a in args], env=WIDE)


@pytest.fixture
def small_pairs(monkeypatch):
    """The four pairs of the small search instead of the ones the budget would define."""
    monkeypatch.setattr(SearchRequest, "candidate_layers", lambda self: [8, 16])
    monkeypatch.setattr(SearchRequest, "candidate_gpu_counts", lambda self: [8, 16])


def test_preview_counts_the_plans_of_every_pair(small_pairs):
    result = run(
        "preview-search", "shortest-training", "--target-loss", 2.6, "--max-gpus", 16,
        "--min-gpus", 8, *SMALL,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Candidate models: 2 (l = 8, 16)" in result.output
    assert "Candidate GPU count): 4" in result.output
    counts = {
        (layers, n): count_plans(
            Model(layers).application(2048),
            load_system("h100"),
            n,
            32,
            datatype="bfloat16",
            no_tp_overlap=True,
        )  # fmt: skip
        for layers in (8, 16)
        for n in (8, 16)
    }
    for layers in (8, 16):
        before = sum(c.generated for (lay, _), c in counts.items() if lay == layers)
        after = sum(c.kept for (lay, _), c in counts.items() if lay == layers)
        assert f"{before:,}" in result.output and f"{after:,}" in result.output
    assert f"{sum(c.generated for c in counts.values()):,}" in result.output


def test_end_to_end(small_search, small_pairs, monkeypatch):
    # the commands leave t free; with t = 8 and --top-k 1 the four pairs are those of the
    # shared small search, so they come from its cache
    monkeypatch.setattr(search, "SearchRequest", partial(SearchRequest, tensor_par=8))
    db = small_search.path
    result = run(
        "shortest-training", "--target-loss", 2.6, "--max-gpus", 16, "--min-gpus", 8,
        "--top-k", 1, *SMALL, "--db", db, "--json",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{") :])
    assert payload["objective"] == "shortest_training" and set(payload["per_model"]) == {"8", "16"}
    assert payload["answer"]["plan"]["tensor_par"] == 8
    search_id = payload["search_id"]

    listed = run("list-searches", "--db", db)
    assert (
        listed.exit_code == 0 and "shortest" in listed.output and "4/4 (4 reused)" in listed.output
    )
    shown = run("show-search", search_id, "--db", db, "--json")
    assert shown.exit_code == 0 and len(json.loads(shown.output)["pairs"]) == 4
    assert "Best parallel plan of each pair" in run("show-search", search_id, "--db", db).output
    for unknown in ("00000000-0000-0000-0000-000000000000", "not-a-uuid"):
        assert run("show-search", unknown, "--db", db).exit_code == 1

    # show-answer: the objective and the budget come from the search; a search made without
    # a request has none
    shown = run("show-answer", search_id, "--db", db)
    assert shown.exit_code == 0 and "Shortest training" in shown.output
    assert run("show-answer", small_search.search_id, "--db", db).exit_code == 1

    # Objective 1 answers on the window range by default (1 to 3 GPUs here)
    narrow = run("fewest-gpus", "--target-loss", 2.6, "--max-days", 1, "--top-k", 1, *SMALL,
                 "--db", db)  # fmt: skip
    assert narrow.exit_code == 0 and "No candidate model meets the budget" in narrow.output
    # plot-search can still use every candidate GPU count
    conn = connect(db)
    every = _objective_result(
        conn, get_search(conn, search_id), "fewest_gpus", max_days=1, all_counts=True
    )
    conn.close()
    assert every.answer is not None
