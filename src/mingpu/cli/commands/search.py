"""The commands that search, one per objective, and the preview of a search before it runs.

mingpu fewest-gpus | shortest-training
mingpu preview-search fewest-gpus | shortest-training
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Annotated, Any

import typer
from rich.table import Table

from ...planner.define_search_space.candidate_models import define_candidate_models
from ...planner.define_search_space.gpu_systems import load_system
from ...planner.define_search_space.request import SearchRequest
from ...planner.define_search_space.scaling_law import LAWS
from ...planner.define_search_space.search_space import SearchSpace
from ...planner.execute_search.inner_search import WorkerPool, count_plans
from ...planner.execute_search.outer_search import PairProgress
from ...planner.workflow import run_request
from ...storage.db import connect
from ...storage.searches import get_models
from ..utils import _console, _default_workers, _fail, _print_result, _range

commands = typer.Typer()
preview = typer.Typer(
    no_args_is_help=True,
    help="Preview the candidate models, GPU counts and parallel plans of a search; run nothing.",
)

# The budgets of each objective.
TargetLoss = Annotated[float, typer.Option("--target-loss", help="Target loss L*.")]
MaxDays = Annotated[float, typer.Option("--max-days", help="Training window T, in days.")]
MfuRange = Annotated[
    str | None,
    typer.Option(
        "--mfu-range", help="MFU range lo:hi bounding the candidate GPU counts. Default: 0.3:0.7."
    ),
]
MaxGpus = Annotated[int, typer.Option("--max-gpus", help="Largest GPU count, N_gpu_max.")]
MinGpus = Annotated[
    int | None, typer.Option("--min-gpus", help="Smallest GPU count. Default: --max-gpus / 3.")
]
GpuStep = Annotated[
    int | None,
    typer.Option("--gpu-step", min=1, help="Step between candidate GPU counts. Default: 256."),
]
# The options that both objectives share.
Gpu = Annotated[
    str | None,
    typer.Option("--gpu", "-g", help="GPU: h100, or the path of a system JSON. Default: h100."),
]
SeqSize = Annotated[
    int | None, typer.Option("--seq-size", help="Sequence length, in tokens. Default: 8192.")
]
BatchSize = Annotated[
    int | None,
    typer.Option("--batch-size", help="Largest global batch, in sequences. Default: 2048."),
]
Vocab = Annotated[
    int | None, typer.Option("--vocab", help="Vocabulary size. Default: 128256 (Llama 3).")
]
Datatype = Annotated[
    str | None,
    typer.Option("--datatype", help="float8, float16 or bfloat16. Default: bfloat16."),
]
Law = Annotated[
    str | None,
    typer.Option("--law", help="Loss law: besiroglu or chinchilla. Default: besiroglu."),
]
NoTpOverlap = Annotated[
    bool | None,
    typer.Option(
        "--no-tp-overlap/--tp-overlap",
        help="Exclude or allow TP-communication overlap. Default: --tp-overlap (allowed).",
    ),
]
TopK = Annotated[
    int | None,
    typer.Option("--top-k", min=1, help="Candidate parallel plans stored per pair. Default: 10."),
]
AllPlans = Annotated[
    bool,
    typer.Option(
        "--all-plans", help="Store every feasible candidate parallel plan, not only --top-k."
    ),
]
Workers = Annotated[
    int | None,
    typer.Option("--workers", "-w", min=1, help="Processes. Default: physical cores, up to 8."),
]
Db = Annotated[str | None, typer.Option("--db", help="Database path. Default: MinGPU's.")]
JsonOut = Annotated[bool, typer.Option("--json", help="Print the answer as JSON.")]


def _request_options(
    gpu: str | None,
    seq_size: int | None,
    batch_size: int | None,
    vocab: int | None,
    datatype: str | None,
    law: str | None,
    no_tp_overlap: bool | None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """The fields of a :class:`SearchRequest` that both objectives share."""
    if law is not None and law not in LAWS:
        _fail(f"unknown law {law!r}: use one of {sorted(LAWS)}")
    return dict(
        gpu=gpu, seq_len=seq_size, b_max=batch_size, vocab=vocab, datatype=datatype,
        law=None if law is None else LAWS[law], no_tp_overlap=no_tp_overlap, top_k=top_k,
    )  # fmt: skip


def _request(
    objective: str, target_loss: float, options: dict[str, Any], all_plans: bool = False
) -> tuple[SearchRequest, SearchSpace]:
    """The request and the search it defines; exits with a message when it is not valid."""
    try:
        given = {k: v for k, v in options.items() if v is not None}
        request = SearchRequest(target_loss=target_loss, objective=objective, **given)
        if all_plans:
            request = replace(request, top_k=None)
        return request, request.to_search_space()
    except (ValueError, OSError) as exc:
        _fail(str(exc))


def _search(
    objective: str,
    target_loss: float,
    options: dict[str, Any],
    all_plans: bool,
    workers: int | None,
    db: str | None,
    json_out: bool,
) -> None:
    """Search the request, store it and print its answer."""
    request, space = _request(objective, target_loss, options, all_plans)
    console = _console()
    lo, hi = request.gpu_range()
    console.print(
        f"L* = {request.target_loss:.6g} ({request.law.name}), l_C = "
        f"{request.compute_optimal_model().model.layers}, candidate models {list(space.layers)}, "
        f"{len(space.n_gpus)} candidate GPU counts from {space.n_gpus[0]:,} "
        f"to {space.n_gpus[-1]:,} "
        f"({len(space.layers) * len(space.n_gpus)} pairs); answer on {lo:,.0f} to {hi:,.0f} GPUs"
    )

    def progress(event: PairProgress) -> None:
        if event.reused:
            detail = "from the cache"
        else:
            r = event.result
            detail = f"{r.count.kept:,} kept, {r.valid:,} valid, {r.seconds:.1f} s"
        console.print(
            f"[{event.done}/{event.total}] l = {event.layers}, N_gpu = {event.n_gpu:,}: {detail}"
        )

    try:
        result = run_request(
            request,
            db_path=db,
            workers=workers or _default_workers(),
            progress=progress,
        )
    except ValueError as exc:
        _fail(str(exc))
    if json_out:
        console.print_json(json.dumps(result.to_json(), default=str))
        return
    conn = connect(db)
    models = get_models(conn, result.search_id)
    conn.close()
    _print_result(result, models)


def _count_one(
    layers: int,
    n_gpu: int,
    application: dict[str, Any],
    system: dict[str, Any],
    b_max: int,
    datatype: str,
    no_tp_overlap: bool,
) -> tuple[int, int, int]:
    """``(layers, plans before the filters, plans after them)`` of one pair."""
    count = count_plans(
        application, system, n_gpu, b_max, datatype=datatype, no_tp_overlap=no_tp_overlap
    )
    return layers, count.generated, count.kept


def _preview(
    objective: str, target_loss: float, options: dict[str, Any], workers: int | None
) -> None:
    """Print what a search would cover; Calculon only generates, and evaluates nothing."""
    request, space = _request(objective, target_loss, options)
    try:
        system = load_system(space.gpu)
    except ValueError as exc:
        _fail(str(exc))

    candidates = define_candidate_models(space.layers, space.target_loss, space.law, space.vocab)
    models = {c.model.layers: c.model for c in candidates if c.reaches_loss}
    # the search skips the models that cannot reach the target loss
    unreachable = [c.model.layers for c in candidates if not c.reaches_loss]

    jobs = [
        (layers, n_gpu, model.application(space.seq_len), system, space.b_max, space.datatype,
         space.no_tp_overlap)
        for layers, model in models.items()
        for n_gpu in space.n_gpus
    ]  # fmt: skip
    workers = workers or _default_workers()
    if workers > 1 and len(jobs) > 1:
        with WorkerPool(workers) as pool:
            counted = pool.starmap(_count_one, jobs)
    else:
        counted = [_count_one(*job) for job in jobs]
    before = dict.fromkeys(models, 0)
    after = dict.fromkeys(models, 0)
    for layers, generated, kept in counted:
        before[layers] += generated
        after[layers] += kept

    console = _console()
    console.print(
        f"Target loss L* = {request.target_loss:.6g} ({request.law.name}), "
        f"compute-optimal model l_C = {request.compute_optimal_model().model.layers}"
    )
    names = ", ".join(str(layers) for layers in models)
    console.print(f"Candidate models: {len(models)} (l = {names})")
    if unreachable:
        console.print(f"Left out, cannot reach the target loss: l = {unreachable}")
    console.print(
        f"Candidate GPU counts: {len(space.n_gpus)}, from {space.n_gpus[0]:,} to "
        f"{space.n_gpus[-1]:,}, step {request.n_gpu_step:,} "
        "(the same GPU count range for all candidate models)"
    )
    console.print(f"Pairs (Candidate model x Candidate GPU count): {len(jobs):,}")

    table = Table(title="Parallel plans over all the candidate GPU counts")
    table.add_column("Candidate model l", justify="right")
    table.add_column("N (B)", justify="right")
    table.add_column("Parallel plans - before the restrictions", justify="right")
    table.add_column("Parallel plans after the restrictions (Candidates)", justify="right")
    for layers, model in models.items():
        table.add_row(
            str(layers),
            f"{model.n_params / 1e9:.1f}",
            f"{before[layers]:,}",
            f"{after[layers]:,}",
        )
    table.add_section()
    table.add_row(
        "Total", "", f"{sum(before.values()):,}", f"{sum(after.values()):,}", style="bold"
    )
    console.print(table)


@commands.command("fewest-gpus")
def fewest_gpus(
    target_loss: TargetLoss,
    max_days: MaxDays,
    mfu_range: MfuRange = None,
    gpu_step: GpuStep = None,
    gpu: Gpu = None,
    seq_size: SeqSize = None,
    batch_size: BatchSize = None,
    vocab: Vocab = None,
    datatype: Datatype = None,
    law: Law = None,
    no_tp_overlap: NoTpOverlap = None,
    top_k: TopK = None,
    all_plans: AllPlans = False,
    workers: Workers = None,
    db: Db = None,
    json_out: JsonOut = False,
) -> None:
    """Objective 1: the fewest GPUs that train a model to the target loss within --max-days.

    Searches every pair and returns the chosen triple (candidate model, candidate GPU count and
    parallel plan), with its tokens, MFU and training time.
    """
    options = _request_options(
        gpu, seq_size, batch_size, vocab, datatype, law, no_tp_overlap, top_k
    )
    options.update(_window(max_days, mfu_range, gpu_step))
    _search("fewest_gpus", target_loss, options, all_plans, workers, db, json_out)


@commands.command("shortest-training")
def shortest_training(
    target_loss: TargetLoss,
    max_gpus: MaxGpus,
    min_gpus: MinGpus = None,
    mfu_tolerance: Annotated[
        float | None,
        typer.Option("--mfu-tolerance", help="MFU tolerance below the highest MFU. Default: 0.15."),
    ] = None,
    gpu_step: GpuStep = None,
    gpu: Gpu = None,
    seq_size: SeqSize = None,
    batch_size: BatchSize = None,
    vocab: Vocab = None,
    datatype: Datatype = None,
    law: Law = None,
    no_tp_overlap: NoTpOverlap = None,
    top_k: TopK = None,
    all_plans: AllPlans = False,
    workers: Workers = None,
    db: Db = None,
    json_out: JsonOut = False,
) -> None:
    """Objective 2: the shortest training to the target loss on at most --max-gpus GPUs.

    Considers only pairs whose best parallel plan has an MFU within --mfu-tolerance of the highest
    reachable. Returns the chosen triple, with its tokens, MFU and training time.
    """
    options = _request_options(
        gpu, seq_size, batch_size, vocab, datatype, law, no_tp_overlap, top_k
    )
    options.update(n_gpu_max=max_gpus, n_gpu_min=min_gpus, delta=mfu_tolerance, n_gpu_step=gpu_step)
    _search("shortest_training", target_loss, options, all_plans, workers, db, json_out)


def _window(max_days: float, mfu_range: str | None, gpu_step: int | None) -> dict[str, Any]:
    """The fields of a :class:`SearchRequest` that define the budget of Objective 1."""
    return dict(
        window_days=max_days,
        mfu_range=None if mfu_range is None else _range(mfu_range),
        n_gpu_step=gpu_step,
    )


@preview.command("fewest-gpus")
def preview_fewest_gpus(
    target_loss: TargetLoss,
    max_days: MaxDays,
    mfu_range: MfuRange = None,
    gpu_step: GpuStep = None,
    gpu: Gpu = None,
    seq_size: SeqSize = None,
    batch_size: BatchSize = None,
    vocab: Vocab = None,
    datatype: Datatype = None,
    law: Law = None,
    no_tp_overlap: NoTpOverlap = None,
    workers: Workers = None,
) -> None:
    """Preview the search that fewest-gpus would make; nothing is evaluated."""
    options = _request_options(gpu, seq_size, batch_size, vocab, datatype, law, no_tp_overlap)
    options.update(_window(max_days, mfu_range, gpu_step))
    _preview("fewest_gpus", target_loss, options, workers)


@preview.command("shortest-training")
def preview_shortest_training(
    target_loss: TargetLoss,
    max_gpus: MaxGpus,
    min_gpus: MinGpus = None,
    gpu_step: GpuStep = None,
    gpu: Gpu = None,
    seq_size: SeqSize = None,
    batch_size: BatchSize = None,
    vocab: Vocab = None,
    datatype: Datatype = None,
    law: Law = None,
    no_tp_overlap: NoTpOverlap = None,
    workers: Workers = None,
) -> None:
    """Preview the search that shortest-training would make; nothing is evaluated."""
    options = _request_options(gpu, seq_size, batch_size, vocab, datatype, law, no_tp_overlap)
    options.update(n_gpu_max=max_gpus, n_gpu_min=min_gpus, n_gpu_step=gpu_step)
    _preview("shortest_training", target_loss, options, workers)
