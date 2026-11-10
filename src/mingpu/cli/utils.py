"""Helpers shared by the command modules: output, parsing and reading a stored search."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import duckdb
import typer
from rich.console import Console
from rich.table import Table

from ..planner.define_search_space.candidate_gpu_counts import budget_gpu_range
from ..planner.define_search_space.request import SearchRequest
from ..planner.objectives.result import ObjectiveResult, run_objective
from ..planner.objectives.rules import DEFAULT_DELTA
from ..storage.db import connect, migrate
from ..storage.searches import get_search

err_console = Console(stderr=True)


def _console() -> Console:
    return Console()


def _range(value: str) -> tuple[float, float]:
    """``"8192:24576"`` -> ``(8192.0, 24576.0)``."""
    try:
        lo, hi = (float(x) for x in value.split(":"))
    except ValueError:
        raise typer.BadParameter(f"not a range lo:hi: {value}") from None
    return lo, hi


def _default_workers() -> int:
    import psutil

    return max(1, min(8, psutil.cpu_count(logical=False) or 1))


def _fail(message: str) -> None:
    err_console.print(f"[red]{message}[/red]")
    raise typer.Exit(1)


def _fmt(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return ""
    if percent and isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _print_result(a: ObjectiveResult, models: list[dict[str, Any]]) -> None:
    console = _console()
    if a.objective == "fewest_gpus":
        title = f"Fewest GPUs within {a.settings['window_days']:g} days"
        where = (
            f"candidate GPU counts {a.settings['n_gpu_lo']:,.0f} to {a.settings['n_gpu_hi']:,.0f}"
        )
    else:
        title = "Shortest training at an MFU within {:.0%} of the highest".format(
            a.settings["delta"]
        )
        where = (
            f"candidate GPU counts {a.settings['n_gpu_min']:,.0f} to {a.settings['n_gpu_max']:,.0f}"
        )
    table = Table(title=f"{title} ({where})")
    columns = ["l", "N (B)", "D (T)", "C (FLOPs)", "N_gpu", "t", "p", "d", "v", "MFU", "days"]
    columns += ["U* of l"] if a.objective == "shortest_training" else []
    for column in columns:
        table.add_column(column, justify="right")
    for m in models:
        c = a.per_model.get(m["layers"])
        chosen = a.answer is not None and c is not None and c.layers == a.answer.layers
        head = [
            f"{m['layers']}{' *' if chosen else ''}",
            f"{m['n_params'] / 1e9:.1f}",
            "" if m["d_req"] is None else f"{m['d_req'] / 1e12:.2f}",
            "unreachable" if m["c_req"] is None else f"{m['c_req']:.3g}",
        ]
        if c is None:
            table.add_row(*head, "none", *[""] * (len(columns) - 5))
            continue
        plan = c.plan
        row = head + [
            f"{c.n_gpu:,}",
            str(plan["tensor_par"]),
            str(plan["pipeline_par"]),
            str(plan["data_par"]),
            str(plan["pipeline_interleaving"]),
            f"{100 * c.mfu:.2f}%",
            f"{c.t_train_days:.1f}",
        ]
        if a.objective == "shortest_training":
            row.append(f"{100 * c.u_star:.2f}%")
        table.add_row(*row, style="bold" if chosen else None)
    console.print(table)
    c = a.answer
    if c is None:
        console.print(
            "[yellow]No candidate model meets the budget on the candidate GPU counts.[/yellow]"
        )
        return
    line = (
        f"[bold]Answer[/bold]: l = {c.layers} ({c.n_params / 1e9:.1f}B parameters, "
        f"{c.d_req / 1e12:.2f}T tokens) on [bold]{c.n_gpu:,} GPUs[/bold], plan (t, p, d) = "
        f"({c.plan['tensor_par']}, {c.plan['pipeline_par']}, {c.plan['data_par']}), "
        f"v {c.plan['pipeline_interleaving']}, MFU {100 * c.mfu:.2f}%, {c.t_train_days:.1f} days"
    )
    if c.u_star is not None:
        line += f"; U* = {100 * c.u_star:.2f}%"
    console.print(line)
    console.print(f"search_id: {a.search_id}")


def _stored_request(record: dict[str, Any]) -> SearchRequest | None:
    """The request a search was made for, when it was made from one."""
    data = record.get("request") or {}
    if "objective" not in data:
        return None
    return SearchRequest.from_json(data)


def _open_search(db: str | None, search_id: str) -> tuple[duckdb.DuckDBPyConnection, dict]:
    """The database and the record of ``search_id``; exits when there is no such search."""
    conn = connect(db)
    migrate(conn)
    try:
        record = get_search(conn, search_id)
    except duckdb.ConversionException:
        record = None
    if record is None:
        conn.close()
        _fail(f"No search with id {search_id}")
    return conn, record


def _objective_result(
    conn: duckdb.DuckDBPyConnection,
    record: dict[str, Any],
    objective: str | None = None,
    *,
    max_days: float | None = None,
    gpus: str | None = None,
    all_counts: bool = False,
    max_gpus: int | None = None,
    min_gpus: int | None = None,
    mfu_tolerance: float | None = None,
) -> ObjectiveResult:
    """An objective on a stored search; what the options leave out comes from its request."""
    search_id = record["id"]
    stored = _stored_request(record)
    if objective is None:
        if stored is None:
            raise ValueError("give --objective: the search has no request to take it from")
        objective = stored.objective
    if objective == "fewest_gpus":
        window = max_days or (stored.window_days if stored else None)
        if window is None:
            raise ValueError("give --max-days")
        if gpus is not None:
            n_gpu_lo, n_gpu_hi = _range(gpus)
        elif all_counts or stored is None:
            n_gpu_lo, n_gpu_hi = 0, math.inf
        else:
            n_gpu_lo, n_gpu_hi = replace(
                stored, objective="fewest_gpus", window_days=window
            ).gpu_range()
        return run_objective(
            conn,
            search_id,
            objective="fewest_gpus",
            window_days=window,
            n_gpu_lo=n_gpu_lo,
            n_gpu_hi=n_gpu_hi,
        )
    if objective == "shortest_training":
        n_gpu_max = max_gpus or (stored.n_gpu_max if stored else None)
        if n_gpu_max is None:
            raise ValueError("give --max-gpus")
        n_gpu_min = min_gpus or (stored.n_gpu_min if stored and max_gpus is None else None)
        return run_objective(
            conn, search_id, objective="shortest_training", n_gpu_max=n_gpu_max,
            n_gpu_min=budget_gpu_range(n_gpu_max, n_gpu_min)[0],
            delta=(
                mfu_tolerance
                if mfu_tolerance is not None
                else (stored.delta if stored else DEFAULT_DELTA)
            ),
        )  # fmt: skip
    raise ValueError(f"unknown objective {objective!r}: use fewest_gpus or shortest_training")
