"""The commands that read the stored results; none of them searches.

mingpu list-searches | show-search | show-answer
"""

from __future__ import annotations

import json

import typer
from rich.table import Table

from ...storage.db import connect, migrate
from ...storage.searches import best_per_pair, get_models, list_searches
from ..utils import (
    _console,
    _fail,
    _fmt,
    _objective_result,
    _open_search,
    _print_result,
    _stored_request,
)

commands = typer.Typer()


@commands.command("list-searches")
def list_searches_cmd(
    db: str | None = typer.Option(None, "--db", help="Database path."),
    limit: int = typer.Option(20, "--limit", "-n", help="Rows to show."),
) -> None:
    """List the stored searches, with their objective and progress."""
    conn = connect(db)
    migrate(conn)
    rows = list_searches(conn, limit=limit)
    conn.close()

    table = Table(title="Searches")
    columns = (
        "id",
        "created_at",
        "status",
        "objective",
        "target_loss",
        "seq_len",
        "b_max",
        "pairs",
    )
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(
            row["id"],
            str(row["created_at"])[:19],
            row["status"],
            row["objective"] or "",
            _fmt(row["target_loss"]),
            str(row["seq_len"]),
            str(row["b_max"]),
            f"{row['pairs_done']}/{row['pairs']} ({row['pairs_reused']} reused)",
        )
    _console().print(table)


@commands.command("show-search")
def show_search(
    search_id: str = typer.Argument(..., help="Search id."),
    db: str | None = typer.Option(None, "--db", help="Database path."),
    json_out: bool = typer.Option(False, "--json", help="Emit raw JSON."),
) -> None:
    """Show a stored search and the best parallel plan of each pair."""
    conn, record = _open_search(db, search_id)
    best = best_per_pair(conn, search_id)
    conn.close()

    if json_out:
        _console().print_json(json.dumps({"search": record, "pairs": best}, default=str))
        return

    console = _console()
    space = record["space"]
    console.print(f"[bold]Search[/bold] {record['id']}  ([green]{record['status']}[/green])")
    console.print(
        f"L* = {record['target_loss']}, s = {record['seq_len']:,}, "
        f"B_max = {record['b_max']:,}, GPU {space['gpu']}, {record['datatype']}, "
        f"restriction steps {record['restrictions']['steps']}, top_k {record['top_k']}"
    )
    if record["error"]:
        console.print(f"[red]error[/red]: {record['error']}")

    table = Table(title="Best parallel plan of each pair")
    for column in ("l", "N (B)", "N_gpu", "t", "p", "d", "v", "MFU", "days"):
        table.add_column(column, justify="right")
    for row in best:
        plan = row["tensor_par"] is not None
        table.add_row(
            str(row["layers"]),
            f"{row['n_params'] / 1e9:.1f}",
            f"{row['n_gpu']:,}",
            *(
                (
                    str(row["tensor_par"]),
                    str(row["pipeline_par"]),
                    str(row["data_par"]),
                    str(row["pipeline_interleaving"]),
                    _fmt(row["mfu"], percent=True),
                    f"{row['t_train_days']:.1f}",
                )
                if plan
                else ("none", "", "", "", "", "")
            ),
        )
    console.print(table)


@commands.command("show-answer")
def show_answer(
    search_id: str = typer.Argument(..., help="Search id."),
    db: str | None = typer.Option(None, "--db", help="Database path. Default: MinGPU's database."),
    json_out: bool = typer.Option(False, "--json", help="Print the answer as JSON."),
) -> None:
    """Show the answer of a stored search, for the objective and budget it was made with."""
    conn, record = _open_search(db, search_id)
    if _stored_request(record) is None:
        conn.close()
        _fail("this search has no objective: it was not made by fewest-gpus or shortest-training")
    try:
        a = _objective_result(conn, record)
        models = get_models(conn, search_id)
    except ValueError as exc:
        conn.close()
        _fail(str(exc))
    conn.close()
    if json_out:
        _console().print_json(json.dumps(a.to_json(), default=str))
        return
    _print_result(a, models)
