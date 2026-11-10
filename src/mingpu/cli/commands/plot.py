"""The command plot-search: MFU and training time against the GPU count of a stored search."""

from __future__ import annotations

import typer

from ..utils import _console, _fail, _objective_result, _open_search

commands = typer.Typer()


@commands.command("plot-search")
def plot_search_cmd(
    search_id: str = typer.Argument(..., help="Search id (mingpu list-searches)."),
    objective: str | None = typer.Option(
        None,
        "--objective",
        "-o",
        help="fewest_gpus or shortest_training. Default: that of the search's request.",
    ),
    max_days: float | None = typer.Option(
        None, "--max-days", help="Training window T, in days (fewest_gpus). Default: the search's."
    ),
    gpus: str | None = typer.Option(
        None,
        "--gpus",
        help="GPU range lo:hi to draw (fewest_gpus). Default: MFU range of the request.",
    ),
    all_counts: bool = typer.Option(
        False, "--all-counts", help="fewest_gpus: draw every count searched."
    ),
    max_gpus: int | None = typer.Option(
        None, "--max-gpus", help="Largest GPU count (shortest_training). Default: the search's."
    ),
    min_gpus: int | None = typer.Option(
        None, "--min-gpus", help="Smallest GPU count (shortest_training). Default: --max-gpus / 3."
    ),
    mfu_tolerance: float | None = typer.Option(
        None,
        "--mfu-tolerance",
        help="MFU tolerance (shortest_training). Default: the search's, else 0.15.",
    ),
    out: str = typer.Option("plots", "--out", help="Directory of the figure. Default: plots."),
    name: str | None = typer.Option(
        None, "--name", help="File name without extension. Default: from the search and range."
    ),
    db: str | None = typer.Option(None, "--db", help="Database path. Default: MinGPU's."),
) -> None:
    """Draw MFU and training time against the GPU count, one curve per candidate model.

    Reads the stored search; nothing is searched, so the figure can be redrawn with another
    window, GPU range or tolerance. Writes a PNG. Needs matplotlib (the plot extra).
    """
    try:
        from ...plotting import plot_search
    except ImportError:
        _fail('plot-search needs matplotlib: pip install -e ".[plot]"')
    conn, record = _open_search(db, search_id)
    try:
        a = _objective_result(
            conn, record, objective, max_days=max_days, gpus=gpus, all_counts=all_counts,
            max_gpus=max_gpus, min_gpus=min_gpus, mfu_tolerance=mfu_tolerance,
        )  # fmt: skip
        paths = plot_search(conn, a, out, name)
    except ValueError as exc:
        conn.close()
        _fail(str(exc))
    conn.close()
    for path in paths:
        _console().print(f"written to {path}")
