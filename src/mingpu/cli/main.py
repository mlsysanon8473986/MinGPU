"""The program ``mingpu``: registers the commands of the other modules.

mingpu fewest-gpus | shortest-training                  search (``commands/search.py``)
mingpu preview-search fewest-gpus | shortest-training    what a search would cover (same module)
mingpu list-searches | show-search | show-answer        stored results (``commands/results.py``)
mingpu plot-search                                      a stored search (``commands/plot.py``)

mingpu preview-search fewest-gpus --target-loss 1.8926 --max-days 78
mingpu fewest-gpus --target-loss 1.8926 --max-days 78
mingpu shortest-training --target-loss 1.8926 --max-gpus 24576 --min-gpus 8192
mingpu show-answer <search_id>
mingpu plot-search <search_id> --out plots
"""

from __future__ import annotations

import typer

from .. import __version__
from .commands import plot, results, search
from .utils import _console

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Plan the GPUs, parallel plan and model size that reach a target loss.",
)


def _version_callback(value: bool) -> None:
    if value:
        _console().print(f"mingpu {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the mingpu version and exit.",
    ),
) -> None:
    """mingpu: plan the GPUs, parallel plan and model size that reach a target loss."""


app.add_typer(search.commands)
app.add_typer(search.preview, name="preview-search")
app.add_typer(results.commands)
app.add_typer(plot.commands)
