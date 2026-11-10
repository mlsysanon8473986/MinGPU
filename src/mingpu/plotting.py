"""Figure of a stored search: MFU and training time against the GPU count, one curve per model.

:func:`plot_search` draws the :class:`~mingpu.planner.objectives.result.ObjectiveResult` of a
stored search from the database; nothing is searched or evaluated, so the figure can be
redrawn whenever the window, the GPU range or the MFU tolerance changes (``mingpu plot-search``).

Two panels share the GPU-count axis, with one colour per candidate model:

* top, the MFU of the best parallel plan of every pair;
* bottom, the training time of that plan.

Objective ``shortest_training`` (on a GPU budget) shades the MFU tolerance band, from
``(1 - delta) U*`` to ``U*``. Objective ``fewest_gpus`` (within a window) draws a horizontal
line at the window and no band. Both mark the answer in the two panels and give, above the panels,
the objective, the target loss, the GPU range evaluated and the GPU type.

The style follows the paper's figures (Times, box axes, ``#3A6EA5`` for the band, ``#D95F02`` for
the answer). Needs matplotlib (``pip install -e ".[plot]"``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
import matplotlib.colors
import matplotlib.ticker
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .planner.objectives.result import ObjectiveResult
from .storage.searches import best_per_pair, get_models, get_search


def _mix(hex_colour: str, pct: float) -> str:
    """xcolor's ``colour!pct``: ``pct`` % of the colour, the rest white."""
    rgb = [int(hex_colour[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(255 - pct / 100 * (255 - c)):02x}" for c in rgb)


BAND, ANSWER = "#3A6EA5", "#D95F02"
FRAME = _mix("#000000", 60)
TINY, SMALL = 5, 7
STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.4, "axes.edgecolor": FRAME,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.4, "ytick.major.width": 0.4, "xtick.major.size": 2.8,
    "ytick.major.size": 2.8, "xtick.color": FRAME, "ytick.color": FRAME,
    "xtick.labelcolor": "black", "ytick.labelcolor": "black", "xtick.labelsize": TINY,
    "ytick.labelsize": TINY, "axes.labelsize": SMALL, "lines.solid_joinstyle": "round",
}  # fmt: skip

#: Training times above this multiple of the fastest one are cut off the time panel.
TIME_CAP = 2.5
LEGEND_COLUMNS = 3


@dataclass(frozen=True)
class Point:
    """The best parallel plan of one pair: GPU count, MFU (%) and training time (days)."""

    n_gpu: int
    mfu: float
    days: float


def curves(rows: list[dict[str, Any]], lo: float, hi: float) -> dict[int, list[Point]]:
    """``{layers: points}`` of the pairs with a plan whose GPU count is in ``[lo, hi]``."""
    out: dict[int, list[Point]] = {}
    for r in rows:
        if r["mfu"] is not None and lo <= r["n_gpu"] <= hi:
            out.setdefault(r["layers"], []).append(
                Point(r["n_gpu"], 100 * r["mfu"], r["t_train_days"])
            )
    return {k: sorted(v, key=lambda p: p.n_gpu) for k, v in sorted(out.items())}


def _colours(layers: list[int]) -> dict[int, str]:
    """One colour per model, in the order of the sizes (viridis, without its pale end)."""
    cmap = matplotlib.colormaps["viridis"]
    steps = max(len(layers) - 1, 1)
    return {n: matplotlib.colors.to_hex(cmap(0.80 * i / steps)) for i, n in enumerate(layers)}


def _size(n_params: float) -> str:
    return rf"$N = {n_params / 1e9:.1f}$B"


def _gpu_name(space: dict[str, Any]) -> str:
    gpu = str(space.get("gpu", "unknown"))
    return Path(gpu).stem if ("/" in gpu or "\\" in gpu or gpu.endswith(".json")) else gpu.upper()


def header(
    record: dict[str, Any], result: ObjectiveResult, lo: float, hi: float
) -> tuple[str, str]:
    """The two lines above the panels: objective; loss, GPU range and GPU type."""
    s = result.settings
    if result.objective == "fewest_gpus":
        objective = f"Objective 1: fewest GPUs within {s['window_days']:g} days"
    else:
        objective = f"Objective 2: shortest training, MFU within {s['delta']:.0%} of the highest"
    detail = (
        f"$L^* = {record['target_loss']:.4f}$, {lo:,.0f} to {hi:,.0f} GPUs, "
        f"{_gpu_name(record['space'])}"
    )
    return objective, detail


def _free_corner(
    xs_ys: list[list[tuple[float, float]]],
    answer_at: tuple[float, float],
    line_y: float | None,
    size: tuple[float, float] = (0.36, 0.36),
) -> tuple[float, float]:
    """Top-left corner (axes fraction) of the note: the spot with fewest curve samples under it.

    ``xs_ys`` are the curves in axes fractions; ties go to the spot nearest the answer. A box
    that spans the answer's vertical line or the window line is penalised.
    """
    w, h = size
    samples = []
    for pts in xs_ys:
        for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
            samples += [(x0 + (x1 - x0) * i / 20, y0 + (y1 - y0) * i / 20) for i in range(21)]
    px, py = answer_at
    best, best_key = (0.585, 0.95), None
    for y_top in (0.97, 0.90, 0.83, 0.76, 0.69, 0.62, 0.55, 0.48, 0.41):
        for i in range(22):
            x0 = 0.02 + 0.03 * i
            if x0 + w > 0.99:
                break
            hits = sum(x0 <= x <= x0 + w and y_top - h <= y <= y_top for x, y in samples)
            if x0 - 0.02 <= px <= x0 + w + 0.02:
                hits += 1000
            if line_y is not None and y_top - h <= line_y <= y_top:
                hits += 40
            key = (hits, math.hypot(x0 + w / 2 - px, y_top - h / 2 - py))
            if best_key is None or key < best_key:
                best, best_key = (x0, y_top), key
    return best


def draw(
    conn: duckdb.DuckDBPyConnection,
    result: ObjectiveResult,
    *,
    width: float = 2.7,
    panel: float = 1.35,
) -> tuple[Figure, dict[str, float]]:
    """The figure of ``result`` and the numbers it shows (range, U*, band, y limits)."""
    search_id = result.search_id
    record = get_search(conn, search_id)
    models = {m["layers"]: m for m in get_models(conn, search_id)}
    s = result.settings
    lo, hi = (
        (s["n_gpu_min"], s["n_gpu_max"])
        if result.objective == "shortest_training"
        else (s["n_gpu_lo"], s["n_gpu_hi"])
    )
    data = curves(best_per_pair(conn, search_id), lo, hi)
    if not data:
        raise ValueError("no plan in the GPU range of the answer: nothing to draw")
    counts = [p.n_gpu for pts in data.values() for p in pts]
    lo_shown, hi_shown = max(lo, min(counts)), min(hi, max(counts))
    if math.isinf(hi) or lo == 0:  # an open range is shown as the counts searched
        lo, hi = lo_shown, hi_shown
    unit = 1e3 if max(counts) >= 4000 else 1.0
    colours = _colours(list(data))
    answer = result.answer
    delta = s.get("delta")
    u_star = 100 * answer.u_star if answer is not None and answer.u_star is not None else None

    times = [p.days for pts in data.values() for p in pts]
    mfus = [p.mfu for pts in data.values() for p in pts]
    window = s.get("window_days")
    t_lo = min(times) if window is None else min(min(times), window)
    t_hi = min(max(times), TIME_CAP * min(times))
    if window is not None:
        t_hi = max(t_hi, 1.3 * window)
    pad = 0.06 * (t_hi - t_lo)
    t_lim = (t_lo - pad, t_hi)
    m_top = max(mfus + ([u_star] if u_star is not None else []))
    m_pad = 0.06 * (m_top - min(mfus))
    m_lim = (min(mfus) - m_pad, m_top + m_pad)

    span = (hi - lo) / unit
    x_lim = (lo / unit - 0.03 * span, hi / unit + 0.03 * span)

    # layout, in inches
    entries = len(data) + (result.objective == "shortest_training")
    rows = math.ceil(entries / LEGEND_COLUMNS)
    left, right, bottom, gap = 0.42, 0.10, 0.36, 0.12
    legend_h, header_h = 0.04 + 0.115 * rows, 0.26
    W, H = left + width + right, bottom + 2 * panel + gap + legend_h + header_h + 0.06
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(W, H))
        ax_t = fig.add_axes((left / W, bottom / H, width / W, panel / H))
        ax_m = fig.add_axes(
            (left / W, (bottom + panel + gap) / H, width / W, panel / H), sharex=ax_t
        )
        centre = (left + width / 2) / W

        if result.objective == "shortest_training" and u_star is not None and delta is not None:
            floor = (1 - delta) * u_star
            ax_m.axhspan(floor, u_star, color=_mix(BAND, 12), lw=0, zorder=0)
        if window is not None:
            ax_t.axhline(window, color="#1F2328", lw=0.5, ls=(0, (6, 2, 1, 2)), zorder=2)
            ax_t.text(0.025, window, f"window {window:g} days",
                      transform=ax_t.get_yaxis_transform(),
                      fontsize=TINY, color="#1F2328", ha="left", va="bottom", zorder=2)  # fmt: skip

        chosen = answer.layers if answer is not None else None
        # answer's model on top
        for layers, pts in sorted(data.items(), key=lambda kv: kv[0] == chosen):
            style = dict(color=colours[layers], marker="o", mec="none")
            if layers == chosen:
                style.update(lw=0.9, ms=2.8, zorder=4)
            else:
                style.update(lw=0.55, ms=2.0, zorder=3)
            x = [p.n_gpu / unit for p in pts]
            ax_m.plot(x, [p.mfu for p in pts], **style)
            ax_t.plot(x, [p.days for p in pts], **style)

        if answer is not None:
            x, m, d = answer.n_gpu / unit, 100 * answer.mfu, answer.t_train_days
            for ax, y in ((ax_m, m), (ax_t, d)):
                ax.plot([x], [y], ls="none", marker="o", ms=4.2, color=ANSWER, mec="none", zorder=6,
                        clip_on=False)  # fmt: skip
            if result.objective == "shortest_training":
                ax_t.axhline(d, color=_mix(ANSWER, 35), lw=0.35, ls=(0, (3, 2)), zorder=1)
                what = "Fastest training\ninside the MFU tolerance"
            else:
                what = "Fewest GPUs\nwithin the window"
            model = rf"$\ell = {answer.layers}$, {_size(answer.n_params)}"
            note = f"{what}\n{model}\n{answer.n_gpu:,} GPUs, {d:.1f} days\nMFU {m:.1f}%"

            def to_axes(p: float, q: float) -> tuple[float, float]:
                return (
                    (p - x_lim[0]) / (x_lim[1] - x_lim[0]),
                    (q - t_lim[0]) / (t_lim[1] - t_lim[0]),
                )

            corner = _free_corner(
                [[to_axes(p.n_gpu / unit, p.days) for p in pts] for pts in data.values()],
                to_axes(x, d),
                None if window is None else to_axes(x, window)[1],
            )
            ax_t.annotate(note, (x, d), xytext=corner, textcoords="axes fraction",
                          fontsize=TINY, color=ANSWER, va="top", ha="left", zorder=7,
                          bbox=dict(boxstyle="square,pad=0.25", facecolor="white",
                                    edgecolor="none"),
                          arrowprops=dict(arrowstyle="-", lw=0.3, color=ANSWER, shrinkA=0,
                                          shrinkB=3))  # fmt: skip
        else:
            ax_t.text(0.5, 0.5, "no model meets the budget\non the GPU counts searched",
                      transform=ax_t.transAxes, fontsize=TINY, ha="center", va="center",
                      color=ANSWER, bbox=dict(facecolor="white", edgecolor="none"))  # fmt: skip

        ax_m.set_ylabel("MFU (%)", labelpad=2)
        ax_t.set_ylabel("Training time (days)", labelpad=2)
        ax_t.set_xlabel("GPU count" + (" (thousands)" if unit > 1 else ""), labelpad=2)
        ax_m.tick_params(labelbottom=False)
        ax_t.set_xlim(*x_lim)
        ax_t.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10])
        )
        ax_t.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax_t.set_ylim(*t_lim)
        ax_m.set_ylim(*m_lim)

        if answer is not None:  # dashed line joining the answer in both panels, through the gap
            top = ax_m.transData.transform((x, m))
            low = ax_t.transData.transform((x, d))
            fig.add_artist(Line2D(*fig.transFigure.inverted().transform([low, top]).T, color=ANSWER,
                                  lw=0.5, ls=(0, (3, 2)), zorder=5))  # fmt: skip

        handles = [
            Line2D([], [], color=colours[n], lw=0.9, marker="o", ms=2.4, mec="none",
                   label=rf"$\ell = {n}$ ({_size(models[n]['n_params'])})")
            for n in data
        ]  # fmt: skip
        if result.objective == "shortest_training" and u_star is not None and delta is not None:
            handles.append(Patch(facecolor=_mix(BAND, 12), edgecolor="none",
                                 label=f"MFU tolerance band ({delta:.0%})"))  # fmt: skip
        top_of_panels = bottom + 2 * panel + gap
        leg = fig.legend(handles=handles, loc="upper center", ncol=LEGEND_COLUMNS, fontsize=TINY,
                         bbox_to_anchor=(centre, (top_of_panels + legend_h + 0.02) / H),
                         frameon=True, fancybox=False, edgecolor=FRAME, framealpha=1,
                         handlelength=1.6, columnspacing=1.0, borderpad=0.3,
                         labelspacing=0.25)  # fmt: skip
        leg.get_frame().set_linewidth(0.4)
        line1, line2 = header(record, result, lo, hi)
        fig.text(
            centre, 1 - 0.02 / H, line1, fontsize=SMALL - 0.5, ha="center", va="top", weight="bold"
        )
        fig.text(centre, 1 - 0.14 / H, line2, fontsize=SMALL - 1, ha="center", va="top")
    shown = {"lo": lo, "hi": hi, "models": float(len(data))}
    if u_star is not None:
        shown["u_star"] = u_star
    return fig, shown


def plot_search(
    conn: duckdb.DuckDBPyConnection,
    result: ObjectiveResult,
    out_dir: str | Path,
    stem: str | None = None,
) -> list[Path]:
    """Draw ``result`` and write ``{stem}.png`` into ``out_dir``; return its path in a list."""
    out = Path(out_dir)
    fig, shown = draw(conn, result)
    out.mkdir(parents=True, exist_ok=True)
    stem = (
        stem
        or f"search_{result.search_id[:8]}_{result.objective}_N{shown['lo']:.0f}-{shown['hi']:.0f}"
    )
    path = out / f"{stem}.png"
    with matplotlib.rc_context(STYLE):  # fonts are looked up when drawing, not when creating
        fig.savefig(path, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    return [path]
