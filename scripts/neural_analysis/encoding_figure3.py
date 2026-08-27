#!/usr/bin/env python3
"""Draw the vertical Figure 3 encoding schematic.

The schematic is intentionally data-free: feature tokens flow through a Gaussian
decoder into a compact neural-activity raster, matching the visual language of
the Figure 3 neural panels.

Example
-------
python src/analyses/neural_analysis/encoding_figure3.py \
  --output /work/nvme/bfsr/xdai3/project3d/plotting/figure3/encoding_schematic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


DEFAULT_OUTPUT = Path("encoding_figure3_vertical")
GREEN = "#8de38e"
PEACH = "#ffd0bb"
DECODER_EDGE = "#7aa7f2"
DECODER_SHADOW = "#b8b8b8"
TRACE_COLOR = "#666666"


def _hide_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_feature_tokens(ax: plt.Axes) -> None:
    """Draw a vertical stack of small feature-token boxes above the decoder."""
    x0 = 0.43
    y0 = 0.775
    width = 0.14
    height = 0.035
    gap = 0.012
    colors = [GREEN, GREEN, GREEN, PEACH, PEACH, PEACH, "white", "white"]

    for idx, color in enumerate(colors):
        y = y0 - idx * (height + gap)
        edge = GREEN if idx < 6 and color == "white" else PEACH if color == "white" else color
        rect = Rectangle(
            (x0, y),
            width,
            height,
            facecolor=color,
            edgecolor=edge,
            linewidth=1.35,
            joinstyle="miter",
        )
        ax.add_patch(rect)


def draw_decoder(ax: plt.Axes) -> tuple[float, float]:
    """Draw the Gaussian decoder box and return the bottom-center anchor."""
    x0, y0 = 0.33, 0.405
    width, height = 0.34, 0.19
    box = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        facecolor="white",
        edgecolor=DECODER_EDGE,
        linewidth=2.5,
    )
    box.set_path_effects(
        [
            patheffects.SimplePatchShadow(
                offset=(2.5, -2.5),
                shadow_rgbFace=DECODER_SHADOW,
                alpha=0.45,
            ),
            patheffects.Normal(),
        ]
    )
    ax.add_patch(box)

    ax.text(
        x0 + width / 2,
        y0 + height * 0.61,
        "Gaussian\nDecoder",
        ha="center",
        va="center",
        fontsize=17,
        fontstyle="italic",
        fontfamily="DejaVu Sans",
    )
    ax.text(
        x0 + width / 2,
        y0 + height * 0.23,
        r"$g$",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
    )
    return x0 + width / 2, y0


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color="black",
        shrinkA=7,
        shrinkB=6,
    )
    ax.add_patch(arrow)


def _synthetic_activity(
    n_neurons: int = 42,
    n_time: int = 150,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic wavy traces with sparse spike-like peaks."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 1, n_time)
    traces = np.empty((n_neurons, n_time), dtype=np.float64)

    for neuron_idx in range(n_neurons):
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(4.0, 8.5)
        trace = 0.10 * np.sin(2 * np.pi * freq * time + phase)
        trace += 0.045 * rng.standard_normal(n_time)

        n_events = rng.integers(3, 7)
        centers = rng.choice(n_time, size=n_events, replace=False)
        for center in centers:
            width = rng.uniform(1.5, 3.5)
            amp = rng.uniform(0.18, 0.38)
            trace += amp * np.exp(-0.5 * ((np.arange(n_time) - center) / width) ** 2)

        traces[neuron_idx] = trace

    return time, traces


def draw_neural_activity(ax: plt.Axes, seed: int) -> None:
    """Draw the neural-activity label and stacked traces."""
    ax.text(
        0.20,
        0.18,
        "Neural\nactivity",
        ha="right",
        va="center",
        fontsize=18,
        fontfamily="DejaVu Serif",
    )

    x_left, x_right = 0.27, 0.84
    y_bottom, y_top = 0.035, 0.325
    time, traces = _synthetic_activity(seed=seed)
    offsets = np.linspace(y_top, y_bottom, traces.shape[0])
    scale = 0.015

    for offset, trace in zip(offsets, traces):
        trace = trace - np.nanmean(trace)
        x = x_left + time * (x_right - x_left)
        y = offset + scale * trace
        ax.plot(x, y, color=TRACE_COLOR, linewidth=1.0, solid_capstyle="round")


def build_figure(seed: int = 7) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 7.2))
    _hide_axes(ax)
    draw_feature_tokens(ax)
    decoder_bottom = draw_decoder(ax)
    draw_arrow(ax, decoder_bottom, (0.50, 0.33))
    draw_neural_activity(ax, seed)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def save_figure(fig: plt.Figure, output: Path, dpi: int, formats: list[str]) -> list[Path]:
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        path = output if output.suffix.lower() == f".{fmt}" else output.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.04)
        saved_paths.append(path)
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without suffix, or with suffix when saving a single format.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["svg", "png"],
        help="One or more output formats understood by Matplotlib.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for the synthetic raster.")
    parser.add_argument("--show", action="store_true", help="Open an interactive preview.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = args.formats
    if args.output.suffix and len(formats) == 1:
        formats = [args.output.suffix.lstrip(".")]

    fig = build_figure(seed=args.seed)
    saved_paths = save_figure(fig, args.output, args.dpi, formats)
    for path in saved_paths:
        print(path)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
