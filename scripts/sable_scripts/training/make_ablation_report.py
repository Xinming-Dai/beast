#!/usr/bin/env python3
"""Generate ablation_table.md and ablation_results.csv from metrics.csv.

This is Phase 8 of the loss-weighting ablation plan. It reads the per-cell
metrics produced by collect_ablation_metrics.py and writes:

- ablation_results.csv   machine-readable per-cell metrics (rows = cells, cols = scalars)
- ablation_table.md      markdown table for the rebuttal text. The "default" row
                         anchors the numbers; the other 7 rows are deltas from
                         default.

The provenance block at the bottom of the table is the one required by the
plan; it is intentionally short so it can be pasted verbatim into the
rebuttal.

Usage:
    python make_ablation_report.py --metrics eval/metrics.csv --out .
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = "/home/jqh/NeuralWorkshops/beast"

# 8 cells (name, train_weights)
CELLS = [
    ("default", 1.0, 0.3, 1.0),
    ("no-percept", 1.0, 0.0, 1.0),
    ("low-percept", 1.0, 0.1, 1.0),
    ("high-percept", 1.0, 1.0, 1.0),
    ("low-recon", 0.5, 0.3, 1.0),
    ("high-recon", 2.0, 0.3, 1.0),
    ("no-geom", 1.0, 0.3, 0.0),
    ("high-geom", 1.0, 0.3, 2.0),
]
EXPECTED_CHECKPOINT_STEP = 10000
EXPECTED_EVAL_PAIRS = 164
REQUIRED_METRICS = (
    "avg_l2_loss",
    "avg_psnr",
    "avg_ssim",
    "avg_perceptual_loss",
    "avg_gs_reg_loss",
    "avg_weighted_loss",
)

PROVENANCE = """\
> Controlled loss-weight sensitivity study with fixed online VDA initialization.
> Single eid (4b00df29-...). Online VDA, encoder=vitb, metric=false, checkpoint
> pinned at SHA256 775e578e8f9431ec0496514aa466bd0a1f67c28d0f518267809f35a43c04329b.
> Batch 12 x grad_accum 2 = 24 effective batch. 10000 steps, warmup 1000.
> Quantitative metrics use all 164 pairs in the frozen validation split; four
> manifest-pinned pairs are used only for the qualitative grid.
> Deviation from Mia's reference run: this study uses online VDA on a single
> session (4b00), while Mia's reference run used precomputed VDA across 3
> sessions (72cb/781b/ecb5). Online VDA and the single-session data contract are
> held fixed across all cells, so only the three loss weights vary.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="Path to metrics.csv from collect_ablation_metrics.py.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("."), help="Output directory for ablation_*.{csv,md}."
    )
    parser.add_argument(
        "--cells",
        type=str,
        default=None,
        help="Comma-ordered cell names to include (default: all 8).",
    )
    parser.add_argument(
        "--default", type=str, default="default", help="Anchor cell name for delta rows."
    )
    return parser.parse_args()


def load_metrics(metrics_csv: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with metrics_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["cell"]] = row
    return rows


def validate_complete_results(rows: dict[str, dict], ordered: list[str]) -> None:
    """Reject partial, failed, or wrong-step inputs before writing a report."""
    errors = []
    for name in ordered:
        row = rows.get(name)
        if row is None:
            errors.append(f"{name}: missing row")
            continue
        if row.get("status") != "ok":
            errors.append(f"{name}: status={row.get('status')!r}")
        if row.get("checkpoint_step") != str(EXPECTED_CHECKPOINT_STEP):
            errors.append(
                f"{name}: checkpoint_step={row.get('checkpoint_step')!r}, "
                f"expected {EXPECTED_CHECKPOINT_STEP}"
            )
        if row.get("num_eval_pairs") != str(EXPECTED_EVAL_PAIRS):
            errors.append(
                f"{name}: num_eval_pairs={row.get('num_eval_pairs')!r}, "
                f"expected {EXPECTED_EVAL_PAIRS}"
            )
        for metric in REQUIRED_METRICS:
            value = safe_float(row.get(metric, ""))
            if value != value:
                errors.append(f"{name}: missing/non-finite {metric}")
    if errors:
        raise RuntimeError(
            "refusing to report incomplete ablation results:\n- " + "\n- ".join(errors)
        )


def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def fmt(x, decimals: int = 3) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:.{decimals}f}"


def fmt_delta(x, decimals: int = 3) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    sign = "+" if x >= 0 else "−"
    return f"{sign}{abs(x):.{decimals}f}"


def write_ablation_results_csv(rows: dict[str, dict], ordered: list[str], out_path: Path) -> None:
    fieldnames = [
        "cell",
        "train_l2",
        "train_perceptual",
        "train_gs_reg",
        "raw_l2_loss",
        "raw_perceptual_loss",
        "raw_gs_reg_loss",
        "raw_lpips_loss",
        "raw_psnr",
        "raw_ssim",
        "weighted_loss",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for name in ordered:
            r = rows.get(name, {})
            writer.writerow(
                [
                    name,
                    r.get("train_l2_loss_weight", r.get("train_l2", "")),
                    r.get("train_perceptual_loss_weight", r.get("train_perceptual", "")),
                    r.get("train_gs_reg_loss_weight", r.get("train_gs_reg", "")),
                    r.get("avg_l2_loss", r.get("raw_l2_loss", "")),
                    r.get("avg_perceptual_loss", r.get("raw_perceptual_loss", "")),
                    r.get("avg_gs_reg_loss", r.get("raw_gs_reg_loss", "")),
                    r.get("avg_lpips_loss", r.get("raw_lpips_loss", "")),
                    r.get("avg_psnr", r.get("raw_psnr", "")),
                    r.get("avg_ssim", r.get("raw_ssim", "")),
                    r.get("avg_weighted_loss", r.get("weighted_loss", "")),
                ]
            )


def write_ablation_table_md(
    rows: dict[str, dict],
    ordered: list[str],
    default_cell: str,
    out_path: Path,
) -> None:
    """Write the markdown table for the rebuttal text.

    The table contains weights, raw losses, image metrics, and the weighted sum.
    """
    default_row = rows.get(default_cell, {})

    lines = []
    lines.append("# Loss-weighting ablation — results")
    lines.append("")
    lines.append(
        "Raw metrics are computed with all loss weights set to 1.0 for fair "
        "comparison across cells. The weighted column uses each cell's training "
        "weights `lambda_i * L_i`."
    )
    lines.append("")
    lines.append(
        "| cell | l2 | p | g | raw L_recon | raw L_percept | raw L_geom | "
        "raw PSNR | raw SSIM | weighted λiLi |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name in ordered:
        r = rows.get(name, {})
        l2_w = safe_float(r.get("train_l2_loss_weight", r.get("train_l2", "")))
        p_w = safe_float(r.get("train_perceptual_loss_weight", r.get("train_perceptual", "")))
        g_w = safe_float(r.get("train_gs_reg_loss_weight", r.get("train_gs_reg", "")))
        raw_l2 = safe_float(r.get("avg_l2_loss", r.get("raw_l2_loss", "")))
        raw_p = safe_float(r.get("avg_perceptual_loss", r.get("raw_perceptual_loss", "")))
        raw_g = safe_float(r.get("avg_gs_reg_loss", r.get("raw_gs_reg_loss", "")))
        raw_psnr = safe_float(r.get("avg_psnr", r.get("raw_psnr", "")))
        raw_ssim = safe_float(r.get("avg_ssim", r.get("raw_ssim", "")))
        weighted = safe_float(r.get("avg_weighted_loss", r.get("weighted_loss", "")))
        lines.append(
            f"| {name} | {fmt(l2_w, 1)} | {fmt(p_w, 1)} | {fmt(g_w, 1)} | "
            f"{fmt(raw_l2, 4)} | {fmt(raw_p, 4)} | {fmt(raw_g, 4)} | "
            f"{fmt(raw_psnr, 2)} | {fmt(raw_ssim, 3)} | {fmt(weighted, 4)} |"
        )

    # Delta rows (everything except `default`).
    lines.append("")
    lines.append(
        "**Deltas vs. `default`** (positive = worse than default for L_*; better for PSNR/SSIM):"
    )
    lines.append("")
    lines.append("| cell | ΔL_recon | ΔL_percept | ΔL_geom | ΔPSNR | ΔSSIM | Δweighted |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in ordered:
        if name == default_cell:
            continue
        r = rows.get(name, {})
        d_l2 = safe_float(r.get("avg_l2_loss", r.get("raw_l2_loss", ""))) - safe_float(
            default_row.get("avg_l2_loss", default_row.get("raw_l2_loss", ""))
        )
        d_p = safe_float(
            r.get("avg_perceptual_loss", r.get("raw_perceptual_loss", ""))
        ) - safe_float(
            default_row.get("avg_perceptual_loss", default_row.get("raw_perceptual_loss", ""))
        )
        d_g = safe_float(r.get("avg_gs_reg_loss", r.get("raw_gs_reg_loss", ""))) - safe_float(
            default_row.get("avg_gs_reg_loss", default_row.get("raw_gs_reg_loss", ""))
        )
        d_psnr = safe_float(r.get("avg_psnr", r.get("raw_psnr", ""))) - safe_float(
            default_row.get("avg_psnr", default_row.get("raw_psnr", ""))
        )
        d_ssim = safe_float(r.get("avg_ssim", r.get("raw_ssim", ""))) - safe_float(
            default_row.get("avg_ssim", default_row.get("raw_ssim", ""))
        )
        d_w = safe_float(r.get("avg_weighted_loss", r.get("weighted_loss", ""))) - safe_float(
            default_row.get("avg_weighted_loss", default_row.get("weighted_loss", ""))
        )
        lines.append(
            f"| {name} | {fmt_delta(d_l2, 4)} | {fmt_delta(d_p, 4)} | {fmt_delta(d_g, 4)} | "
            f"{fmt_delta(d_psnr, 2)} | {fmt_delta(d_ssim, 3)} | {fmt_delta(d_w, 4)} |"
        )

    lines.append("")
    lines.append("**Provenance:**")
    lines.append("")
    lines.append(PROVENANCE.rstrip())

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main() -> int:
    args = parse_args()
    rows = load_metrics(args.metrics)
    if args.cells:
        ordered = [c.strip() for c in args.cells.split(",")]
    else:
        ordered = [c[0] for c in CELLS]

    validate_complete_results(rows, ordered)
    args.out.mkdir(parents=True, exist_ok=True)
    write_ablation_results_csv(rows, ordered, args.out / "ablation_results.csv")
    write_ablation_table_md(rows, ordered, args.default, args.out / "ablation_table.md")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
