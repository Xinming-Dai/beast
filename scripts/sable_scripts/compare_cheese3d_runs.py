#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

BASE = Path("/home/jqh/NeuralWorkshops/beast")
OUT_DIR = BASE / "outputs" / "cheese3d_rebuttal_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATTERN = re.compile(
    r"step=(?P<step>\d+)/5000 loss=(?P<loss>[0-9.]+) l2=(?P<l2>[0-9.]+)\s+psnr=(?P<psnr>[0-9.]+) gs_reg=(?P<gs>[0-9.]+) perceptual=(?P<perc>[0-9.]+)"
)


def parse_metrics(path: Path):
    steps, psnr, loss, gs = [], [], [], []
    text = path.read_text()
    for match in PATTERN.finditer(text):
        steps.append(int(match.group("step")))
        loss.append(float(match.group("loss")))
        psnr.append(float(match.group("psnr")))
        gs.append(float(match.group("gs")))
    return {"steps": steps, "psnr": psnr, "loss": loss, "gs": gs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Cheese3D training logs")
    parser.add_argument("--phase1-log", type=Path, required=True)
    parser.add_argument("--phase3-log", type=Path, required=True)
    args = parser.parse_args()

    logs = {
        "phase1": args.phase1_log,
        "phase3": args.phase3_log,
    }
    parsed = {name: parse_metrics(path) for name, path in logs.items()}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, color in [("phase1", "tab:blue"), ("phase3", "tab:orange")]:
        data = parsed[name]
        if not data["steps"]:
            continue
        axes[0].plot(data["steps"], data["psnr"], label=name, color=color)
        axes[1].plot(data["steps"], data["loss"], label=name, color=color)
        axes[2].plot(data["steps"], data["gs"], label=name, color=color)

    axes[0].set_title("PSNR")
    axes[1].set_title("Loss")
    axes[2].set_title("GS Reg")
    for ax in axes:
        ax.set_xlabel("step")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase1_vs_phase3_early_curves.png", dpi=150)

    summary_lines = []
    for name in ["phase1", "phase3"]:
        data = parsed[name]
        if not data["steps"]:
            summary_lines.append(f"{name}: no metrics parsed")
            continue
        idx = -1
        summary_lines.append(
            f"{name}: last_step={data['steps'][idx]}, psnr={data['psnr'][idx]:.4f}, loss={data['loss'][idx]:.4f}, gs_reg={data['gs'][idx]:.4f}"
        )
        best_psnr = max(data["psnr"])
        best_step = data["steps"][data["psnr"].index(best_psnr)]
        summary_lines.append(
            f"{name}: best_psnr={best_psnr:.4f} at step={best_step}"
        )
    (OUT_DIR / "early_comparison_summary.txt").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
