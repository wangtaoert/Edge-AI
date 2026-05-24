from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as exc:
    missing = exc.name
    print(
        f"Missing dependency: {missing}\n\n"
        "Install dependencies in the Python environment you use to run this script:\n"
        "  python -m pip install -r code\\02-softmax-from-scratch\\requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

from softmax_from_scratch import (
    compare_outputs,
    int8_dequant_softmax,
    make_demo_logits,
    naive_softmax,
    stable_softmax,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize softmax stability and precision effects.")
    parser.add_argument("--case", choices=["small", "large", "close", "wide", "manual"], default="manual", help="Logit distribution.")
    parser.add_argument("--row", type=int, default=0, help="Row to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--batch", type=int, default=4, help="Number of rows.")
    parser.add_argument("--classes", type=int, default=5, help="Number of classes per row.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("code/02-softmax-from-scratch/softmax_visualization.png"),
        help="Path of the generated PNG image.",
    )
    return parser.parse_args()


def plot_softmax(args: argparse.Namespace) -> dict[str, float]:
    logits = make_demo_logits(args.batch, args.classes, args.seed, args.case)
    row_index = args.row % logits.shape[0]
    row = logits[row_index]

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        naive = naive_softmax(logits, axis=1, dtype=np.float32)
        naive_exp = np.exp(row.astype(np.float32))
    stable = stable_softmax(logits, axis=1, dtype=np.float32)
    fp16 = stable_softmax(logits, axis=1, dtype=np.float16).astype(np.float32)
    int8_y, _qx, deq_x, qparams = int8_dequant_softmax(logits, axis=1)

    shifted = row - np.max(row)
    stable_exp = np.exp(shifted.astype(np.float32))
    classes = np.arange(row.shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("Softmax from Scratch: Stability and Precision", fontsize=14)

    ax = axes[0, 0]
    width = 0.36
    ax.bar(classes - width / 2, row, width=width, label="FP32 logits")
    ax.bar(classes + width / 2, deq_x[row_index], width=width, label="INT8 dequant logits")
    ax.set_title("Logits before softmax")
    ax.set_xlabel("class")
    ax.set_ylabel("logit")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[0, 1]
    ax.plot(classes, naive_exp, marker="o", label="exp(logits)")
    ax.plot(classes, stable_exp, marker="o", label="exp(logits - max)")
    ax.set_title("Subtracting max keeps exp bounded")
    ax.set_xlabel("class")
    ax.set_ylabel("exp value")
    ax.set_yscale("symlog")
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.plot(classes, stable[row_index], marker="o", linewidth=2, label="stable fp32")
    ax.plot(classes, fp16[row_index], marker="s", linestyle="--", label="fp16")
    ax.plot(classes, int8_y[row_index], marker="^", linestyle="--", label="int8 dequant")
    ax.set_title("Softmax output")
    ax.set_xlabel("class")
    ax.set_ylabel("probability")
    ax.set_ylim(bottom=-0.02, top=max(1.02, float(np.max(stable[row_index])) * 1.15))
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    errors = {
        "naive": compare_outputs(stable, naive, axis=1)["max_abs_error"],
        "fp16": compare_outputs(stable, fp16, axis=1)["max_abs_error"],
        "int8": compare_outputs(stable, int8_y, axis=1)["max_abs_error"],
    }
    ax.bar(errors.keys(), errors.values(), color=["#7f8c8d", "#2e86ab", "#c44e52"])
    ax.set_title("Max absolute error vs stable FP32")
    ax.set_ylabel("max abs error")
    ax.grid(True, axis="y", alpha=0.25)

    metric_text = "\n".join(
        [
            f"case: {args.case}",
            f"row: {row_index}",
            f"q scale: {qparams.scale:.6f}",
            f"q zero_point: {qparams.zero_point}",
            f"naive has NaN: {np.isnan(naive).any()}",
        ]
    )
    axes[0, 0].text(
        0.02,
        0.98,
        metric_text,
        transform=axes[0, 0].transAxes,
        ha="left",
        va="top",
        family="monospace",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    return compare_outputs(stable, int8_y, axis=1)


def main() -> None:
    args = parse_args()
    metrics = plot_softmax(args)
    print(f"saved: {args.output}")
    for name, value in metrics.items():
        print(f"{name}: {value:.8f}")


if __name__ == "__main__":
    main()
