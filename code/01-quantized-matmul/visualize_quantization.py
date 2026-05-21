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
        "  python -m pip install -r code\\01-quantized-matmul\\requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

from quantized_matmul import (
    compare_outputs,
    dequantize,
    fp32_matmul,
    make_demo_inputs,
    quantize,
    quantized_matmul,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the int8 quantization process.")
    parser.add_argument("--m", type=int, default=8, help="Rows of A.")
    parser.add_argument("--k", type=int, default=16, help="Columns of A / rows of B.")
    parser.add_argument("--n", type=int, default=8, help="Columns of B.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--symmetric", action="store_true", help="Use symmetric int8 quantization.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("code/01-quantized-matmul/quantization_visualization.png"),
        help="Path of the generated PNG image.",
    )
    return parser.parse_args()


def add_matrix_image(ax: plt.Axes, matrix: np.ndarray, title: str) -> None:
    image = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def plot_quantization(args: argparse.Namespace) -> dict[str, float]:
    a, b, bias = make_demo_inputs(args.m, args.k, args.n, args.seed)
    fp32_out = fp32_matmul(a, b, bias)
    int8_out, qa, _qb, a_params, _b_params = quantized_matmul(
        a,
        b,
        bias,
        symmetric=args.symmetric,
    )
    deq_a = dequantize(qa, a_params)
    metrics = compare_outputs(fp32_out, int8_out)

    flat_a = a.ravel()
    flat_deq_a = deq_a.ravel()
    order = np.argsort(flat_a)
    sorted_a = flat_a[order]
    sorted_deq_a = flat_deq_a[order]
    sorted_q = quantize(sorted_a, a_params).astype(np.int16)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("INT8 Quantization: FP32 -> INT8 -> Dequantized Output", fontsize=14)

    ax = axes[0, 0]
    sample_count = min(96, sorted_a.size)
    xs = np.arange(sample_count)
    ax.plot(xs, sorted_a[:sample_count], label="FP32 value", linewidth=2)
    ax.step(xs, sorted_deq_a[:sample_count], label="dequantized value", where="mid")
    ax.set_title("Quantization turns continuous values into discrete steps")
    ax.set_xlabel("sorted sample index")
    ax.set_ylabel("value")
    ax.legend()

    ax = axes[0, 1]
    ax.scatter(sorted_a, sorted_q, s=18, alpha=0.8)
    ax.axhline(a_params.zero_point, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"FP32 value mapped to INT8 code (zero_point={a_params.zero_point})")
    ax.set_xlabel("FP32 value")
    ax.set_ylabel("INT8 code")
    ax.grid(True, alpha=0.25)

    add_matrix_image(axes[1, 0], a, "Original matrix A (FP32)")
    add_matrix_image(axes[1, 1], deq_a - a, "Quantization error of A (dequantized - FP32)")

    metric_text = "\n".join(
        [
            f"A scale: {a_params.scale:.6f}",
            f"A zero_point: {a_params.zero_point}",
            f"max_abs_error: {metrics['max_abs_error']:.6f}",
            f"mean_abs_error: {metrics['mean_abs_error']:.6f}",
            f"mean_relative_error: {metrics['mean_relative_error']:.6f}",
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
    return metrics


def main() -> None:
    args = parse_args()
    metrics = plot_quantization(args)
    print(f"saved: {args.output}")
    for name, value in metrics.items():
        print(f"{name}: {value:.8f}")


if __name__ == "__main__":
    main()
