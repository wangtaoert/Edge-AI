from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from softmax_from_scratch import int8_dequant_softmax, make_demo_logits, stable_softmax


def softmax_rowwise_loop(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"row-wise softmax expects a 2D array, got shape {x.shape}.")

    out = np.empty_like(x, dtype=np.float32)
    for row_idx in range(x.shape[0]):
        row = x[row_idx]
        shifted = row - np.max(row)
        exp_row = np.exp(shifted)
        out[row_idx] = exp_row / np.sum(exp_row)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark softmax variants.")
    parser.add_argument("--batch", type=int, default=64, help="Number of rows.")
    parser.add_argument("--classes", type=int, default=128, help="Number of classes per row.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--case", choices=["small", "large", "close", "wide"], default="small", help="Logit distribution.")
    parser.add_argument("--repeat", type=int, default=200, help="Repeat count for vectorized implementations.")
    parser.add_argument("--loop-repeat", type=int, default=10, help="Repeat count for the slower Python loop.")
    return parser.parse_args()


def measure(name: str, repeat: int, fn) -> tuple[str, float]:
    start = perf_counter()
    for _ in range(repeat):
        fn()
    elapsed_ms = (perf_counter() - start) * 1000.0 / repeat
    return name, elapsed_ms


def main() -> None:
    args = parse_args()
    logits = make_demo_logits(args.batch, args.classes, args.seed, args.case)

    cases = [
        (
            "stable fp32 numpy",
            args.repeat,
            lambda: stable_softmax(logits, axis=1, dtype=np.float32),
        ),
        (
            "stable fp16 numpy",
            args.repeat,
            lambda: stable_softmax(logits, axis=1, dtype=np.float16),
        ),
        (
            "int8 dequant + stable fp32",
            args.repeat,
            lambda: int8_dequant_softmax(logits, axis=1),
        ),
        (
            "stable fp32 python row loop",
            args.loop_repeat,
            lambda: softmax_rowwise_loop(logits),
        ),
    ]

    print(f"shape: logits=({args.batch}, {args.classes}), case={args.case}")
    for name, repeat, fn in cases:
        case_name, elapsed_ms = measure(name, repeat, fn)
        print(f"{case_name:32s} {elapsed_ms:10.4f} ms/run  repeat={repeat}")


if __name__ == "__main__":
    main()
