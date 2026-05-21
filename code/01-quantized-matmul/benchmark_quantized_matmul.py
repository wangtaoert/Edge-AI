from __future__ import annotations

import argparse
from time import perf_counter

from quantized_matmul import (
    fp32_matmul,
    make_demo_inputs,
    quantized_matmul,
    quantized_matmul_per_channel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark quantized matmul variants.")
    parser.add_argument("--m", type=int, default=16, help="Rows of A.")
    parser.add_argument("--k", type=int, default=32, help="Columns of A / rows of B.")
    parser.add_argument("--n", type=int, default=16, help="Columns of B.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--repeat", type=int, default=100, help="Repeat count for vectorized implementations.")
    parser.add_argument("--loop-repeat", type=int, default=3, help="Repeat count for the slow Python loop.")
    return parser.parse_args()


def measure(name: str, repeat: int, fn) -> tuple[str, float]:
    start = perf_counter()
    for _ in range(repeat):
        fn()
    elapsed_ms = (perf_counter() - start) * 1000.0 / repeat
    return name, elapsed_ms


def main() -> None:
    args = parse_args()
    a, b, bias = make_demo_inputs(args.m, args.k, args.n, args.seed)

    cases = [
        (
            "fp32 numpy matmul",
            args.repeat,
            lambda: fp32_matmul(a, b, bias),
        ),
        (
            "int8 tensor numpy accumulator",
            args.repeat,
            lambda: quantized_matmul(a, b, bias, symmetric=True, quantize_bias=True),
        ),
        (
            "int8 per-channel numpy accumulator",
            args.repeat,
            lambda: quantized_matmul_per_channel(a, b, bias, symmetric=True, quantize_bias=True),
        ),
        (
            "int8 tensor python-loop accumulator",
            args.loop_repeat,
            lambda: quantized_matmul(a, b, bias, symmetric=True, use_loop=True, quantize_bias=True),
        ),
    ]

    print(f"shape: A=({args.m}, {args.k}), B=({args.k}, {args.n})")
    for name, repeat, fn in cases:
        case_name, elapsed_ms = measure(name, repeat, fn)
        print(f"{case_name:36s} {elapsed_ms:10.4f} ms/run  repeat={repeat}")


if __name__ == "__main__":
    main()
