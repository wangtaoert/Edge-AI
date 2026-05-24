from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys

try:
    import numpy as np
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    print(
        "Missing dependency: numpy\n\n"
        "Install it in the Python environment you use to run this script:\n"
        "  python -m pip install -r code\\02-softmax-from-scratch\\requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


@dataclass(frozen=True)
class QuantizationParams:
    scale: float
    zero_point: int
    qmin: int = -128
    qmax: int = 127


def naive_softmax(x: np.ndarray, axis: int = -1, dtype: np.dtype = np.float32) -> np.ndarray:
    x = np.asarray(x, dtype=dtype)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def stable_softmax(x: np.ndarray, axis: int = -1, dtype: np.dtype = np.float32) -> np.ndarray:
    x = np.asarray(x, dtype=dtype)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_rowwise(x: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"row-wise softmax expects a 2D array, got shape {x.shape}.")
    return stable_softmax(x, axis=1, dtype=dtype)


def calculate_qparams(x: np.ndarray, *, num_bits: int = 8, symmetric: bool = False) -> QuantizationParams:
    if num_bits != 8:
        raise ValueError("This exercise only implements int8 quantization.")

    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1
    x = np.asarray(x, dtype=np.float32)

    if symmetric:
        max_abs = float(np.max(np.abs(x)))
        scale = max_abs / qmax if max_abs != 0.0 else 1.0
        return QuantizationParams(scale=scale, zero_point=0, qmin=qmin, qmax=qmax)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_min == x_max:
        return QuantizationParams(scale=1.0, zero_point=0, qmin=qmin, qmax=qmax)

    scale = (x_max - x_min) / float(qmax - qmin)
    zero_point = int(round(qmin - x_min / scale))
    zero_point = int(np.clip(zero_point, qmin, qmax))
    return QuantizationParams(scale=scale, zero_point=zero_point, qmin=qmin, qmax=qmax)


def quantize_int8(x: np.ndarray, params: QuantizationParams) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    q = np.round(x / params.scale + params.zero_point)
    q = np.clip(q, params.qmin, params.qmax)
    return q.astype(np.int8)


def dequantize_int8(q: np.ndarray, params: QuantizationParams) -> np.ndarray:
    q = np.asarray(q, dtype=np.int8)
    return params.scale * (q.astype(np.float32) - params.zero_point)


def int8_dequant_softmax(
    x: np.ndarray,
    *,
    axis: int = -1,
    symmetric: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, QuantizationParams]:
    params = calculate_qparams(x, symmetric=symmetric)
    qx = quantize_int8(x, params)
    deq_x = dequantize_int8(qx, params)
    y = stable_softmax(deq_x, axis=axis, dtype=np.float32)
    return y, qx, deq_x, params


def compare_outputs(reference: np.ndarray, actual: np.ndarray, *, axis: int = -1) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    abs_error = np.abs(reference - actual)
    relative_error = abs_error / np.maximum(np.abs(reference), 1e-6)
    sum_error = np.abs(np.sum(actual, axis=axis) - 1.0)

    return {
        "max_abs_error": float(np.max(abs_error)),
        "mean_abs_error": float(np.mean(abs_error)),
        "mean_relative_error": float(np.mean(relative_error)),
        "max_sum_error": float(np.max(sum_error)),
    }


def make_demo_logits(batch: int, classes: int, seed: int, case: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if case == "small":
        return rng.normal(loc=0.0, scale=1.0, size=(batch, classes)).astype(np.float32)
    if case == "large":
        return rng.normal(loc=1000.0, scale=2.0, size=(batch, classes)).astype(np.float32)
    if case == "close":
        base = rng.normal(loc=0.0, scale=0.01, size=(batch, classes))
        return base.astype(np.float32)
    if case == "wide":
        return rng.normal(loc=0.0, scale=12.0, size=(batch, classes)).astype(np.float32)
    if case == "manual":
        values = np.array(
            [
                [1.0, 2.0, 3.0, 0.0, -1.0],
                [10.0, 20.0, 30.0, 0.0, -10.0],
                [1000.0, 1001.0, 1002.0, 999.0, 998.0],
                [0.01, 0.02, 0.03, 0.00, -0.01],
            ],
            dtype=np.float32,
        )
        if classes != values.shape[1]:
            raise ValueError("--case manual requires --classes 5.")
        repeats = int(np.ceil(batch / values.shape[0]))
        return np.tile(values, (repeats, 1))[:batch]
    raise ValueError(f"Unknown case: {case}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Softmax from scratch: stability and low-precision experiments.")
    parser.add_argument("--batch", type=int, default=4, help="Number of rows.")
    parser.add_argument("--classes", type=int, default=5, help="Number of classes per row.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--case", choices=["small", "large", "close", "wide", "manual"], default="manual", help="Logit distribution.")
    parser.add_argument("--symmetric", action="store_true", help="Use symmetric int8 quantization for the int8-dequant experiment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logits = make_demo_logits(args.batch, args.classes, args.seed, args.case)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        naive = naive_softmax(logits, axis=1, dtype=np.float32)
    stable = softmax_rowwise(logits, dtype=np.float32)
    fp16 = softmax_rowwise(logits, dtype=np.float16).astype(np.float32)
    int8_y, qx, deq_x, qparams = int8_dequant_softmax(logits, axis=1, symmetric=args.symmetric)

    print(f"shape: logits={logits.shape}, case={args.case}")
    print(f"logits range: [{np.min(logits):.6f}, {np.max(logits):.6f}]")
    print(f"naive contains_nan={np.isnan(naive).any()}, contains_inf={np.isinf(naive).any()}")
    print(f"stable row sum range: [{np.sum(stable, axis=1).min():.8f}, {np.sum(stable, axis=1).max():.8f}]")
    print(f"int8 qparams: scale={qparams.scale:.8f}, zero_point={qparams.zero_point}")
    print(f"int8 logits range: [{qx.min()}, {qx.max()}]")

    metrics = {
        "naive_vs_stable": compare_outputs(stable, naive, axis=1),
        "fp16_vs_fp32": compare_outputs(stable, fp16, axis=1),
        "int8_dequant_vs_fp32": compare_outputs(stable, int8_y, axis=1),
    }
    for title, values in metrics.items():
        print(title)
        for name, value in values.items():
            print(f"  {name}: {value:.8f}")

    row = 0
    print("sample row 0")
    print(f"  logits:       {np.array2string(logits[row], precision=5, floatmode='fixed')}")
    print(f"  dequantized:  {np.array2string(deq_x[row], precision=5, floatmode='fixed')}")
    print(f"  stable fp32:  {np.array2string(stable[row], precision=6, floatmode='fixed')}")
    print(f"  fp16:         {np.array2string(fp16[row], precision=6, floatmode='fixed')}")
    print(f"  int8-dequant: {np.array2string(int8_y[row], precision=6, floatmode='fixed')}")


if __name__ == "__main__":
    main()
