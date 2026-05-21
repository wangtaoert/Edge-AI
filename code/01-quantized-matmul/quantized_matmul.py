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
        "  python -m pip install -r code\\01-quantized-matmul\\requirements.txt\n\n"
        "If VS Code still reports this error, check that the selected Python "
        "interpreter is the same one where NumPy was installed.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


@dataclass(frozen=True)
class QuantizationParams:
    scale: float
    zero_point: int
    qmin: int = -128
    qmax: int = 127


def calculate_qparams(
    x: np.ndarray,
    *,
    num_bits: int = 8,
    symmetric: bool = False,
) -> QuantizationParams:
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


def quantize(x: np.ndarray, params: QuantizationParams) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    q = np.round(x / params.scale + params.zero_point)
    q = np.clip(q, params.qmin, params.qmax)
    return q.astype(np.int8)


def dequantize(q: np.ndarray, params: QuantizationParams) -> np.ndarray:
    q = np.asarray(q, dtype=np.int8)
    return params.scale * (q.astype(np.float32) - params.zero_point)


def fp32_matmul(a: np.ndarray, b: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    y = np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32)
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float32)
    return y


def quantized_matmul(
    a: np.ndarray,
    b: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    symmetric: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, QuantizationParams, QuantizationParams]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be 2D matrices.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Shape mismatch: A{a.shape} cannot multiply B{b.shape}.")

    a_params = calculate_qparams(a, symmetric=symmetric)
    b_params = calculate_qparams(b, symmetric=symmetric)
    qa = quantize(a, a_params)
    qb = quantize(b, b_params)

    a_centered = qa.astype(np.int32) - a_params.zero_point
    b_centered = qb.astype(np.int32) - b_params.zero_point
    acc_int32 = a_centered @ b_centered

    y = acc_int32.astype(np.float32) * (a_params.scale * b_params.scale)
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float32)
        if bias.shape != (b.shape[1],):
            raise ValueError(f"Bias must have shape ({b.shape[1]},), got {bias.shape}.")
        y = y + bias

    return y, qa, qb, a_params, b_params


def compare_outputs(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    abs_error = np.abs(reference - actual)
    relative_error = abs_error / np.maximum(np.abs(reference), 1e-6)

    return {
        "max_abs_error": float(np.max(abs_error)),
        "mean_abs_error": float(np.mean(abs_error)),
        "mean_relative_error": float(np.mean(relative_error)),
    }


def make_demo_inputs(m: int, k: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=0.2, scale=0.8, size=(m, k)).astype(np.float32)
    b = rng.normal(loc=-0.1, scale=0.6, size=(k, n)).astype(np.float32)
    bias = rng.normal(loc=0.0, scale=0.1, size=(n,)).astype(np.float32)
    return a, b, bias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal int8 quantized matmul demo.")
    parser.add_argument("--m", type=int, default=4, help="Rows of A.")
    parser.add_argument("--k", type=int, default=8, help="Columns of A / rows of B.")
    parser.add_argument("--n", type=int, default=5, help="Columns of B.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--symmetric", action="store_true", help="Use symmetric int8 quantization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    a, b, bias = make_demo_inputs(args.m, args.k, args.n, args.seed)

    reference = fp32_matmul(a, b, bias)
    actual, qa, qb, a_params, b_params = quantized_matmul(
        a,
        b,
        bias,
        symmetric=args.symmetric,
    )
    metrics = compare_outputs(reference, actual)

    print(f"shape: A={a.shape}, B={b.shape}, bias={bias.shape}")
    print(f"A qparams: scale={a_params.scale:.8f}, zero_point={a_params.zero_point}")
    print(f"B qparams: scale={b_params.scale:.8f}, zero_point={b_params.zero_point}")
    print(f"QA dtype/range: {qa.dtype}, [{qa.min()}, {qa.max()}]")
    print(f"QB dtype/range: {qb.dtype}, [{qb.min()}, {qb.max()}]")
    for name, value in metrics.items():
        print(f"{name}: {value:.8f}")


if __name__ == "__main__":
    main()
