"""Pure-Python matrix and vector linear algebra operations for neural networks.

Engineered for the from-scratch Transformer decoder, optimized for clean execution
without external C/CUDA library dependencies.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

Vector = list[float]
Matrix = list[list[float]]


def zeros(rows: int, cols: int) -> Matrix:
    """Return a 2D matrix of zeros with dimensions rows x cols."""
    return [[0.0] * cols for _ in range(rows)]


def zeros_vec(size: int) -> Vector:
    """Return a 1D vector of zeros."""
    return [0.0] * size


def random_matrix(rows: int, cols: int, scale: float = 0.02, seed: int | None = None) -> Matrix:
    """Initialize a matrix with small random values drawn from normal/uniform distribution."""
    rng = random.Random(seed) if seed is not None else random
    return [[(rng.uniform(-1.0, 1.0) * scale) for _ in range(cols)] for _ in range(rows)]


def random_vec(size: int, scale: float = 0.02, seed: int | None = None) -> Vector:
    """Initialize a vector with small random values."""
    rng = random.Random(seed) if seed is not None else random
    return [(rng.uniform(-1.0, 1.0) * scale) for _ in range(size)]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def vector_add(a: Sequence[float], b: Sequence[float]) -> Vector:
    """Elementwise addition of two vectors: a + b."""
    return [x + y for x, y in zip(a, b)]


def vector_sub(a: Sequence[float], b: Sequence[float]) -> Vector:
    """Elementwise subtraction of two vectors: a - b."""
    return [x - y for x, y in zip(a, b)]


def vector_scale(v: Sequence[float], scalar: float) -> Vector:
    """Multiply vector by scalar."""
    return [x * scalar for x in v]


def mat_vec_mul(W: Matrix, x: Sequence[float]) -> Vector:
    """Multiply matrix W (rows x cols) by vector x (cols), returning vector of length rows.
    Computes y = W * x.
    """
    return [dot(row, x) for row in W]


def vec_mat_mul(x: Sequence[float], W: Matrix) -> Vector:
    """Multiply vector x (rows) by matrix W (rows x cols), returning vector of length cols.
    Computes y = x * W.
    """
    rows = len(W)
    cols = len(W[0]) if rows > 0 else 0
    out = [0.0] * cols
    for i, xi in enumerate(x):
        row = W[i]
        for j in range(cols):
            out[j] += xi * row[j]
    return out


def matmul(A: Matrix, B: Matrix) -> Matrix:
    """Matrix multiplication C = A * B."""
    rows_A = len(A)
    cols_A = len(A[0]) if rows_A > 0 else 0
    rows_B = len(B)
    cols_B = len(B[0]) if rows_B > 0 else 0
    if cols_A != rows_B:
        raise ValueError(f"Incompatible dimensions for matmul: {rows_A}x{cols_A} vs {rows_B}x{cols_B}")

    # Transpose B for memory and cache locality during row-row dot products
    B_T = [[B[r][c] for r in range(rows_B)] for c in range(cols_B)]
    return [[dot(row_A, col_B) for col_B in B_T] for row_A in A]


def softmax(x: Sequence[float], temperature: float = 1.0) -> Vector:
    """Numerically stable softmax with temperature scaling."""
    if not x:
        return []
    temp = max(1e-6, temperature)
    scaled = [val / temp for val in x]
    max_val = max(scaled)
    exps = [math.exp(val - max_val) for val in scaled]
    sum_exps = sum(exps)
    if sum_exps == 0.0:
        return [1.0 / len(x)] * len(x)
    return [e / sum_exps for e in exps]


def gelu(x: float) -> float:
    """Gaussian Error Linear Unit (GELU) activation (Hendrycks & Gimpel approximation)."""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3))))


def gelu_vec(v: Sequence[float]) -> Vector:
    """Elementwise GELU activation on vector."""
    return [gelu(x) for x in v]


def rmsnorm(x: Sequence[float], weight: Sequence[float], eps: float = 1e-5) -> Vector:
    """Root Mean Square Layer Normalization (RMSNorm) used in modern LLMs (Llama, Gemma).
    y = (x / sqrt(mean(x^2) + eps)) * weight
    """
    mean_square = sum(val * val for val in x) / max(1, len(x))
    scale = 1.0 / math.sqrt(mean_square + eps)
    return [val * scale * w for val, w in zip(x, weight)]


def layernorm(
    x: Sequence[float], gamma: Sequence[float], beta: Sequence[float] | None = None, eps: float = 1e-5
) -> Vector:
    """Standard Layer Normalization with mean subtraction and variance scaling."""
    n = max(1, len(x))
    mean = sum(x) / n
    variance = sum((val - mean) ** 2 for val in x) / n
    std_inv = 1.0 / math.sqrt(variance + eps)
    if beta:
        return [(val - mean) * std_inv * g + b for val, g, b in zip(x, gamma, beta)]
    return [(val - mean) * std_inv * g for val, g in zip(x, gamma)]


def cross_entropy_loss(logits: Sequence[float], target_idx: int) -> float:
    """Cross entropy loss -log(softmax(logits)[target_idx])."""
    probs = softmax(logits)
    target_prob = max(1e-12, probs[target_idx])
    return -math.log(target_prob)
