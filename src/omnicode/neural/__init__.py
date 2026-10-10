"""OmniCode Neural: Transformer LLM architecture built from scratch."""

from omnicode.neural.matrix import (
    Vector,
    Matrix,
    matmul,
    vector_add,
    softmax,
    gelu,
    rmsnorm,
    layernorm,
)
from omnicode.neural.tokenizer import ScratchTokenizer
from omnicode.neural.transformer import TransformerConfig, TransformerModel

__all__ = [
    "Vector",
    "Matrix",
    "matmul",
    "vector_add",
    "softmax",
    "gelu",
    "rmsnorm",
    "layernorm",
    "ScratchTokenizer",
    "TransformerConfig",
    "TransformerModel",
]
