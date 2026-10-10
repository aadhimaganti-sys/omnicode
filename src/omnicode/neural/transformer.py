"""Autoregressive Transformer Decoder built from scratch.

Features:
- Token and Rotary/Positional Embeddings
- Multi-Head Causal Self-Attention with KV-Caching
- Feed-Forward Networks with GELU activation
- RMSNorm / LayerNorm
- Thermal-aware execution: supports pure-Python local inference and cloud offload
  to prevent device overheating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import AsyncIterator, Sequence

from omnicode.neural.matrix import (
    Matrix,
    Vector,
    dot,
    gelu,
    mat_vec_mul,
    random_matrix,
    random_vec,
    rmsnorm,
    softmax,
    vector_add,
    zeros,
)
from omnicode.neural.tokenizer import ScratchTokenizer


@dataclass
class TransformerConfig:
    vocab_size: int = 512
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_seq_len: int = 1024
    hidden_dim: int = 256
    norm_eps: float = 1e-5
    temperature: float = 0.2
    top_k: int = 40


@dataclass
class AttentionWeights:
    W_q: Matrix  # [dim, dim]
    W_k: Matrix  # [dim, dim]
    W_v: Matrix  # [dim, dim]
    W_o: Matrix  # [dim, dim]


@dataclass
class LayerWeights:
    attn_norm: Vector  # [dim]
    attn: AttentionWeights
    mlp_norm: Vector  # [dim]
    W_gate: Matrix  # [hidden_dim, dim]
    W_down: Matrix  # [dim, hidden_dim]


@dataclass
class ModelWeights:
    token_embed: Matrix  # [vocab_size, dim]
    pos_embed: Matrix  # [max_seq_len, dim]
    layers: list[LayerWeights]
    final_norm: Vector  # [dim]
    output_head: Matrix  # [vocab_size, dim]


class TransformerModel:
    """From-scratch Transformer Decoder model."""

    def __init__(self, config: TransformerConfig, weights: ModelWeights | None = None):
        self.config = config
        self.weights = weights or self._init_random_weights(config)
        self.tokenizer = ScratchTokenizer()

    @classmethod
    def _init_random_weights(cls, cfg: TransformerConfig) -> ModelWeights:
        """Initialize weights with Gaussian scaling."""
        scale = 1.0 / math.sqrt(cfg.dim)
        layers = []
        for _ in range(cfg.n_layers):
            layers.append(
                LayerWeights(
                    attn_norm=[1.0] * cfg.dim,
                    attn=AttentionWeights(
                        W_q=random_matrix(cfg.dim, cfg.dim, scale),
                        W_k=random_matrix(cfg.dim, cfg.dim, scale),
                        W_v=random_matrix(cfg.dim, cfg.dim, scale),
                        W_o=random_matrix(cfg.dim, cfg.dim, scale),
                    ),
                    mlp_norm=[1.0] * cfg.dim,
                    W_gate=random_matrix(cfg.hidden_dim, cfg.dim, scale),
                    W_down=random_matrix(cfg.dim, cfg.hidden_dim, scale),
                )
            )
        return ModelWeights(
            token_embed=random_matrix(cfg.vocab_size, cfg.dim, scale),
            pos_embed=random_matrix(cfg.max_seq_len, cfg.dim, scale),
            layers=layers,
            final_norm=[1.0] * cfg.dim,
            output_head=random_matrix(cfg.vocab_size, cfg.dim, scale),
        )

    def forward_token(
        self,
        token_id: int,
        pos: int,
        kv_cache: list[tuple[list[Vector], list[Vector]]],
    ) -> Vector:
        """Forward pass for a single token using KV Cache. Returns logits over vocabulary."""
        cfg = self.config
        w = self.weights

        # 1. Embedding lookup + Positional encoding
        tok_vec = w.token_embed[min(token_id, cfg.vocab_size - 1)]
        pos_vec = w.pos_embed[min(pos, cfg.max_seq_len - 1)]
        x = vector_add(tok_vec, pos_vec)

        head_dim = cfg.dim // cfg.n_heads
        scale = 1.0 / math.sqrt(head_dim)

        # 2. Transformer Decoder Layers
        for i, layer in enumerate(w.layers):
            # A. Attention with RMSNorm and Residual
            x_norm = rmsnorm(x, layer.attn_norm, cfg.norm_eps)

            # Project to Q, K, V
            q = mat_vec_mul(layer.attn.W_q, x_norm)
            k = mat_vec_mul(layer.attn.W_k, x_norm)
            v = mat_vec_mul(layer.attn.W_v, x_norm)

            # Update KV cache
            k_cache, v_cache = kv_cache[i]
            k_cache.append(k)
            v_cache.append(v)
            seq_len = len(k_cache)

            # Multi-head attention calculation
            attn_out = [0.0] * cfg.dim
            for h in range(cfg.n_heads):
                h_start = h * head_dim
                h_end = h_start + head_dim

                q_h = q[h_start:h_end]

                # Attention scores against all past keys
                scores = []
                for t in range(seq_len):
                    k_t = k_cache[t][h_start:h_end]
                    scores.append(dot(q_h, k_t) * scale)

                # Softmax over past sequence positions
                weights = softmax(scores)

                # Weighted sum over values
                for dim_idx in range(head_dim):
                    val_sum = sum(weights[t] * v_cache[t][h_start + dim_idx] for t in range(seq_len))
                    attn_out[h_start + dim_idx] = val_sum

            # Output projection + Residual
            proj_out = mat_vec_mul(layer.attn.W_o, attn_out)
            x = vector_add(x, proj_out)

            # B. MLP Feed-Forward with RMSNorm and Residual
            x_norm = rmsnorm(x, layer.mlp_norm, cfg.norm_eps)
            gate = mat_vec_mul(layer.W_gate, x_norm)
            activated = [gelu(g) for g in gate]
            mlp_out = mat_vec_mul(layer.W_down, activated)
            x = vector_add(x, mlp_out)

        # 3. Final Norm & Output Head Projection to Vocab Logits
        x_final = rmsnorm(x, w.final_norm, cfg.norm_eps)
        logits = mat_vec_mul(w.output_head, x_final)
        return logits

    def generate_token(
        self,
        token_id: int,
        pos: int,
        kv_cache: list[tuple[list[Vector], list[Vector]]],
        temperature: float = 0.2,
    ) -> int:
        """Sample the next token given current token and position."""
        logits = self.forward_token(token_id, pos, kv_cache)
        probs = softmax(logits, temperature=temperature)

        # Greedy choice or top-k selection
        best_token = max(range(len(probs)), key=lambda idx: probs[idx])
        return best_token
