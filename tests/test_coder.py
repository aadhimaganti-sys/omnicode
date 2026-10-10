"""Tests for the Coder model and the from-scratch Neural Engine (unittest)."""

import unittest

from omnicode.config import Config, ProviderSpec
from omnicode.neural.matrix import (
    dot,
    gelu,
    matmul,
    rmsnorm,
    softmax,
    vector_add,
)
from omnicode.neural.tokenizer import ScratchTokenizer
from omnicode.neural.transformer import TransformerConfig, TransformerModel
from omnicode.providers.engine import LLMEngine
from omnicode.providers.registry import PROVIDER_TYPES, create_provider


class TestCoderAndNeuralEngine(unittest.TestCase):
    def test_matrix_operations(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        C = matmul(A, B)
        self.assertEqual(C, [[19.0, 22.0], [43.0, 50.0]])

        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 5.0, 6.0]
        self.assertEqual(dot(v1, v2), 32.0)
        self.assertEqual(vector_add(v1, v2), [5.0, 7.0, 9.0])

        s = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(s), 1.0, places=5)
        self.assertTrue(s[2] > s[1] > s[0])

        self.assertAlmostEqual(gelu(0.0), 0.0, places=5)
        norm = rmsnorm([2.0, 2.0], [1.0, 1.0])
        self.assertAlmostEqual(norm[0], 1.0, places=3)

    def test_tokenizer_scratch(self):
        tok = ScratchTokenizer()
        code = "def fib(n: int) -> int:\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)"
        encoded = tok.encode(code)
        self.assertTrue(len(encoded) > 0)
        decoded = tok.decode(encoded)
        self.assertEqual(decoded, code)

    def test_transformer_scratch_forward(self):
        cfg = TransformerConfig(vocab_size=260, dim=16, n_layers=2, n_heads=2, hidden_dim=32)
        model = TransformerModel(cfg)
        kv_cache = [([], []) for _ in range(cfg.n_layers)]
        token_id = 42
        next_token = model.generate_token(token_id, 0, kv_cache)
        self.assertIsInstance(next_token, int)
        self.assertTrue(0 <= next_token < cfg.vocab_size)

    def test_coder_provider_registered(self):
        self.assertIn("coder", PROVIDER_TYPES)
        spec = ProviderSpec(type="coder", api_key="test-key")
        provider = create_provider("coder", spec)
        self.assertEqual(provider.type_name, "coder")
        self.assertEqual(provider.model, "gemini-3.8-flash")

    def test_coder_config_and_engine(self):
        cfg = Config()
        cfg.active = "coder"
        cfg.providers["coder"] = ProviderSpec(type="coder", api_key="test-key")
        engine = LLMEngine.from_config(cfg)
        self.assertEqual(engine.primary.name, "coder")
        self.assertEqual(engine.primary.provider.type_name, "coder")


if __name__ == "__main__":
    unittest.main()
