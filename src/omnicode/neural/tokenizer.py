"""Byte-level and Subword Tokenizer built from scratch for the OmniCode Neural LLM.

Supports arbitrary UTF-8 text with zero out-of-vocabulary (OOV) errors,
featuring dedicated token IDs for code structures, indentation, and special agent symbols.
"""

from __future__ import annotations

import re
from typing import Sequence

# Special Token Constants
PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|startoftext|>"
EOS_TOKEN = "<|endoftext|>"
THOUGHT_TOKEN = "<|thought|>"
CALL_TOKEN = "<|call|>"
RESULT_TOKEN = "<|result|>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    THOUGHT_TOKEN,
    CALL_TOKEN,
    RESULT_TOKEN,
]

COMMON_CODE_PIECES = [
    # Indentation & whitespace
    "    ",
    "  ",
    "\n    ",
    "\n  ",
    "\n",
    " ",
    # Syntax & operators
    " -> ",
    " => ",
    " == ",
    " != ",
    " <= ",
    " >= ",
    " += ",
    " -= ",
    " = ",
    ": ",
    "::",
    "...",
    "//",
    "/*",
    "*/",
    "# ",
    "()",
    "{}",
    "[]",
    # Python keywords
    "def ",
    "class ",
    "return ",
    "import ",
    "from ",
    "async def ",
    "await ",
    "if ",
    "else:",
    "elif ",
    "for ",
    "while ",
    "try:",
    "except ",
    "raise ",
    "with ",
    "yield ",
    "lambda ",
    "self.",
    "self",
    "None",
    "True",
    "False",
    "print(",
    # Multi-language keywords (C, C++, Rust, Go, JS/TS)
    "function ",
    "const ",
    "let ",
    "var ",
    "fn ",
    "pub ",
    "struct ",
    "impl ",
    "package ",
    "func ",
    "void ",
    "int ",
    "char ",
    "float ",
    "bool ",
    "auto ",
    "public ",
    "private ",
    "protected ",
    "include ",
    "#include ",
    "printf(",
    "console.log(",
    "std::",
    "vector<",
    "string",
    "interface ",
    "export ",
    "type ",
]


class ScratchTokenizer:
    """Byte-pair and subword tokenizer with 100% code vocabulary coverage."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self) -> None:
        idx = 0
        # 1. Special tokens
        for sp in SPECIAL_TOKENS:
            self.vocab[sp] = idx
            self.inverse_vocab[idx] = sp
            idx += 1

        # 2. All 256 raw bytes (guarantees zero OOV errors on any UTF-8 string)
        for b in range(256):
            byte_repr = bytes([b]).decode("latin-1")
            if byte_repr not in self.vocab:
                self.vocab[byte_repr] = idx
                self.inverse_vocab[idx] = byte_repr
                idx += 1

        # 3. High-frequency code tokens & subwords
        for piece in COMMON_CODE_PIECES:
            if piece not in self.vocab:
                self.vocab[piece] = idx
                self.inverse_vocab[idx] = piece
                idx += 1

        self.vocab_size = idx
        # Build regex pattern for greedy longest-match tokenization
        sorted_keys = sorted(
            [k for k in self.vocab.keys() if len(k) > 1 and k not in SPECIAL_TOKENS],
            key=len,
            reverse=True,
        )
        escaped_pieces = [re.escape(k) for k in sorted_keys]
        escaped_specials = [re.escape(sp) for sp in SPECIAL_TOKENS]
        self._pattern = re.compile("|".join(escaped_specials + escaped_pieces) if escaped_pieces else "$^")

    @property
    def pad_id(self) -> int:
        return self.vocab[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.vocab[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.vocab[EOS_TOKEN]

    @property
    def thought_id(self) -> int:
        return self.vocab[THOUGHT_TOKEN]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text into a list of integer token IDs."""
        if not text:
            return []

        tokens: list[int] = []
        if add_special_tokens:
            tokens.append(self.bos_id)

        pos = 0
        text_len = len(text)

        while pos < text_len:
            match = self._pattern.match(text, pos)
            if match:
                token_str = match.group(0)
                tokens.append(self.vocab[token_str])
                pos = match.end()
            else:
                # Fall back to single character / UTF-8 byte
                char = text[pos]
                # Encode char bytes into latin-1 byte representations
                for byte_val in char.encode("utf-8"):
                    byte_char = bytes([byte_val]).decode("latin-1")
                    tokens.append(self.vocab[byte_char])
                pos += 1

        if add_special_tokens:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, tokens: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of integer token IDs back to a string."""
        raw_byte_buffer = bytearray()
        string_accumulator: list[str] = []

        def flush_bytes():
            nonlocal raw_byte_buffer
            if raw_byte_buffer:
                string_accumulator.append(raw_byte_buffer.decode("utf-8", errors="replace"))
                raw_byte_buffer = bytearray()

        for t in tokens:
            piece = self.inverse_vocab.get(t)
            if piece is None:
                continue
            if skip_special_tokens and piece in SPECIAL_TOKENS:
                continue

            # Check if piece is a single byte token
            if len(piece) == 1 and ord(piece) < 256:
                raw_byte_buffer.append(ord(piece))
            else:
                flush_bytes()
                string_accumulator.append(piece)

        flush_bytes()
        return "".join(string_accumulator)
