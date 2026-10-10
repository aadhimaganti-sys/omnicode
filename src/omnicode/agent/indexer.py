"""Git-aware project indexer: file discovery, symbol extraction (Python AST +
regex for other languages) and BM25 search over code chunks. Pure stdlib."""

from __future__ import annotations

import ast
import math
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build", ".idea",
    ".vscode", ".mypy_cache", ".pytest_cache", ".ruff_cache", "target", ".tox", ".next",
    ".omnicode", ".cache", "site-packages",
}
BINARY_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".tar", ".bz2", ".xz",
    ".7z", ".so", ".dll", ".dylib", ".exe", ".bin", ".class", ".jar", ".pyc", ".woff", ".woff2",
    ".ttf", ".otf", ".mp3", ".mp4", ".mov", ".avi", ".sqlite", ".db", ".lock", ".min.js", ".map",
}

_REGEX_SYMBOLS: dict[str, list[tuple[re.Pattern[str], str]]] = {}


def _rx(*pairs: tuple[str, str]) -> list[tuple[re.Pattern[str], str]]:
    return [(re.compile(p, re.M), kind) for p, kind in pairs]


_JS = _rx(
    (r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(\w+)", "function"),
    (r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)", "class"),
    (r"^\s*(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>", "function"),
    (r"^\s*(?:export\s+)?(?:interface|type|enum)\s+(\w+)", "type"),
)
for _ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
    _REGEX_SYMBOLS[_ext] = _JS
_REGEX_SYMBOLS[".go"] = _rx((r"^func\s+(?:\([^)]*\)\s*)?(\w+)", "function"), (r"^type\s+(\w+)\s+(?:struct|interface)", "type"))
_REGEX_SYMBOLS[".rs"] = _rx(
    (r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)", "function"),
    (r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)", "type"),
)
_JAVALIKE = _rx(
    (r"^\s*(?:public|private|protected|internal|abstract|final|static|\s)*(?:class|interface|enum|record)\s+(\w+)", "class"),
    (r"^\s*(?:public|private|protected|static|final|\s)+[\w<>\[\],\s]+\s+(\w+)\s*\([^;]*\)\s*(?:throws [\w, .]+)?\s*\{", "method"),
)
for _ext in (".java", ".cs", ".kt", ".swift", ".scala"):
    _REGEX_SYMBOLS[_ext] = _JAVALIKE
_REGEX_SYMBOLS[".rb"] = _rx((r"^\s*def\s+(?:self\.)?(\w+)", "function"), (r"^\s*(?:class|module)\s+(\w+)", "class"))
_REGEX_SYMBOLS[".php"] = _rx((r"^\s*(?:public|private|protected|static|\s)*function\s+(\w+)", "function"), (r"^\s*class\s+(\w+)", "class"))
_REGEX_SYMBOLS[".c"] = _rx((r"^[\w\s\*]+?\b(\w+)\s*\([^;]*\)\s*\{", "function"))
_REGEX_SYMBOLS[".cpp"] = _REGEX_SYMBOLS[".h"] = _REGEX_SYMBOLS[".hpp"] = _REGEX_SYMBOLS[".c"]

_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
_TOKEN = re.compile(r"[A-Za-z]+|\d+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(_CAMEL.sub(r"\1 \2", text)) if len(t) > 1]


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str
    line: int


@dataclass
class SearchHit:
    path: str
    start: int
    end: int
    score: float
    snippet: str


@dataclass
class _Chunk:
    path: str
    start: int
    end: int
    tf: Counter
    length: int
    text: str


def extract_symbols(path: str, text: str) -> list[Symbol]:
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return []
        out: list[Symbol] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                out.append(Symbol(node.name, "class", node.lineno))
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out.append(Symbol(f"{node.name}.{sub.name}", "method", sub.lineno))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(Symbol(node.name, "function", node.lineno))
        return out
    out = []
    for rx, kind in _REGEX_SYMBOLS.get(suffix, []):
        for m in rx.finditer(text):
            out.append(Symbol(m.group(1), kind, text.count("\n", 0, m.start()) + 1))
    return sorted(set(out), key=lambda s: s.line)


class ProjectIndex:
    def __init__(self, root: Path, max_files: int = 5000, max_file_bytes: int = 300_000, chunk_lines: int = 40):
        self.root = Path(root).resolve()
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self.chunk_lines = chunk_lines
        self._built = False
        self.files: list[str] = []
        self.symbols: dict[str, list[Symbol]] = {}
        self._chunks: list[_Chunk] = []
        self._df: Counter = Counter()
        self._avgdl = 1.0

    # --------------------------------------------------------------- git
    def _git(self, *args: str) -> str | None:
        try:
            r = subprocess.run(
                ["git", "-C", str(self.root), *args], capture_output=True, text=True, timeout=20
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout if r.returncode == 0 else None

    @property
    def is_git_repo(self) -> bool:
        return self._git("rev-parse", "--is-inside-work-tree") is not None

    def git_status(self, limit: int = 40) -> str:
        out = self._git("status", "--short", "--branch")
        if out is None:
            return ""
        lines = out.strip().splitlines()
        if len(lines) > limit:
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)

    # ----------------------------------------------------------- discovery
    def _discover(self) -> list[str]:
        out = self._git("ls-files", "-z", "--cached", "--others", "--exclude-standard")
        if out is not None:
            paths = [p for p in out.split("\0") if p]
        else:
            paths = []
            for dirpath, dirnames, filenames in os.walk(self.root):
                dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".git"))
                for fn in sorted(filenames):
                    paths.append(os.path.relpath(os.path.join(dirpath, fn), self.root))
        keep = [
            p for p in paths
            if not (set(Path(p).parts[:-1]) & IGNORE_DIRS) and Path(p).suffix.lower() not in BINARY_EXT
            and (self.root / p).is_file()
        ]
        return sorted(keep)[: self.max_files]

    def _read(self, rel: str) -> str | None:
        p = self.root / rel
        try:
            if p.stat().st_size > self.max_file_bytes:
                return None
            data = p.read_bytes()
        except OSError:
            return None
        if b"\0" in data[:1024]:
            return None
        return data.decode("utf-8", errors="replace")

    # --------------------------------------------------------------- build
    def invalidate(self) -> None:
        self._built = False

    def ensure(self) -> None:
        if not self._built:
            self.build()

    def build(self) -> None:
        self.files = []
        self.symbols = {}
        self._chunks = []
        self._df = Counter()
        for rel in self._discover():
            text = self._read(rel)
            if text is None:
                continue
            self.files.append(rel)
            syms = extract_symbols(rel, text)
            if syms:
                self.symbols[rel] = syms
            path_tokens = tokenize(rel)
            lines = text.splitlines()
            for start in range(0, max(len(lines), 1), self.chunk_lines):
                body = "\n".join(lines[start : start + self.chunk_lines])
                tokens = tokenize(body) + path_tokens
                if not tokens:
                    continue
                tf = Counter(tokens)
                self._df.update(tf.keys())
                self._chunks.append(
                    _Chunk(rel, start + 1, min(start + self.chunk_lines, len(lines)), tf, len(tokens), body)
                )
        total = sum(c.length for c in self._chunks)
        self._avgdl = (total / len(self._chunks)) if self._chunks else 1.0
        self._built = True

    # -------------------------------------------------------------- queries
    def search(self, query: str, limit: int = 8) -> list[SearchHit]:
        """BM25 ranking over ~40-line chunks (identifiers are camel/snake split)."""
        self.ensure()
        q = tokenize(query)
        if not q or not self._chunks:
            return []
        n = len(self._chunks)
        k1, b = 1.5, 0.75
        scored: list[tuple[float, _Chunk]] = []
        for c in self._chunks:
            score = 0.0
            for term in set(q):
                f = c.tf.get(term)
                if not f:
                    continue
                df = self._df[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * c.length / self._avgdl))
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        return [
            SearchHit(c.path, c.start, c.end, round(s, 3), "\n".join(c.text.splitlines()[:12]))
            for s, c in scored[:limit]
        ]

    def repo_map(self, max_chars: int = 6000) -> str:
        """Compact tree of files with their top-level symbols, shallow files first."""
        self.ensure()
        ordered = sorted(self.files, key=lambda p: (p.count("/"), p))
        out: list[str] = []
        used = 0
        for i, rel in enumerate(ordered):
            block = [rel]
            for s in self.symbols.get(rel, [])[:8]:
                block.append(f"    {s.kind} {s.name} (L{s.line})")
            text = "\n".join(block)
            if used + len(text) > max_chars:
                out.append(f"... {len(ordered) - i} more files (use list_dir / search_code)")
                break
            out.append(text)
            used += len(text) + 1
        return "\n".join(out)
