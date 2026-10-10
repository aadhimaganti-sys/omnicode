"""Unified-diff creation and tolerant application.

LLMs rarely produce perfectly numbered hunks, so application ignores the line
numbers in hunk headers (using them only as a tie-breaker hint) and locates
each hunk by matching its context/removed lines, falling back to
whitespace-insensitive matching.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field


class DiffError(Exception):
    pass


@dataclass
class Hunk:
    old_start: int | None
    ops: list[tuple[str, str]] = field(default_factory=list)  # (" " | "-" | "+", text)


@dataclass
class FilePatch:
    old_path: str | None  # None => /dev/null (new file)
    new_path: str | None  # None => /dev/null (deletion)
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def path(self) -> str:
        return (self.new_path or self.old_path or "").strip()

    @property
    def is_new(self) -> bool:
        return self.old_path is None


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")


def make_unified_diff(old: str | None, new: str, path: str) -> str:
    """Unified diff text; ``old=None`` means the file is being created."""
    a = (old or "").splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    fromfile = "/dev/null" if old is None else f"a/{path}"
    out: list[str] = []
    for line in difflib.unified_diff(a, b, fromfile, f"b/{path}", lineterm="\n"):
        out.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return "".join(out)


def _strip_prefix(raw: str) -> str | None:
    p = raw.split("\t")[0].strip()
    if p == "/dev/null":
        return None
    if p.startswith(("a/", "b/")):
        p = p[2:]
    return p


def parse_unified_diff(text: str) -> list[FilePatch]:
    lines = [ln for ln in text.replace("\r\n", "\n").split("\n") if not ln.startswith("```")]
    while lines and lines[-1] == "":
        lines.pop()
    patches: list[FilePatch] = []
    cur: FilePatch | None = None
    hunk: Hunk | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            cur = FilePatch(_strip_prefix(line[4:]), _strip_prefix(lines[i + 1][4:]))
            patches.append(cur)
            hunk = None
            i += 2
            continue
        if line.startswith("@@"):
            if cur is None:
                raise DiffError("Found a hunk (@@) before any '--- / +++' file header")
            m = _HUNK_RE.match(line)
            hunk = Hunk(int(m.group(1)) if m else None)
            cur.hunks.append(hunk)
        elif line.startswith(("diff ", "index ")):
            hunk = None
        elif hunk is not None:
            tag = line[:1]
            if tag in (" ", "-", "+"):
                hunk.ops.append((tag, line[1:]))
            elif line == "":
                hunk.ops.append((" ", ""))  # models often strip the space of blank context lines
            elif not line.startswith("\\"):
                raise DiffError(f"Unexpected line inside hunk (missing +/-/space prefix?): {line!r}")
        i += 1
    for p in patches:
        for h in p.hunks:
            while h.ops and h.ops[0] == (" ", ""):
                h.ops.pop(0)
            while h.ops and h.ops[-1] == (" ", ""):
                h.ops.pop()
    if not patches:
        raise DiffError("No '--- a/file' / '+++ b/file' headers found in the patch")
    return patches


def _locate(lines: list[str], old: list[str], cursor: int, expected: int) -> int | None:
    n = len(old)
    if n == 0:
        return min(max(expected, cursor), len(lines))
    for norm in (lambda s: s, str.rstrip, str.strip):
        target = [norm(x) for x in old]
        cands = [
            i
            for i in range(cursor, len(lines) - n + 1)
            if norm(lines[i]) == target[0] and [norm(x) for x in lines[i : i + n]] == target
        ]
        if cands:
            return min(cands, key=lambda i: abs(i - expected))
    return None


def apply_hunks(original: str, hunks: list[Hunk], path: str = "file") -> str:
    crlf = "\r\n" in original
    text = original.replace("\r\n", "\n")
    ends_nl = text.endswith("\n") or text == ""
    lines = text.split("\n")
    if text.endswith("\n") or text == "":
        lines.pop()
    cursor = 0
    delta = 0
    for n, h in enumerate(hunks, 1):
        old = [t for tag, t in h.ops if tag != "+"]
        expected = (h.old_start - 1 + delta) if h.old_start else cursor
        idx = _locate(lines, old, cursor, expected)
        if idx is None:
            preview = "\n".join(old[:5])
            raise DiffError(
                f"Hunk #{n} for {path} does not match the file content. "
                f"Re-read the file and regenerate the patch. First lines expected:\n{preview}"
            )
        new: list[str] = []
        j = idx
        for tag, t in h.ops:
            if tag == " ":
                new.append(lines[j])
                j += 1
            elif tag == "-":
                j += 1
            else:
                new.append(t)
        lines[idx : idx + len(old)] = new
        delta += len(new) - len(old)
        cursor = idx + len(new)
    out = "\n".join(lines)
    if lines and ends_nl:
        out += "\n"
    return out.replace("\n", "\r\n") if crlf else out
