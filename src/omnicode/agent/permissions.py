"""Permission safeguards: classify shell commands and gate risky actions on
explicit user approval."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal

Decision = bool | Literal["always"]


@dataclass
class ApprovalRequest:
    kind: Literal["create", "edit", "command"]
    title: str
    detail: str = ""  # unified diff for edits, cwd for commands
    dangerous: bool = False
    reason: str = ""
    read_only: bool = False
    command: str = ""

    @property
    def key(self) -> str:
        return f"cmd:{self.command}" if self.kind == "command" else "edits"


Approver = Callable[[ApprovalRequest], Awaitable[Decision]]

# (regex, reason) - matching ANY of these forces a confirmation in every mode.
_DANGEROUS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.I), why)
    for p, why in [
        (r"\brm\b|\brmdir\b|\bunlink\b|\bshred\b|\bsrm\b", "deletes files"),
        (r"\bmv\b", "moves/overwrites files"),
        (r"\bdd\b|\bmkfs|\bfdisk\b|\bparted\b|\bformat\b", "writes to disks"),
        (r"\bsudo\b|\bsu\b|\bdoas\b", "runs with elevated privileges"),
        (r"\bchmod\b|\bchown\b|\bchgrp\b", "changes permissions"),
        (r"\bgit\s+(reset\s+--hard|clean|restore|checkout\s+(--|\.)|branch\s+-D|stash\s+(drop|clear)|rebase|filter-branch)", "discards git history/changes"),
        (r"\bgit\s+push\b.*(--force|-f\b|--delete)", "force-pushes / deletes remote refs"),
        (r"\bgit\s+push\b", "publishes to a remote"),
        (r"(curl|wget)\b.*\|\s*(sudo\s+)?(ba|z|da)?sh\b", "pipes a download into a shell"),
        (r"\|\s*(sudo\s+)?(ba|z|da)?sh\b", "pipes into a shell"),
        (r"(^|[^>&\d])>(?![>&])", "redirects output over a file"),
        (r"\btruncate\b|\bfind\b.*(-delete|-exec)", "deletes or rewrites files"),
        (r"\bkill(all)?\b|\bpkill\b|\bshutdown\b|\breboot\b|\bhalt\b", "stops processes / the machine"),
        (r"\bdrop\s+(table|database|schema)\b|\bdelete\s+from\b", "destroys database data"),
        (r":\(\)\s*\{", "fork bomb"),
        (r"\bpip3?\s+uninstall\b|\bnpm\s+(uninstall|rm)\b|\bapt(-get)?\s+(remove|purge)\b", "uninstalls packages"),
        (r"\bRemove-Item\b|\bdel\s|\brd\s+/s\b", "deletes files (Windows)"),
    ]
]

_READ_ONLY_PREFIXES = (
    "ls", "pwd", "cat", "head", "tail", "wc", "echo", "which", "whoami", "date", "tree",
    "grep", "rg", "file", "stat", "du", "df", "env", "uname", "python --version",
    "git status", "git diff", "git log", "git show", "git branch", "git ls-files", "git blame",
)
_SHELL_META = re.compile(r"[;&|<>`]|\$\(")


def classify_command(command: str) -> tuple[bool, str]:
    """Return ``(is_dangerous, reason)``."""
    for pattern, reason in _DANGEROUS:
        if pattern.search(command):
            return True, reason
    return False, ""


def is_read_only_command(command: str) -> bool:
    """Conservative allowlist of inspection commands that can't modify anything."""
    cmd = command.strip()
    if not cmd or _SHELL_META.search(cmd):
        return False
    if cmd.split()[0] == "find" and re.search(r"-(delete|exec|fprint|ok)", cmd):
        return False
    try:
        shlex.split(cmd)
    except ValueError:
        return False
    if cmd.split()[0] == "find":
        return True
    if cmd.split()[0] == "git" and re.search(r"\bbranch\b.*(-[dDmMcC]|--delete|--move|--copy)", cmd):
        return False
    return any(cmd == p or cmd.startswith(p + " ") for p in _READ_ONLY_PREFIXES)


class PermissionManager:
    """Decides whether an action may proceed.

    mode:
      ask         confirm every edit and every non-read-only command (default)
      auto_edits  apply file edits without asking; still confirm commands
      yolo        also run non-destructive commands without asking
    Destructive commands are confirmed in EVERY mode, and can never be
    pre-approved with "always".
    """

    MODES = ("ask", "auto_edits", "yolo")

    def __init__(
        self,
        approver: Approver | None,
        mode: str = "ask",
        allowed_commands: list[str] | None = None,
    ):
        self.approver = approver
        self.mode = mode if mode in self.MODES else "ask"
        self.allowed_commands = list(allowed_commands or [])
        self._session_allowed: set[str] = set()

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {', '.join(self.MODES)}")
        self.mode = mode

    def _auto_allowed(self, req: ApprovalRequest) -> bool:
        if req.dangerous:
            return False
        if req.kind in ("create", "edit"):
            return self.mode in ("auto_edits", "yolo") or "edits" in self._session_allowed
        if req.read_only or req.key in self._session_allowed or self.mode == "yolo":
            return True
        cmd = req.command.strip()
        return any(cmd == a or cmd.startswith(a + " ") for a in self.allowed_commands)

    async def check(self, req: ApprovalRequest) -> bool:
        if self._auto_allowed(req):
            return True
        if self.approver is None:
            return False  # headless: refuse anything that needs confirmation
        decision = await self.approver(req)
        if decision == "always" and not req.dangerous:
            self._session_allowed.add(req.key)
            return True
        return decision is True
