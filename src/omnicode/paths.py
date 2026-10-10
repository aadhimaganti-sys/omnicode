"""Path safety helpers: keep every file operation inside the project root."""

from __future__ import annotations

from pathlib import Path


class PathError(Exception):
    pass


def resolve_in_root(root: Path, rel: str) -> Path:
    """Resolve ``rel`` against ``root`` and refuse anything that escapes it."""
    root = root.resolve()
    candidate = Path(rel)
    p = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise PathError(f"Path '{rel}' is outside the project root") from None
    return p


def is_protected(root: Path, p: Path) -> bool:
    """Files the agent must never write (git internals, OmniCode state, API keys, credentials)."""
    try:
        rel = p.resolve().relative_to(root.resolve())
        parts = rel.parts
        name = p.name.lower()
    except ValueError:
        return True

    # Protect git internals and omnicode internals
    if ".git" in parts or ".omnicode" in parts:
        return True

    # Protect API keys, secrets, credentials, and environment files
    if any(k in name for k in ("api_key", "apikey", "secret", "credential", "token")) or name.startswith(".env") or name.endswith(".key"):
        return True

    return False

