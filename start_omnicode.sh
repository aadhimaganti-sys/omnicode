#!/usr/bin/env bash
# ==========================================
# OmniCode 1-Click Studio Launcher
# ==========================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=========================================="
echo "⚡ Starting OmniCode Studio..."
echo "=========================================="

if [ -f ".venv/bin/python" ]; then
    .venv/bin/python -m omnicode gui
else
    python3 -m omnicode gui
fi
