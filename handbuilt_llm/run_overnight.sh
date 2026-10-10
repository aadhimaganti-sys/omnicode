#!/usr/bin/env bash
# ==========================================
# Run 10,000 Steps Overnight Training Script
# ==========================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=========================================="
echo "⚡ Starting 10,000 Steps Overnight Training"
echo "=========================================="
echo "Log file: $DIR/train.log"
echo "Checkpoints: $DIR/checkpoint_latest.pt & $DIR/model_best.pt"
echo ""

# Activate virtual environment if present
if [ -f "$DIR/../.venv/bin/activate" ]; then
    source "$DIR/../.venv/bin/activate"
fi

# Run in background with nohup
nohup python train.py > "$DIR/train.log" 2>&1 &
PID=$!

echo "✓ Training started in background with PID: $PID"
echo ""
echo "To monitor progress in real-time, run:"
echo "  tail -f $DIR/train.log"
echo ""
echo "To stop training anytime, run:"
echo "  kill $PID"
echo "=========================================="
