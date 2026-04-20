#!/usr/bin/env bash
# Train the τ-aligned LAM (SeqΔ-REPA) with a frozen visual teacher.
# Usage: bash scripts/lam/train_lam_align.sh
# Set VJEPA2_ROOT to your local V-JEPA 2 checkout before running.

set -euo pipefail

CFG="${CFG:-configs/lam/lam_vjepa_align.yaml}"

if [[ -n "${VJEPA2_ROOT:-}" ]]; then
    export PYTHONPATH="${VJEPA2_ROOT}:${PYTHONPATH:-}"
fi

python -m lam.main fit --model-type lam_align --config "$CFG"
