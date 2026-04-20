#!/usr/bin/env bash
# Train the baseline LAM (VAE variant).
# Usage: bash scripts/lam/train_lam.sh [resume_from_ckpt]

set -euo pipefail
CFG="${CFG:-configs/lam/lam.yaml}"
CKPT="${1:-}"

CMD=(python -m lam.main fit --model-type lam --config "$CFG")
if [[ -n "$CKPT" ]]; then
  CMD+=(--ckpt_path "$CKPT")
fi
"${CMD[@]}"
