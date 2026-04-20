#!/usr/bin/env bash
# Zero-shot action transfer inference.
# Thin wrapper around world_model/inference/action_transfer.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python "${ROOT_DIR}/world_model/inference/action_transfer.py" "$@"
