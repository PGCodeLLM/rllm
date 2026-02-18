#!/usr/bin/env bash
# Run NCCL distributed test. Use --early-exit-rank 0 to reproduce TCPStore "Broken pipe" error.
#
# Single node (e.g. 2 GPUs):
#   ./scripts/run_test_nccl_distributed.sh
#   ./scripts/run_test_nccl_distributed.sh --early-exit-rank 0
#
# From repo root:
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
NPROC="${NPROC_PER_NODE:-2}"
echo "Running with torchrun --nproc_per_node=$NPROC (set NPROC_PER_NODE to override)"
torchrun --nproc_per_node="$NPROC" scripts/test_nccl_distributed.py "$@"
