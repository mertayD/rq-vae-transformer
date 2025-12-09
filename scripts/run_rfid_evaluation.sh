#!/bin/bash
# =============================================================================
# Run rFID Evaluation for All RQ-VAE Experiments
# =============================================================================
#
# Usage:
#   ./scripts/run_rfid_evaluation.sh
#   ./scripts/run_rfid_evaluation.sh --checkpoint-dir /path/to/checkpoints
#
# =============================================================================

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${BASE_DIR}/scripts/compute_all_rfid.py"

# Default paths
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${BASE_DIR}/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/results}"
BATCH_SIZE="${BATCH_SIZE:-100}"

echo "============================================================================="
echo "rFID Evaluation for RQ-VAE Experiments"
echo "============================================================================="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "============================================================================="

# Check if checkpoints exist
if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: Checkpoint directory not found: ${CHECKPOINT_DIR}"
    exit 1
fi

# List available experiments
echo ""
echo "Available experiments:"
for exp in baseline coarse fine mid; do
    if [ -d "${CHECKPOINT_DIR}/${exp}" ]; then
        ckpt_count=$(ls -1 "${CHECKPOINT_DIR}/${exp}"/epoch*_model.pt 2>/dev/null | wc -l)
        echo "  ✓ ${exp}: ${ckpt_count} checkpoint(s)"
    else
        echo "  ✗ ${exp}: not found"
    fi
done
echo ""

# Run evaluation
python3 "${SCRIPT}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    "$@"

echo ""
echo "============================================================================="
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================================="
