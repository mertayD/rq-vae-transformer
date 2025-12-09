#!/bin/bash
# =============================================================================
# RQ-VAE Training Experiments: Commitment Loss Weight Ablation
# =============================================================================
# Runs 4 experiments sequentially with different commitment weight configurations
#
# Usage:
#   ./scripts/run_all_experiments.sh [NUM_GPUS]
#
# Arguments:
#   NUM_GPUS: Number of GPUs to use (default: 8)
#
# =============================================================================

set -e  # Exit on error

# Configuration
NUM_GPUS=${1:-8}
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${BASE_DIR}/configs/imagenet256/stage1"
CHECKPOINT_BASE="${BASE_DIR}/checkpoints"
LOG_DIR="${BASE_DIR}/logs"

# Create directories
mkdir -p "${CHECKPOINT_BASE}"
mkdir -p "${LOG_DIR}"

# Experiment configurations
declare -A EXPERIMENTS=(
    ["baseline"]="config_baseline.yaml"
    ["coarse"]="config_coarse.yaml"
    ["fine"]="config_fine.yaml"
    ["mid"]="config_mid.yaml"
)

# Experiment descriptions
declare -A DESCRIPTIONS=(
    ["baseline"]="Uniform weights [1.0, 1.0, 1.0, 1.0]"
    ["coarse"]="Coarse-first weights [4.0, 2.0, 1.0, 0.5]"
    ["fine"]="Fine-first weights [0.5, 1.0, 2.0, 4.0]"
    ["mid"]="Mid-focus weights [0.5, 1.0, 1.0, 0.5]"
)

# Order of experiments
EXPERIMENT_ORDER=("baseline" "coarse" "fine" "mid")

# =============================================================================
# Helper Functions
# =============================================================================

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $1"
}

clear_gpu_memory() {
    log "Clearing GPU memory..."
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    # Give GPUs time to release memory
    sleep 5
}

print_separator() {
    echo "============================================================================="
}

print_experiment_header() {
    local exp_name=$1
    local exp_num=$2
    local total=$3

    print_separator
    echo "EXPERIMENT ${exp_num}/${total}: ${exp_name}"
    echo "Description: ${DESCRIPTIONS[$exp_name]}"
    echo "Config: ${EXPERIMENTS[$exp_name]}"
    echo "Started at: $(timestamp)"
    print_separator
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    local total_experiments=${#EXPERIMENT_ORDER[@]}
    local exp_num=0
    local start_time=$(date +%s)

    print_separator
    log "RQ-VAE Commitment Loss Weight Ablation Study"
    log "Total experiments: ${total_experiments}"
    log "GPUs: ${NUM_GPUS}"
    log "Base directory: ${BASE_DIR}"
    print_separator
    echo ""

    # Summary log file
    SUMMARY_LOG="${LOG_DIR}/experiment_summary_$(date +%Y%m%d_%H%M%S).log"

    {
        echo "RQ-VAE Experiment Summary"
        echo "Started: $(timestamp)"
        echo "GPUs: ${NUM_GPUS}"
        echo ""
    } > "${SUMMARY_LOG}"

    for exp_name in "${EXPERIMENT_ORDER[@]}"; do
        ((exp_num++))

        local config_file="${CONFIG_DIR}/${EXPERIMENTS[$exp_name]}"
        local checkpoint_dir="${CHECKPOINT_BASE}/${exp_name}"
        local log_file="${LOG_DIR}/${exp_name}_$(date +%Y%m%d_%H%M%S).log"

        # Create checkpoint directory
        mkdir -p "${checkpoint_dir}"

        # Print header
        print_experiment_header "${exp_name}" "${exp_num}" "${total_experiments}"

        # Clear GPU memory before starting
        clear_gpu_memory

        # Record start time
        local exp_start=$(date +%s)
        log "Log file: ${log_file}"
        log "Checkpoint directory: ${checkpoint_dir}"
        echo ""

        # Run training
        log "Starting training..."

        if torchrun \
            --nproc_per_node="${NUM_GPUS}" \
            "${BASE_DIR}/main_stage1.py" \
            -m "${config_file}" \
            -r "${checkpoint_dir}" \
            2>&1 | tee "${log_file}"; then

            local exp_end=$(date +%s)
            local exp_duration=$((exp_end - exp_start))
            local exp_duration_fmt=$(printf '%02d:%02d:%02d' $((exp_duration/3600)) $((exp_duration%3600/60)) $((exp_duration%60)))

            log "Experiment '${exp_name}' completed successfully"
            log "Duration: ${exp_duration_fmt}"

            # Append to summary
            {
                echo "${exp_name}:"
                echo "  Status: SUCCESS"
                echo "  Duration: ${exp_duration_fmt}"
                echo "  Config: ${EXPERIMENTS[$exp_name]}"
                echo "  Checkpoint: ${checkpoint_dir}"
                echo ""
            } >> "${SUMMARY_LOG}"
        else
            local exp_end=$(date +%s)
            local exp_duration=$((exp_end - exp_start))
            local exp_duration_fmt=$(printf '%02d:%02d:%02d' $((exp_duration/3600)) $((exp_duration%3600/60)) $((exp_duration%60)))

            log "ERROR: Experiment '${exp_name}' failed!"
            log "Check log file: ${log_file}"

            # Append to summary
            {
                echo "${exp_name}:"
                echo "  Status: FAILED"
                echo "  Duration: ${exp_duration_fmt}"
                echo "  Config: ${EXPERIMENTS[$exp_name]}"
                echo "  Log: ${log_file}"
                echo ""
            } >> "${SUMMARY_LOG}"

            # Continue with next experiment instead of exiting
            log "Continuing with next experiment..."
        fi

        echo ""
    done

    # Final summary
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    local total_duration_fmt=$(printf '%02d:%02d:%02d' $((total_duration/3600)) $((total_duration%3600/60)) $((total_duration%60)))

    print_separator
    log "All experiments completed!"
    log "Total duration: ${total_duration_fmt}"
    log "Summary log: ${SUMMARY_LOG}"
    print_separator

    # Append final summary
    {
        echo "============================================"
        echo "Completed: $(timestamp)"
        echo "Total duration: ${total_duration_fmt}"
    } >> "${SUMMARY_LOG}"

    # Print summary
    echo ""
    echo "Experiment Summary:"
    cat "${SUMMARY_LOG}"
}

# Run main function
main "$@"
