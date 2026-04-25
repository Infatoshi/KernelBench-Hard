#!/bin/bash
# Full active sweep: every (harness, model) × every CUDA problem.
#
# Usage: ./scripts/sweep.sh
#
# Edit ACTIVE_MATRIX below to add/remove models. ccr-rust must be running
# for ccr-claude entries (see docs/ccr-rust-setup.md).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# (harness, model, reasoning_effort) tuples.
# Empty reasoning_effort = use default.
declare -a ACTIVE_MATRIX=(
    "claude claude-opus-4-7 "
    "codex gpt-5.5 xhigh"
    "kimi kimi-k2.6 "
    "ccr-claude glm-5.1 "
    "ccr-claude minimax-m2.7 "
    "ccr-claude deepseek-v4-pro-max "
    "ccr-claude deepseek-v4-flash "
)

# All CUDA problems. Metal (08) runs separately from the macbook.
declare -a CUDA_PROBLEMS=(
    "problems/01_fp8_gemm"
    "problems/02_kda_cutlass"
    "problems/03_paged_attention"
    "problems/04_kahan_softmax"
    "problems/05_topk_bitonic"
    "problems/06_sonic_moe_swiglu"
    "problems/07_w4a16_gemm"
)

SWEEP_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_LOG="outputs/runs/sweep_${SWEEP_TIMESTAMP}.log"
mkdir -p "$(dirname "$SWEEP_LOG")"

echo "========================================" | tee "$SWEEP_LOG"
echo "KERNELBENCH-HARD SWEEP" | tee -a "$SWEEP_LOG"
echo "Started: $(date)" | tee -a "$SWEEP_LOG"
echo "Models:  ${#ACTIVE_MATRIX[@]}" | tee -a "$SWEEP_LOG"
echo "Probs:   ${#CUDA_PROBLEMS[@]}" | tee -a "$SWEEP_LOG"
echo "Runs:    $((${#ACTIVE_MATRIX[@]} * ${#CUDA_PROBLEMS[@]}))" | tee -a "$SWEEP_LOG"
echo "========================================" | tee -a "$SWEEP_LOG"

for problem in "${CUDA_PROBLEMS[@]}"; do
    for mh in "${ACTIVE_MATRIX[@]}"; do
        read -r HARNESS MODEL EFFORT <<< "$mh"
        echo "" | tee -a "$SWEEP_LOG"
        echo "=== $(date +%H:%M:%S) $HARNESS/$MODEL × $(basename "$problem") ===" | tee -a "$SWEEP_LOG"
        ./scripts/run_hard.sh "$HARNESS" "$MODEL" "$problem" "$EFFORT" 2>&1 | tee -a "$SWEEP_LOG" || true
    done
done

echo "" | tee -a "$SWEEP_LOG"
echo "========================================" | tee -a "$SWEEP_LOG"
echo "SWEEP COMPLETE" | tee -a "$SWEEP_LOG"
echo "Finished: $(date)" | tee -a "$SWEEP_LOG"
echo "========================================" | tee -a "$SWEEP_LOG"
