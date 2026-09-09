#!/bin/bash
set -e

# Confirmation set for truncated_eval_vs_full.py (seeds 4-6, never used
# for this experiment before), then pools with the discovery set
# (seeds 1-3, already run manually) for the final answer.
#
# Run from the REPO ROOT.

if [ ! -d "latent_diag" ] || [ ! -d "models" ]; then
    echo "ERROR: run this from the repo root (essim2d3d/), not from latent_diag/."
    exit 1
fi

export PYTHONPATH=.
POP="${1:-30}"
FULL_STEPS="${2:-2000}"
TRUNCATED_STEPS="${3:-1000}"
SIGMA="${4:-0.03}"

SEEDS=(4 5 6)
OUT_ROOT="results_truncated_eval"

REPORT_PATHS=()
LABELS=()

echo ">>> Confirmation sweep: ${#SEEDS[@]} fresh seeds, pop=$POP, full=$FULL_STEPS, truncated=$TRUNCATED_STEPS"
echo ""

for SEED in "${SEEDS[@]}"; do
    OUT_DIR="${OUT_ROOT}/seed${SEED}"
    echo ">>> [seed${SEED}]"
    python3 latent_diag/truncated_eval_vs_full.py \
        --base_model models/social3d_v15_full_final.json \
        --pop "$POP" --episodes 9 --full_steps "$FULL_STEPS" --truncated_steps "$TRUNCATED_STEPS" \
        --sigma "$SIGMA" --seed "$SEED" \
        --out_dir "$OUT_DIR" \
        > "log_truncated_seed${SEED}.txt" 2>&1
    echo "    done -- see log_truncated_seed${SEED}.txt"
    echo ""
done

echo "=================================================="
echo "Pooling discovery (seeds 1-3) + confirmation (seeds 4-6):"
echo "=================================================="
ALL_PATHS=""
for SEED in 1 2 3 4 5 6; do
    FOUND=$(ls results_truncated_eval/seed${SEED}/truncated_eval_report_*.csv 2>/dev/null | head -1)
    if [ -z "$FOUND" ]; then
        echo "WARNING: no report found for seed${SEED} -- skipping."
        continue
    fi
    ALL_PATHS="${ALL_PATHS}${ALL_PATHS:+,}${FOUND}"
done
ALL_LABELS="seed1,seed2,seed3,seed4,seed5,seed6"

python3 latent_diag/spearman_significance.py \
    --csvs "$ALL_PATHS" --labels "$ALL_LABELS" \
    --col_x full_fitness --col_y truncated_fitness
