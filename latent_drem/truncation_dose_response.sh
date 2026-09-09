#!/bin/bash
set -e

# Dose-response for truncated_eval_vs_full.py: brackets the already-confirmed
# truncated_steps=1000 point (pooled rho=0.259, p=0.0008 over 6 seeds) with
# 750 and 1500, using seeds 1-3 for each (reused deliberately -- same seeds
# as the 1000-step point, for direct comparability of the trend; this is a
# secondary/exploratory sweep characterizing a curve, not a new standalone
# significance claim, so lighter than the full discovery+confirmation
# protocol used for the headline result).
#
# Run from the REPO ROOT.

if [ ! -d "latent_diag" ] || [ ! -d "models" ]; then
    echo "ERROR: run this from the repo root (essim2d3d/), not from latent_diag/."
    exit 1
fi

export PYTHONPATH=.
POP="${1:-30}"
FULL_STEPS="${2:-2000}"
SIGMA="${3:-0.03}"

TRUNC_VALUES=(750 1500)
SEEDS=(1 2 3)
OUT_ROOT="results_truncated_eval"

echo ">>> Dose-response sweep: truncated_steps in ${TRUNC_VALUES[*]}, seeds ${SEEDS[*]}, full=$FULL_STEPS"
echo ""

for TRUNC in "${TRUNC_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUT_DIR="${OUT_ROOT}/trunc${TRUNC}_seed${SEED}"
        echo ">>> [truncated_steps=$TRUNC, seed=$SEED]"
        python3 latent_diag/truncated_eval_vs_full.py \
            --base_model models/social3d_v15_full_final.json \
            --pop "$POP" --episodes 9 --full_steps "$FULL_STEPS" --truncated_steps "$TRUNC" \
            --sigma "$SIGMA" --seed "$SEED" \
            --out_dir "$OUT_DIR" \
            > "log_dose_trunc${TRUNC}_seed${SEED}.txt" 2>&1
        echo "    done -- see log_dose_trunc${TRUNC}_seed${SEED}.txt"
    done
    echo ""
done

echo "=================================================="
echo "Dose-response summary (each truncation level, 3 seeds pooled):"
echo "=================================================="
for TRUNC in "${TRUNC_VALUES[@]}"; do
    echo ""
    echo "--- truncated_steps=$TRUNC ---"
    PATHS=""
    for SEED in 1 2 3; do
        FOUND=$(ls ${OUT_ROOT}/trunc${TRUNC}_seed${SEED}/truncated_eval_report_*.csv 2>/dev/null | head -1)
        if [ -z "$FOUND" ]; then
            echo "WARNING: no report found for trunc${TRUNC}_seed${SEED} -- skipping."
            continue
        fi
        PATHS="${PATHS}${PATHS:+,}${FOUND}"
    done
    python3 latent_diag/spearman_significance.py \
        --csvs "$PATHS" --labels "seed1,seed2,seed3" \
        --col_x full_fitness --col_y truncated_fitness
done

echo ""
echo "=================================================="
echo "For reference, truncated_steps=1000 (already confirmed, 6 seeds):"
echo "  rho=0.259  p=0.000756  95% CI=(0.110, 0.396)"
echo "=================================================="
