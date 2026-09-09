#!/bin/bash
set -e

# Step 2 of the robustness plan: 3 seeds x 2 sigmas on v15, horizon=100
# (the horizon that gave the higher of the two rho's so far, 0.134,
# still non-significant -- checking whether that holds or was itself
# noise). Ends by running spearman_significance.py on all 6 reports.
#
# Run from the REPO ROOT, same as collect_pool_data.sh.

if [ ! -d "latent_diag" ] || [ ! -d "models" ]; then
    echo "ERROR: run this from the repo root (essim2d3d/), not from latent_diag/."
    exit 1
fi

export PYTHONPATH=.
POP="${1:-30}"
HORIZON="${2:-100}"
EPISODES="${3:-9}"

SEEDS=(4 5 6)
SIGMAS=(0.03 0.06)

OUT_ROOT="results_confirmation"
mkdir -p "$OUT_ROOT"

REPORT_PATHS=()
LABELS=()

echo ">>> Robustness sweep: ${#SEEDS[@]} seeds x ${#SIGMAS[@]} sigmas, pop=$POP, horizon=$HORIZON, episodes=$EPISODES"
echo ""

for SEED in "${SEEDS[@]}"; do
    for SIGMA in "${SIGMAS[@]}"; do
        LABEL="seed${SEED}_sigma${SIGMA}"
        OUT_DIR="${OUT_ROOT}/${LABEL}"
        echo ">>> [$LABEL]"
        python3 latent_diag/dream_screen_vs_real.py \
            --base_model models/social3d_v15_full_final.json \
            --train_csv h_x_actraw_300.csv \
            --pop "$POP" --episodes "$EPISODES" --horizon "$HORIZON" \
            --sigma "$SIGMA" --seed "$SEED" \
            --out_dir "$OUT_DIR" \
            > "log_confirmation_${LABEL}.txt" 2>&1
        REPORT_PATHS+=("${OUT_DIR}/dream_screen_vs_real_report.csv")
        LABELS+=("$LABEL")
        echo "    done -- see log_confirmation_${LABEL}.txt for full output."
        echo ""
    done
done

CSVS_JOINED=$(IFS=,; echo "${REPORT_PATHS[*]}")
LABELS_JOINED=$(IFS=,; echo "${LABELS[*]}")

echo "=================================================="
echo "All 6 runs done. Significance summary:"
echo "=================================================="
python3 latent_diag/spearman_significance.py --csvs "$CSVS_JOINED" --labels "$LABELS_JOINED"
