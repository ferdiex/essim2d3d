#!/bin/bash
set -e

# Collects small h_x_actraw datasets from three already-trained models
# (different from social3d_v15_full_final.json, which h_x_actraw_300.csv
# already covers), for the multi-policy pooling experiment discussed in
# README_world_model_addition.md ("Cross-policy generalization check").
#
# No new evolution is run here -- these are existing trained weights,
# just used to generate fresh episodes for data collection.
#
# Run from the REPO ROOT (same place you run caracterizacion_baseline_v15.sh
# from) -- collect_h_x_actraw_vectors.py lives in latent_diag/ but needs
# config_loader.py/world_builder.py (repo root) and robots/epuck_sim.urdf
# resolved relative to the working directory, same reasoning as the
# PYTHONPATH fix from earlier in the session.

if [ ! -d "latent_diag" ] || [ ! -d "models" ]; then
    echo "ERROR: run this from the repo root (essim2d3d/), not from latent_diag/."
    echo "       Expected to see both a latent_diag/ and a models/ directory here."
    exit 1
fi

EPISODES="${1:-30}"   # first argument overrides episode count, default 30
export PYTHONPATH=.

MODELS=(
    "social3d_v10_nn_annealing_gen143"
    "social3d_indep2_final"
    "social3d_sesgo_check1_150_final"
)

echo ">>> Collecting $EPISODES episodes each from ${#MODELS[@]} models: ${MODELS[*]}"
echo ""

for NAME in "${MODELS[@]}"; do
    OUT="h_x_actraw_${NAME}_${EPISODES}.csv"
    echo ">>> [$NAME] -> $OUT"
    python3 latent_diag/collect_h_x_actraw_vectors.py \
        --model "models/${NAME}.json" \
        --episodes "$EPISODES" \
        --out "$OUT" \
        > "log_collect_${NAME}.txt" 2>&1
    ROWS=$(($(wc -l < "$OUT") - 1))  # minus header
    echo "    done, $ROWS rows written."
    echo ""
done

echo "=================================================="
echo "Done. Files written to repo root:"
for NAME in "${MODELS[@]}"; do
    echo "  h_x_actraw_${NAME}_${EPISODES}.csv"
done
echo "Logs: log_collect_<model>.txt (only worth checking if a model failed)"
echo "=================================================="
