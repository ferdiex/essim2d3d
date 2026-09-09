#!/bin/bash
set -e

# Discovery sweep for episode_count_baseline.py (seeds 1-3 only).
# Deliberately does NOT auto-run confirmation (seeds 4-6) -- check the
# pooled discovery result first, then decide if it's worth confirming.
#
# TIMING: each candidate needs two full evaluate_individual_3d calls
# (9 episodes + 7 episodes, both max_steps=2000). Do not assume this
# finishes in 2-3 minutes -- likely several minutes per seed, similar
# order to today's other sweeps or a bit more. No reliable estimate
# without timing it once.
#
# Run from the REPO ROOT.

if [ ! -d "latent_diag" ] || [ ! -d "models" ]; then
    echo "ERROR: run this from the repo root (essim2d3d/), not from latent_diag/."
    exit 1
fi

export PYTHONPATH=.
POP="${1:-30}"
EPISODES="${2:-9}"
REDUCED_EPISODES="${3:-7}"
MAX_STEPS="${4:-2000}"
SIGMA="${5:-0.03}"

SEEDS=(1 2 3)
OUT_ROOT="results_episode_baseline"

echo ">>> Episode-count baseline, DISCOVERY: seeds ${SEEDS[*]}, pop=$POP, "
echo "    full=$EPISODES episodes, reduced=$REDUCED_EPISODES episodes, max_steps=$MAX_STEPS"
echo ">>> Timing unknown in advance -- watch the per-candidate output as it runs."
echo ""

START=$(date +%s)
for SEED in "${SEEDS[@]}"; do
    OUT_DIR="${OUT_ROOT}/seed${SEED}"
    echo ">>> [seed${SEED}] started at $(date +%H:%M:%S)"
    python3 latent_drem/episode_count_baseline.py \
        --base_model models/social3d_v15_full_final.json \
        --pop "$POP" --episodes "$EPISODES" --reduced_episodes "$REDUCED_EPISODES" \
        --max_steps "$MAX_STEPS" --sigma "$SIGMA" --seed "$SEED" \
        --out_dir "$OUT_DIR" \
        > "log_episode_baseline_seed${SEED}.txt" 2>&1
    echo "    done at $(date +%H:%M:%S) -- see log_episode_baseline_seed${SEED}.txt"
    echo ""
done
END=$(date +%s)
echo ">>> All 3 discovery runs took $((END-START))s total ($((($END-$START)/3))s avg per seed)"

echo "=================================================="
echo "Discovery summary (seeds 1-3):"
echo "=================================================="
PATHS=""
for SEED in 1 2 3; do
    FOUND=$(ls ${OUT_ROOT}/seed${SEED}/episode_baseline_report_*.csv 2>/dev/null | head -1)
    if [ -z "$FOUND" ]; then
        echo "WARNING: no report found for seed${SEED} -- skipping."
        continue
    fi
    PATHS="${PATHS}${PATHS:+,}${FOUND}"
done
python3 latent_drem/spearman_significance.py \
    --csvs "$PATHS" --labels "seed1,seed2,seed3" \
    --col_x full_fitness --col_y reduced_fitness

echo ""
echo "If this looks worth confirming, run the same command with --seed 4/5/6"
echo "(fresh, never used for this experiment) before trusting it."
