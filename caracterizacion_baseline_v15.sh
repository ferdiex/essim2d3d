#!/bin/bash
set -e

MODEL="models/social3d_v15_full_final.json"
STEPS="${1:-4000}"   # primer argumento del script, default 4000 si no se pasa nada
                      # (default de test_physics_gru.py es 2000 -- subido a
                      # proposito para probar si parte de los fracasos son
                      # episodios cortados a mitad de una recuperacion lenta)

echo ">>> Usando STEPS=$STEPS"

echo ">>> [1/4] Test de exito, señal ACTIVA (n=300, steps=$STEPS)..."
python3 test_physics_gru.py --model "$MODEL" --episodes 300 --steps "$STEPS" \
    --world random_obstacles --physics slippery \
    > log_baseline_activa.txt 2>&1

echo ">>> [2/4] Test de exito, señal MUDA (n=300, steps=$STEPS)..."
python3 test_physics_gru.py --model "$MODEL" --episodes 300 --steps "$STEPS" \
    --world random_obstacles --physics slippery --mute_social \
    > log_baseline_muda.txt 2>&1

echo ">>> [3/4] Tiempo total atascado por episodio (n=150, steps=$STEPS)..."
python3 diagnose_stall_time_total.py --model "$MODEL" --episodes 150 --max_steps "$STEPS" \
    > log_baseline_stall.txt 2>&1

echo ">>> [4/4] Efecto de desatoro por señal (n=150, steps=$STEPS)..."
python3 diagnose_signal_effect_windowed3d.py --model "$MODEL" --episodes 150 --max_steps "$STEPS" \
    > log_baseline_desatoro.txt 2>&1

echo ""
echo "=================================================="
echo "CARACTERIZACION DEL BASELINE (v15 + fix histeresis)"
echo "=================================================="

echo ""
echo "--- [1] Exito: señal ACTIVA ---"
grep -A 6 "resumen" log_baseline_activa.txt

echo ""
echo "--- [2] Exito: señal MUDA ---"
grep -A 6 "resumen" log_baseline_muda.txt

echo ""
echo "--- [3] Tiempo total atascado por episodio ---"
tail -20 log_baseline_stall.txt

echo ""
echo "--- [4] Efecto de desatoro por señal ---"
tail -25 log_baseline_desatoro.txt

echo ""
echo "=================================================="
echo "Listo. Logs completos guardados en log_baseline_*.txt"
echo "=================================================="
