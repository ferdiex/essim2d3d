# Paper 3 -- Sussillo & Barak, session scripts

End-to-end usage order. Everything runs in pure NumPy (without JAX/PyTorch)
except for the two data collection scripts, which require PyBullet and
run on the simulator.

## 1. Mathematical core (validated; should not need modification)

- `gru_dynamics.py` -- Pure F(h,x) (extracted from GRUNetwork.forward,
controllers.py lines 80-98) and q(h,x) = 0.5*||h-F(h,x)||^2. 
Verified against the actual GRUNetwork.forward: exact 0.0 difference.

- `jacobian.py` -- Analytical Jacobian of F with respect to h (derived
by hand). Verified against finite differences: difference ~1e-10,
using both random weights and the actual points found.

- `find_fixed_points.py` -- The search itself (Adam + analytical
gradient). Two versions:
- `search_fixed_points()`: without tether (original step 3). 
- `search_fixed_points_con_correa()`: WITH tether (recommended
for real data; see "Note on the tether" below). 
Also contains the synthetic validation (run
`python3 find_fixed_points.py` to repeat it).

## 2. Real data collection (run on your machine, with PyBullet)

- `collect_h_x_vectors.py` -- Like collect_h_vectors.py from the original
Paper 3, but *also* saves x (11 dims) per decision, not just h. 
Does not modify foraging_env3d.py (frozen) -- intercepts x via a
small subclass of the controller. Usage:
python3 collect_h_x_vectors.py --model models/social3d_v15_full_final.json --episodes 300 --out h_x_vectors_300.csv

- `compute_x_fijo.py` -- reads that CSV and calculates x_fijo per agent:
option 1 (average of x across all rows where stalled=True) and
option 2 (average of x only at the first timestep of each streak). 

Usage:
python3 compute_x_fijo.py --csv h_x_vectors_300.csv --out x_fijo_300.json

## 3. Search on real data and analysis

- `run_real_search.py` -- runs the fixed-point search (with a
leash) on real data, for agents A and B, using both
x_fijo options. For each fixed point found, it calculates the
Jacobian, the spectral radius (stability), and the action
towards which the dominant direction points (largest eigenvector,
projected via w_out). 

Usage:
python3 run_real_search.py ​​--csv h_x_vectors_300.csv --x_fijo x_fijo_300.json --out resultado_puntos_fijos_300.json

Default parameters: 80 initial h values ​​sampled from real rows
where stalled=True + 10 random ones, per agent and per x_fijo option; 
5000 Adam iterations, lr=0.05, lambda_leash=0.001.

- `diagnostico_streak_length.py` -- additional diagnostic: correlates
the spectral radius of each fixed point found with the actual
duration (streak_length) of the stall streak from which that initial
h originated. It also reports the actual frequency of each action
during the stall (subject to the methodological caveat below). Usage:
python3 diagnostico_streak_length.py
(assumes h_x_vectors_300.csv, x_fijo_300.json, and
social3d_v15_full_final.json are in the same directory)

## Main result (n=300 episodes, session 2026-08-29)

Spectral radius (largest |eigenvalue| of the Jacobian) at fixed points
found from actual jamming h-values:

- Agent A: median ~1.03 (UNSTABLE), high dispersion, only 10–23%
of points are stable. 
- Agent B: median ~0.999 (STABLE), very low dispersion, 76–87%
of points are stable. 
- Mann-Whitney U: p=2.7e-14 (option 1) and p=1.1e-05 (option 2). 
- The result held up when scaling from n=30 to n=300 (unlike other
patterns in this research that washed out at scale)—the
magnitude of A's instability dropped slightly (from 1.69 to 1.03),
but the separation remains very clear.

Interpretation: B's latent space, near its actual jamming states, is
locally stable (traps/holds)—A's is locally unstable (releases).
Consistent with behavioral observations (A recovers more easily;
B gets more stuck and requires more time to break free).

- `analisis_act_id_raw.py` — the final check, which really solidified
the finding. Uses `act_id_raw` for measurement, free from heuristic noise:
1. Frequency of `giro_der` (turn-right, action 2) during jamming,
A vs. B, using a chi-square test. 2. Persistence ("sustained turn"): given that the previous step was
`giro_der` (right turn), the probability of repeating it. 

Usage:
python3 analisis_act_id_raw.py --csv h_x_actraw_300.csv

## FINAL Result (the most solid finding, session 2026-08-29)

Using `act_id_raw` (the network's raw decision, BEFORE the
assistance instinct—captured by `collect_h_x_actraw_vectors.py`,
not available in previous CSVs), across n=300 episodes:

- Frequency of `giro_der` during the jam: A=27.0% (5477/20313),
B=76.2% (23383/30669). Difference of 49.3 percentage points. 
Chi-square: chi2=12079.3, p≈0 (huge n, massive difference). 

- Persistence ("sustained turn"): given that the previous step was
`giro_der`, the probability of repeating it: A=64.6% (n=5477 transitions