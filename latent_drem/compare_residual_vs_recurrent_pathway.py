"""
compare_residual_vs_recurrent_pathway.py

Follow-up to a question raised live this session: could the GRU-Residual
architecture itself (not just the physics predictor's accuracy) explain
why dreamed rollouts diverge? Specifically: h(t+1)=F(h,x) genuinely does
NOT depend on the residual pathway (w_res) at all -- confirmed earlier
this session by reading controllers.py -- so the residual can't explain
why the x(t+1) predictor itself struggles. But the residual DOES feed
directly into the DECISION (the logits an evolved candidate uses to pick
an action in dream_screen_vs_real.py), as an unfiltered bypass from x
straight to the output, alongside the path mediated by h (which carries
recurrent "memory" and is presumably smoother). If a small error in
dreamed x lands near a sensitive threshold (e.g. near_wall=0.82), the
residual path could flip the chosen action abruptly in a way the
h-mediated path wouldn't -- a candidate mechanical explanation for fast
error accumulation in dreamed rollouts, not yet tested.

This script tests the PRECONDITION for that hypothesis using only
already-collected real data (h_x_actraw_300.csv) and real weights --
no simulation: how large is the residual pathway's contribution to the
logits, relative to the h-mediated pathway's, in practice? If the
residual dominates (or is comparable) especially near the near_wall
threshold, the hypothesis is worth pursuing further (e.g. an ablation
inside dream_screen_vs_real.py). If the residual is consistently small
relative to the h-mediated term, the architecture is not a promising
explanation and this specific thread can be closed.

Logits formula replicated EXACTLY from controllers.py (GRUNetwork.forward,
with ENABLE_SOCIAL_FILTER=False, ENABLE_BEACON=False, the defaults):

    near_wall = max(x[:8]) > NEAR_WALL_THRESHOLD (0.82)
    multi = 0.3 if near_wall else 5.0
    res = x @ w_res
    in_conflict = |x[8]| > 0.05 and |x[10]| > 0.05 and sign(x[8]) != sign(x[10])
    effective_boost = 3.0 * (1.0 if in_conflict else 1.0)   # JALON_ATTENUATION_FACTOR=1.0,
                                                              # currently a no-op either way
    res = res + x[10] * w_res[10,:] * (effective_boost - 1.0)
    logits = h @ w_out + b_out + res * multi

This script only computes the two additive TERMS of that last line
(h @ w_out + b_out, and res * multi) and compares their magnitudes --
it does not change or re-derive the decision itself.

Usage:
    python3 compare_residual_vs_recurrent_pathway.py \
        --csv ../h_x_actraw_300.csv \
        --weights ../models/social3d_v15_full_final.json \
        --out_dir results_residual_analysis/

Usage (self-test, synthetic weights/data, no real files needed):
    python3 compare_residual_vs_recurrent_pathway.py --selftest
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_THIS_DIR, os.path.join(_THIS_DIR, "latent_diag"),
                   os.getcwd(), os.path.join(os.getcwd(), "latent_diag")):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

NEAR_WALL_THRESHOLD = 0.82
SOCIAL_RESIDUAL_BOOST = 3.0
JALON_ATTENUATION_FACTOR = 1.0
JALON_CONFLICT_MIN_MAG = 0.05


def logit_contributions(h, x, weights):
    """Returns (h_mediated_term, residual_term, near_wall) for one
    (h, x) pair -- the two additive pieces of the real logits formula,
    not a re-derivation of the decision itself."""
    w_out = np.array(weights["w_out"])
    b_out = np.array(weights["b_out"])
    w_res = np.array(weights["w_res"])

    near_wall = bool(np.max(x[:8]) > NEAR_WALL_THRESHOLD)
    multi = 0.3 if near_wall else 5.0

    res = x @ w_res
    in_conflict = (abs(x[8]) > JALON_CONFLICT_MIN_MAG and abs(x[10]) > JALON_CONFLICT_MIN_MAG
                   and np.sign(x[8]) != np.sign(x[10]))
    effective_boost = SOCIAL_RESIDUAL_BOOST * JALON_ATTENUATION_FACTOR if in_conflict else SOCIAL_RESIDUAL_BOOST
    res = res + x[10] * w_res[10, :] * (effective_boost - 1.0)
    residual_term = res * multi

    h_mediated_term = h @ w_out + b_out
    return h_mediated_term, residual_term, near_wall


def run(csv_path, weights_path, out_dir):
    df = pd.read_csv(csv_path)
    h_cols = [f"h_{k}" for k in range(16)]
    x_cols = [f"x_{k}" for k in range(11)]
    missing = [c for c in h_cols + x_cols + ["agent"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")
    has_real_near_wall = "near_wall" in df.columns

    with open(weights_path) as f:
        weights = json.load(f)

    rows = []
    h_vals = df[h_cols].values
    x_vals = df[x_cols].values
    agents = df["agent"].values
    real_near_wall_col = df["near_wall"].astype(bool).values if has_real_near_wall else None

    for i in range(len(df)):
        w = weights["brain_a"] if agents[i] == "a" else weights["brain_b"]
        h_term, res_term, computed_near_wall = logit_contributions(h_vals[i], x_vals[i], w)
        h_norm = float(np.linalg.norm(h_term))
        res_norm = float(np.linalg.norm(res_term))
        row = {
            "agent": agents[i],
            "near_wall_computed": computed_near_wall,
            "h_mediated_norm": h_norm, "residual_norm": res_norm,
            "ratio_residual_over_h": res_norm / max(h_norm, 1e-9),
        }
        if has_real_near_wall:
            row["near_wall_real"] = bool(real_near_wall_col[i])
        rows.append(row)

    out = pd.DataFrame(rows)

    if has_real_near_wall:
        agree = (out["near_wall_computed"] == out["near_wall_real"]).mean() * 100
        n_real_true = out["near_wall_real"].sum()
        n_computed_true = out["near_wall_computed"].sum()
        print(f"[near_wall check] CSV's own column: {n_real_true} True out of {len(out)}. "
              f"Recomputed (threshold {NEAR_WALL_THRESHOLD} on x[:8]): {n_computed_true} True. "
              f"Agreement: {agree:.2f}%.")
        if agree < 99.0:
            print("  MISMATCH -- recomputed near_wall does not reliably match the real "
                  "column. Using the REAL column (near_wall_real) for the grouped analysis "
                  "below, not the recomputed one.")
        near_wall_col_to_use = "near_wall_real"
    else:
        print("[near_wall check] CSV has no near_wall column -- falling back to recomputed "
              "value, unverified against ground truth.")
        near_wall_col_to_use = "near_wall_computed"
    print()

    def summarize(sub, label):
        if len(sub) == 0:
            print(f"  {label}: no rows")
            return
        pct_res_dominant = (sub["residual_norm"] > sub["h_mediated_norm"]).mean() * 100
        print(f"  {label:24s} n={len(sub):6d}  "
              f"median h_norm={sub['h_mediated_norm'].median():7.3f}  "
              f"median res_norm={sub['residual_norm'].median():7.3f}  "
              f"median ratio={sub['ratio_residual_over_h'].median():.3f}  "
              f"residual-dominant in {pct_res_dominant:.1f}% of rows")

    print("=== Residual pathway vs h-mediated pathway, magnitude comparison ===")
    print("(ratio > 1 means the residual/bypass term is LARGER than the h-mediated term "
          "for that row -- i.e. x has more direct sway over the decision than accumulated "
          "recurrent state does)\n")
    summarize(out, "ALL ROWS")
    summarize(out[out[near_wall_col_to_use]], f"{near_wall_col_to_use}=True")
    summarize(out[~out[near_wall_col_to_use]], f"{near_wall_col_to_use}=False")
    for agent in sorted(out["agent"].unique()):
        summarize(out[out["agent"] == agent], f"agent={agent}")

    print("\nInterpretation guide:")
    print("  - If residual is dominant mostly/only near near_wall=True: supports the "
          "hypothesis that threshold-crossing errors in dreamed x could flip decisions "
          "abruptly via the unfiltered bypass -- worth an ablation inside "
          "dream_screen_vs_real.py next.")
    print("  - If residual is small/rare everywhere: the architecture is not a promising "
          "explanation for dreamed-rollout divergence -- close this specific thread.")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, "residual_vs_recurrent_report.csv")
        out.to_csv(report_path, index=False)
        print(f"\nFull per-row report written to {report_path}")
    return out


def run_selftest():
    print("=== SELF-TEST (synthetic weights/data, no PyBullet needed) ===\n")
    rng = np.random.default_rng(0)
    weights = {
        "brain_a": {
            "w_out": (rng.normal(size=(16, 5)) * 0.5).tolist(),
            "b_out": (rng.normal(size=(5,)) * 0.1).tolist(),
            "w_res": (rng.normal(size=(11, 5)) * 0.5).tolist(),
        },
        "brain_b": {
            "w_out": (rng.normal(size=(16, 5)) * 0.5).tolist(),
            "b_out": (rng.normal(size=(5,)) * 0.1).tolist(),
            "w_res": (rng.normal(size=(11, 5)) * 0.5).tolist(),
        },
    }
    n = 200
    rows = []
    for i in range(n):
        h = rng.normal(size=16) * 0.3
        x = rng.normal(size=11) * 0.3
        if i % 5 == 0:
            x[:8] = 0.9  # force some near_wall=True rows
        rows.append({
            "agent": "a" if i % 2 == 0 else "b",
            "near_wall": bool(i % 5 == 0),  # matches the forced x[:8]=0.9 rows exactly,
            # exercises the agreement-check branch (should show ~100% agreement here)
            **{f"h_{k}": h[k] for k in range(16)},
            **{f"x_{k}": x[k] for k in range(11)},
        })
    df = pd.DataFrame(rows)
    csv_path = "/tmp/_residual_selftest.csv"
    weights_path = "/tmp/_residual_selftest_weights.json"
    df.to_csv(csv_path, index=False)
    with open(weights_path, "w") as f:
        json.dump(weights, f)

    out = run(csv_path, weights_path, out_dir=None)
    assert len(out) == n and "ratio_residual_over_h" in out.columns
    print("\nSelf-test finished without errors.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--weights")
    ap.add_argument("--out_dir", default="results_residual_analysis")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    if not args.csv or not args.weights:
        raise SystemExit("Need --csv and --weights (or use --selftest).")

    run(args.csv, args.weights, args.out_dir)


if __name__ == "__main__":
    main()
