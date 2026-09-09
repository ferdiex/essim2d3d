"""
diagnose_social_signal_effect.py

Two cheap diagnostics for a question raised live this session: does the
GRU-Residual architecture underuse its own memory (h) specifically for
social coordination, consistent with the residual-pathway-dominance
finding (97.8% of real decisions) and the author's own qualitative
impression that social/goal-seeking behavior never looked as clean as
obstacle avoidance? Both reuse data already on disk -- no new
simulation, no PyBullet.

DIAGNOSTIC 1 -- residual dominance, split by heard_signal:
Re-groups the per-row h-mediated vs residual contribution magnitudes
(already computed by compare_residual_vs_recurrent_pathway.py) by
whether the agent was receiving the partner's social signal at that
decision. If residual dominance is just as high WITH heard_signal=True
as without, that's direct evidence memory isn't being leaned on more
even in the moment it would matter most for coordination.

DIAGNOSTIC 2 -- does h carry useful predictive state at all?
Trains a tiny linear probe (logistic regression) to predict whether the
EPISODE eventually succeeds (own_success at any later row in the same
episode/agent trajectory), separately from h alone and from x alone,
on held-out episodes. If h barely beats (or loses to) x at this, it
suggests h isn't accumulating useful state for the outcome that
matters -- consistent with the residual pathway making memory close to
irrelevant, not just less important.

Neither diagnostic proves the transfer-learning idea would work -- they
only test the PRECONDITION (is memory underused for social/outcome-
relevant information) before committing to that separate project.

Usage:
    python3 diagnose_social_signal_effect.py \
        --csv ../h_x_actraw_300.csv \
        --residual_report ../latent_diag/results_residual_analysis/residual_vs_recurrent_report.csv \
        --out_dir results_social_diagnosis/

Usage (self-test, synthetic data, no real files needed):
    python3 diagnose_social_signal_effect.py --selftest
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_THIS_DIR, os.path.join(_THIS_DIR, "latent_diag"),
                   os.getcwd(), os.path.join(os.getcwd(), "latent_diag")):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)


def diagnostic_1_heard_signal_split(df, residual_report_path):
    if not os.path.exists(residual_report_path):
        print(f"[Diagnostic 1] SKIPPED -- residual report not found at "
              f"{residual_report_path}. Run compare_residual_vs_recurrent_pathway.py "
              f"first, then point --residual_report at its output.")
        return
    rep = pd.read_csv(residual_report_path)
    if len(rep) != len(df):
        print(f"[Diagnostic 1] SKIPPED -- row count mismatch between CSV ({len(df)}) "
              f"and residual report ({len(rep)}). They need to come from the exact same "
              f"run of the same CSV.")
        return
    if "heard_signal" not in df.columns:
        print("[Diagnostic 1] SKIPPED -- CSV has no heard_signal column.")
        return

    merged = rep.copy()
    merged["heard_signal"] = df["heard_signal"].astype(bool).values

    print("=== Diagnostic 1: residual dominance, with vs. without social signal ===")
    for val in [True, False]:
        sub = merged[merged["heard_signal"] == val]
        if len(sub) == 0:
            print(f"  heard_signal={val}: no rows")
            continue
        pct_res_dominant = (sub["residual_norm"] > sub["h_mediated_norm"]).mean() * 100
        print(f"  heard_signal={val!s:5s}  n={len(sub):6d}  "
              f"median ratio(residual/h)={sub['ratio_residual_over_h'].median():.3f}  "
              f"residual-dominant in {pct_res_dominant:.1f}% of rows")
    print("  (if these two rows look similar, memory isn't being leaned on more even "
          "when social signal is present)\n")


def diagnostic_2_h_vs_x_predicts_outcome(df, seed=42, test_frac=0.2):
    print("=== Diagnostic 2: does h alone predict eventual episode success "
          "better than x alone? ===")
    h_cols = [f"h_{k}" for k in range(16)]
    x_cols = [f"x_{k}" for k in range(11)]
    missing = [c for c in h_cols + x_cols + ["episode", "agent", "own_success"] if c not in df.columns]
    if missing:
        print(f"  SKIPPED -- CSV missing columns: {missing}")
        return

    # Label each row with whether ITS episode/agent trajectory eventually succeeds
    # (not just this row's own_success, which is usually False until the end)
    eventual = df.groupby(["episode", "agent"])["own_success"].transform("max")
    labels = eventual.astype(int).values

    if len(np.unique(labels)) < 2:
        print("  SKIPPED -- only one class present (no variation in eventual success).")
        return

    episodes = df["episode"].values
    uniq_eps = np.unique(episodes)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq_eps)
    n_test = max(1, int(len(uniq_eps) * test_frac))
    test_eps = set(uniq_eps[:n_test].tolist())
    is_test = np.array([e in test_eps for e in episodes])

    for name, cols in [("h alone", h_cols), ("x alone", x_cols)]:
        X = df[cols].values
        model = LogisticRegression(max_iter=2000)
        model.fit(X[~is_test], labels[~is_test])
        probs = model.predict_proba(X[is_test])[:, 1]
        auc = roc_auc_score(labels[is_test], probs)
        print(f"  {name:10s}  test AUC = {auc:.3f}  "
              f"(0.5 = no better than chance, 1.0 = perfect)")

    print("  (if 'h alone' isn't clearly better than 'x alone' at this, h isn't "
          "accumulating much useful state beyond what's already visible right now)\n")


def run(csv_path, residual_report_path, out_dir):
    df = pd.read_csv(csv_path)
    diagnostic_1_heard_signal_split(df, residual_report_path)
    diagnostic_2_h_vs_x_predicts_outcome(df)


def run_and_save(csv_path, residual_report_path, out_dir):
    """Same as run(), but also writes a CSV report -- needed so the
    numbers behind Tabla 4 come from a file a reviewer could reproduce,
    not from a chat transcript."""
    df = pd.read_csv(csv_path)
    rows = []

    if residual_report_path and os.path.exists(residual_report_path):
        rep = pd.read_csv(residual_report_path)
        if len(rep) == len(df) and "heard_signal" in df.columns:
            merged = rep.copy()
            merged["heard_signal"] = df["heard_signal"].astype(bool).values
            for val in [True, False]:
                sub = merged[merged["heard_signal"] == val]
                if len(sub) == 0:
                    continue
                pct_res_dominant = (sub["residual_norm"] > sub["h_mediated_norm"]).mean() * 100
                rows.append({
                    "diagnostico": "residual_vs_heard_signal",
                    "grupo": f"heard_signal={val}",
                    "n": len(sub),
                    "mediana_ratio_residual_sobre_h": float(sub["ratio_residual_over_h"].median()),
                    "pct_residual_dominante": float(pct_res_dominant),
                })

    h_cols = [f"h_{k}" for k in range(16)]
    x_cols = [f"x_{k}" for k in range(11)]
    if all(c in df.columns for c in h_cols + x_cols + ["episode", "agent", "own_success"]):
        eventual = df.groupby(["episode", "agent"])["own_success"].transform("max")
        labels = eventual.astype(int).values
        if len(np.unique(labels)) >= 2:
            episodes = df["episode"].values
            uniq_eps = np.unique(episodes)
            rng = np.random.RandomState(42)
            rng.shuffle(uniq_eps)
            n_test = max(1, int(len(uniq_eps) * 0.2))
            test_eps = set(uniq_eps[:n_test].tolist())
            is_test = np.array([e in test_eps for e in episodes])
            for name, cols in [("h_solo", h_cols), ("x_solo", x_cols)]:
                X = df[cols].values
                model = LogisticRegression(max_iter=2000)
                model.fit(X[~is_test], labels[~is_test])
                probs = model.predict_proba(X[is_test])[:, 1]
                auc = roc_auc_score(labels[is_test], probs)
                rows.append({
                    "diagnostico": "h_vs_x_predice_exito",
                    "grupo": name,
                    "n": int(is_test.sum()),
                    "auc": float(auc),
                })

    out = pd.DataFrame(rows)
    print("=== Resumen guardado ===")
    print(out.to_string(index=False))

    # También corre la version impresa normal, para lectura en consola
    diagnostic_1_heard_signal_split(df, residual_report_path)
    diagnostic_2_h_vs_x_predicts_outcome(df)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, "social_signal_diagnosis_report.csv")
        out.to_csv(report_path, index=False)
        print(f"\nReporte completo escrito en {report_path}")
    return out


def run_selftest():
    print("=== SELF-TEST (synthetic data) ===\n")
    rng = np.random.default_rng(0)
    n_episodes = 40
    rows = []
    fixed_success_direction = rng.normal(size=16)  # ONE consistent direction in h-space
    # that correlates with success across episodes -- a linear probe can only detect
    # a signal that's consistent, not a fresh random direction re-drawn per episode.
    for ep in range(n_episodes):
        succeeds = rng.random() < 0.5
        n_dec = rng.integers(10, 30)
        h_bias = fixed_success_direction * (1.0 if succeeds else -1.0) * 0.8
        for dec in range(n_dec):
            h = h_bias + rng.normal(size=16) * 0.3
            x = rng.normal(size=11) * 0.3
            own_success = bool(succeeds and dec == n_dec - 1)
            rows.append({
                "episode": ep, "agent": "a", "decision": dec,
                "heard_signal": bool(rng.random() < 0.3),
                "own_success": own_success,
                **{f"h_{k}": h[k] for k in range(16)},
                **{f"x_{k}": x[k] for k in range(11)},
            })
    df = pd.DataFrame(rows)
    csv_path = "/tmp/_social_diag_selftest.csv"
    df.to_csv(csv_path, index=False)

    diagnostic_1_heard_signal_split(df, "/nonexistent/path.csv")  # exercises the SKIPPED branch
    diagnostic_2_h_vs_x_predicts_outcome(df)
    print("Self-test finished without errors.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--residual_report", default=None,
                     help="output of compare_residual_vs_recurrent_pathway.py on the SAME csv")
    ap.add_argument("--out_dir", default="results_social_diagnosis")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    if not args.csv:
        raise SystemExit("Need --csv (or use --selftest).")

    run_and_save(args.csv, args.residual_report, args.out_dir)


if __name__ == "__main__":
    main()
