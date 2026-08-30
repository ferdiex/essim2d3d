"""
PCA over h vectors collected by collect_h_vectors.py.

Reduces the 16 dimensions of h to 2 (PCA), and generates two plots:
  1. colored by agent (a/b)
  2. colored by stalled state (stuck/normal)

It also trains a simple classifier (logistic regression) on the full
16-dimensional h vectors (not on the 2D PCA projection) to give an
objective separability score (AUC) for each label -- a high AUC means
h DOES contain clear information about that distinction; ~0.5 means it
does not.

Usage:
    python3 analyze_h_pca.py --csv h_vectors_pilot.csv
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_prefix", default="pca_h")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Rows loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")

    h_cols = [c for c in df.columns if c.startswith("h_")]
    X = df[h_cols].values
    print(f"h dimensionality: {X.shape[1]} (expected: 16)")
    print(f"Rows per agent: {df['agent'].value_counts().to_dict()}")
    print(f"Rows with stalled=True: {df['stalled'].sum()} / {len(df)}\n")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_
    print(f"Variance explained by the 2 components: {var_explained[0]:.1%} + {var_explained[1]:.1%} "
          f"= {sum(var_explained):.1%} of total")

    fig, ax = plt.subplots(figsize=(7, 6))
    for agent, color in [("a", "tab:blue"), ("b", "tab:orange")]:
        mask = df["agent"] == agent
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=8, alpha=0.5, label=f"agent {agent.upper()}", color=color)
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
    ax.set_title("GRU memory space (h) reduced by PCA -- colored by agent")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_by_agent.png", dpi=150)
    print(f"Saved: {args.out_prefix}_by_agent.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    for stalled, color, label in [(False, "tab:green", "normal"), (True, "tab:red", "stuck")]:
        mask = df["stalled"] == stalled
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=8, alpha=0.5, label=label, color=color)
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
    ax.set_title("GRU memory space (h) reduced by PCA -- colored by stuck state")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_by_stalled.png", dpi=150)
    print(f"Saved: {args.out_prefix}_by_stalled.png\n")

    def auc_for_label(y, name):
        if y.nunique() < 2:
            print(f"{name}: only one class present in this sample, cannot compute AUC yet.")
            return
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        print(f"{name}: AUC = {auc:.3f} (0.5 = not separable, 1.0 = perfectly separable)")

    print("--- h separability (logistic regression on the full 16-dim h vector) ---")
    auc_for_label((df["agent"] == "b").astype(int), "agent (A vs B)")
    auc_for_label(df["stalled"].astype(int), "stalled (stuck vs normal)")
    auc_for_label(df["heard_signal"].astype(int), "heard_signal (heard vs not)")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
