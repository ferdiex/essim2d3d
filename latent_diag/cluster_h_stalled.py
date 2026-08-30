"""
Clustering on the full 16-dimensional h vectors for stuck-state rows.

Unlike analyze_h_pca.py (which reduces h to 2D for visualization), this
script clusters directly on the full 16-dim h -- important because the
large-scale collection run (n=300) showed that the clean "trailing
loop" pattern visible in the 2D PCA projection on a small pilot (n=30)
dissolved at scale, even though the AUC separability of stalled vs
normal stayed high (~0.82-0.87). This suggests the real structure lives
in the full 16-dim space in a way a 2D linear projection does not
capture well -- so clustering must be done there directly, not on the
PCA-reduced coordinates.

Two methods:
  1. K-means, with silhouette score used to pick the best k (from a
     small candidate range), giving a hard "N clusters" answer.
  2. DBSCAN, which does not require choosing k in advance and can
     report noise points separately -- useful as a cross-check, since
     it makes no assumption about cluster shape (K-means assumes
     roughly spherical clusters).

Runs separately for each agent, since h-space is already known to be
almost perfectly separable between agents (AUC=1.0) -- pooling both
together would just rediscover that split, not anything about the
attractor structure itself.

Usage:
    python3 cluster_h_stalled.py --csv h_vectors_300.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def analyze_agent(X, agent_name):
    print(f"\n=== Agent {agent_name.upper()} -- {len(X)} stuck-state rows ===")

    if len(X) < 10:
        print("Too few rows to cluster meaningfully. Skipping.")
        return

    X_scaled = StandardScaler().fit_transform(X)

    print("\n-- K-means (silhouette-based k selection, k=2..6) --")
    best_k, best_score = None, -1.0
    for k in range(2, 7):
        if len(X_scaled) <= k:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"  k={k}: silhouette = {score:.3f}")
        if score > best_score:
            best_score, best_k = score, k

    if best_k is not None:
        km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        sizes = pd.Series(labels).value_counts().sort_index()
        print(f"  Best k = {best_k} (silhouette = {best_score:.3f})")
        print(f"  Cluster sizes: {sizes.to_dict()}")
        if best_score < 0.25:
            print("  NOTE: silhouette below 0.25 typically indicates weak/overlapping "
                  "clusters, not a clean multi-attractor structure.")

    print("\n-- DBSCAN (density-based, no fixed k, eps=1.5, min_samples=10) --")
    db = DBSCAN(eps=1.5, min_samples=10)
    labels_db = db.fit_predict(X_scaled)
    n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    n_noise = int(np.sum(labels_db == -1))
    print(f"  Clusters found: {n_clusters_db}")
    print(f"  Noise points (not assigned to any cluster): {n_noise} / {len(X_scaled)} "
          f"({n_noise/len(X_scaled):.1%})")
    if n_clusters_db > 0:
        sizes_db = pd.Series(labels_db[labels_db != -1]).value_counts().sort_index()
        print(f"  Cluster sizes (excluding noise): {sizes_db.to_dict()}")
    print("  NOTE: DBSCAN's eps=1.5 is a starting guess on standardized data -- "
          "if this gives 0 or 1 clusters, or almost everything as noise, try adjusting "
          "eps up/down (smaller eps = stricter/more clusters, larger eps = looser/fewer).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    h_cols = [c for c in df.columns if c.startswith("h_")]

    stuck = df[df["stalled"] == True]
    print(f"Total rows: {len(df)}, stuck rows: {len(stuck)} ({len(stuck)/len(df):.1%})")

    for agent in ["a", "b"]:
        X = stuck[stuck["agent"] == agent][h_cols].values
        analyze_agent(X, agent)


if __name__ == "__main__":
    main()
