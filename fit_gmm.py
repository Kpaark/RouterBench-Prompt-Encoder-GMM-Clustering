"""
Fit a Gaussian Mixture Model on RouterBench prompt embeddings.

Pipeline:
  1. Load data/embeddings.npy + data/embeddings.ids.csv.
  2. PCA-reduce to 50 dims (full-cov GMM in 768-d is unstable on ~36k samples).
  3. Sweep K via BIC, pick the best (or use --k to skip).
  4. Fit final GMM, compute majority `family` label per cluster.

Outputs (data/):
  pca.joblib, gmm.joblib, cluster_summary.csv,
  cluster_assignments.csv, bic_sweep.csv

Usage:
    python fit_gmm.py
    python fit_gmm.py --k 30                 # skip sweep
    python fit_gmm.py --k-sweep 5,10,20,40   # custom sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def fit(X: np.ndarray, k: int, n_init: int, seed: int):
    from sklearn.mixture import GaussianMixture

    return GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=n_init,
        random_state=seed,
        reg_covar=1e-5,
    ).fit(X)


def majority_table(assignments: np.ndarray, labels: pd.Series, k: int) -> pd.DataFrame:
    rows = []
    for c in range(k):
        sub = labels[assignments == c]
        if len(sub) == 0:
            rows.append(
                {"cluster_id": c, "size": 0, "majority_label": None, "purity": 0.0}
            )
            continue
        vc = sub.value_counts()
        rows.append(
            {
                "cluster_id": c,
                "size": int(len(sub)),
                "majority_label": str(vc.index[0]),
                "purity": float(vc.iloc[0] / vc.sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings", type=Path, default=Path("data/embeddings.npy")
    )
    parser.add_argument(
        "--ids", type=Path, default=Path("data/embeddings.ids.csv")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--k-sweep", default="10,15,20,25,30,40,50")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    X = np.load(args.embeddings)
    ids_df = pd.read_csv(args.ids)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=args.pca_dim, random_state=args.seed)
    X_fit = pca.fit_transform(X).astype(np.float32)
    joblib.dump(pca, args.out_dir / "pca.joblib")
    print(
        f"PCA -> {args.pca_dim} dims, retained variance "
        f"= {pca.explained_variance_ratio_.sum():.3f}"
    )

    if args.k is None:
        ks = [int(x) for x in args.k_sweep.split(",")]
        sweep = []
        for k in ks:
            g = fit(X_fit, k=k, n_init=1, seed=args.seed)
            sweep.append(
                {"k": k, "bic": float(g.bic(X_fit)), "aic": float(g.aic(X_fit))}
            )
            print(f"  K={k:>3}: BIC={sweep[-1]['bic']:,.1f}")
        sweep_df = pd.DataFrame(sweep)
        sweep_df.to_csv(args.out_dir / "bic_sweep.csv", index=False)
        best_k = int(sweep_df.loc[sweep_df["bic"].idxmin(), "k"])
        print(f"Best K = {best_k}")
    else:
        best_k = args.k

    gmm = fit(X_fit, k=best_k, n_init=3, seed=args.seed)
    assignments = gmm.predict(X_fit)
    summary = majority_table(assignments, ids_df["family"].astype(str), best_k)
    weighted_purity = (
        summary["purity"] * summary["size"]
    ).sum() / summary["size"].sum()
    print(
        f"K={best_k}, weighted purity={weighted_purity:.3f}, "
        f"BIC={gmm.bic(X_fit):,.1f}"
    )

    joblib.dump(gmm, args.out_dir / "gmm.joblib")
    summary.to_csv(args.out_dir / "cluster_summary.csv", index=False)

    out_df = ids_df.copy()
    out_df["cluster_id"] = assignments
    out_df["majority_label"] = out_df["cluster_id"].map(
        dict(zip(summary["cluster_id"], summary["majority_label"]))
    )
    out_df.to_csv(args.out_dir / "cluster_assignments.csv", index=False)


if __name__ == "__main__":
    main()
