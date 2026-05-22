"""
Fit a Gaussian Mixture Model on RouterBench prompt embeddings.

Pipeline:
  1. Load `data/embeddings.npy` and `data/embeddings.ids.csv`
     (produced by encode_prompts.py).
  2. Optionally reduce dimensionality with PCA before clustering. Full-cov
     GMM in the raw 768-dim embedding space is statistically unstable with
     only ~36k samples per component, so PCA -> 50 is the default.
  3. Sweep K via BIC/AIC to pick the number of components, or skip the
     sweep with `--k`.
  4. Fit the final GMM, compute per-sample cluster assignments.
  5. Compute majority RouterBench label per cluster, plus purity and size.

Outputs (in `data/`):
  gmm.joblib                  fitted GaussianMixture
  pca.joblib                  fitted PCA (if --pca-dim > 0)
  cluster_assignments.csv     per-prompt cluster id + majority label
  cluster_summary.csv         per-cluster size, majority label, purity
  bic_sweep.csv               sweep results (if a sweep was run)
  gmm.manifest.txt            run summary

Usage:
    python fit_gmm.py                          # PCA->50, sweep K, BIC pick
    python fit_gmm.py --k 30                   # skip sweep, fit K=30
    python fit_gmm.py --pca-dim 0              # cluster in full embedding space
    python fit_gmm.py --covariance-type diag   # cheaper, axis-aligned ellipses
    python fit_gmm.py --label-col eval_name    # finer-grained majority class
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def parse_k_sweep(arg: str) -> list[int]:
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def fit_one_gmm(
    X: np.ndarray,
    k: int,
    covariance_type: str,
    n_init: int,
    seed: int,
    max_iter: int = 300,
):
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        n_init=n_init,
        random_state=seed,
        max_iter=max_iter,
        reg_covar=1e-5,
    )
    gmm.fit(X)
    return gmm


def majority_class_table(
    assignments: np.ndarray, labels: pd.Series, n_clusters: int
) -> pd.DataFrame:
    """For each cluster, return size, majority label, purity, and top-3 labels."""
    rows = []
    for k in range(n_clusters):
        mask = assignments == k
        sub = labels[mask]
        if len(sub) == 0:
            rows.append(
                {
                    "cluster_id": k,
                    "size": 0,
                    "majority_label": None,
                    "purity": float("nan"),
                    "top3_labels": "",
                }
            )
            continue
        vc = sub.value_counts()
        top3 = vc.head(3)
        purity = float(vc.iloc[0] / vc.sum())
        rows.append(
            {
                "cluster_id": k,
                "size": int(len(sub)),
                "majority_label": str(vc.index[0]),
                "purity": purity,
                "top3_labels": "; ".join(f"{name}({cnt})" for name, cnt in top3.items()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/embeddings.npy"),
        help="Embedding matrix from encode_prompts.py.",
    )
    parser.add_argument(
        "--ids",
        type=Path,
        default=Path("data/embeddings.ids.csv"),
        help="Row-aligned ids/labels CSV from encode_prompts.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Directory for the fitted GMM + cluster CSVs (default: ./data).",
    )
    parser.add_argument(
        "--label-col",
        default="family",
        choices=["family", "eval_name"],
        help="Label column used to compute majority class per cluster.",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=50,
        help="Reduce embeddings to this many PCA components before GMM (0 = skip).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Fixed number of mixture components; skips the BIC sweep.",
    )
    parser.add_argument(
        "--k-sweep",
        default="10,15,20,25,30,40,50",
        help="Comma-separated K values to sweep when --k is not given.",
    )
    parser.add_argument(
        "--covariance-type",
        default="full",
        choices=["full", "tied", "diag", "spherical"],
        help="GMM covariance parameterization (default: full).",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=3,
        help="Random restarts for the final fit (default: 3).",
    )
    parser.add_argument(
        "--sweep-n-init",
        type=int,
        default=1,
        help="Random restarts for each K during the sweep (default: 1, for speed).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Max EM iterations per fit (default: 300).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.embeddings.exists():
        raise SystemExit(
            f"Embeddings not found: {args.embeddings}. Run encode_prompts.py first."
        )
    if not args.ids.exists():
        raise SystemExit(
            f"IDs CSV not found: {args.ids}. Run encode_prompts.py first."
        )

    X = np.load(args.embeddings)
    ids_df = pd.read_csv(args.ids)
    if len(X) != len(ids_df):
        raise SystemExit(
            f"Row mismatch: {len(X)} embeddings vs {len(ids_df)} ids."
        )
    print(f"Embeddings: shape={X.shape}, dtype={X.dtype}")
    print(f"IDs       : {len(ids_df):,} rows, columns={list(ids_df.columns)}")

    pca = None
    if args.pca_dim and args.pca_dim > 0:
        from sklearn.decomposition import PCA

        n_comp = min(args.pca_dim, X.shape[1], X.shape[0])
        print(f"\nFitting PCA -> {n_comp} components...")
        pca = PCA(n_components=n_comp, random_state=args.seed)
        X_fit = pca.fit_transform(X).astype(np.float32)
        ev = float(pca.explained_variance_ratio_.sum())
        print(f"  retained variance = {ev:.3f}")
        pca_path = args.out_dir / "pca.joblib"
        joblib.dump(pca, pca_path)
        print(f"  saved -> {pca_path.resolve()}")
    else:
        X_fit = X.astype(np.float32, copy=False)
        ev = None

    print(f"\nClustering space: shape={X_fit.shape}")

    sweep_results = None
    if args.k is None:
        ks = parse_k_sweep(args.k_sweep)
        print(
            f"\nSweeping K over {ks} | covariance_type='{args.covariance_type}' | "
            f"n_init={args.sweep_n_init}"
        )
        rows = []
        for k in ks:
            gmm = fit_one_gmm(
                X_fit,
                k=k,
                covariance_type=args.covariance_type,
                n_init=args.sweep_n_init,
                seed=args.seed,
                max_iter=args.max_iter,
            )
            bic = float(gmm.bic(X_fit))
            aic = float(gmm.aic(X_fit))
            rows.append(
                {"k": k, "bic": bic, "aic": aic, "converged": bool(gmm.converged_)}
            )
            print(
                f"  K={k:>3}: BIC={bic:>14,.1f}  AIC={aic:>14,.1f}  "
                f"converged={gmm.converged_}"
            )
        sweep_results = pd.DataFrame(rows)
        best_k = int(sweep_results.loc[sweep_results["bic"].idxmin(), "k"])
        print(f"\nBest K by BIC = {best_k}")
    else:
        best_k = args.k
        print(f"\nUsing fixed K = {best_k}")

    print(f"\nFitting final GMM: K={best_k}, n_init={args.n_init}")
    gmm = fit_one_gmm(
        X_fit,
        k=best_k,
        covariance_type=args.covariance_type,
        n_init=args.n_init,
        seed=args.seed,
        max_iter=args.max_iter,
    )
    final_bic = float(gmm.bic(X_fit))
    final_aic = float(gmm.aic(X_fit))
    print(
        f"  converged={gmm.converged_}  iters={gmm.n_iter_}  "
        f"BIC={final_bic:,.1f}  AIC={final_aic:,.1f}"
    )

    assignments = gmm.predict(X_fit)

    labels = ids_df[args.label_col].astype(str)
    summary = majority_class_table(assignments, labels, n_clusters=best_k)
    nonempty = summary["size"] > 0
    weighted_purity = float(
        (summary.loc[nonempty, "purity"] * summary.loc[nonempty, "size"]).sum()
        / summary.loc[nonempty, "size"].sum()
    )
    print(f"\nWeighted average purity (by '{args.label_col}') = {weighted_purity:.3f}")
    print("\nTop 15 clusters by size:")
    print(
        summary.sort_values("size", ascending=False)
        .head(15)
        .to_string(index=False)
    )

    gmm_path = args.out_dir / "gmm.joblib"
    joblib.dump(gmm, gmm_path)
    print(f"\nSaved -> {gmm_path.resolve()}")

    summary_path = args.out_dir / "cluster_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved -> {summary_path.resolve()}")

    majority_lookup = dict(zip(summary["cluster_id"], summary["majority_label"]))
    assignments_df = ids_df.copy()
    assignments_df["cluster_id"] = assignments
    assignments_df["majority_label"] = assignments_df["cluster_id"].map(majority_lookup)
    assignments_df["is_majority"] = (
        assignments_df[args.label_col].astype(str) == assignments_df["majority_label"]
    )
    assignments_path = args.out_dir / "cluster_assignments.csv"
    assignments_df.to_csv(assignments_path, index=False)
    print(f"Saved -> {assignments_path.resolve()}")

    if sweep_results is not None:
        sweep_path = args.out_dir / "bic_sweep.csv"
        sweep_results.to_csv(sweep_path, index=False)
        print(f"Saved -> {sweep_path.resolve()}")

    manifest = args.out_dir / "gmm.manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                f"embeddings: {args.embeddings.resolve()}",
                f"ids: {args.ids.resolve()}",
                f"label_col: {args.label_col}",
                f"pca_dim: {args.pca_dim if pca is not None else 0}",
                f"pca_explained_variance: {ev if ev is not None else 'NA'}",
                f"k: {best_k}",
                f"covariance_type: {args.covariance_type}",
                f"n_init: {args.n_init}",
                f"seed: {args.seed}",
                f"converged: {bool(gmm.converged_)}",
                f"n_iter: {int(gmm.n_iter_)}",
                f"bic: {final_bic:.1f}",
                f"aic: {final_aic:.1f}",
                f"weighted_purity: {weighted_purity:.4f}",
            ]
        )
        + "\n"
    )
    print(f"Saved -> {manifest.resolve()}")


if __name__ == "__main__":
    main()
