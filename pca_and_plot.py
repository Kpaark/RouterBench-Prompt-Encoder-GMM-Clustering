"""
PCA projection + ellipse plot of the fitted GMM.

Reads the artifacts produced by fit_gmm.py and draws the reference-style
figure: GMM components as ellipses in 2D PCA space, colored by their majority
RouterBench label.

Projection logic:
  * If fit_gmm.py was run with --pca-dim > 0, the GMM means/covariances live
    in that PCA-reduced space already. "PC1/PC2" in the figure then maps to
    the first two dimensions of that space, so projection is a simple slice.
  * If fit_gmm.py was run with --pca-dim 0, the GMM is in the raw embedding
    space. This script then fits a fresh 2D PCA on the embeddings and uses
    Sigma_2d = W Sigma_d W^T to project the component covariances.

Outputs:
  figures/gmm_pca_majority_class.png    main figure
  figures/bic_sweep.png                 BIC/AIC curve (if bic_sweep.csv exists)

Usage:
    python pca_and_plot.py
    python pca_and_plot.py --scatter-points --scatter-frac 0.05
    python pca_and_plot.py --n-std 2.5 --alpha 0.35
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse


def expand_covariances(gmm) -> np.ndarray:
    """Return per-component (d, d) covariance regardless of covariance_type."""
    K = gmm.n_components
    d = gmm.means_.shape[1]
    cov = gmm.covariances_
    ctype = gmm.covariance_type
    if ctype == "full":
        return cov
    if ctype == "tied":
        return np.repeat(cov[None, :, :], K, axis=0)
    if ctype == "diag":
        return np.stack([np.diag(cov[k]) for k in range(K)])
    if ctype == "spherical":
        return np.stack([np.eye(d) * cov[k] for k in range(K)])
    raise ValueError(f"Unsupported covariance_type: {ctype}")


def project_2d(means_d: np.ndarray, covs_d: np.ndarray, W: np.ndarray):
    """Apply mu_2d = W mu_d, Sigma_2d = W Sigma_d W^T. W has shape (2, d)."""
    means_2d = means_d @ W.T
    covs_2d = np.einsum("il,klm,jm->kij", W, covs_d, W)
    return means_2d, covs_2d


def ellipse_params(cov_2d: np.ndarray, n_std: float):
    """Width, height, angle (deg) for matplotlib.patches.Ellipse from 2x2 cov."""
    vals, vecs = np.linalg.eigh(cov_2d)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    width = float(2.0 * n_std * np.sqrt(max(vals[0], 0.0)))
    height = float(2.0 * n_std * np.sqrt(max(vals[1], 0.0)))
    return width, height, angle


def categorical_palette(labels: list[str]):
    """Map each distinct label to a matplotlib color."""
    uniq = sorted(set(labels))
    if len(uniq) <= 10:
        cmap = plt.get_cmap("tab10")
    elif len(uniq) <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("tab20b")
    return {lab: cmap(i % cmap.N) for i, lab in enumerate(uniq)}, uniq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmm", type=Path, default=Path("data/gmm.joblib"))
    parser.add_argument("--pca", type=Path, default=Path("data/pca.joblib"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument(
        "--summary", type=Path, default=Path("data/cluster_summary.csv")
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("data/cluster_assignments.csv"),
    )
    parser.add_argument(
        "--bic-sweep", type=Path, default=Path("data/bic_sweep.csv")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/gmm_pca_majority_class.png"),
    )
    parser.add_argument(
        "--label-by",
        default="majority",
        choices=["majority", "cluster_id"],
        help="Color clusters by majority label (default) or cluster id.",
    )
    parser.add_argument(
        "--n-std",
        type=float,
        default=2.0,
        help="Ellipse drawn at this many standard deviations (default 2 ~ 95%% mass).",
    )
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--figsize",
        default="9,7",
        help="Figure size in inches, formatted W,H (default: 9,7).",
    )
    parser.add_argument(
        "--scatter-points",
        action="store_true",
        help="Overlay sampled embeddings as a faint scatter underneath the ellipses.",
    )
    parser.add_argument(
        "--scatter-frac",
        type=float,
        default=0.05,
        help="Fraction of points to scatter when --scatter-points is set (default 0.05).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.gmm.exists():
        raise SystemExit(f"GMM not found: {args.gmm}. Run fit_gmm.py first.")
    if not args.summary.exists():
        raise SystemExit(f"Cluster summary not found: {args.summary}.")

    gmm = joblib.load(args.gmm)
    summary = pd.read_csv(args.summary)
    d_cluster = gmm.means_.shape[1]
    print(
        f"GMM: K={gmm.n_components}, covariance_type={gmm.covariance_type}, "
        f"clustering-space dim={d_cluster}"
    )
    print(f"Summary: {len(summary)} cluster rows")

    fresh_pca_2d = None
    if args.pca.exists():
        pca = joblib.load(args.pca)
        W = np.zeros((2, d_cluster), dtype=np.float64)
        W[0, 0] = 1.0
        W[1, 1] = 1.0
        ev_first2 = float(pca.explained_variance_ratio_[:2].sum())
        print(
            f"Loaded pca.joblib (pca_dim={pca.n_components_}); first 2 PCs "
            f"retain {ev_first2:.3f} of original-embedding variance."
        )
    else:
        if not args.embeddings.exists():
            raise SystemExit(
                f"No pca.joblib at {args.pca} and no embeddings at {args.embeddings}; "
                "cannot fit a 2D PCA for visualization."
            )
        from sklearn.decomposition import PCA

        print(
            f"No pca.joblib found; fitting fresh 2D PCA on {args.embeddings} ..."
        )
        X_full = np.load(args.embeddings)
        if X_full.shape[1] != d_cluster:
            raise SystemExit(
                f"Embedding dim {X_full.shape[1]} != GMM dim {d_cluster}."
            )
        fresh_pca_2d = PCA(n_components=2, random_state=args.seed)
        fresh_pca_2d.fit(X_full)
        W = fresh_pca_2d.components_
        ev_first2 = float(fresh_pca_2d.explained_variance_ratio_.sum())
        print(f"  Fresh 2D PCA retained variance = {ev_first2:.3f}")

    covs_d = expand_covariances(gmm)
    means_2d, covs_2d = project_2d(gmm.means_, covs_d, W)
    print(f"means_2d: {means_2d.shape}, covs_2d: {covs_2d.shape}")

    if args.label_by == "majority":
        per_cluster_labels = (
            summary.sort_values("cluster_id")["majority_label"]
            .fillna("(empty)")
            .astype(str)
            .tolist()
        )
    else:
        per_cluster_labels = [str(cid) for cid in summary["cluster_id"]]

    color_map, unique_labels = categorical_palette(per_cluster_labels)

    figsize = tuple(float(x) for x in args.figsize.split(","))
    fig, ax = plt.subplots(figsize=figsize)

    if args.scatter_points:
        if not args.embeddings.exists():
            raise SystemExit("--scatter-points requires data/embeddings.npy")
        if not args.assignments.exists():
            raise SystemExit("--scatter-points requires cluster_assignments.csv")
        X_full = np.load(args.embeddings)
        assignments_df = pd.read_csv(args.assignments)
        n = X_full.shape[0]
        rng = np.random.default_rng(args.seed)
        sample_n = max(1, int(n * args.scatter_frac))
        idx = rng.choice(n, size=sample_n, replace=False)
        if args.pca.exists():
            X_proj = pca.transform(X_full[idx])[:, :2]
        else:
            X_proj = fresh_pca_2d.transform(X_full[idx])  # type: ignore[union-attr]
        cluster_ids = assignments_df["cluster_id"].values[idx]
        scatter_colors = [
            color_map[per_cluster_labels[int(c)]] for c in cluster_ids
        ]
        ax.scatter(
            X_proj[:, 0],
            X_proj[:, 1],
            c=scatter_colors,
            s=4,
            alpha=0.18,
            linewidths=0,
        )

    for k in range(gmm.n_components):
        mu = means_2d[k]
        cov = covs_2d[k]
        width, height, angle = ellipse_params(cov, n_std=args.n_std)
        color = color_map[per_cluster_labels[k]]
        ax.add_patch(
            Ellipse(
                xy=mu,
                width=width,
                height=height,
                angle=angle,
                facecolor=color,
                edgecolor=color,
                alpha=args.alpha,
                linewidth=1.0,
            )
        )
        ax.plot(mu[0], mu[1], "k.", markersize=2)

    x_pad = 0.05 * float(means_2d[:, 0].max() - means_2d[:, 0].min() + 1e-9)
    y_pad = 0.05 * float(means_2d[:, 1].max() - means_2d[:, 1].min() + 1e-9)
    largest_ellipse = float(
        np.max(np.sqrt(np.maximum(np.linalg.eigvalsh(covs_2d), 0.0)))
        * 2.0
        * args.n_std
    )
    pad = max(x_pad, y_pad, 0.6 * largest_ellipse)
    ax.set_xlim(means_2d[:, 0].min() - pad, means_2d[:, 0].max() + pad)
    ax.set_ylim(means_2d[:, 1].min() - pad, means_2d[:, 1].max() + pad)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color_map[lab],
            markeredgecolor=color_map[lab],
            markersize=9,
            label=lab,
        )
        for lab in unique_labels
    ]
    ncol = 2 if len(unique_labels) > 6 else 1
    ax.legend(handles=handles, fontsize=9, loc="best", ncol=ncol, framealpha=0.9)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    title_suffix = (
        "Majority Class Label"
        if args.label_by == "majority"
        else "Cluster ID"
    )
    ax.set_title(
        f"GMM Components in PCA Space - Colored by {title_suffix}  "
        f"(K={gmm.n_components}, PC1+PC2 var={ev_first2:.2f})"
    )
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"\nSaved -> {args.out.resolve()}")

    if args.bic_sweep.exists():
        sweep = pd.read_csv(args.bic_sweep)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(sweep["k"], sweep["bic"], "o-", label="BIC")
        ax2.plot(sweep["k"], sweep["aic"], "s--", label="AIC")
        best_k = int(sweep.loc[sweep["bic"].idxmin(), "k"])
        ax2.axvline(best_k, color="red", linestyle=":", alpha=0.6, label=f"min BIC @ K={best_k}")
        ax2.set_xlabel("K (number of components)")
        ax2.set_ylabel("Information criterion")
        ax2.set_title("GMM K selection - BIC / AIC sweep")
        ax2.grid(True, linestyle=":", alpha=0.4)
        ax2.legend()
        plt.tight_layout()
        bic_out = args.out.parent / "bic_sweep.png"
        fig2.savefig(bic_out, dpi=args.dpi)
        plt.close(fig2)
        print(f"Saved -> {bic_out.resolve()}")


if __name__ == "__main__":
    main()
