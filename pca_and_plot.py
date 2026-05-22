"""
Plot fitted GMM components as ellipses in 2D PCA space, colored by majority
RouterBench family per cluster.

Reads data/{gmm.joblib, pca.joblib, cluster_summary.csv} and writes
figures/gmm_pca_majority_class.png.

Assumes fit_gmm.py was run with the default full covariance and PCA enabled,
so gmm.means_ and gmm.covariances_ already live in PCA space - the 2D
projection is just a slice of the first two dimensions.

Usage:
    python pca_and_plot.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmm", type=Path, default=Path("data/gmm.joblib"))
    parser.add_argument("--pca", type=Path, default=Path("data/pca.joblib"))
    parser.add_argument(
        "--summary", type=Path, default=Path("data/cluster_summary.csv")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/gmm_pca_majority_class.png"),
    )
    args = parser.parse_args()

    gmm = joblib.load(args.gmm)
    pca = joblib.load(args.pca)
    summary = pd.read_csv(args.summary).sort_values("cluster_id")

    means_2d = gmm.means_[:, :2]
    covs_2d = gmm.covariances_[:, :2, :2]
    labels = summary["majority_label"].fillna("(empty)").astype(str).tolist()

    cmap = plt.get_cmap("tab20" if len(set(labels)) <= 20 else "tab20b")
    color = {lab: cmap(i % cmap.N) for i, lab in enumerate(sorted(set(labels)))}

    fig, ax = plt.subplots(figsize=(9, 7))
    for k in range(gmm.n_components):
        vals, vecs = np.linalg.eigh(covs_2d[k])
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
        w, h = 2 * 2 * np.sqrt(np.maximum(vals, 0.0))
        ax.add_patch(
            Ellipse(
                xy=means_2d[k],
                width=float(w),
                height=float(h),
                angle=angle,
                facecolor=color[labels[k]],
                edgecolor=color[labels[k]],
                alpha=0.35,
                linewidth=1.0,
            )
        )
        ax.plot(*means_2d[k], "k.", markersize=2)

    pad = float(2 * 2 * np.sqrt(np.linalg.eigvalsh(covs_2d).max()))
    ax.set_xlim(means_2d[:, 0].min() - pad, means_2d[:, 0].max() + pad)
    ax.set_ylim(means_2d[:, 1].min() - pad, means_2d[:, 1].max() + pad)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=c,
            markeredgecolor=c,
            markersize=9,
            label=lab,
        )
        for lab, c in sorted(color.items())
    ]
    ax.legend(handles=handles, fontsize=9, loc="best", ncol=2, framealpha=0.9)
    ax.set(
        xlabel="PC1",
        ylabel="PC2",
        title=(
            f"GMM Components in PCA Space (K={gmm.n_components}, "
            f"PC1+PC2 var={pca.explained_variance_ratio_[:2].sum():.2f})"
        ),
    )
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
