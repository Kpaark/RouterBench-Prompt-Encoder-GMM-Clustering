# RouterBench Prompt-Encoder + GMM Clustering

Pipeline that takes every prompt in the [RouterBench](https://huggingface.co/datasets/withmartian/routerbench)
corpus, encodes it with a sentence encoder, fits a Gaussian Mixture Model in
the embedding space, and visualizes the result in 2D via PCA (colored by
majority RouterBench task label per component).

## Setup

```bash
cd routerbench_gmm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline

```bash
# Phase 2: load + dedupe prompts
python load_RouterBench.py --pkl ../RouterBench_stats/data/routerbench_raw.pkl

# Phase 3: encode (default: sentence-transformers/all-mpnet-base-v2, 768-d)
python encode_prompts.py
python encode_prompts.py --model intfloat/e5-large-v2     # swap encoder

# Phase 4: PCA -> 50, BIC sweep, fit final GMM
python fit_gmm.py
python fit_gmm.py --k 30                                  # skip sweep

# Phase 5: plot ellipses colored by majority family
python pca_and_plot.py
```

## Outputs

| File | Phase | Contents |
|---|---|---|
| `data/prompts.csv` | 2 | `sample_id, prompt, eval_name, family` (36,497 rows) |
| `data/embeddings.npy` | 3 | `(N, dim)` float32 |
| `data/embeddings.ids.csv` | 3 | `sample_id, eval_name, family` aligned with `embeddings.npy` |
| `data/pca.joblib` | 4 | fitted `sklearn.decomposition.PCA` |
| `data/gmm.joblib` | 4 | fitted `sklearn.mixture.GaussianMixture` |
| `data/cluster_summary.csv` | 4 | per-cluster size, majority label, purity |
| `data/cluster_assignments.csv` | 4 | per-prompt cluster_id and majority label |
| `data/bic_sweep.csv` | 4 | BIC/AIC for each K (when a sweep ran) |
| `figures/gmm_pca_majority_class.png` | 5 | the reference-style figure |

## Design notes

- **Encoder**: pluggable via `--model`; once the paper's prompt-encoder section is confirmed, override the default.
- **PCA before GMM**: full-covariance GMM in 768-d is statistically unstable with ~36k samples; fitting in 50-d PCA space stabilizes covariance estimation. The 2D projection in Phase 5 is then just the first two dimensions of that PCA space.
- **Covariance type**: hardcoded to `full` so component ellipses can rotate, matching the reference figure.
