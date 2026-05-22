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

## Workflow phases

| Phase | Script | Purpose |
|---|---|---|
| 1 | `requirements.txt` | Pin dependencies |
| 2 | `load_RouterBench.py` | Load pickle, dedupe to unique `(prompt, label)` rows |
| 3 | `encode_prompts.py` | Apply prompt encoder -> `embeddings.npy` |
| 4 | `fit_gmm.py` | GMM in full embedding space, BIC sweep |
| 5 | `pca_and_plot.py` | PCA projection + ellipses colored by majority class |

## Load prompts

The RouterBench raw pickle is ~1.1 GB. If you've already cached it for the
`RouterBench_stats` project, point this script at it directly:

```bash
python load_RouterBench.py \
    --pkl ../RouterBench_stats/data/routerbench_raw.pkl \
    --out-dir data
```

Otherwise it will download the pickle into `data/` on first run.

Output: `data/prompts.csv` with columns `sample_id, prompt, eval_name, family`.

## Encode prompts

Runs a sentence encoder over every prompt and writes an aligned embedding
matrix. The default encoder is `sentence-transformers/all-mpnet-base-v2`
(768-dim, 384-token cap); swap once the paper's prompt-encoder section is
confirmed.

```bash
python encode_prompts.py
python encode_prompts.py --model intfloat/e5-large-v2 --batch-size 16
python encode_prompts.py --limit 1000          # quick smoke test
```

Outputs (in `data/`):

- `embeddings.npy` - shape `(N, dim)`, float32
- `embeddings.ids.csv` - `sample_id, eval_name, family` aligned row-for-row
- `embeddings.manifest.txt` - encoder name, dim, normalization, etc.

## Fit GMM

Fits a Gaussian Mixture Model in the embedding space (after optional PCA) and
records the majority RouterBench label per cluster. Full-covariance GMM in
768-dim is statistically shaky with only ~36k samples, so the default reduces
to 50 PCA dims first (override with `--pca-dim 0` to cluster in the raw space).

```bash
python fit_gmm.py                          # PCA->50, sweep K, BIC pick
python fit_gmm.py --k 30                   # skip sweep, fit K=30 directly
python fit_gmm.py --covariance-type diag   # faster axis-aligned ellipses
python fit_gmm.py --label-col eval_name    # finer-grained majority class
```

Outputs (in `data/`):

- `gmm.joblib` - fitted `sklearn.mixture.GaussianMixture`
- `pca.joblib` - fitted `sklearn.decomposition.PCA` (if `--pca-dim > 0`)
- `cluster_assignments.csv` - per-prompt `cluster_id`, `majority_label`, `is_majority`
- `cluster_summary.csv` - per-cluster `size`, `majority_label`, `purity`, top-3 labels
- `bic_sweep.csv` - BIC/AIC for each K (when a sweep was run)
- `gmm.manifest.txt` - run summary (K, covariance type, weighted purity, etc.)
