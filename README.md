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
| 2 | `load_routerbench.py` | Load pickle, dedupe to unique `(prompt, label)` rows |
| 3 | `encode_prompts.py` | Apply prompt encoder -> `embeddings.npy` |
| 4 | `fit_gmm.py` | GMM in full embedding space, BIC sweep |
| 5 | `pca_and_plot.py` | PCA projection + ellipses colored by majority class |

## Load prompts

The RouterBench raw pickle is ~1.1 GB. If you've already cached it for the
`RouterBench_stats` project, point this script at it directly:

```bash
python 01_load_routerbench.py \
    --pkl ../RouterBench_stats/data/routerbench_raw.pkl \
    --out-dir data
```

Otherwise it will download the pickle into `data/` on first run.

Output: `data/prompts.csv` with columns `sample_id, prompt, eval_name, family`.
