"""
Encode RouterBench prompts into a single embedding matrix.

Reads data/prompts.csv (from load_RouterBench.py) and writes:
  data/embeddings.npy        shape (N, dim), float32
  data/embeddings.ids.csv    sample_id, eval_name, family (row-aligned)

Usage:
    python encode_prompts.py
    python encode_prompts.py --model intfloat/e5-large-v2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=Path("data/prompts.csv"))
    parser.add_argument(
        "--model", default="sentence-transformers/all-mpnet-base-v2"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("data/embeddings.npy"))
    args = parser.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    df = pd.read_csv(args.prompts)
    print(f"Encoding {len(df):,} prompts with {args.model} on {device}")

    model = SentenceTransformer(args.model, device=device)
    embeddings = model.encode(
        df["prompt"].astype(str).tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, embeddings)
    df[["sample_id", "eval_name", "family"]].to_csv(
        args.out.with_suffix(".ids.csv"), index=False
    )
    print(f"Saved {embeddings.shape} -> {args.out}")


if __name__ == "__main__":
    main()
