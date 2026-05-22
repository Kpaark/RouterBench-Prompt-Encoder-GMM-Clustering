"""
Encode RouterBench prompts into a single embedding matrix.

Reads `data/prompts.csv` (produced by load_RouterBench.py), runs every prompt
through a sentence encoder, and writes:

  - data/embeddings.npy           shape (N, d), float32
  - data/embeddings.ids.csv       sample_id, eval_name, family  (row-aligned)
  - data/embeddings.manifest.txt  encoder name, dim, normalization, etc.

The default encoder is `sentence-transformers/all-mpnet-base-v2` (768-dim,
384-token cap). Override with `--model` once the paper's prompt-encoder
section pins down the exact checkpoint.

Usage:
    python encode_prompts.py
    python encode_prompts.py --model intfloat/e5-large-v2 --batch-size 16
    python encode_prompts.py --limit 1000           # quick smoke test
    python encode_prompts.py --device cpu           # force CPU
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("data/prompts.csv"),
        help="CSV produced by load_RouterBench.py (default: data/prompts.csv).",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-mpnet-base-v2",
        help=(
            "Sentence-transformers / HF encoder checkpoint. Swap once the "
            "paper's prompt-encoder section is confirmed."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoding batch size (default: 32).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help=(
            "Override the encoder's default max_seq_length. RouterBench p95 "
            "prompt length is ~8k chars (~2k tokens), so longer is more "
            "faithful but slower."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device. 'auto' picks cuda > mps > cpu.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip L2 normalization (default is normalized embeddings).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/embeddings.npy"),
        help="Path for the embedding matrix (default: data/embeddings.npy).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Encode only the first N prompts (useful for smoke tests).",
    )
    args = parser.parse_args()

    if not args.prompts.exists():
        raise SystemExit(
            f"Prompts file not found: {args.prompts}. "
            f"Run load_RouterBench.py first."
        )

    df = pd.read_csv(args.prompts)
    required_cols = {"sample_id", "prompt", "eval_name", "family"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Prompts CSV is missing columns: {missing}")

    if args.limit:
        df = df.head(args.limit).copy()

    n_prompts = len(df)
    print(f"Loaded {n_prompts:,} prompts from {args.prompts}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit(
            "sentence-transformers not installed. "
            "Activate the venv and run: pip install -r requirements.txt"
        ) from e

    device = pick_device(args.device)
    print(f"Device: {device}")
    print(f"Loading encoder: {args.model}")
    model = SentenceTransformer(args.model, device=device)
    if args.max_seq_length is not None:
        model.max_seq_length = args.max_seq_length

    dim = model.get_sentence_embedding_dimension()
    print(f"  max_seq_length = {model.max_seq_length}")
    print(f"  embedding dim  = {dim}")

    prompts = df["prompt"].astype(str).tolist()
    normalize = not args.no_normalize
    print(
        f"Encoding {n_prompts:,} prompts | batch_size={args.batch_size} | "
        f"normalize={normalize}"
    )
    embeddings = model.encode(
        prompts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    ).astype(np.float32)

    print(f"Embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
    if embeddings.shape[0] != n_prompts:
        raise SystemExit(
            f"Row mismatch: got {embeddings.shape[0]} embeddings for "
            f"{n_prompts} prompts."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, embeddings)
    print(f"Saved -> {args.out.resolve()}")

    ids_path = args.out.with_suffix(".ids.csv")
    df[["sample_id", "eval_name", "family"]].to_csv(ids_path, index=False)
    print(f"Saved -> {ids_path.resolve()}")

    manifest = args.out.with_suffix(".manifest.txt")
    manifest.write_text(
        "\n".join(
            [
                f"encoder: {args.model}",
                f"device: {device}",
                f"n_prompts: {n_prompts}",
                f"dim: {dim}",
                f"max_seq_length: {model.max_seq_length}",
                f"normalized: {normalize}",
                f"batch_size: {args.batch_size}",
                f"source_prompts: {args.prompts.resolve()}",
                f"embeddings_path: {args.out.resolve()}",
                f"ids_path: {ids_path.resolve()}",
            ]
        )
        + "\n"
    )
    print(f"Saved -> {manifest.resolve()}")


if __name__ == "__main__":
    main()
