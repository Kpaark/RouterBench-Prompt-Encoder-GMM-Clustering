"""
Phase 2 - Load the RouterBench corpus.

The RouterBench raw pickle has one row per (query x model) pair (~401k rows).
For embedding/clustering we only need each *unique* prompt once, paired with
its task label.

This script:
  1. Loads (or downloads) routerbench_raw.pkl.
  2. Deduplicates rows to one entry per `sample_id` (== unique prompt).
  3. Adds a top-level benchmark `family` column (MMLU, HellaSwag, ...).
  4. Prints class distributions and a few prompt previews per family.
  5. Saves `data/prompts.csv` with columns: sample_id, prompt, eval_name, family.

Usage:
    python load_RouterBench.py --pkl ../RouterBench_stats/data/routerbench_raw.pkl
    python load_RouterBench.py
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd


def unwrap_prompt(value) -> str:
    """
    RouterBench stores `prompt` as a string-encoded Python list literal,
    e.g. the cell text is literally `"['turn 1', 'turn 2']"`. MT-Bench rows
    are multi-turn (2+ elements); the other tasks are single-turn.

    Safely parse with ast.literal_eval and join turns with a blank line so the
    downstream encoder sees readable text instead of a `repr` with quotes
    and brackets.
    """
    if isinstance(value, (list, tuple)):
        return "\n\n".join(str(v) for v in value)
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple)):
                return "\n\n".join(str(v) for v in parsed)
        return value
    return str(value)


def to_family(eval_name: str) -> str:
    name = eval_name.lower()
    if name.startswith("mmlu"):
        return "MMLU"
    if name.startswith("hellaswag"):
        return "HellaSwag"
    if name.startswith("grade-school-math") or "gsm" in name:
        return "GSM8K"
    if name.startswith("arc"):
        return "ARC-Challenge"
    if name.startswith("winogrande"):
        return "Winogrande"
    if name.startswith("mbpp"):
        return "MBPP"
    if name.startswith("mtbench"):
        return "MT-Bench"
    return "RAG"


def download_pickle(data_dir: Path) -> Path:
    """Download routerbench_raw.pkl from Hugging Face (cached after first run)."""
    from huggingface_hub import hf_hub_download

    data_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id="withmartian/routerbench",
        filename="routerbench_raw.pkl",
        repo_type="dataset",
        local_dir=str(data_dir),
    )
    return Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pkl",
        type=Path,
        default=None,
        help="Path to an existing routerbench_raw.pkl (skips download).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Directory to write prompts.csv into (default: ./data).",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=160,
        help="How many characters of each prompt to show in the preview block.",
    )
    args = parser.parse_args()

    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 160)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.pkl is not None:
        pkl_path = args.pkl.expanduser().resolve()
        if not pkl_path.exists():
            raise SystemExit(f"--pkl path does not exist: {pkl_path}")
    else:
        pkl_path = download_pickle(args.out_dir)
    print(f"File on disk: {pkl_path}")
    print(f"Size (MB)   : {pkl_path.stat().st_size / 1024**2:.1f}")

    df = pd.read_pickle(pkl_path)
    print(f"Loaded DataFrame with shape {df.shape}")
    expected_cols = {"sample_id", "prompt", "eval_name"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Pickle is missing expected columns: {missing}")

    prompts_df = (
        df[["sample_id", "prompt", "eval_name"]]
        .drop_duplicates(subset=["sample_id"])
        .reset_index(drop=True)
        .copy()
    )
    prompts_df["prompt"] = prompts_df["prompt"].map(unwrap_prompt)

    n_sample_ids = prompts_df["sample_id"].nunique()
    n_unique_prompts = prompts_df["prompt"].nunique()
    print(f"Unique sample_id     : {n_sample_ids:,}")
    print(f"Unique prompt strings: {n_unique_prompts:,}")
    if n_sample_ids != n_unique_prompts:
        print(
            f"NOTE: {n_sample_ids - n_unique_prompts} sample_ids share text "
            f"with another sample_id (expected for RouterBench)."
        )

    char_lens = prompts_df["prompt"].str.len()
    print(
        f"Prompt char length: min={char_lens.min()}, "
        f"median={int(char_lens.median())}, "
        f"mean={char_lens.mean():.1f}, "
        f"p95={int(char_lens.quantile(0.95))}, "
        f"max={char_lens.max()}"
    )

    prompts_df["family"] = prompts_df["eval_name"].map(to_family)

    print("\nPer-family counts:")
    print(prompts_df["family"].value_counts().to_frame("n_prompts"))

    print("\nTop 15 fine-grained eval_name labels:")
    print(
        prompts_df["eval_name"]
        .value_counts()
        .head(15)
        .to_frame("n_prompts")
    )
    print(
        f"\nTotal fine-grained eval_name labels: "
        f"{prompts_df['eval_name'].nunique()}"
    )

    print("\nSample prompts per family:")
    for family, grp in prompts_df.groupby("family"):
        print(f"\n--- {family} (n={len(grp):,}) ---")
        for prompt in grp["prompt"].head(2):
            preview = prompt[: args.preview_chars].replace("\n", " ")
            ellipsis = "..." if len(prompt) > args.preview_chars else ""
            print(f"  > {preview}{ellipsis}")

    out_csv = args.out_dir / "prompts.csv"
    prompts_df.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv.resolve()}")
    print(f"Rows  : {len(prompts_df):,}")
    print(f"Cols  : {list(prompts_df.columns)}")

    manifest = args.out_dir / "prompts.manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                f"source_pickle: {pkl_path}",
                f"n_prompts: {len(prompts_df)}",
                f"n_eval_names: {prompts_df['eval_name'].nunique()}",
                f"n_families: {prompts_df['family'].nunique()}",
                f"columns: {','.join(prompts_df.columns)}",
            ]
        )
        + "\n"
    )
    print(f"Saved -> {manifest.resolve()}")


if __name__ == "__main__":
    main()
