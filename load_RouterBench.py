"""
Load the RouterBench corpus, deduplicate to one row per unique prompt, and
write data/prompts.csv with columns: sample_id, prompt, eval_name, family.

Usage:
    python load_RouterBench.py --pkl ../RouterBench_stats/data/routerbench_raw.pkl
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd


def unwrap_prompt(value) -> str:
    """RouterBench stores `prompt` as a string-encoded list literal,
    e.g. "['turn 1', 'turn 2']". Parse and join multi-turn rows with a blank line."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/prompts.csv"))
    args = parser.parse_args()

    df = pd.read_pickle(args.pkl)
    prompts_df = (
        df[["sample_id", "prompt", "eval_name"]]
        .drop_duplicates(subset=["sample_id"])
        .reset_index(drop=True)
    )
    prompts_df["prompt"] = prompts_df["prompt"].map(unwrap_prompt)
    prompts_df["family"] = prompts_df["eval_name"].map(to_family)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    prompts_df.to_csv(args.out, index=False)
    print(f"Saved {len(prompts_df):,} prompts -> {args.out}")
    print(prompts_df["family"].value_counts().to_string())


if __name__ == "__main__":
    main()
