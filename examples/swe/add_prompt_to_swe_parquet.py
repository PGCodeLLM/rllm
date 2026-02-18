#!/usr/bin/env python3
"""Add a 'prompt' column (copy of problem_statement) to SWE Bench Verified test.parquet."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Add 'prompt' column to SWE parquet (copy of problem_statement).")
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=Path("/shared_workspace_mfs/datasets/r2e/R2E_Gym_Subset.parquet"),
        help="Path to input parquet file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to output parquet (default: overwrite input)",
    )
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output or input_path

    df = pd.read_parquet(input_path)
    df["prompt"] = df["problem_statement"]
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path} with new column 'prompt' (= problem_statement).")


if __name__ == "__main__":
    main()
