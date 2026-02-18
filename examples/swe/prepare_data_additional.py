#!/usr/bin/env python3
"""Add an extra_info column to a SWE parquet for env creation in agent_ppo_trainer.

Each row gets extra_info = json.dumps({...}) so that when the batch is loaded
and env_class.from_dict(env_args[i]) is called, the env receives the SWE entry.

SWEEnvRemote only uses: repo_name, docker_image, problem_statement. Other columns
(modified_files, modified_entity_summaries, relevant_files as ndarray; num_* as
np.int64) are not needed for the env and are not JSON-serializable as-is.

Default: include only env-required fields (no serialization issues).
Use --full to include all columns (converts numpy types; list columns become lists).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Fields actually used by SWEEnvRemote (swe_remote.py)
ENV_REQUIRED_COLUMNS = ("repo_name", "docker_image", "problem_statement")


def _to_json_serializable(obj):
    """Convert numpy/pandas types to native Python for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return [_to_json_serializable(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        f = float(obj)
        return None if np.isnan(f) else f
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is pd.NA or (isinstance(obj, float) and np.isnan(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Add extra_info column (JSON dict with repo_name, docker_image, etc.) to SWE parquet."
    )
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
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Only include repo_name and docker_image in extra_info (env may need more; use full row by default)",
    )
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output or input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    df = pd.read_parquet(input_path)

    required = {"repo_name", "docker_image"}
    if not required.issubset(df.columns):
        raise ValueError(f"Parquet must have columns {required}; got {list(df.columns)}")

    # Optional: problem_statement required for env reset; skip if missing
    env_cols = [c for c in ENV_REQUIRED_COLUMNS if c in df.columns]

    def row_extra_info(row):
        if args.minimal:
            d = {"repo_name": row["repo_name"], "docker_image": row["docker_image"]}
        elif args.full:
            d = _to_json_serializable(row.to_dict())
        else:
            # Default: only env-required fields (all JSON-serializable, no ndarray/int64)
            d = {k: row[k] for k in env_cols}
        return d

    df["extra_info"] = df.apply(row_extra_info, axis=1)

    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path} with new column 'extra_info' (JSON dict).")
    if args.minimal:
        print("  extra_info: repo_name, docker_image only.")
    elif args.full:
        print("  extra_info: full row (numpy types converted).")
    else:
        print(f"  extra_info: env-required fields only: {env_cols}.")


if __name__ == "__main__":
    main()
