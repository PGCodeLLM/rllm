#!/usr/bin/env python3
"""Test SWEAgent with SWEEnvRemote using OpenAI client on one instance from R2E_Gym_Subset.parquet.

Loads dataset from /shared_workspace_mfs/datasets/r2e/R2E_Gym_Subset.parquet, runs one episode
with full step logging. Requires OPENAI_API_KEY and mindforge_harness (SyncRemoteRuntime).
"""

import json
import logging
import os
import sys

import pandas as pd

# Ensure rllm is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openai import BadRequestError, OpenAI

from rllm.agents.swe_agent import SWEAgent
from rllm.environments.swe.swe_remote import SWEEnvRemote


DATASET_PATH = "/shared_workspace_mfs/datasets/r2e/R2E_Gym_Subset.parquet"
MAX_STEPS = 100
LOG_LEVEL = logging.DEBUG
TRAJ_OUTPUT_PATH = os.environ.get("TRAJ_OUTPUT_PATH", "swe_trajectory.txt")


def setup_logging():
    """Configure logging to stdout with step-by-step detail."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # Reduce noise from third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def load_one_entry(parquet_path: str, row_index: int = 0) -> dict:
    """Load one row from parquet as env entry (problem_statement, repo_name, docker_image, expected_output_json)."""
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    required = {"repo_name", "docker_image", "problem_statement"}
    missing = required - set(df.columns)
    if missing:
        # Try to extract values from 'extra_info' if present
        row = df.iloc[row_index]
        if "extra_info" in row and isinstance(row["extra_info"], dict):
            entry = {
                "problem_statement": row["extra_info"].get("problem_statement", ""),
                "repo_name": str(row["extra_info"].get("repo_name", "")),
                "docker_image": str(row["extra_info"].get("docker_image", "")),
            }
            # expected_output_json from extra_info if available
            if "expected_output_json" in row["extra_info"]:
                entry["expected_output_json"] = row["extra_info"].get("expected_output_json", "")
            else:
                entry["expected_output_json"] = ""
            return entry
        else:
            raise ValueError(f"Parquet missing columns {missing} and no usable 'extra_info'. Available: {list(df.columns)}")
    row = df.iloc[row_index]
    entry = {
        "problem_statement": row["problem_statement"] if pd.notna(row["problem_statement"]) else "",
        "repo_name": str(row["repo_name"]),
        "docker_image": str(row["docker_image"]),
    }
    if "expected_output_json" in df.columns and pd.notna(row.get("expected_output_json")):
        entry["expected_output_json"] = row["expected_output_json"]
    else:
        entry["expected_output_json"] = ""
    return entry


def write_trajectory_txt(
    path: str,
    entry: dict,
    initial_observation: str,
    steps: list[dict],
    eval_result: dict,
    final_reward: float,
) -> None:
    """Write full trajectory to a human-readable txt file."""
    lines = []
    lines.append("=" * 80)
    lines.append("SWE AGENT REMOTE – FULL TRAJECTORY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("--- ENTRY ---")
    lines.append(f"repo_name: {entry.get('repo_name', '')}")
    lines.append(f"docker_image: {entry.get('docker_image', '')}")
    lines.append("")
    lines.append("--- INITIAL INSTRUCTION (problem_statement) ---")
    lines.append(initial_observation or "")
    lines.append("")
    for i, s in enumerate(steps, start=1):
        lines.append("=" * 80)
        lines.append(f"STEP {i}")
        lines.append("=" * 80)
        lines.append("")
        lines.append("--- Model response ---")
        lines.append(s.get("model_response", ""))
        lines.append("")
        lines.append("--- Action ---")
        lines.append(s.get("action", ""))
        lines.append("")
        lines.append("--- Observation ---")
        lines.append(s.get("observation", ""))
        lines.append("")
        lines.append(f"reward: {s.get('reward', 0)}  done: {s.get('done', False)}")
        if s.get("step_info"):
            lines.append(f"step_info: {s['step_info']}")
        lines.append("")
    lines.append("=" * 80)
    lines.append("EVALUATION RESULT (from runtime)")
    lines.append("=" * 80)
    lines.append(json.dumps(eval_result, indent=2))
    lines.append("")
    lines.append(f"Final reward (resolved): {final_reward}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_episode(entry: dict, agent: SWEAgent, env: SWEEnvRemote, client: OpenAI, model: str, logger: logging.Logger, traj_path: str | None = None):
    """Run one episode: reset env, then loop (model -> agent -> env.step) until done or max_steps."""
    observation, info = env.reset()
    info["max_steps"] = MAX_STEPS

    agent.reset()
    agent.update_from_env(observation, 0.0, False, info)

    logger.info("=== Episode started ===")
    logger.info("Initial instruction (first 500 chars): %s", (observation or "")[:500])

    steps: list[dict] = []
    for step in range(1, MAX_STEPS + 1):
        logger.info("--- Step %d ---", step)

        messages = agent.chat_completions
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=4096,
                temperature=0.0,
            )
        except BadRequestError as e:
            err_msg = str(e)
            if "max_tokens" in err_msg or "max_completion_tokens" in err_msg:
                logger.error("Context length exceeded (max_tokens too large): %s", err_msg)
                env.close()
                raise
            raise
        content = response.choices[0].message.content or ""
        logger.debug("Model response length: %d chars", len(content))

        action = agent.update_from_model(content)
        action_str = action.action if hasattr(action, "action") else action
        logger.info("Action (first 400 chars): %s", (action_str or "")[:400])

        next_obs, reward, done, step_info = env.step(action_str)
        next_obs_str = str(next_obs)
        steps.append({
            "model_response": content,
            "action": action_str or "",
            "observation": next_obs_str,
            "reward": reward,
            "done": done,
            "step_info": step_info,
        })
        logger.info("Observation length: %d chars; reward=%.2f; done=%s", len(next_obs_str), reward, done)
        if len(next_obs_str) <= 600:
            logger.info("Observation: %s", next_obs_str)
        else:
            logger.info("Observation (first 600 chars): %s", next_obs_str[:600])

        next_info = {**info, **step_info, "max_steps": MAX_STEPS}
        agent.update_from_env(next_obs_str, reward, done, next_info)
        info = next_info

        if done:
            logger.info("Episode finished at step %d (env reported done).", step)
            break
    else:
        logger.info("Reached max_steps=%d without done.", MAX_STEPS)

    eval_result = env.get_evaluation_result()
    logger.info("=== Episode ended === Evaluation result (full dict from runtime): %s", json.dumps(eval_result, indent=2))
    final_reward = 1.0 if eval_result.get("resolved") else 0.0
    logger.info("Final reward (resolved): %.2f", final_reward)

    out_path = traj_path or TRAJ_OUTPUT_PATH
    write_trajectory_txt(out_path, entry, observation or "", steps, eval_result, final_reward)
    logger.info("Full trajectory written to: %s", os.path.abspath(out_path))
    return final_reward


def main():
    logger = setup_logging()
    logger.info("Loading dataset: %s", DATASET_PATH)
    entry = load_one_entry(DATASET_PATH, row_index=2000)
    logger.info("Entry: repo=%s, docker_image=%s", entry["repo_name"], entry["docker_image"])

    base_url = os.environ.get("BASE_URL", "http://localhost:8000")
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    model = os.environ.get("OPENAI_MODEL", "/shared_workspace_mfs/original_models/Qwen3-Coder-30B-A3B-Instruct")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = "EMPTY"

    client = OpenAI(api_key=api_key, base_url=base_url)
    agent = SWEAgent(use_fn_calling=False, scaffold="r2egym")
    env = SWEEnvRemote(entry=entry, scaffold="r2egym", verbose=True)

    try:
        run_episode(entry, agent, env, client, model, logger)
    finally:
        env.close()
        logger.info("Environment closed.")


if __name__ == "__main__":
    main()
