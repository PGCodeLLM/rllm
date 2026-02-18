import json
import os
import re
import time
import uuid
import logging

import numpy as np
from datasets import Dataset, load_dataset


from r2egym.agenthub.observation import Observation as R2EGymObservation
from r2egym.agenthub.action import Action as R2EGymAction
from r2egym.agenthub.agent.commands import ParseCommandBash, Command

try:
    from mindforge_harness.agent.runtime import SyncRemoteRuntime
except ImportError as e:
    SyncRemoteRuntime = None

from rllm.environments.base.base_env import BaseEnv
from rllm.environments.swe.swe import *

cmd_parser = ParseCommandBash()


def _is_shebang_script(path: str) -> bool:
    """Return True if the file has a shebang line (e.g. #!/usr/bin/env bash)."""
    try:
        with open(path, "rb") as f:
            first = f.readline()
        return first.startswith(b"#!")
    except OSError:
        return False

def _upload_to_remote(
    runtime: SyncRemoteRuntime,
    local_path: str,
    container_dir: str,
    container_path: str,
) -> None:
    """Upload a file using put_archive if available, else _copy_to_remote."""
    runtime.put_archive(local_path, container_dir)
    # put_archive extracts to container_dir, so file is at container_dir/basename(local_path)
    # If we need a different final path (e.g. strip .py), move it
    actual_name = os.path.basename(local_path)
    if container_path != f"{container_dir.rstrip('/')}/{actual_name}":
        runtime.exec_cmd(f"mv {container_dir.rstrip('/')}/{actual_name} {container_path}")


def add_r2egym_commands(
    runtime: SyncRemoteRuntime,
    cmd_files: list[str],
    logger: logging.Logger,
) -> list:
    """
    Adds command files to the environment by parsing them,
    copying them to the remote runtime, and making them executable or sourced.

    Args:
        runtime: The sync remote runtime to add commands to.
        cmd_files: List of paths to command files.

    Returns:
        List of parsed commands (for reference / attaching to runtime.commands if needed).
    """

    cmds = []
    for cmd_file in cmd_files:
        parsed_commands = cmd_parser.parse_command_file(cmd_file)
        cmds.extend(parsed_commands)

        _, ext = os.path.splitext(cmd_file)
        cmd_name = os.path.basename(cmd_file)

        container_dir = "/usr/local/bin"
        if ext == ".py" or _is_shebang_script(cmd_file):
            if ext == ".py":
                container_cmd_name = cmd_name[:-3]
            else:
                container_cmd_name = cmd_name
            container_path = f"{container_dir}/{container_cmd_name}"
            _upload_to_remote(runtime, cmd_file, container_dir, container_path)
            runtime.exec_cmd(f"chmod +x {container_path}")

        elif ext == ".sh":
            container_cmd_name = cmd_name
            container_path = f"{container_dir}/{container_cmd_name}"
            _upload_to_remote(runtime, cmd_file, container_dir, container_path)
            runtime.exec_cmd(f"bash -c 'source {container_path}'")

        else:
            container_cmd_name = cmd_name
            container_path = f"{container_dir}/{container_cmd_name}"
            _upload_to_remote(runtime, cmd_file, container_dir, container_path)
            runtime.exec_cmd(f"chmod +x {container_path}")
            runtime.exec_cmd(f"bash -c 'source {container_path}'")

    logger.info(f"Added {len(cmds)} commands to the environment.")
    return cmds


def run_action(
    runtime: SyncRemoteRuntime,
    commands: list["Command"],
    action: "R2EGymAction",
    timeout: int,
    logger: logging.Logger | None = None,
) -> tuple[str, int, float]:
    """
    Run an R2E-style Action on the runtime: validate against allowed commands,
    convert to bash via action.to_bashcmd(), execute, and return (output, exit_code, time_taken).

    Args:
        runtime: SyncRemoteRuntime to run the command in.
        commands: List of parsed command objects (each must have .name for allowed list).
        action: Action with .function_name and .to_bashcmd() (e.g. R2EGym Action).
        timeout: Max seconds for the command.
        logger: Optional logger; uses module logger if None.

    Returns:
        (bash_output, error_code, total_time). For empty/invalid action or on exception,
        error_code is -1 and output/time are set accordingly.
    """
    assert isinstance(action, R2EGymAction), "action must be an instance of R2EGymAction"
    assert isinstance(commands[0], Command), "commands must be a list of Command"
    
    if logger is None:
        logger = logging.getLogger(__name__)

    # Empty or no function call
    if not action.function_name:
        return "", 0, 0.0

    start_time = time.time()
    try:
        action_name = getattr(action, "function_name", "")
        allowed_cmds = [x.name for x in commands]
        if action_name not in allowed_cmds:
            raise AssertionError(
                f"Invalid Action: input action must be one of allowed actions\n"
                f"Allowed actions: {allowed_cmds}\n"
                f"Input action: {action_name}\t"
            )
        bash_cmd = action.to_bashcmd()
        output_lines, inspect = runtime.exec_cmd(bash_cmd, timeout=timeout)
        bash_output = "\n".join(output_lines) if output_lines else ""
        error_code = inspect.get("ExitCode", -1)
        if error_code is None:
            error_code = -1
    except Exception as e:
        logger.error("Exception occurred: %s", e)
        bash_output = ""
        error_code = -1
    end_time = time.time()
    total_time = end_time - start_time
    return bash_output, error_code, total_time


class SWEEnvRemote(BaseEnv):
    
    def __init__(self,
        entry: dict | None = None,
        dataset: Dataset | None = None,
        idx: int | None = None,
        step_timeout: int = 90,
        reward_timeout: int = 300,
        verbose: bool = False,
        name_server_url: str = "http://10.10.110.129:9401",
        scaffold: str = "r2egym",
    ):
        if entry is not None:
            self.entry = entry.copy()
            self.dataset = None
            self.idx = None
        else:
            if dataset is None:
                dataset = load_dataset(R2E_ENV_IDS[0], split="train")
            self.dataset = dataset

            if idx is None:
                idx = np.random.randint(0, len(self.dataset))
            assert 0 <= idx < len(self.dataset), "Selected index out of range"
            self.idx = idx
            self.entry = dict(self.dataset[idx])
            
        assert scaffold in ["r2egym", "sweagent"], f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent']"

        self.step_timeout = step_timeout
        self.reward_timeout = reward_timeout
        self.total_steps = 0
        self.commands = []
        self.done = False
        self.env = None
        self.verbose = verbose
        self.scaffold = scaffold
        self.name_server_url = name_server_url

        self.logger = logging.getLogger(f"r2e-train-{uuid.uuid4().hex[:8]}")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.propagate = False
        self.logger.info(f"Initializing environment for repo: {self.entry['repo_name']}")
        self.runtime_name = f"r2e-train-{self.entry['repo_name']}-{uuid.uuid4().hex[:8]}"
        
    def reset(self) -> tuple[str, dict]:
        if self.env:
            self.env.stop()
            self.env = None

        self.env = SyncRemoteRuntime(
            instance={
                "docker_image": self.entry['docker_image'],
                "repo_name": self.entry['repo_name'],
                "expected_output_json": self.entry['expected_output_json'],
                "data_source": "swebench" if "swebench" in self.entry['docker_image'] else "r2egym",
            },
            name_server_url=self.name_server_url,
            lookup_keys=[self.entry['docker_image']],
            runtime_name=self.runtime_name,
            logger=self.logger,
        )
        self.env.start()
    
        if self.scaffold == "r2egym":
            self.commands = add_r2egym_commands(runtime=self.env, cmd_files=R2EGYM_COMMAND_FILES, logger=self.logger)
        else:
            self.commands = add_r2egym_commands(runtime=self.env, cmd_files=SWEAGENT_COMMAND_FILES, logger=self.logger)
        self.total_steps = 0

        try:
            content = self.entry["problem_statement"]
            instruction = re.search(r"\[ISSUE\](.*)\[/ISSUE\]", content, re.DOTALL).group(1)
        except Exception as e:
            instruction = self.entry["problem_statement"]

        return (
            instruction,
            {
                # 'gt_patch': gt_patch,
            },
        )
    
    def compute_final_reward(self):
        eval_info = self.env.evaluate("")
        return 1.0 if eval_info["resolved"] else 0.0

    def get_evaluation_result(self) -> dict:
        """Return the full dictionary from the runtime evaluate call."""
        return self.env.evaluate("")

    def step(self, action: str | R2EGymAction) -> tuple[str, float, bool, bool, dict]:
        if isinstance(action, str):
            action_obj: R2EGymAction = R2EGymAction.from_string(action)
        else:
            action_obj = action if isinstance(action, R2EGymAction) else R2EGymAction.from_string(action)

        if not action_obj.function_name:
            return "", 0, False, {}

        bash_output, error_code, total_time = run_action(
            runtime=self.env, 
            commands=self.commands,
            action=action_obj, 
            logger=self.logger,
            timeout=self.step_timeout
        )
        self.observation = R2EGymObservation(bash_output, error_code, action_obj)
        if "finish" in action_obj.function_name.lower() or "submit" in action_obj.function_name.lower():
            self.done = True

        self.total_steps += 1
        # Reward is 0 during the execution, will be evaluated by the remote runtime.
        return self.observation, 0.0, self.done, {"total_time": total_time}

    def close(self) -> None:
        if self.env:
            self.env.stop()
            self.env = None

    @staticmethod
    def from_dict(extra_info: dict | str) -> "SWEEnvRemote":
        """Create an environment instance from JSON configuration.

        Args:
            extra_info: Dictionary containing configuration parameters.
                       The entire dict will be used as 'entry', and any keys
                       matching __init__ parameters will be extracted and passed.

        Returns:
            Initialized SWEEnvRemote instance
        """
        import inspect

        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        # Use inner "extra_info" when present (VERL batch format: {prompt, reward_model, extra_info: <swe_record>})
        entry = extra_info.get("extra_info", extra_info)

        sig = inspect.signature(SWEEnvRemote.__init__)
        init_params = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param_name in extra_info:
                init_params[param_name] = extra_info[param_name]
        init_params["entry"] = entry
        return SWEEnvRemote(**init_params)
