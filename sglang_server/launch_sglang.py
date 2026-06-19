from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch an SGLang OpenAI-compatible server.")
    parser.add_argument(
        "--config",
        default="sglang_server/server_config.json",
        help="Path to the SGLang server JSON config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launch command without starting the server.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if not config.get("model_path"):
        raise ValueError(f"{path} must define model_path")

    return config


def add_optional_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    command.extend([flag, str(value)])


def build_command(config: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(config["model_path"]),
        "--host",
        str(config.get("host", "0.0.0.0")),
        "--port",
        str(config.get("port", 30000)),
    ]

    add_optional_arg(command, "--served-model-name", config.get("served_model_name"))
    add_optional_arg(command, "--tp-size", config.get("tensor_parallel_size"))
    add_optional_arg(command, "--dtype", config.get("dtype"))
    add_optional_arg(command, "--context-length", config.get("context_length"))
    add_optional_arg(command, "--mem-fraction-static", config.get("mem_fraction_static"))
    add_optional_arg(command, "--dllm-algorithm", config.get("dllm_algorithm"))
    add_optional_arg(command, "--dllm-algorithm-config", config.get("dllm_algorithm_config"))

    if config.get("enable_deterministic_inference", False):
        command.append("--enable-deterministic-inference")

    if config.get("trust_remote_code", True):
        command.append("--trust-remote-code")

    command.extend(str(item) for item in config.get("extra_launch_args", []))
    return command


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    command = build_command(config)

    printable = " ".join(shlex.quote(part) for part in command)
    print(f"Config: {config_path}")
    print(f"Launch command: {printable}", flush=True)

    if args.dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
