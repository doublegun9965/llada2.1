from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llada_experiments import SGLangClient, load_settings
from sglang_server.launch_sglang import build_command, load_config

import httpx

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only on minimal server envs
    tqdm = None


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


@dataclass(frozen=True)
class Example:
    example_id: str
    question: str
    answer: str
    gold: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GSM8K accuracy and generation speed across LLaDA2.1 thresholds."
    )
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--input-jsonl",
        default=None,
        help="Optional local JSONL with question and answer fields. Skips Hugging Face datasets.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Number of examples to evaluate.")
    parser.add_argument(
        "--thresholds",
        "--confidence-thresholds",
        dest="thresholds",
        default="0.5",
        help="Comma-separated M2T threshold values for SGLang JointThreshold.",
    )
    parser.add_argument(
        "--edit-thresholds",
        default="0.0",
        help="Comma-separated T2T edit_threshold values, e.g. 0.0,0.2,0.4",
    )
    parser.add_argument(
        "--generation-config",
        default=None,
        help=(
            "Optional request extra_body JSON config. Do not put LLaDA2.1 thresholds here; "
            "SGLang 0.5.12.post1 reads them from --dllm-algorithm-config at server startup."
        ),
    )
    parser.add_argument(
        "--server-config",
        default=None,
        help=(
            "SGLang server JSON config. Defaults to server_config.local.json when present, "
            "otherwise server_config.json."
        ),
    )
    parser.add_argument(
        "--use-running-server",
        action="store_true",
        help="Do not launch SGLang. Evaluate the already-running server once.",
    )
    parser.add_argument("--startup-timeout-seconds", type=float, default=1200)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=60)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--gold-prefix-tokens",
        type=int,
        default=0,
        help=(
            "Prepend the first N whitespace-separated tokens from the gold answer "
            "as a correct-solution prefix. Default 0 disables this context."
        ),
    )
    parser.add_argument(
        "--gold-prefix-style",
        choices=["instructed", "direct"],
        default="instructed",
        help=(
            "Prompt style when --gold-prefix-tokens is enabled. "
            "'instructed' adds explicit continuation instructions; "
            "'direct' sends question plus the gold prefix only."
        ),
    )
    parser.add_argument(
        "--gold-noise-ratio",
        type=float,
        default=None,
        help=(
            "Directly append the full gold answer after the question, but replace this "
            "fraction of whitespace-separated gold-answer tokens with --gold-noise-token. "
            "The final answer number in the last '#### <number>' line is always masked "
            "when this option is enabled. Omit this option to disable."
        ),
    )
    parser.add_argument("--gold-noise-token", default="[MASK]")
    parser.add_argument("--gold-noise-seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/gsm8k")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of concurrent SGLang requests for each threshold pair. Default 1.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def parse_thresholds(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one threshold value is required.")
    return values


def default_generation_config_path() -> Path:
    local_path = Path("sglang_server/generation_config.local.json")
    if local_path.exists():
        return local_path
    return Path("sglang_server/generation_config.json")


def default_server_config_path() -> Path:
    local_path = Path("sglang_server/server_config.local.json")
    if local_path.exists():
        return local_path
    return Path("sglang_server/server_config.json")


def default_server_env_path() -> Path:
    return Path("sglang_server/server_env.local")


def load_server_env(path: Path | None = None) -> dict[str, str]:
    env_path = path or default_server_env_path()
    values: dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def load_base_extra_body(path: str | None) -> dict[str, Any]:
    config_path = Path(path) if path else default_generation_config_path()
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    extra_body = config.get("extra_body", {})
    if not isinstance(extra_body, dict):
        raise ValueError(f"{config_path}: extra_body must be a JSON object")
    return dict(extra_body)


def write_dllm_algorithm_config(
    path: Path,
    *,
    threshold: float,
    edit_threshold: float,
    max_post_edit_steps: int = 16,
    penalty_lambda: float = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"threshold: {threshold}\n"
        f"edit_threshold: {edit_threshold}\n"
        f"max_post_edit_steps: {max_post_edit_steps}\n"
        f"penalty_lambda: {penalty_lambda}\n"
    )
    path.write_text(content, encoding="utf-8")


def managed_base_url(server_config: dict[str, Any]) -> str:
    port = int(server_config.get("port", 30000))
    return f"http://127.0.0.1:{port}/v1"


def start_sglang_server(
    *,
    server_config: dict[str, Any],
    dllm_config_path: Path,
    log_path: Path,
) -> subprocess.Popen:
    config = dict(server_config)
    config["dllm_algorithm"] = config.get("dllm_algorithm") or "JointThreshold"
    config["dllm_algorithm_config"] = str(dllm_config_path)

    command = build_command(config)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    log_handle.write("$ " + " ".join(command) + "\n")
    env = os.environ.copy()
    server_env = load_server_env()
    if server_env:
        env.update(server_env)
        log_handle.write(
            "# Loaded env: " + ", ".join(f"{key}={value}" for key, value in sorted(server_env.items())) + "\n"
        )
    log_handle.flush()

    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
        env=env,
        text=True,
    )
    process._llada_log_handle = log_handle  # type: ignore[attr-defined]
    return process


def wait_for_server(base_url: str, process: subprocess.Popen, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    models_url = f"{base_url.rstrip('/')}/models"
    last_error = ""

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang exited early with code {process.returncode}.")

        try:
            response = httpx.get(models_url, timeout=5)
            if response.status_code < 500:
                print(f"SGLang is ready: {models_url}")
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        time.sleep(5)

    raise TimeoutError(f"SGLang did not become ready within {timeout_seconds}s. Last error: {last_error}")


def stop_sglang_server(process: subprocess.Popen, timeout_seconds: float) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)

    log_handle = getattr(process, "_llada_log_handle", None)
    if log_handle is not None:
        log_handle.close()


def normalize_number(raw: str | None) -> str | None:
    if raw is None:
        return None

    cleaned = raw.replace(",", "").strip()
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None

    if value == 0:
        return "0"

    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def extract_answer(text: str) -> str | None:
    answer_matches = ANSWER_RE.findall(text)
    if answer_matches:
        return normalize_number(answer_matches[-1])

    number_matches = NUMBER_RE.findall(text)
    if number_matches:
        return normalize_number(number_matches[-1])

    return None


def load_examples_from_jsonl(path: Path, limit: int | None) -> list[Example]:
    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            answer = str(row["answer"])
            examples.append(
                Example(
                    example_id=str(row.get("id", index)),
                    question=str(row["question"]),
                    answer=answer,
                    gold=extract_answer(answer),
                )
            )
            if limit and len(examples) >= limit:
                break
    return examples


def load_examples(args: argparse.Namespace) -> list[Example]:
    if args.input_jsonl:
        return load_examples_from_jsonl(Path(args.input_jsonl), args.limit)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required to load GSM8K from Hugging Face. "
            "Install with: pip install -e ."
        ) from exc

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    examples: list[Example] = []
    for index, row in enumerate(dataset):
        answer = str(row["answer"])
        examples.append(
            Example(
                example_id=str(row.get("id", index)),
                question=str(row["question"]),
                answer=answer,
                gold=extract_answer(answer),
            )
        )
    return examples


def gold_prefix(answer: str, token_count: int) -> str:
    if token_count <= 0:
        return ""

    pieces = re.findall(r"\S+\s*", answer)
    return "".join(pieces[:token_count]).strip()


def stable_seed(seed: int, example_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def mask_token_piece(piece: str, noise_token: str) -> str:
    match = re.match(r"(\S+)(\s*)", piece)
    if not match:
        return piece
    return noise_token + match.group(2)


def noised_gold_answer(
    answer: str,
    *,
    ratio: float,
    noise_token: str,
    seed: int,
    example_id: str,
) -> tuple[str, list[int], list[int]]:
    if ratio < 0 or ratio > 1:
        raise ValueError("--gold-noise-ratio must be between 0.0 and 1.0")

    pieces = list(re.finditer(r"\S+\s*", answer))
    if not pieces:
        return answer, [], []

    num_noised = round(len(pieces) * ratio)
    rng = random.Random(stable_seed(seed, example_id))
    indices = sorted(rng.sample(range(len(pieces)), k=num_noised)) if num_noised else []
    index_set = set(indices)

    final_answer_indices: list[int] = []
    answer_matches = list(ANSWER_RE.finditer(answer))
    if answer_matches:
        final_answer_span = answer_matches[-1].span(1)
        for index, piece_match in enumerate(pieces):
            token_start, token_end = piece_match.span()
            if token_start < final_answer_span[1] and final_answer_span[0] < token_end:
                index_set.add(index)
                final_answer_indices.append(index)

    all_indices = sorted(index_set)
    noised_pieces = [
        mask_token_piece(piece.group(0), noise_token) if index in index_set else piece.group(0)
        for index, piece in enumerate(pieces)
    ]
    return "".join(noised_pieces).strip(), all_indices, final_answer_indices


def build_prompt(
    question: str,
    answer_prefix: str = "",
    gold_prefix_style: str = "instructed",
    noisy_gold_answer: str | None = None,
) -> str:
    if noisy_gold_answer is not None:
        return f"{question}\n{noisy_gold_answer}"

    if answer_prefix:
        if gold_prefix_style == "direct":
            return f"{question}\n{answer_prefix}"

        return (
            "Solve the following grade-school math problem. "
            "You are given the beginning of a correct solution. Continue from it, "
            "then end with a final line exactly like: #### <number>\n\n"
            f"Problem:\n{question}\n\n"
            f"Beginning of a correct solution:\n{answer_prefix}\n\n"
            "Continue the solution:"
        )

    return (
        "Solve the following grade-school math problem. "
        "Show concise reasoning, then end with a final line exactly like: #### <number>\n\n"
        f"Problem:\n{question}"
    )


def completion_tokens(raw: dict[str, Any]) -> int | None:
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return None

    value = usage.get("completion_tokens")
    if isinstance(value, int):
        return value
    return None


def safe_name(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def make_run_output_dir(base_output_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_output_dir) / f"run_{timestamp}"


def progress_write(message: str, *, enabled: bool) -> None:
    if enabled and tqdm is not None:
        tqdm.write(message)
    else:
        print(message, flush=True)


def evaluate_one_example(
    *,
    client: SGLangClient,
    model: str,
    example: Example,
    index: int,
    total_examples: int,
    extra_body: dict[str, Any],
    threshold: float,
    edit_threshold: float,
    temperature: float,
    max_tokens: int,
    gold_prefix_tokens: int,
    gold_prefix_style: str,
    gold_noise_ratio: float | None,
    gold_noise_token: str,
    gold_noise_seed: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    answer_prefix = gold_prefix(example.answer, gold_prefix_tokens)
    noised_answer = None
    noise_indices: list[int] = []
    final_answer_noise_indices: list[int] = []
    if gold_noise_ratio is not None:
        noised_answer, noise_indices, final_answer_noise_indices = noised_gold_answer(
            example.answer,
            ratio=gold_noise_ratio,
            noise_token=gold_noise_token,
            seed=gold_noise_seed,
            example_id=example.example_id,
        )
    prompt = build_prompt(
        example.question,
        answer_prefix=answer_prefix,
        gold_prefix_style=gold_prefix_style,
        noisy_gold_answer=noised_answer,
    )
    started = time.perf_counter()
    result = client.chat_completion(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    latency = time.perf_counter() - started
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    prediction = extract_answer(result.text)
    is_correct = prediction is not None and example.gold is not None and prediction == example.gold
    tokens = completion_tokens(result.raw)
    return {
        "id": example.example_id,
        "index": index,
        "total_examples": total_examples,
        "threshold": threshold,
        "edit_threshold": edit_threshold,
        "question": example.question,
        "gold_answer": example.gold,
        "gold_solution": example.answer,
        "gold_prefix_tokens": gold_prefix_tokens,
        "gold_prefix_style": gold_prefix_style,
        "gold_prefix": answer_prefix,
        "gold_noise_ratio": gold_noise_ratio,
        "gold_noise_token": gold_noise_token if gold_noise_ratio is not None else None,
        "gold_noise_seed": gold_noise_seed if gold_noise_ratio is not None else None,
        "gold_noise_indices": noise_indices,
        "gold_final_answer_noise_indices": final_answer_noise_indices,
        "gold_noised_solution": noised_answer,
        "prompt": prompt,
        "predicted_answer": prediction,
        "correct": is_correct,
        "latency_seconds": latency,
        "completion_tokens": tokens,
        "completion_chars": len(result.text),
        "completion": result.text,
    }


def evaluate_threshold_pair(
    *,
    client: SGLangClient,
    model: str,
    examples: list[Example],
    output_path: Path,
    base_extra_body: dict[str, Any],
    threshold: float,
    edit_threshold: float,
    temperature: float,
    max_tokens: int,
    gold_prefix_tokens: int,
    gold_prefix_style: str,
    gold_noise_ratio: float | None,
    gold_noise_token: str,
    gold_noise_seed: int,
    batch_size: int,
    sleep_seconds: float,
    show_progress: bool,
) -> dict[str, Any]:
    extra_body = dict(base_extra_body)

    correct_count = 0
    total_latency = 0.0
    total_completion_tokens = 0
    token_count_available = 0
    total_completion_chars = 0
    results: list[dict[str, Any]] = []
    wall_started = time.perf_counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_bar = None
    if show_progress and tqdm is not None:
        description = f"threshold={threshold} edit={edit_threshold} batch={batch_size}"
        progress_bar = tqdm(total=len(examples), desc=description, unit="ex")

    try:
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [
                executor.submit(
                    evaluate_one_example,
                    client=client,
                    model=model,
                    example=example,
                    index=index,
                    total_examples=len(examples),
                    extra_body=extra_body,
                    threshold=threshold,
                    edit_threshold=edit_threshold,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    gold_prefix_tokens=gold_prefix_tokens,
                    gold_prefix_style=gold_prefix_style,
                    gold_noise_ratio=gold_noise_ratio,
                    gold_noise_token=gold_noise_token,
                    gold_noise_seed=gold_noise_seed,
                    sleep_seconds=sleep_seconds,
                )
                for index, example in enumerate(examples, start=1)
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                results.append(record)
                correct_count += int(record["correct"])
                total_latency += float(record["latency_seconds"])
                total_completion_chars += int(record["completion_chars"])

                tokens = record["completion_tokens"]
                if tokens is not None:
                    total_completion_tokens += int(tokens)
                    token_count_available += 1

                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        acc=f"{correct_count / completed:.3f}",
                        latency=f"{record['latency_seconds']:.2f}s",
                        pred=record["predicted_answer"],
                        gold=record["gold_answer"],
                    )
                elif not show_progress:
                    print(
                        f"[threshold={threshold} edit={edit_threshold}] "
                        f"{completed}/{len(examples)} correct={correct_count}/{completed} "
                        f"latency={record['latency_seconds']:.2f}s "
                        f"pred={record['predicted_answer']} gold={record['gold_answer']}",
                        flush=True,
                    )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    with output_path.open("w", encoding="utf-8") as handle:
        for record in sorted(results, key=lambda item: int(item["index"])):
            record.pop("total_examples", None)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(examples)
    wall_time = time.perf_counter() - wall_started
    summary = {
        "threshold": threshold,
        "edit_threshold": edit_threshold,
        "batch_size": batch_size,
        "gold_prefix_tokens": gold_prefix_tokens,
        "gold_prefix_style": gold_prefix_style if gold_prefix_tokens > 0 else "none",
        "gold_noise_ratio": gold_noise_ratio,
        "gold_noise_token": gold_noise_token if gold_noise_ratio is not None else None,
        "gold_noise_seed": gold_noise_seed if gold_noise_ratio is not None else None,
        "num_examples": total,
        "correct": correct_count,
        "accuracy": correct_count / total if total else 0.0,
        "total_latency_seconds": total_latency,
        "avg_latency_seconds": total_latency / total if total else 0.0,
        "wall_time_seconds": wall_time,
        "requests_per_second": total / wall_time if wall_time > 0 else None,
        "completion_tokens_available": token_count_available == total,
        "total_completion_tokens": total_completion_tokens if token_count_available else None,
        "tokens_per_second": (
            total_completion_tokens / total_latency
            if token_count_available == total and total_latency > 0
            else None
        ),
        "wall_tokens_per_second": (
            total_completion_tokens / wall_time
            if token_count_available == total and wall_time > 0
            else None
        ),
        "chars_per_second": total_completion_chars / total_latency if total_latency > 0 else None,
        "wall_chars_per_second": total_completion_chars / wall_time if wall_time > 0 else None,
        "details_path": str(output_path),
    }
    return summary


def write_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    summary_json = output_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "runs": summaries,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    summary_csv = output_dir / "summary.csv"
    fieldnames = [
        "threshold",
        "edit_threshold",
        "batch_size",
        "gold_prefix_tokens",
        "gold_prefix_style",
        "gold_noise_ratio",
        "gold_noise_token",
        "gold_noise_seed",
        "num_examples",
        "correct",
        "accuracy",
        "total_latency_seconds",
        "avg_latency_seconds",
        "wall_time_seconds",
        "requests_per_second",
        "completion_tokens_available",
        "total_completion_tokens",
        "tokens_per_second",
        "wall_tokens_per_second",
        "chars_per_second",
        "wall_chars_per_second",
        "details_path",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    args = parse_args()
    settings = load_settings()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    thresholds = parse_thresholds(args.thresholds)
    edit_thresholds = parse_thresholds(args.edit_thresholds)
    threshold_pairs = [(threshold, edit) for threshold in thresholds for edit in edit_thresholds]
    if args.gold_noise_ratio is not None and args.gold_prefix_tokens > 0:
        raise ValueError("--gold-noise-ratio and --gold-prefix-tokens are mutually exclusive.")
    if args.use_running_server and len(threshold_pairs) != 1:
        raise ValueError(
            "--use-running-server can only evaluate one threshold pair because SGLang "
            "0.5.12.post1 reads thresholds at server startup."
        )

    examples = load_examples(args)
    if not examples:
        raise RuntimeError("No GSM8K examples loaded.")

    base_extra_body = load_base_extra_body(args.generation_config)
    output_dir = make_run_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing GSM8K results to {output_dir}")

    server_config_path = Path(args.server_config) if args.server_config else default_server_config_path()
    server_config = load_config(server_config_path)
    client_base_url = settings.base_url if args.use_running_server else managed_base_url(server_config)

    summaries: list[dict[str, Any]] = []
    for threshold, edit_threshold in threshold_pairs:
        pair_name = f"threshold_{safe_name(threshold)}_edit_{safe_name(edit_threshold)}"
        details_path = output_dir / f"details_{pair_name}.jsonl"
        dllm_config_path = output_dir / "dllm_configs" / f"{pair_name}.yaml"
        server_log_path = output_dir / "server_logs" / f"{pair_name}.log"
        process = None

        if not args.use_running_server:
            write_dllm_algorithm_config(
                dllm_config_path,
                threshold=threshold,
                edit_threshold=edit_threshold,
            )
            print(f"Starting SGLang for threshold={threshold} edit_threshold={edit_threshold}")
            print(f"DLLM config: {dllm_config_path}")
            print(f"Server log: {server_log_path}")
            process = start_sglang_server(
                server_config=server_config,
                dllm_config_path=dllm_config_path,
                log_path=server_log_path,
            )
            try:
                wait_for_server(client_base_url, process, args.startup_timeout_seconds)
            except Exception as exc:
                stop_sglang_server(process, args.shutdown_timeout_seconds)
                raise RuntimeError(
                    f"SGLang failed to become ready for {pair_name}. "
                    f"Check the server log: {server_log_path}"
                ) from exc

        try:
            client = SGLangClient(
                base_url=client_base_url,
                api_key=settings.api_key,
                timeout_seconds=settings.timeout_seconds,
            )
            summary = evaluate_threshold_pair(
                client=client,
                model=settings.model,
                examples=examples,
                output_path=details_path,
                base_extra_body=base_extra_body,
                threshold=threshold,
                edit_threshold=edit_threshold,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                gold_prefix_tokens=args.gold_prefix_tokens,
                gold_prefix_style=args.gold_prefix_style,
                gold_noise_ratio=args.gold_noise_ratio,
                gold_noise_token=args.gold_noise_token,
                gold_noise_seed=args.gold_noise_seed,
                batch_size=args.batch_size,
                sleep_seconds=args.sleep_seconds,
                show_progress=not args.no_progress,
            )
        except Exception as exc:
            if process is not None:
                exit_code = process.poll()
                status = (
                    f"SGLang exited with code {exit_code}"
                    if exit_code is not None
                    else "SGLang process is still running but disconnected"
                )
                raise RuntimeError(
                    f"Request failed for {pair_name}: {status}. "
                    f"Check the server log: {server_log_path}"
                ) from exc
            raise
        finally:
            if process is not None:
                print("Stopping SGLang")
                stop_sglang_server(process, args.shutdown_timeout_seconds)

        summaries.append(summary)
        write_summary(output_dir, summaries)
        progress_write(
            f"SUMMARY threshold={threshold} edit_threshold={edit_threshold} "
            f"gold_prefix_tokens={args.gold_prefix_tokens} "
            f"gold_prefix_style={args.gold_prefix_style if args.gold_prefix_tokens > 0 else 'none'} "
            f"gold_noise_ratio={args.gold_noise_ratio} "
            f"accuracy={summary['accuracy']:.4f} "
            f"avg_latency={summary['avg_latency_seconds']:.2f}s "
            f"wall_time={summary['wall_time_seconds']:.2f}s "
            f"requests_per_second={summary['requests_per_second']} "
            f"tokens_per_second={summary['tokens_per_second']} "
            f"wall_tokens_per_second={summary['wall_tokens_per_second']} "
            f"chars_per_second={summary['chars_per_second']:.2f}",
            enabled=not args.no_progress,
        )

    write_summary(output_dir, summaries)
    print(f"Wrote summary to {output_dir / 'summary.csv'} and {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
