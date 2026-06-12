from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
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
    parser.add_argument("--output-dir", default="outputs/gsm8k")
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
    log_handle.flush()

    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
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

    return format(value.normalize(), "f").rstrip("0").rstrip(".") or "0"


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


def build_prompt(question: str) -> str:
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
    sleep_seconds: float,
    show_progress: bool,
) -> dict[str, Any]:
    extra_body = dict(base_extra_body)

    correct_count = 0
    total_latency = 0.0
    total_completion_tokens = 0
    token_count_available = 0
    total_completion_chars = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        progress_bar = None
        iterable = examples
        if show_progress and tqdm is not None:
            description = f"threshold={threshold} edit={edit_threshold}"
            progress_bar = tqdm(examples, desc=description, unit="ex")
            iterable = progress_bar

        for index, example in enumerate(iterable, start=1):
            started = time.perf_counter()
            result = client.chat_completion(
                model=model,
                prompt=build_prompt(example.question),
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            latency = time.perf_counter() - started

            prediction = extract_answer(result.text)
            is_correct = prediction is not None and example.gold is not None and prediction == example.gold
            correct_count += int(is_correct)
            total_latency += latency
            total_completion_chars += len(result.text)

            tokens = completion_tokens(result.raw)
            if tokens is not None:
                total_completion_tokens += tokens
                token_count_available += 1

            record = {
                "id": example.example_id,
                "index": index,
                "threshold": threshold,
                "edit_threshold": edit_threshold,
                "question": example.question,
                "gold_answer": example.gold,
                "predicted_answer": prediction,
                "correct": is_correct,
                "latency_seconds": latency,
                "completion_tokens": tokens,
                "completion_chars": len(result.text),
                "completion": result.text,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            if progress_bar is not None:
                progress_bar.set_postfix(
                    acc=f"{correct_count / index:.3f}",
                    latency=f"{latency:.2f}s",
                    pred=prediction,
                    gold=example.gold,
                )
            elif not show_progress:
                print(
                    f"[threshold={threshold} edit={edit_threshold}] "
                    f"{index}/{len(examples)} correct={correct_count}/{index} "
                    f"latency={latency:.2f}s pred={prediction} gold={example.gold}",
                    flush=True,
                )

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    total = len(examples)
    summary = {
        "threshold": threshold,
        "edit_threshold": edit_threshold,
        "num_examples": total,
        "correct": correct_count,
        "accuracy": correct_count / total if total else 0.0,
        "total_latency_seconds": total_latency,
        "avg_latency_seconds": total_latency / total if total else 0.0,
        "completion_tokens_available": token_count_available == total,
        "total_completion_tokens": total_completion_tokens if token_count_available else None,
        "tokens_per_second": (
            total_completion_tokens / total_latency
            if token_count_available == total and total_latency > 0
            else None
        ),
        "chars_per_second": total_completion_chars / total_latency if total_latency > 0 else None,
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
        "num_examples",
        "correct",
        "accuracy",
        "total_latency_seconds",
        "avg_latency_seconds",
        "completion_tokens_available",
        "total_completion_tokens",
        "tokens_per_second",
        "chars_per_second",
        "details_path",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    args = parse_args()
    settings = load_settings()

    thresholds = parse_thresholds(args.thresholds)
    edit_thresholds = parse_thresholds(args.edit_thresholds)
    threshold_pairs = [(threshold, edit) for threshold in thresholds for edit in edit_thresholds]
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
            f"accuracy={summary['accuracy']:.4f} "
            f"avg_latency={summary['avg_latency_seconds']:.2f}s "
            f"tokens_per_second={summary['tokens_per_second']} "
            f"chars_per_second={summary['chars_per_second']:.2f}",
            enabled=not args.no_progress,
        )

    write_summary(output_dir, summaries)
    print(f"Wrote summary to {output_dir / 'summary.csv'} and {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
