from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llada_experiments import SGLangClient, load_settings

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
        "--confidence-thresholds",
        default="0.85",
        help="Comma-separated values, e.g. 0.6,0.7,0.8,0.9",
    )
    parser.add_argument(
        "--edit-thresholds",
        default="0.85",
        help="Comma-separated values, e.g. 0.6,0.7,0.8,0.9",
    )
    parser.add_argument("--confidence-key", default="confidence_threshold")
    parser.add_argument("--edit-key", default="edit_threshold")
    parser.add_argument(
        "--generation-config",
        default=None,
        help=(
            "Base JSON config. Defaults to generation_config.local.json when present, "
            "otherwise generation_config.json."
        ),
    )
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
    confidence_key: str,
    edit_key: str,
    confidence_threshold: float,
    edit_threshold: float,
    temperature: float,
    max_tokens: int,
    sleep_seconds: float,
    show_progress: bool,
) -> dict[str, Any]:
    extra_body = dict(base_extra_body)
    extra_body[confidence_key] = confidence_threshold
    extra_body[edit_key] = edit_threshold

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
            description = f"conf={confidence_threshold} edit={edit_threshold}"
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
                "confidence_threshold": confidence_threshold,
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
                    f"[conf={confidence_threshold} edit={edit_threshold}] "
                    f"{index}/{len(examples)} correct={correct_count}/{index} "
                    f"latency={latency:.2f}s pred={prediction} gold={example.gold}",
                    flush=True,
                )

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    total = len(examples)
    summary = {
        "confidence_threshold": confidence_threshold,
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
        "confidence_threshold",
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
    client = SGLangClient(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout_seconds=settings.timeout_seconds,
    )

    examples = load_examples(args)
    if not examples:
        raise RuntimeError("No GSM8K examples loaded.")

    confidence_thresholds = parse_thresholds(args.confidence_thresholds)
    edit_thresholds = parse_thresholds(args.edit_thresholds)
    base_extra_body = load_base_extra_body(args.generation_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for confidence_threshold in confidence_thresholds:
        for edit_threshold in edit_thresholds:
            details_path = output_dir / (
                f"details_conf_{safe_name(confidence_threshold)}"
                f"_edit_{safe_name(edit_threshold)}.jsonl"
            )
            summary = evaluate_threshold_pair(
                client=client,
                model=settings.model,
                examples=examples,
                output_path=details_path,
                base_extra_body=base_extra_body,
                confidence_key=args.confidence_key,
                edit_key=args.edit_key,
                confidence_threshold=confidence_threshold,
                edit_threshold=edit_threshold,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                sleep_seconds=args.sleep_seconds,
                show_progress=not args.no_progress,
            )
            summaries.append(summary)
            write_summary(output_dir, summaries)
            progress_write(
                f"SUMMARY conf={confidence_threshold} edit={edit_threshold} "
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
