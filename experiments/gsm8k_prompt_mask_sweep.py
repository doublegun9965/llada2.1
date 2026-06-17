from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gsm8k_threshold_sweep import (
    SGLangClient,
    build_prompt,
    completion_tokens,
    default_server_config_path,
    extract_answer,
    load_base_extra_body,
    load_config,
    load_examples,
    load_settings,
    managed_base_url,
    progress_write,
    safe_name,
    start_sglang_server,
    stop_sglang_server,
    wait_for_server,
    write_dllm_algorithm_config,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only on minimal server envs
    tqdm = None


@dataclass(frozen=True)
class MaskVariant:
    name: str
    position: str
    count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GSM8K with SGLang while adding explicit LLaDA mask tokens "
            "before or after the prompt."
        )
    )
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--input-jsonl",
        default=None,
        help="Optional local JSONL with question and answer fields. Skips Hugging Face datasets.",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/gsm8k_prompt_mask")
    parser.add_argument(
        "--variants",
        default="none,head:8,head:4,tail:4,tail:8",
        help=(
            "Comma-separated mask variants. Use 'none' or '<head|tail>:<count>', "
            "for example: none,head:8,head:4,tail:4,tail:8"
        ),
    )
    parser.add_argument("--mask-token", default="<|mask|>")
    parser.add_argument(
        "--mask-separator",
        default=" ",
        help="String used between repeated mask tokens and between masks and prompt.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--edit-threshold", type=float, default=0.0)
    parser.add_argument(
        "--generation-config",
        default=None,
        help="Optional request extra_body JSON config. Thresholds stay in SGLang DLLM config.",
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
        help="Do not launch SGLang. Evaluate the already-running server.",
    )
    parser.add_argument("--startup-timeout-seconds", type=float, default=1200)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=60)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def parse_variants(raw: str) -> list[MaskVariant]:
    variants: list[MaskVariant] = []
    for item in raw.split(","):
        value = item.strip().lower()
        if not value:
            continue
        if value in {"none", "no_mask", "nomask"}:
            variants.append(MaskVariant(name="no_mask", position="none", count=0))
            continue

        if ":" not in value:
            raise ValueError(f"Invalid variant {item!r}. Expected 'none' or '<head|tail>:<count>'.")
        position, raw_count = value.split(":", 1)
        if position not in {"head", "tail"}:
            raise ValueError(f"Invalid mask position {position!r}. Expected 'head' or 'tail'.")
        count = int(raw_count)
        if count < 0:
            raise ValueError("Mask count must be non-negative.")
        name = "no_mask" if count == 0 else f"{position}_{count}"
        variants.append(MaskVariant(name=name, position=position if count else "none", count=count))

    if not variants:
        raise ValueError("At least one mask variant is required.")
    return variants


def make_run_output_dir(base_output_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_output_dir) / f"run_{timestamp}"
    counter = 1
    while run_dir.exists():
        run_dir = Path(base_output_dir) / f"run_{timestamp}_{counter:02d}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def add_prompt_masks(
    *,
    prompt: str,
    variant: MaskVariant,
    mask_token: str,
    separator: str,
) -> str:
    if variant.count == 0:
        return prompt

    masks = separator.join([mask_token] * variant.count)
    if variant.position == "head":
        return f"{masks}{separator}{prompt}"
    if variant.position == "tail":
        return f"{prompt}{separator}{masks}"
    raise ValueError(f"Unsupported mask position: {variant.position}")


def evaluate_variant(
    *,
    client: SGLangClient,
    model: str,
    examples: list[Any],
    variant: MaskVariant,
    output_path: Path,
    base_extra_body: dict[str, Any],
    mask_token: str,
    mask_separator: str,
    threshold: float,
    edit_threshold: float,
    temperature: float,
    max_tokens: int,
    sleep_seconds: float,
    show_progress: bool,
) -> dict[str, Any]:
    correct_count = 0
    total_latency = 0.0
    total_completion_tokens = 0
    token_count_available = 0
    total_completion_chars = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        iterable = examples
        progress_bar = None
        if show_progress and tqdm is not None:
            progress_bar = tqdm(examples, desc=variant.name, unit="ex")
            iterable = progress_bar

        for index, example in enumerate(iterable, start=1):
            base_prompt = build_prompt(example.question)
            prompt = add_prompt_masks(
                prompt=base_prompt,
                variant=variant,
                mask_token=mask_token,
                separator=mask_separator,
            )

            started = time.perf_counter()
            result = client.chat_completion(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=dict(base_extra_body),
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
                "variant": variant.name,
                "mask_position": variant.position,
                "mask_count": variant.count,
                "mask_token": mask_token,
                "threshold": threshold,
                "edit_threshold": edit_threshold,
                "question": example.question,
                "gold_answer": example.gold,
                "gold_solution": example.answer,
                "base_prompt": base_prompt,
                "prompt": prompt,
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
                    f"[{variant.name}] {index}/{len(examples)} "
                    f"correct={correct_count}/{index} latency={latency:.2f}s "
                    f"pred={prediction} gold={example.gold}",
                    flush=True,
                )

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    total = len(examples)
    return {
        "variant": variant.name,
        "mask_position": variant.position,
        "mask_count": variant.count,
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


def write_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "runs": summaries,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    fieldnames = [
        "variant",
        "mask_position",
        "mask_count",
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
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    args = parse_args()
    settings = load_settings()
    variants = parse_variants(args.variants)
    examples = load_examples(args)
    if not examples:
        raise RuntimeError("No GSM8K examples loaded.")

    base_extra_body = load_base_extra_body(args.generation_config)
    output_dir = make_run_output_dir(args.output_dir)
    print(f"Writing GSM8K prompt-mask results to {output_dir}")

    server_config_path = Path(args.server_config) if args.server_config else default_server_config_path()
    server_config = load_config(server_config_path)
    client_base_url = settings.base_url if args.use_running_server else managed_base_url(server_config)

    pair_name = f"threshold_{safe_name(args.threshold)}_edit_{safe_name(args.edit_threshold)}"
    dllm_config_path = output_dir / "dllm_configs" / f"{pair_name}.yaml"
    server_log_path = output_dir / "server_logs" / f"{pair_name}.log"
    process = None

    if not args.use_running_server:
        write_dllm_algorithm_config(
            dllm_config_path,
            threshold=args.threshold,
            edit_threshold=args.edit_threshold,
        )
        print(f"Starting SGLang for threshold={args.threshold} edit_threshold={args.edit_threshold}")
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
            raise RuntimeError(f"SGLang failed to become ready. Check the server log: {server_log_path}") from exc

    summaries: list[dict[str, Any]] = []
    try:
        client = SGLangClient(
            base_url=client_base_url,
            api_key=settings.api_key,
            timeout_seconds=settings.timeout_seconds,
        )
        for variant in variants:
            details_path = output_dir / f"details_{variant.name}.jsonl"
            summary = evaluate_variant(
                client=client,
                model=settings.model,
                examples=examples,
                variant=variant,
                output_path=details_path,
                base_extra_body=base_extra_body,
                mask_token=args.mask_token,
                mask_separator=args.mask_separator,
                threshold=args.threshold,
                edit_threshold=args.edit_threshold,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                sleep_seconds=args.sleep_seconds,
                show_progress=not args.no_progress,
            )
            summaries.append(summary)
            write_summary(output_dir, summaries)
            progress_write(
                f"SUMMARY variant={variant.name} "
                f"accuracy={summary['accuracy']:.4f} "
                f"avg_latency={summary['avg_latency_seconds']:.2f}s "
                f"tokens_per_second={summary['tokens_per_second']} "
                f"chars_per_second={summary['chars_per_second']:.2f}",
                enabled=not args.no_progress,
            )
    except Exception as exc:
        if process is not None:
            exit_code = process.poll()
            status = (
                f"SGLang exited with code {exit_code}"
                if exit_code is not None
                else "SGLang process is still running but disconnected"
            )
            raise RuntimeError(f"Request failed: {status}. Check the server log: {server_log_path}") from exc
        raise
    finally:
        if process is not None:
            print("Stopping SGLang")
            stop_sglang_server(process, args.shutdown_timeout_seconds)

    write_summary(output_dir, summaries)
    print(f"Wrote summary to {output_dir / 'summary.csv'} and {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
