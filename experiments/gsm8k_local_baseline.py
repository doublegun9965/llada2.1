from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only on minimal server envs
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from gsm8k_mask_reconstruct import (
    extract_answer,
    load_jsonl_examples,
    load_model_and_tokenizer,
    timestamped_run_dir,
)
from prompt_mask_generation import build_input_ids, with_added_masks
from trace_llada_generation import traced_generate, write_markdown_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed-threshold local LLaDA2.1 GSM8K baseline. Use this as "
            "the reference run for later dynamic-threshold experiments."
        )
    )
    parser.add_argument("--input-jsonl", required=True, help="Local GSM8K JSONL file.")
    parser.add_argument("--model-path", default="model/llada2.1")
    parser.add_argument("--output-dir", default="outputs/gsm8k_local_baseline")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--baseline-name", default="fixed_threshold_no_edit")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--editing-threshold",
        default="off",
        help="Float edit threshold, or off/none/disable. Default: off.",
    )
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--max-post-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--num-to-transfer", type=int, default=1)
    parser.add_argument("--minimal-topk", type=int, default=1)
    parser.add_argument("--mask-count", type=int, default=0)
    parser.add_argument("--mask-position", choices=["head", "tail"], default="tail")
    parser.add_argument("--mask-separator", default=" ")
    parser.add_argument("--trace-limit", type=int, default=0)
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=None,
        help=(
            "Write decoded snapshots every N trace iterations. Default: 1 when "
            "--trace-limit is enabled, otherwise 0."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-eos-early-stop", action="store_true")
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Tokenize prompts directly instead of using tokenizer.apply_chat_template.",
    )
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map. Use 'none' to call model.to(--device) instead.",
    )
    parser.add_argument("--device", default=None, help="Used only when --device-map none.")
    return parser.parse_args()


def parse_editing_threshold(raw: Any) -> tuple[float, str]:
    if raw is None:
        return 1.0, "off"
    value = str(raw).strip().lower()
    if value in {"off", "none", "disable", "disabled", "null"}:
        return 1.0, "off"
    parsed = float(raw)
    return parsed, str(parsed)


def validate_args(args: argparse.Namespace) -> None:
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.gen_length <= 0:
        raise ValueError("--gen-length must be positive")
    if args.block_length <= 0:
        raise ValueError("--block-length must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.max_post_steps < 0:
        raise ValueError("--max-post-steps must be non-negative")
    if args.num_to_transfer <= 0:
        raise ValueError("--num-to-transfer must be positive")
    if args.minimal_topk <= 0:
        raise ValueError("--minimal-topk must be positive")
    if args.mask_count < 0:
        raise ValueError("--mask-count must be non-negative")
    if args.trace_limit < 0:
        raise ValueError("--trace-limit must be non-negative")
    if args.snapshot_every is not None and args.snapshot_every < 0:
        raise ValueError("--snapshot-every must be non-negative")
    parse_editing_threshold(args.editing_threshold)


def gsm8k_prompt(question: str) -> str:
    return (
        "Solve the following grade-school math problem. "
        "Show concise reasoning, then end with a final line exactly like: #### <number>\n\n"
        f"Problem:\n{question}"
    )


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "id",
        "correct",
        "gold_answer",
        "predicted_answer",
        "latency_seconds",
        "input_tokens",
        "generated_token_count",
        "trace_path",
        "trace_markdown_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)
    effective_editing_threshold, editing_threshold_label = parse_editing_threshold(
        args.editing_threshold
    )
    generate_args = argparse.Namespace(**vars(args))
    generate_args.editing_threshold = effective_editing_threshold
    if generate_args.snapshot_every is None:
        generate_args.snapshot_every = 1 if args.trace_limit else 0

    run_dir = timestamped_run_dir(Path(args.output_dir))
    details_path = run_dir / "details.jsonl"
    summary_path = run_dir / "summary.json"
    summary_csv_path = run_dir / "summary.csv"
    trace_dir = run_dir / "traces"
    if args.trace_limit:
        trace_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args)
    if tokenizer.mask_token is None or tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer does not define a mask token.")

    examples = load_jsonl_examples(Path(args.input_jsonl), args.limit)
    if not examples:
        raise RuntimeError("No GSM8K examples loaded.")

    correct_count = 0
    total_latency = 0.0
    total_generated_tokens = 0
    rows: list[dict[str, Any]] = []

    iterable = examples
    progress_bar = None
    if not args.no_progress and tqdm is not None:
        progress_bar = tqdm(examples, desc=f"baseline {args.baseline_name}", unit="ex")
        iterable = progress_bar

    with details_path.open("w", encoding="utf-8") as details_handle:
        for index, example in enumerate(iterable, start=1):
            prompt = gsm8k_prompt(example.question)
            masked_prompt = with_added_masks(
                prompt=prompt,
                mask_token=tokenizer.mask_token,
                mask_count=args.mask_count,
                mask_position=args.mask_position,
                separator=args.mask_separator,
            )
            input_ids = build_input_ids(
                tokenizer=tokenizer,
                prompt=masked_prompt,
                use_chat_template=not args.no_chat_template,
            )
            trace_path = trace_dir / f"example_{index:04d}.jsonl" if index <= args.trace_limit else None
            trace_markdown_path = (
                trace_dir / f"example_{index:04d}.md" if index <= args.trace_limit else None
            )
            output_ids, block_summaries, latency = traced_generate(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                args=generate_args,
                trace_path=trace_path or Path(os.devnull),
            )

            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_text_with_special = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            predicted_answer = extract_answer(generated_text)
            is_correct = (
                predicted_answer is not None
                and example.gold is not None
                and predicted_answer == example.gold
            )
            correct_count += int(is_correct)
            total_latency += latency
            generated_token_count = int(output_ids.shape[1])
            total_generated_tokens += generated_token_count

            if trace_path is not None and trace_markdown_path is not None:
                trace_summary = {
                    "model_path": args.model_path,
                    "prompt": prompt,
                    "masked_prompt": masked_prompt,
                    "use_chat_template": not args.no_chat_template,
                    "model_input_text": tokenizer.decode(input_ids[0], skip_special_tokens=False),
                    "input_tokens": int(input_ids.shape[1]),
                    "generated_token_count": generated_token_count,
                    "generated_text": generated_text,
                    "generated_text_with_special_tokens": generated_text_with_special,
                    "latency_seconds": latency,
                    "gen_length": args.gen_length,
                    "block_length": args.block_length,
                    "steps": args.steps,
                    "threshold": args.threshold,
                    "editing_threshold": editing_threshold_label,
                    "max_post_steps": args.max_post_steps,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "num_to_transfer": args.num_to_transfer,
                    "minimal_topk": args.minimal_topk,
                    "mask_count": args.mask_count,
                    "mask_position": args.mask_position,
                    "block_summaries": block_summaries,
                    "trace_path": str(trace_path),
                    "markdown_path": str(trace_markdown_path),
                }
                write_markdown_trace(
                    trace_path=trace_path,
                    markdown_path=trace_markdown_path,
                    summary=trace_summary,
                )

            record = {
                "id": example.example_id,
                "index": index,
                "baseline_name": args.baseline_name,
                "question": example.question,
                "gold_answer": example.gold,
                "gold_solution": example.answer,
                "prompt": prompt,
                "masked_prompt": masked_prompt,
                "threshold": args.threshold,
                "editing_threshold": editing_threshold_label,
                "predicted_answer": predicted_answer,
                "correct": is_correct,
                "latency_seconds": latency,
                "input_tokens": int(input_ids.shape[1]),
                "generated_token_count": generated_token_count,
                "generated_text": generated_text,
                "generated_text_with_special_tokens": generated_text_with_special,
                "block_summaries": block_summaries,
                "trace_path": str(trace_path) if trace_path is not None else None,
                "trace_markdown_path": (
                    str(trace_markdown_path) if trace_markdown_path is not None else None
                ),
            }
            details_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows.append(
                {
                    "index": index,
                    "id": example.example_id,
                    "correct": is_correct,
                    "gold_answer": example.gold,
                    "predicted_answer": predicted_answer,
                    "latency_seconds": latency,
                    "input_tokens": int(input_ids.shape[1]),
                    "generated_token_count": generated_token_count,
                    "trace_path": str(trace_path) if trace_path is not None else None,
                    "trace_markdown_path": (
                        str(trace_markdown_path) if trace_markdown_path is not None else None
                    ),
                }
            )

            if progress_bar is not None:
                progress_bar.set_postfix(
                    acc=f"{correct_count / index:.3f}",
                    pred=predicted_answer,
                    gold=example.gold,
                )
            else:
                print(
                    f"[{index}/{len(examples)}] correct={correct_count}/{index} "
                    f"pred={predicted_answer} gold={example.gold} latency={latency:.2f}s",
                    flush=True,
                )

    if progress_bar is not None:
        progress_bar.close()

    total = len(examples)
    summary = {
        "experiment": "gsm8k_local_baseline",
        "baseline_name": args.baseline_name,
        "input_jsonl": args.input_jsonl,
        "model_path": args.model_path,
        "num_examples": total,
        "correct": correct_count,
        "accuracy": correct_count / total if total else 0.0,
        "total_latency_seconds": total_latency,
        "avg_latency_seconds": total_latency / total if total else 0.0,
        "total_generated_tokens": total_generated_tokens,
        "generated_tokens_per_second": total_generated_tokens / total_latency
        if total_latency > 0
        else None,
        "threshold": args.threshold,
        "editing_threshold": editing_threshold_label,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "steps": args.steps,
        "max_post_steps": args.max_post_steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_to_transfer": args.num_to_transfer,
        "minimal_topk": args.minimal_topk,
        "mask_count": args.mask_count,
        "mask_position": args.mask_position,
        "details_path": str(details_path),
        "summary_csv_path": str(summary_csv_path),
        "trace_dir": str(trace_dir) if args.trace_limit else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_csv(summary_csv_path, rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Results written to {run_dir}")


if __name__ == "__main__":
    main()
