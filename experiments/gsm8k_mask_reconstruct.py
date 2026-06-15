from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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


@dataclass(frozen=True)
class MaskedInput:
    text: str
    input_ids: Any
    masked_input_ids: Any
    target_mask: Any
    answer_mask: Any
    answer_token_indices: list[int]
    masked_token_indices: list[int]
    forced_final_answer_indices: list[int]
    masked_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LLaDA2.1 masked reconstruction on GSM8K gold solutions without SGLang."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Local GSM8K JSONL with question and answer fields.",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model-path", default="model/llada2.1")
    parser.add_argument("--output-dir", default="outputs/gsm8k_mask_reconstruct")
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument(
        "--no-force-final-answer-mask",
        action="store_true",
        help="Do not force-mask the final answer number in the GSM8K '#### <number>' line.",
    )
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--editing-threshold",
        type=float,
        default=None,
        help=(
            "Enable reconstruction + editing. Non-mask tokens inside the gold solution "
            "may be rewritten when the sampled token confidence is above this value. "
            "Omit for strict reconstruction."
        ),
    )
    parser.add_argument(
        "--max-edit-steps",
        type=int,
        default=16,
        help="Maximum extra refinement steps after all mask tokens are filled.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--num-to-transfer", type=int, default=1)
    parser.add_argument(
        "--attention-mode",
        choices=["full", "block-causal"],
        default="full",
        help=(
            "'full' uses the model's bidirectional mask over the whole corrupted text; "
            "'block-causal' mirrors the local generation block attention."
        ),
    )
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map. Use 'none' to call model.to(--device) instead.",
    )
    parser.add_argument("--device", default=None, help="Used only when --device-map none.")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


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


def load_jsonl_examples(path: Path, limit: int | None) -> list[Example]:
    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            answer = str(row["answer"])
            examples.append(
                Example(
                    example_id=str(row.get("id", line_number - 1)),
                    question=str(row["question"]),
                    answer=answer,
                    gold=extract_answer(answer),
                )
            )
            if limit and len(examples) >= limit:
                break
    return examples


def stable_seed(seed: int, example_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def timestamped_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    counter = 1
    while run_dir.exists():
        run_dir = base_dir / f"run_{timestamp}_{counter:02d}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def intersecting_token_indices(
    offsets: list[tuple[int, int]],
    span_start: int,
    span_end: int,
) -> list[int]:
    return [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_start != token_end and token_start < span_end and span_start < token_end
    ]


def build_masked_input(
    *,
    tokenizer: Any,
    question: str,
    answer: str,
    example_id: str,
    mask_ratio: float,
    mask_seed: int,
    force_final_answer_mask: bool,
) -> MaskedInput:
    if mask_ratio < 0 or mask_ratio > 1:
        raise ValueError("--mask-ratio must be between 0.0 and 1.0")
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer does not define mask_token_id.")

    prefix = (
        "Show concise reasoning, then end with a final line exactly like: #### <number>\n\n"
        f"Problem:\n{question}\n\nGround truth solution:\n"
    )
    text = prefix + answer
    answer_start = len(prefix)
    answer_end = len(text)

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"][0].tolist()]

    answer_token_indices = intersecting_token_indices(offsets, answer_start, answer_end)
    if not answer_token_indices:
        raise ValueError("No answer tokens found to mask.")

    num_random_masks = round(len(answer_token_indices) * mask_ratio)
    rng = random.Random(stable_seed(mask_seed, example_id))
    random_mask_indices = (
        rng.sample(answer_token_indices, k=num_random_masks) if num_random_masks else []
    )

    forced_final_answer_indices: list[int] = []
    if force_final_answer_mask:
        answer_matches = list(ANSWER_RE.finditer(answer))
        if answer_matches:
            final_span_start, final_span_end = answer_matches[-1].span(1)
            forced_final_answer_indices = intersecting_token_indices(
                offsets,
                answer_start + final_span_start,
                answer_start + final_span_end,
            )

    masked_token_indices = sorted(set(random_mask_indices) | set(forced_final_answer_indices))

    masked_input_ids = input_ids.clone()
    target_mask = input_ids.new_zeros(input_ids.shape, dtype=bool)
    answer_mask = input_ids.new_zeros(input_ids.shape, dtype=bool)
    answer_mask[0, answer_token_indices] = True
    if masked_token_indices:
        masked_input_ids[0, masked_token_indices] = tokenizer.mask_token_id
        target_mask[0, masked_token_indices] = True

    masked_text = tokenizer.decode(masked_input_ids[0], skip_special_tokens=False)
    return MaskedInput(
        text=text,
        input_ids=input_ids,
        masked_input_ids=masked_input_ids,
        target_mask=target_mask,
        answer_mask=answer_mask,
        answer_token_indices=answer_token_indices,
        masked_token_indices=masked_token_indices,
        forced_final_answer_indices=forced_final_answer_indices,
        masked_text=masked_text,
    )


def torch_dtype_from_name(dtype_name: str) -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: torch. Run this script in the server model environment "
            "or install a PyTorch build compatible with the server GPU."
        ) from exc

    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: torch/transformers. Run this script in the server model "
            "environment, or install torch, transformers, accelerate, and safetensors."
        ) from exc

    model_path = str(Path(args.model_path))
    dtype = torch_dtype_from_name(args.dtype)
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if args.device_map.lower() == "none":
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    if dtype != "auto":
        model = model.to(dtype)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def model_device(model: Any) -> Any:
    return next(model.parameters()).device


def reconstruct_masked_tokens(
    *,
    model: Any,
    masked_input_ids: Any,
    target_mask: Any,
    answer_mask: Any,
    mask_id: int,
    eos_id: int,
    block_length: int,
    threshold: float,
    editing_threshold: float | None,
    max_edit_steps: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    num_to_transfer: int,
    attention_mode: str,
) -> tuple[Any, int, list[int]]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: torch. Run this script in the server model environment."
        ) from exc

    if block_length <= 0:
        raise ValueError("--block-length must be positive")
    if num_to_transfer <= 0:
        raise ValueError("--num-to-transfer must be positive")

    device = model_device(model)
    input_ids = masked_input_ids.to(device)
    target_mask = target_mask.to(device)
    answer_mask = answer_mask.to(device)
    if attention_mode == "full":
        return reconstruct_masked_tokens_full_attention(
            model=model,
            input_ids=input_ids,
            target_mask=target_mask,
            answer_mask=answer_mask,
            mask_id=mask_id,
            threshold=threshold,
            editing_threshold=editing_threshold,
            max_edit_steps=max_edit_steps,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_to_transfer=num_to_transfer,
        )
    if attention_mode != "block-causal":
        raise ValueError(f"Unsupported attention mode: {attention_mode}")

    seq_len = input_ids.shape[1]
    num_blocks = math.ceil(seq_len / block_length)
    total_length = num_blocks * block_length

    x = torch.full((1, total_length), eos_id, dtype=torch.long, device=device)
    x[:, :seq_len] = input_ids
    full_target_mask = torch.zeros((1, total_length), dtype=torch.bool, device=device)
    full_target_mask[:, :seq_len] = target_mask
    full_answer_mask = torch.zeros((1, total_length), dtype=torch.bool, device=device)
    full_answer_mask[:, :seq_len] = answer_mask

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    attention_mask = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
    ).to(torch.bfloat16)
    position_ids = torch.arange(total_length, device=device).unsqueeze(0)

    iterations = 0
    with torch.no_grad():
        for block_index in range(num_blocks):
            block_start = block_index * block_length
            block_end = block_start + block_length
            block_target_mask = full_target_mask[:, block_start:block_end]
            if not block_target_mask.any():
                continue

            current_window_end = block_end
            cur_attn_mask = attention_mask[:, :, :current_window_end, :current_window_end]
            cur_position_ids = position_ids[:, :current_window_end]
            block_answer_mask = full_answer_mask[:, block_start:block_end]

            edit_steps = 0
            while True:
                old_block_tokens = x[:, block_start:block_end].clone()
                active_block_mask = block_target_mask & (x[:, block_start:block_end] == mask_id)
                if not active_block_mask.any():
                    edit_steps += 1
                if not active_block_mask.any() and editing_threshold is None:
                    break
                if edit_steps > max_edit_steps:
                    break

                cur_x = x[:, :current_window_end]
                outputs = model.forward(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                )
                logits = outputs.logits[:, -block_length:, :]
                sampled_tokens, sampled_probs = model._sample_with_temperature_topk_topp(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

                confidence = torch.where(active_block_mask, sampled_probs, -torch.inf)
                high_confidence = (confidence[0] > threshold) & active_block_mask[0]
                transfer_mask = torch.zeros_like(active_block_mask)
                available = int(active_block_mask.sum().item())
                if available > 0:
                    if high_confidence.sum().item() >= num_to_transfer:
                        transfer_mask[0] = high_confidence
                    else:
                        _, selected = torch.topk(
                            confidence[0], k=min(num_to_transfer, available)
                        )
                        transfer_mask[0, selected] = True

                block_tokens = x[:, block_start:block_end]
                block_tokens[transfer_mask] = sampled_tokens[transfer_mask]

                editing_transfer_mask = torch.zeros_like(active_block_mask)
                if editing_threshold is not None:
                    editable_positions = block_answer_mask & ~active_block_mask
                    editing_confidence = torch.where(editable_positions, sampled_probs, -torch.inf)
                    token_changed = sampled_tokens != old_block_tokens
                    editing_transfer_mask = (
                        (editing_confidence > editing_threshold)
                        & editable_positions
                        & token_changed
                    )
                    block_tokens[editing_transfer_mask] = sampled_tokens[editing_transfer_mask]

                x[:, block_start:block_end] = block_tokens
                iterations += 1
                if not active_block_mask.any() and not editing_transfer_mask.any():
                    break

    edited_mask = (x[:, :seq_len] != input_ids) & answer_mask & ~target_mask
    edited_indices = edited_mask[0].nonzero(as_tuple=True)[0].tolist()
    return x[:, :seq_len], iterations, [int(index) for index in edited_indices]


def reconstruct_masked_tokens_full_attention(
    *,
    model: Any,
    input_ids: Any,
    target_mask: Any,
    answer_mask: Any,
    mask_id: int,
    threshold: float,
    editing_threshold: float | None,
    max_edit_steps: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    num_to_transfer: int,
) -> tuple[Any, int, list[int]]:
    import torch

    x = input_ids.clone()
    iterations = 0
    edit_steps = 0
    with torch.no_grad():
        while True:
            old_tokens = x.clone()
            active_mask = target_mask & (x == mask_id)
            if not active_mask.any():
                edit_steps += 1
            if not active_mask.any() and editing_threshold is None:
                break
            if edit_steps > max_edit_steps:
                break

            outputs = model.forward(x)
            sampled_tokens, sampled_probs = model._sample_with_temperature_topk_topp(
                outputs.logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            confidence = torch.where(active_mask, sampled_probs, -torch.inf)
            high_confidence = (confidence[0] > threshold) & active_mask[0]
            transfer_mask = torch.zeros_like(active_mask)
            available = int(active_mask.sum().item())
            if available > 0:
                if high_confidence.sum().item() >= num_to_transfer:
                    transfer_mask[0] = high_confidence
                else:
                    _, selected = torch.topk(confidence[0], k=min(num_to_transfer, available))
                    transfer_mask[0, selected] = True

            x[transfer_mask] = sampled_tokens[transfer_mask]

            editing_transfer_mask = torch.zeros_like(active_mask)
            if editing_threshold is not None:
                editable_positions = answer_mask & ~active_mask
                editing_confidence = torch.where(editable_positions, sampled_probs, -torch.inf)
                token_changed = sampled_tokens != old_tokens
                editing_transfer_mask = (
                    (editing_confidence > editing_threshold)
                    & editable_positions
                    & token_changed
                )
                x[editing_transfer_mask] = sampled_tokens[editing_transfer_mask]

            iterations += 1
            if not active_mask.any() and not editing_transfer_mask.any():
                break

    edited_mask = (x != input_ids) & answer_mask & ~target_mask
    edited_indices = edited_mask[0].nonzero(as_tuple=True)[0].tolist()
    return x, iterations, [int(index) for index in edited_indices]


def decode_answer_slice(tokenizer: Any, token_ids: Any, answer_token_indices: list[int]) -> str:
    if not answer_token_indices:
        return ""
    start = min(answer_token_indices)
    end = max(answer_token_indices) + 1
    return tokenizer.decode(token_ids[0, start:end], skip_special_tokens=False)


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    preferred_fieldnames = [
        "mask_ratio",
        "attention_mode",
        "editing_threshold",
        "max_edit_steps",
        "num_examples",
        "num_masked_tokens",
        "num_edited_tokens",
        "mask_token_accuracy",
        "exact_reconstruction_rate",
        "final_answer_accuracy",
        "avg_latency_seconds",
        "masked_tokens_per_second",
    ]
    fieldnames = list(preferred_fieldnames)
    for summary in summaries:
        for key in summary:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    args = parse_args()
    run_dir = timestamped_run_dir(Path(args.output_dir))
    details_path = run_dir / "details.jsonl"
    pretty_path = run_dir / "details_pretty.md"
    summary_path = run_dir / "summary.json"
    summary_csv_path = run_dir / "summary.csv"

    examples = load_jsonl_examples(Path(args.input_jsonl), args.limit)
    model, tokenizer = load_model_and_tokenizer(args)
    mask_id = int(tokenizer.mask_token_id)
    eos_id = int(tokenizer.eos_token_id or tokenizer.pad_token_id)

    total_masked = 0
    total_correct_masked = 0
    total_edited_tokens = 0
    exact_count = 0
    final_answer_correct_count = 0
    total_latency = 0.0

    iterable = examples
    progress_bar = None
    if not args.no_progress and tqdm is not None:
        progress_bar = tqdm(examples, desc="mask reconstruction", unit="ex")
        iterable = progress_bar

    with details_path.open("w", encoding="utf-8") as details_handle, pretty_path.open(
        "w", encoding="utf-8"
    ) as pretty_handle:
        pretty_handle.write("# GSM8K Mask Reconstruction Details\n\n")

        for index, example in enumerate(iterable, start=1):
            masked = build_masked_input(
                tokenizer=tokenizer,
                question=example.question,
                answer=example.answer,
                example_id=example.example_id,
                mask_ratio=args.mask_ratio,
                mask_seed=args.mask_seed,
                force_final_answer_mask=not args.no_force_final_answer_mask,
            )

            started = time.perf_counter()
            reconstructed_ids, iterations, edited_token_indices = reconstruct_masked_tokens(
                model=model,
                masked_input_ids=masked.masked_input_ids,
                target_mask=masked.target_mask,
                answer_mask=masked.answer_mask,
                mask_id=mask_id,
                eos_id=eos_id,
                block_length=args.block_length,
                threshold=args.threshold,
                editing_threshold=args.editing_threshold,
                max_edit_steps=args.max_edit_steps,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_to_transfer=args.num_to_transfer,
                attention_mode=args.attention_mode,
            )
            latency = time.perf_counter() - started

            original_ids = masked.input_ids.to(reconstructed_ids.device)
            masked_positions = masked.masked_token_indices
            correct_positions = [
                token_index
                for token_index in masked_positions
                if int(reconstructed_ids[0, token_index]) == int(original_ids[0, token_index])
            ]
            masked_count = len(masked_positions)
            correct_count = len(correct_positions)
            exact = masked_count > 0 and correct_count == masked_count

            reconstructed_text = tokenizer.decode(
                reconstructed_ids[0], skip_special_tokens=False
            )
            reconstructed_answer = decode_answer_slice(
                tokenizer,
                reconstructed_ids,
                masked.answer_token_indices,
            )
            predicted_final_answer = extract_answer(reconstructed_answer)
            final_answer_correct = (
                predicted_final_answer is not None
                and example.gold is not None
                and predicted_final_answer == example.gold
            )

            total_masked += masked_count
            total_correct_masked += correct_count
            total_edited_tokens += len(edited_token_indices)
            exact_count += int(exact)
            final_answer_correct_count += int(final_answer_correct)
            total_latency += latency

            record = {
                "id": example.example_id,
                "index": index,
                "question": example.question,
                "gold_answer": example.gold,
                "gold_solution": example.answer,
                "mask_ratio": args.mask_ratio,
                "mask_seed": args.mask_seed,
                "mask_token": tokenizer.mask_token,
                "mask_token_id": mask_id,
                "masked_token_indices": masked_positions,
                "forced_final_answer_indices": masked.forced_final_answer_indices,
                "num_masked_tokens": masked_count,
                "num_correct_masked_tokens": correct_count,
                "mask_token_accuracy": correct_count / masked_count if masked_count else 0.0,
                "exact_reconstruction": exact,
                "predicted_final_answer": predicted_final_answer,
                "final_answer_correct": final_answer_correct,
                "latency_seconds": latency,
                "iterations": iterations,
                "attention_mode": args.attention_mode,
                "editing_threshold": args.editing_threshold,
                "max_edit_steps": args.max_edit_steps,
                "edited_token_indices": edited_token_indices,
                "num_edited_tokens": len(edited_token_indices),
                "original_text": masked.text,
                "masked_text": masked.masked_text,
                "reconstructed_text": reconstructed_text,
                "reconstructed_answer": reconstructed_answer,
            }
            details_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            pretty_handle.write(f"## Example {index} / id={example.example_id}\n\n")
            pretty_handle.write(f"- gold_answer: `{example.gold}`\n")
            pretty_handle.write(f"- predicted_final_answer: `{predicted_final_answer}`\n")
            pretty_handle.write(f"- final_answer_correct: `{final_answer_correct}`\n")
            pretty_handle.write(
                f"- mask_token_accuracy: `{record['mask_token_accuracy']:.4f}` "
                f"({correct_count}/{masked_count})\n\n"
            )
            pretty_handle.write(f"- num_edited_tokens: `{len(edited_token_indices)}`\n\n")
            pretty_handle.write("### Masked Text\n\n```text\n")
            pretty_handle.write(masked.masked_text)
            pretty_handle.write("\n```\n\n### Reconstructed Answer Slice\n\n```text\n")
            pretty_handle.write(reconstructed_answer)
            pretty_handle.write("\n```\n\n")

            if progress_bar is not None:
                progress_bar.set_postfix(
                    mask_acc=f"{total_correct_masked / max(total_masked, 1):.3f}",
                    final_acc=f"{final_answer_correct_count / index:.3f}",
                )

    num_examples = len(examples)
    summary = {
        "mask_ratio": args.mask_ratio,
        "mask_seed": args.mask_seed,
        "model_path": args.model_path,
        "input_jsonl": args.input_jsonl,
        "num_examples": num_examples,
        "num_masked_tokens": total_masked,
        "num_edited_tokens": total_edited_tokens,
        "mask_token_accuracy": total_correct_masked / total_masked if total_masked else 0.0,
        "exact_reconstruction_rate": exact_count / num_examples if num_examples else 0.0,
        "final_answer_accuracy": (
            final_answer_correct_count / num_examples if num_examples else 0.0
        ),
        "avg_latency_seconds": total_latency / num_examples if num_examples else 0.0,
        "masked_tokens_per_second": total_masked / total_latency if total_latency else None,
        "block_length": args.block_length,
        "threshold": args.threshold,
        "editing_threshold": args.editing_threshold,
        "max_edit_steps": args.max_edit_steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_to_transfer": args.num_to_transfer,
        "attention_mode": args.attention_mode,
        "force_final_answer_mask": not args.no_force_final_answer_mask,
        "details_path": str(details_path),
        "pretty_path": str(pretty_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_csv(summary_csv_path, [summary])

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Results written to {run_dir}")


if __name__ == "__main__":
    main()
