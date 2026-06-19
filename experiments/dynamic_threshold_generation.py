from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from gsm8k_mask_reconstruct import load_model_and_tokenizer, timestamped_run_dir
from prompt_mask_generation import build_input_ids, read_prompt, with_added_masks
from trace_llada_generation import decode_tokens, first_eos_offset, token_text


@dataclass(frozen=True)
class PhaseConfig:
    name: str
    threshold: float
    editing_threshold: float | None


SCHEDULE_CONFIG_KEYS = {
    "early_ratio",
    "late_ratio",
    "early_threshold",
    "mid_threshold",
    "late_threshold",
    "early_editing_threshold",
    "mid_editing_threshold",
    "late_editing_threshold",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local LLaDA2.1 generation with dynamic threshold schedules based on "
            "how many non-prompt tokens are already generated inside the current block."
        )
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Prompt text to send to the model.")
    prompt_group.add_argument("--prompt-file", help="UTF-8 text file containing the prompt.")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Dynamic schedule JSON config. Defaults to "
            "experiments/dynamic_threshold_config.local.json when present, "
            "otherwise experiments/dynamic_threshold_config.json."
        ),
    )
    parser.add_argument("--model-path", default="model/llada2.1")
    parser.add_argument("--output-dir", default="outputs/dynamic_threshold_generation")
    parser.add_argument("--mask-count", type=int, default=0)
    parser.add_argument("--mask-position", choices=["head", "tail"], default="tail")
    parser.add_argument("--mask-separator", default=" ")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--num-to-transfer", type=int, default=1)
    parser.add_argument("--minimal-topk", type=int, default=1)
    parser.add_argument("--max-post-steps", type=int, default=16)
    parser.add_argument("--no-eos-early-stop", action="store_true")
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Tokenize the masked prompt directly instead of using tokenizer.apply_chat_template.",
    )
    parser.add_argument(
        "--early-ratio",
        type=float,
        default=0.33,
        help="Use the early phase while generated_count / generatable_count is below this value.",
    )
    parser.add_argument(
        "--late-ratio",
        type=float,
        default=0.75,
        help="Use the mid phase below this value, then the late phase.",
    )
    parser.add_argument("--early-threshold", type=float, default=0.9)
    parser.add_argument("--mid-threshold", type=float, default=0.0)
    parser.add_argument("--late-threshold", type=float, default=0.0)
    parser.add_argument(
        "--early-editing-threshold",
        default="off",
        help="Float threshold or off/none/disable. Default: off.",
    )
    parser.add_argument(
        "--mid-editing-threshold",
        default="0.9",
        help="Float threshold or off/none/disable. Default: 0.9.",
    )
    parser.add_argument(
        "--late-editing-threshold",
        default="off",
        help="Float threshold or off/none/disable. Default: off.",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=1,
        help="Write decoded generated-text snapshots every N iterations. Use 0 to disable snapshots.",
    )
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map. Use 'none' to call model.to(--device) instead.",
    )
    parser.add_argument("--device", default=None, help="Used only when --device-map none.")
    args = parser.parse_args()
    apply_config_defaults(args, sys.argv[1:])
    return args


def default_config_path() -> Path:
    local_path = Path("experiments/dynamic_threshold_config.local.json")
    if local_path.exists():
        return local_path
    return Path("experiments/dynamic_threshold_config.json")


def cli_provided(argv: list[str], arg_name: str) -> bool:
    flag = "--" + arg_name.replace("_", "-")
    return any(item == flag or item.startswith(flag + "=") for item in argv)


def load_schedule_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a JSON object")

    unknown_keys = sorted(set(config) - SCHEDULE_CONFIG_KEYS)
    if unknown_keys:
        raise ValueError(f"{path}: unsupported keys: {', '.join(unknown_keys)}")
    return config


def apply_config_defaults(args: argparse.Namespace, argv: list[str]) -> None:
    explicit_config = args.config is not None
    config_path = Path(args.config) if explicit_config else default_config_path()
    if not config_path.exists():
        if explicit_config:
            raise FileNotFoundError(f"Dynamic threshold config not found: {config_path}")
        return

    config = load_schedule_config(config_path)
    args.config = str(config_path)
    for key, value in config.items():
        if not cli_provided(argv, key):
            setattr(args, key, value)


def parse_optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    value = raw.strip().lower()
    if value in {"off", "none", "disable", "disabled", "null"}:
        return None
    return float(raw)


def validate_args(args: argparse.Namespace) -> None:
    if args.block_length <= 0:
        raise ValueError("--block-length must be positive")
    if args.gen_length <= 0:
        raise ValueError("--gen-length must be positive")
    if args.minimal_topk <= 0:
        raise ValueError("--minimal-topk must be positive")
    if args.num_to_transfer <= 0:
        raise ValueError("--num-to-transfer must be positive")
    if not (0 <= args.early_ratio <= 1):
        raise ValueError("--early-ratio must be between 0 and 1")
    if not (0 <= args.late_ratio <= 1):
        raise ValueError("--late-ratio must be between 0 and 1")
    if args.early_ratio > args.late_ratio:
        raise ValueError("--early-ratio must be <= --late-ratio")


def phase_for_ratio(ratio: float, args: argparse.Namespace) -> str:
    if ratio < args.early_ratio:
        return "early"
    if ratio < args.late_ratio:
        return "mid"
    return "late"


def phase_config(phase: str, args: argparse.Namespace) -> PhaseConfig:
    if phase == "early":
        return PhaseConfig(
            name="early",
            threshold=args.early_threshold,
            editing_threshold=parse_optional_float(args.early_editing_threshold),
        )
    if phase == "mid":
        return PhaseConfig(
            name="mid",
            threshold=args.mid_threshold,
            editing_threshold=parse_optional_float(args.mid_editing_threshold),
        )
    if phase == "late":
        return PhaseConfig(
            name="late",
            threshold=args.late_threshold,
            editing_threshold=parse_optional_float(args.late_editing_threshold),
        )
    raise ValueError(f"Unsupported phase: {phase}")


def event_records(
    *,
    tokenizer: Any,
    transfer_mask: Any,
    sampled_tokens: Any,
    sampled_probs: Any,
    old_tokens: Any,
    prompt_mask_in_block: Any,
    prompt_length: int,
    block_start_pos: int,
    event_type: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    positions = transfer_mask[0].nonzero(as_tuple=True)[0].tolist()
    for block_offset in positions:
        block_offset = int(block_offset)
        absolute_pos = block_start_pos + block_offset
        old_token_id = int(old_tokens[0, block_offset].item())
        new_token_id = int(sampled_tokens[0, block_offset].item())
        records.append(
            {
                "type": event_type,
                "absolute_pos": absolute_pos,
                "generated_pos": absolute_pos - prompt_length,
                "block_offset": block_offset,
                "in_prompt": bool(prompt_mask_in_block[block_offset].item()),
                "old_token_id": old_token_id,
                "old_token": token_text(tokenizer, old_token_id),
                "new_token_id": new_token_id,
                "new_token": token_text(tokenizer, new_token_id),
                "prob": float(sampled_probs[0, block_offset].item()),
            }
        )
    return records


def dynamic_generate(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: Any,
    args: argparse.Namespace,
    trace_path: Path,
) -> tuple[Any, list[dict[str, Any]], float]:
    import torch

    started = time.perf_counter()
    _ = min(args.steps, args.gen_length // args.minimal_topk)
    device = getattr(model, "device", next(model.parameters()).device)
    input_ids = input_ids.to(device)
    prompt_length = input_ids.shape[1]
    num_blocks = (prompt_length + args.gen_length + args.block_length - 1) // args.block_length
    total_length = num_blocks * args.block_length
    eos_id = int(tokenizer.eos_token_id or tokenizer.pad_token_id)
    mask_id = int(tokenizer.mask_token_id)

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    block_diffusion_attention_mask = (
        block_mask.repeat_interleave(args.block_length, dim=0)
        .repeat_interleave(args.block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
    ).to(torch.bfloat16)

    position_ids = torch.arange(total_length, device=device).unsqueeze(0)
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids.clone()
    prefill_blocks = prompt_length // args.block_length

    block_summaries: list[dict[str, Any]] = []
    global_iteration = 0

    with trace_path.open("w", encoding="utf-8") as trace_handle:
        with torch.no_grad():
            for num_block in range(prefill_blocks, num_blocks):
                current_window_end = (num_block + 1) * args.block_length
                cur_x = x[:, :current_window_end]
                cur_attn_mask = block_diffusion_attention_mask[
                    :, :, :current_window_end, :current_window_end
                ]
                cur_position_ids = position_ids[:, :current_window_end]
                block_start_pos = num_block * args.block_length
                post_steps = 0
                block_iteration = 0
                block_summary = {
                    "block": num_block,
                    "block_start_pos": block_start_pos,
                    "block_end_pos": current_window_end,
                    "iterations": 0,
                    "mask_fills": 0,
                    "edits": 0,
                    "phase_counts": {"early": 0, "mid": 0, "late": 0},
                }

                while True:
                    global_iteration += 1
                    block_iteration += 1
                    old_block_tokens = cur_x[:, -args.block_length :].clone()
                    active_block_mask = cur_x[:, -args.block_length :] == mask_id
                    active_mask_count_before = int(active_block_mask.sum().item())
                    if torch.any(active_block_mask) == False:
                        post_steps += 1
                    if post_steps > args.max_post_steps:
                        break

                    prompt_mask_in_block = torch.zeros(
                        args.block_length, dtype=torch.bool, device=device
                    )
                    if block_start_pos < prompt_length:
                        prompt_end_in_block = min(
                            prompt_length - block_start_pos, args.block_length
                        )
                        prompt_mask_in_block[:prompt_end_in_block] = True

                    non_prompt_positions = ~prompt_mask_in_block
                    generated_positions = (~active_block_mask[0]) & non_prompt_positions
                    active_non_prompt_masks = active_block_mask[0] & non_prompt_positions
                    generated_count = int(generated_positions.sum().item())
                    generatable_count = int(non_prompt_positions.sum().item())
                    active_non_prompt_mask_count = int(active_non_prompt_masks.sum().item())
                    generated_ratio = generated_count / generatable_count if generatable_count else 1.0
                    phase = phase_for_ratio(generated_ratio, args)
                    config = phase_config(phase, args)
                    block_summary["phase_counts"][phase] += 1

                    outputs = model.forward(
                        cur_x,
                        attention_mask=cur_attn_mask,
                        position_ids=cur_position_ids,
                        output_attentions=True,
                    )
                    active_logits = outputs.logits[:, -args.block_length :, :]
                    sampled_tokens, sampled_probs = model._sample_with_temperature_topk_topp(
                        active_logits,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                    )

                    mask_transfer_index = torch.zeros_like(sampled_tokens, dtype=torch.bool)
                    mask_selected_by = "none"
                    if active_block_mask.sum() > 0:
                        mask_confidence = torch.where(active_block_mask, sampled_probs, -torch.inf)
                        high_conf_mask = (
                            mask_confidence[0] > config.threshold
                        ) & active_block_mask[0]
                        num_high_confidence = int(high_conf_mask.sum().item())
                        if num_high_confidence >= args.num_to_transfer:
                            mask_transfer_index[0] = high_conf_mask
                            mask_selected_by = f"{phase}_threshold"
                        else:
                            num_available = int(active_block_mask.sum().item())
                            if num_available > 0:
                                _, idx = torch.topk(
                                    mask_confidence[0],
                                    k=min(args.num_to_transfer, num_available),
                                )
                                mask_transfer_index[0, idx] = True
                                mask_selected_by = f"{phase}_topk_fallback"

                    editing_transfer_index = torch.zeros_like(sampled_tokens, dtype=torch.bool)
                    if config.editing_threshold is not None:
                        non_mask_positions = ~active_block_mask
                        editable_positions = non_mask_positions & non_prompt_positions[None, :]
                        editing_confidence = torch.where(
                            editable_positions, sampled_probs, -torch.inf
                        )
                        high_conf_editing = (
                            editing_confidence[0] > config.editing_threshold
                        ) & editable_positions[0]
                        token_changed = sampled_tokens[0] != old_block_tokens[0]
                        editing_transfer_index[0] = high_conf_editing & token_changed

                    mask_events = event_records(
                        tokenizer=tokenizer,
                        transfer_mask=mask_transfer_index,
                        sampled_tokens=sampled_tokens,
                        sampled_probs=sampled_probs,
                        old_tokens=old_block_tokens,
                        prompt_mask_in_block=prompt_mask_in_block,
                        prompt_length=prompt_length,
                        block_start_pos=block_start_pos,
                        event_type="mask_fill",
                    )
                    edit_events = event_records(
                        tokenizer=tokenizer,
                        transfer_mask=editing_transfer_index,
                        sampled_tokens=sampled_tokens,
                        sampled_probs=sampled_probs,
                        old_tokens=old_block_tokens,
                        prompt_mask_in_block=prompt_mask_in_block,
                        prompt_length=prompt_length,
                        block_start_pos=block_start_pos,
                        event_type="edit",
                    )

                    final_transfer_index = mask_transfer_index | editing_transfer_index
                    if final_transfer_index.any():
                        cur_x[:, -args.block_length :][final_transfer_index] = sampled_tokens[
                            final_transfer_index
                        ]

                    active_mask_count_after = int(
                        (cur_x[:, -args.block_length :] == mask_id).sum().item()
                    )
                    block_summary["iterations"] = block_iteration
                    block_summary["mask_fills"] += len(mask_events)
                    block_summary["edits"] += len(edit_events)

                    snapshot = None
                    if args.snapshot_every > 0 and global_iteration % args.snapshot_every == 0:
                        generated_so_far = cur_x[0, prompt_length:current_window_end]
                        snapshot = decode_tokens(tokenizer, generated_so_far)

                    event = {
                        "global_iteration": global_iteration,
                        "block": num_block,
                        "block_iteration": block_iteration,
                        "phase": phase,
                        "generated_count": generated_count,
                        "generatable_count": generatable_count,
                        "generated_ratio": generated_ratio,
                        "active_non_prompt_mask_count": active_non_prompt_mask_count,
                        "threshold": config.threshold,
                        "editing_threshold": config.editing_threshold,
                        "block_start_pos": block_start_pos,
                        "block_end_pos": current_window_end,
                        "post_steps": post_steps,
                        "active_mask_count_before": active_mask_count_before,
                        "active_mask_count_after": active_mask_count_after,
                        "mask_selected_by": mask_selected_by,
                        "mask_fills": mask_events,
                        "edits": edit_events,
                        "snapshot": snapshot,
                    }
                    trace_handle.write(json.dumps(event, ensure_ascii=False) + "\n")

                    if active_block_mask.sum() == 0 and not editing_transfer_index.any():
                        break

                block_summaries.append(block_summary)
                x[:, :current_window_end] = cur_x
                if not args.no_eos_early_stop:
                    generated_part = x[0, prompt_length:current_window_end]
                    if (generated_part == mask_id).sum() == 0:
                        eos_positions = (generated_part == eos_id).nonzero(as_tuple=True)[0]
                        if len(eos_positions) > 0:
                            break

    generated_answer = x[:, : prompt_length + args.gen_length]
    first_eos = first_eos_offset(generated_answer[0][prompt_length:], eos_id, args.gen_length)
    output_ids = generated_answer[:, prompt_length : prompt_length + first_eos + 1]
    latency = time.perf_counter() - started
    return output_ids, block_summaries, latency


def write_markdown(
    *,
    trace_path: Path,
    markdown_path: Path,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Dynamic Threshold Generation",
        "",
        "## Summary",
        "",
        f"- Model: `{summary['model_path']}`",
        f"- Prompt tokens: `{summary['input_tokens']}`",
        f"- Generated tokens: `{summary['generated_token_count']}`",
        f"- Block length: `{summary['block_length']}`",
        f"- Early ratio: `{summary['early_ratio']}`",
        f"- Late ratio: `{summary['late_ratio']}`",
        f"- Schedule: `{summary['schedule']}`",
        f"- Latency seconds: `{summary['latency_seconds']:.4f}`",
        "",
        "## Prompt",
        "",
        "```text",
        summary["masked_prompt"],
        "```",
        "",
        "## Final Generated Text",
        "",
        "```text",
        summary["generated_text"],
        "```",
        "",
        "## Iterations",
        "",
    ]
    with trace_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            event = json.loads(raw_line)
            lines.extend(
                [
                    (
                        f"### Iteration {event['global_iteration']} "
                        f"(block {event['block']}, {event['phase']})"
                    ),
                    "",
                    (
                        f"- Generated in block: `{event['generated_count']}` / "
                        f"`{event['generatable_count']}` "
                        f"({event['generated_ratio']:.3f})"
                    ),
                    f"- Active non-prompt masks: `{event['active_non_prompt_mask_count']}`",
                    f"- Threshold: `{event['threshold']}`",
                    f"- Editing threshold: `{event['editing_threshold']}`",
                    f"- Active masks: `{event['active_mask_count_before']}` -> `{event['active_mask_count_after']}`",
                    f"- Mask selection: `{event['mask_selected_by']}`",
                    "",
                ]
            )
            if event["mask_fills"]:
                lines.extend(["Mask fills:", ""])
                for item in event["mask_fills"]:
                    lines.append(
                        "- "
                        f"pos={item['generated_pos']} "
                        f"abs={item['absolute_pos']} "
                        f"in_prompt={item['in_prompt']} "
                        f"prob={item['prob']:.6f} "
                        f"`{item['old_token']}` -> `{item['new_token']}`"
                    )
                lines.append("")
            if event["edits"]:
                lines.extend(["Edits:", ""])
                for item in event["edits"]:
                    lines.append(
                        "- "
                        f"pos={item['generated_pos']} "
                        f"abs={item['absolute_pos']} "
                        f"prob={item['prob']:.6f} "
                        f"`{item['old_token']}` -> `{item['new_token']}`"
                    )
                lines.append("")
            if not event["mask_fills"] and not event["edits"]:
                lines.extend(["No token changes in this iteration.", ""])
            if event.get("snapshot") is not None:
                lines.extend(["Snapshot:", "", "```text", event["snapshot"], "```", ""])

    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def schedule_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "early": {
            "threshold": args.early_threshold,
            "editing_threshold": parse_optional_float(args.early_editing_threshold),
        },
        "mid": {
            "threshold": args.mid_threshold,
            "editing_threshold": parse_optional_float(args.mid_editing_threshold),
        },
        "late": {
            "threshold": args.late_threshold,
            "editing_threshold": parse_optional_float(args.late_editing_threshold),
        },
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    run_dir = timestamped_run_dir(Path(args.output_dir))
    trace_path = run_dir / "trace.jsonl"
    summary_path = run_dir / "summary.json"
    markdown_path = run_dir / "trace.md"

    model, tokenizer = load_model_and_tokenizer(args)
    if tokenizer.mask_token is None or tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer does not define a mask token.")

    prompt = read_prompt(args)
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
    output_ids, block_summaries, latency = dynamic_generate(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        args=args,
        trace_path=trace_path,
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text_with_special = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    model_input_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    summary = {
        "model_path": args.model_path,
        "config": args.config,
        "prompt": prompt,
        "masked_prompt": masked_prompt,
        "use_chat_template": not args.no_chat_template,
        "model_input_text": model_input_text,
        "input_tokens": int(input_ids.shape[1]),
        "generated_token_count": int(output_ids.shape[1]),
        "generated_text": generated_text,
        "generated_text_with_special_tokens": generated_text_with_special,
        "latency_seconds": latency,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "steps": args.steps,
        "early_ratio": args.early_ratio,
        "late_ratio": args.late_ratio,
        "schedule": schedule_summary(args),
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
        "markdown_path": str(markdown_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(trace_path=trace_path, markdown_path=markdown_path, summary=summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Results written to {run_dir}")


if __name__ == "__main__":
    main()
