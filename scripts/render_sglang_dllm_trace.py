from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render SGLang dLLM trace JSONL into a readable Markdown report."
    )
    parser.add_argument("trace_path", help="Trace JSONL file written by SGLang.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Model/tokenizer path, for example /mnt/workspace/models/inclusionAI/LLaDA2.1-mini.",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Output Markdown path. Default: trace_path with .md suffix.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoTokenizer.from_pretrained.",
    )
    return parser.parse_args()


def markdown_token(value: Any) -> str:
    rendered = json.dumps(str(value), ensure_ascii=False)
    if "`" not in rendered:
        return f"`{rendered}`"
    max_ticks = max(len(match.group(0)) for match in re.finditer(r"`+", rendered))
    fence = "`" * (max_ticks + 1)
    return f"{fence} {rendered} {fence}"


def read_events(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".json":
        loaded = json.loads(text)
        if isinstance(loaded, list):
            return loaded
        return [loaded]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def render_event(tokenizer: Any, event: dict[str, Any]) -> list[str]:
    lines = [
        (
            f"### Iteration {event.get('global_iteration')} "
            f"(batch index {event.get('batch_index')})"
        ),
        "",
        f"- Active masks: `{event.get('active_mask_count_before')}` -> `{event.get('active_mask_count_after')}`",
        f"- Post edit steps: `{event.get('post_edit_steps')}`",
        f"- Mask selection: `{event.get('mask_selected_by')}`",
        "",
    ]

    mask_fills = event.get("mask_fills") or []
    edits = event.get("edits") or []
    remasks = event.get("remasks") or []

    if mask_fills:
        lines.extend(["Mask fills:", ""])
        for item in mask_fills:
            old_token = decode_token(tokenizer, item["old_token_id"])
            new_token = decode_token(tokenizer, item["new_token_id"])
            lines.append(
                "- "
                f"block_offset={item['block_offset']} "
                f"prob={item['prob']:.6f} "
                f"{markdown_token(old_token)} -> {markdown_token(new_token)}"
            )
        lines.append("")

    if edits:
        lines.extend(["Edits:", ""])
        for item in edits:
            old_token = decode_token(tokenizer, item["old_token_id"])
            new_token = decode_token(tokenizer, item["new_token_id"])
            lines.append(
                "- "
                f"block_offset={item['block_offset']} "
                f"prob={item['prob']:.6f} "
                f"{markdown_token(old_token)} -> {markdown_token(new_token)}"
            )
        lines.append("")

    if remasks:
        lines.extend(["Remasks:", ""])
        for item in remasks:
            old_token = decode_token(tokenizer, item["old_token_id"])
            new_token = decode_token(tokenizer, item["new_token_id"])
            lines.append(
                "- "
                f"block_offset={item['block_offset']} "
                f"prob={item['prob']:.6f} "
                f"{markdown_token(old_token)} -> {markdown_token(new_token)}"
            )
        lines.append("")

    if not mask_fills and not edits and not remasks:
        lines.extend(["No token changes recorded in this iteration.", ""])

    block_token_ids = event.get("block_token_ids")
    if block_token_ids is not None:
        lines.extend(
            [
                "Block snapshot:",
                "",
                "```text",
                tokenizer.decode(block_token_ids, skip_special_tokens=False),
                "```",
                "",
            ]
        )

    return lines


def main() -> None:
    args = parse_args()

    from transformers import AutoTokenizer

    trace_path = Path(args.trace_path)
    output_md = Path(args.output_md) if args.output_md else trace_path.with_suffix(".md")
    events = read_events(trace_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )

    lines = [
        "# SGLang dLLM Trace",
        "",
        "## Summary",
        "",
        f"- Trace path: `{trace_path}`",
        f"- Model path: `{args.model_path}`",
        f"- Events: `{len(events)}`",
        "",
    ]

    if events:
        first = events[0]
        lines.extend(
            [
                f"- Run id: `{first.get('run_id')}`",
                f"- Batch size: `{first.get('batch_size')}`",
                f"- Block size: `{first.get('block_size')}`",
                f"- Threshold: `{first.get('threshold')}`",
                f"- Edit threshold: `{first.get('edit_threshold')}`",
                f"- Max post edit steps: `{first.get('max_post_edit_steps')}`",
                f"- Confidence remask threshold: `{first.get('confidence_remask_threshold')}`",
                f"- Confidence remask max count: `{first.get('confidence_remask_max_count')}`",
                "",
            ]
        )

    lines.extend(["## Iterations", ""])
    for event in events:
        lines.extend(render_event(tokenizer, event))

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
