from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "sample_id",
    "block_index",
    "block_iteration",
    "block_offset",
    "event_type",
    "old_token",
    "new_token",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one sample/block from token_events.csv as a Markdown sequence "
            "of per-iteration snapshots."
        )
    )
    parser.add_argument("token_events_csv", help="Path to token_events.csv.")
    parser.add_argument("--sample-id", required=True, help="Exact sample_id to render.")
    parser.add_argument("--block-index", required=True, type=int, help="Block index to render.")
    parser.add_argument(
        "--output-md",
        default=None,
        help="Output path. Default: a sample/block-specific Markdown file next to the CSV.",
    )
    return parser.parse_args()


def visible_token(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)[1:-1]


def markdown_code(value: Any) -> str:
    rendered = visible_token(value)
    if "`" not in rendered:
        return f"`{rendered}`"
    max_ticks = max(len(match.group(0)) for match in re.finditer(r"`+", rendered))
    fence = "`" * (max_ticks + 1)
    return f"{fence} {rendered} {fence}"


def read_block_events(
    path: Path, *, sample_id: str, block_index: int
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = sorted(REQUIRED_FIELDS.difference(fieldnames))
        if missing:
            hint = (
                " This looks like token_summary.csv; use token_events.csv instead."
                if "final_token" in fieldnames and "old_token" not in fieldnames
                else ""
            )
            raise ValueError(
                f"CSV is missing required field(s): {', '.join(missing)}.{hint}"
            )
        return [
            row
            for row in reader
            if str(row["sample_id"]) == sample_id
            and int(row["block_index"]) == block_index
        ]


def initial_state(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    max_offset = max(int(row["block_offset"]) for row in rows)
    state: list[str | None] = [None] * (max_offset + 1)
    warnings: list[str] = []
    for row in rows:
        offset = int(row["block_offset"])
        if state[offset] is None:
            state[offset] = row["old_token"]

    for offset, token in enumerate(state):
        if token is None:
            state[offset] = "<unknown>"
            warnings.append(
                f"No event exists for block offset {offset}; its state is shown as <unknown>."
            )
    return [str(token) for token in state], warnings


def token_map(state: list[str], changed_offsets: set[int]) -> str:
    parts = []
    for offset, token in enumerate(state):
        slot = markdown_code(f"[{offset:02d}] {token}")
        parts.append(f"**{slot}**" if offset in changed_offsets else slot)
    return " ".join(parts)


def text_snapshot(state: list[str]) -> list[str]:
    text = "".join(state)
    fence = "````" if "```" in text else "```"
    return [f"{fence}text", text, fence]


def format_probability(value: str | None) -> str:
    if value is None or not value.strip():
        return ""
    return f"{float(value):.6f}"


def markdown_table_code(value: Any) -> str:
    return markdown_code(value).replace("|", r"\|")


def render_block(
    *, csv_path: Path, sample_id: str, block_index: int, rows: list[dict[str, str]]
) -> str:
    state, warnings = initial_state(rows)
    by_iteration: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_iteration[int(row["block_iteration"])].append(row)

    lines = [
        "# Token Event Block Evolution",
        "",
        "## Summary",
        "",
        f"- Source: `{csv_path}`",
        f"- Sample ID: `{sample_id}`",
        f"- Block index: `{block_index}`",
        f"- Block width: `{len(state)}` token positions",
        f"- Recorded iterations: `{len(by_iteration)}`",
        f"- Token events: `{len(rows)}`",
        "",
        "Bold token slots changed in the current iteration. Escapes such as `\\n` are "
        "shown literally in token maps; text snapshots preserve the decoded token text.",
        "",
    ]
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")

    lines.extend(
        [
            "## Initial state",
            "",
            token_map(state, set()),
            "",
            *text_snapshot(state),
            "",
            "## Iterations",
            "",
        ]
    )

    for iteration in sorted(by_iteration):
        iteration_rows = by_iteration[iteration]
        changed_offsets: set[int] = set()
        consistency_warnings: list[str] = []
        for row in iteration_rows:
            offset = int(row["block_offset"])
            if state[offset] != row["old_token"]:
                consistency_warnings.append(
                    f"offset {offset}: reconstructed {visible_token(state[offset])!r}, "
                    f"CSV old_token {visible_token(row['old_token'])!r}"
                )
            state[offset] = row["new_token"]
            if row["old_token"] != row["new_token"]:
                changed_offsets.add(offset)

        global_iterations = sorted(
            {
                row.get("global_iteration", "")
                for row in iteration_rows
                if row.get("global_iteration", "")
            },
            key=int,
        )
        global_suffix = (
            f" (global iteration {', '.join(global_iterations)})" if global_iterations else ""
        )
        lines.extend(
            [
                f"### Block iteration {iteration}{global_suffix}",
                "",
                token_map(state, changed_offsets),
                "",
                *text_snapshot(state),
                "",
                "| Offset | Event | Old token | New token | Probability | Changed |",
                "| ---: | --- | --- | --- | ---: | :---: |",
            ]
        )
        for row in iteration_rows:
            changed = row["old_token"] != row["new_token"]
            lines.append(
                "| "
                f"{int(row['block_offset'])} | {row['event_type']} | "
                f"{markdown_table_code(row['old_token'])} | "
                f"{markdown_table_code(row['new_token'])} | "
                f"{format_probability(row.get('prob'))} | {'yes' if changed else 'no'} |"
            )
        lines.append("")
        if consistency_warnings:
            lines.append(
                "> Reconstruction warning: " + "; ".join(consistency_warnings) + "."
            )
            lines.append("")

    return "\n".join(lines)


def default_output_path(csv_path: Path, sample_id: str, block_index: int) -> Path:
    safe_sample_id = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("_") or "sample"
    return csv_path.with_name(
        f"{csv_path.stem}.sample_{safe_sample_id}.block_{block_index}.md"
    )


def main() -> None:
    args = parse_args()
    csv_path = Path(args.token_events_csv)
    try:
        rows = read_block_events(
            csv_path, sample_id=str(args.sample_id), block_index=args.block_index
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    if not rows:
        raise SystemExit(
            f"No token events found for sample_id={args.sample_id!r}, "
            f"block_index={args.block_index}."
        )

    output_md = (
        Path(args.output_md)
        if args.output_md
        else default_output_path(csv_path, str(args.sample_id), args.block_index)
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        render_block(
            csv_path=csv_path,
            sample_id=str(args.sample_id),
            block_index=args.block_index,
            rows=rows,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
