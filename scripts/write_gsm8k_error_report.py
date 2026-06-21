from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DetailsFile:
    path: Path
    records: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a Markdown report for wrong GSM8K detail records from "
            "details_threshold_*_edit_*.jsonl files."
        )
    )
    parser.add_argument(
        "details_files",
        nargs="+",
        help="One or more details JSONL/JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gsm8k_error_report",
        help="Base output directory. A timestamped run directory is created under it.",
    )
    parser.add_argument(
        "--output-name",
        default="errors.md",
        help="Markdown filename inside the timestamped run directory.",
    )
    parser.add_argument(
        "--max-completion-chars",
        type=int,
        default=0,
        help="Truncate completion fields to this many characters. Default 0 keeps full text.",
    )
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=0,
        help="Truncate prompt fields to this many characters. Default 0 keeps full text.",
    )
    parser.add_argument(
        "--include-correct",
        action="store_true",
        help="Include correct records too. Default only writes records whose correct field is not true.",
    )
    return parser.parse_args()


def timestamped_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    counter = 1
    while run_dir.exists():
        run_dir = base_dir / f"run_{timestamp}_{counter:02d}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_details(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    if path.suffix == ".json":
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            records = load_jsonl_records(path, text)
        else:
            if isinstance(loaded, list):
                records = loaded
            elif isinstance(loaded, dict) and isinstance(loaded.get("records"), list):
                records = loaded["records"]
            elif isinstance(loaded, dict) and isinstance(loaded.get("details"), list):
                records = loaded["details"]
            elif isinstance(loaded, dict) and isinstance(loaded.get("results"), list):
                records = loaded["results"]
            elif isinstance(loaded, dict):
                records = [loaded]
            else:
                raise ValueError(f"{path}: expected JSON object or list")
    else:
        records = load_jsonl_records(path, text)

    bad_types = [index for index, record in enumerate(records, start=1) if not isinstance(record, dict)]
    if bad_types:
        raise ValueError(f"{path}: expected JSON objects at record indexes {bad_types[:5]}")
    return records


def load_jsonl_records(path: Path, text: str) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
    return records


def truncate_text(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "\n\n...[truncated]..."
    return text


def code_block(value: Any, *, max_chars: int = 0) -> list[str]:
    text = truncate_text(value, max_chars)
    fence = "```"
    if fence in text:
        fence = "````"
    return [f"{fence}text", text, fence]


def markdown_value(value: Any) -> str:
    if value is None:
        return "`null`"
    return f"`{value}`"


def is_wrong(record: dict[str, Any]) -> bool:
    return record.get("correct") is not True


def error_reason(record: dict[str, Any]) -> str:
    if record.get("correct") is True:
        return "correct"
    if record.get("gold_answer") is None:
        return "missing_gold_answer"
    if record.get("predicted_answer") is None:
        return "missing_predicted_answer"
    return "wrong_predicted_answer"


def threshold_label(record: dict[str, Any]) -> str:
    threshold = record.get("threshold")
    edit_threshold = record.get("edit_threshold")
    if threshold is None and edit_threshold is None:
        return "unknown"
    return f"threshold={threshold}, edit={edit_threshold}"


def write_report(
    *,
    output_path: Path,
    details_files: list[DetailsFile],
    include_correct: bool,
    max_completion_chars: int,
    max_prompt_chars: int,
) -> None:
    total_records = sum(len(item.records) for item in details_files)
    selected: list[tuple[Path, int, dict[str, Any]]] = []
    for details_file in details_files:
        for row_index, record in enumerate(details_file.records, start=1):
            if include_correct or is_wrong(record):
                selected.append((details_file.path, row_index, record))

    lines: list[str] = [
        "# GSM8K Error Report",
        "",
        f"- Created at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Source files: {len(details_files)}",
        f"- Total records: {total_records}",
        f"- Records shown: {len(selected)}",
        f"- Mode: `{'all records' if include_correct else 'wrong records only'}`",
        "",
        "## Sources",
        "",
    ]

    for details_file in details_files:
        wrong_count = sum(1 for record in details_file.records if is_wrong(record))
        lines.append(
            f"- `{details_file.path}`: {len(details_file.records)} records, {wrong_count} wrong"
        )

    lines.extend(["", "## Error Cases", ""])
    if not selected:
        lines.append("No wrong records found.")

    for case_index, (source_path, row_index, record) in enumerate(selected, start=1):
        title_id = record.get("id", row_index)
        lines.extend(
            [
                f"### {case_index}. {source_path.name} row {row_index}, id {title_id}",
                "",
                f"- Source: `{source_path}`",
                f"- Index: {markdown_value(record.get('index'))}",
                f"- Thresholds: `{threshold_label(record)}`",
                f"- Error reason: `{error_reason(record)}`",
                f"- Correct field: {markdown_value(record.get('correct'))}",
                f"- Gold answer: {markdown_value(record.get('gold_answer'))}",
                f"- Predicted answer: {markdown_value(record.get('predicted_answer'))}",
                f"- Completion tokens: {markdown_value(record.get('completion_tokens'))}",
                f"- Latency seconds: {markdown_value(record.get('latency_seconds'))}",
                "",
                "#### Question",
                "",
                *code_block(record.get("question")),
                "",
            ]
        )

        user_prompt = record.get("user_prompt", record.get("prompt"))
        if user_prompt is not None:
            lines.extend(
                [
                    "#### Prompt",
                    "",
                    *code_block(user_prompt, max_chars=max_prompt_chars),
                    "",
                ]
            )

        templated_prompt = record.get("templated_prompt")
        if templated_prompt:
            lines.extend(
                [
                    "#### Templated Prompt",
                    "",
                    *code_block(templated_prompt, max_chars=max_prompt_chars),
                    "",
                ]
            )

        assistant_prefix = record.get("assistant_prefix")
        if assistant_prefix:
            lines.extend(
                [
                    "#### Assistant Prefix",
                    "",
                    *code_block(assistant_prefix),
                    "",
                ]
            )

        gold_solution = record.get("gold_solution")
        if gold_solution is not None:
            lines.extend(
                [
                    "#### Gold Solution",
                    "",
                    *code_block(gold_solution),
                    "",
                ]
            )

        completion_with_prefix = record.get("completion_with_assistant_prefix")
        completion = completion_with_prefix if completion_with_prefix is not None else record.get("completion")
        lines.extend(
            [
                "#### Model Output",
                "",
                *code_block(completion, max_chars=max_completion_chars),
                "",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = [Path(value) for value in args.details_files]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing details files: " + ", ".join(str(path) for path in missing))

    details_files = [DetailsFile(path=path, records=load_details(path)) for path in paths]
    run_dir = timestamped_run_dir(Path(args.output_dir))
    output_path = run_dir / args.output_name
    write_report(
        output_path=output_path,
        details_files=details_files,
        include_correct=args.include_correct,
        max_completion_chars=args.max_completion_chars,
        max_prompt_chars=args.max_prompt_chars,
    )
    print(f"Wrote error report to {output_path}")


if __name__ == "__main__":
    main()
