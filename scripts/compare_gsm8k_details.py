from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceFile:
    label: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple GSM8K details JSONL files and write a Markdown report "
            "for questions where any run is wrong or missing."
        )
    )
    parser.add_argument(
        "details_files",
        nargs="+",
        help="details_*.jsonl files to compare.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gsm8k_detail_compare",
        help="Base output directory. A timestamped run directory is created under it.",
    )
    parser.add_argument(
        "--output-name",
        default="comparison.md",
        help="Markdown filename inside the timestamped run directory.",
    )
    parser.add_argument(
        "--key",
        choices=["question", "id"],
        default="question",
        help="Field used to align records across files. Default: question.",
    )
    parser.add_argument(
        "--label-from",
        choices=["filename", "variant", "path"],
        default="filename",
        help="How to label each source in the Markdown report.",
    )
    parser.add_argument(
        "--max-completion-chars",
        type=int,
        default=0,
        help="Truncate completion text to this many characters. Default 0 keeps full text.",
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(row)
    return records


def source_label(path: Path, records: list[dict[str, Any]], label_from: str) -> str:
    if label_from == "path":
        return str(path)
    if label_from == "variant":
        variants = {str(record.get("variant")) for record in records if record.get("variant") is not None}
        if len(variants) == 1:
            return variants.pop()
    return path.name


def make_sources(paths: list[Path], records_by_path: dict[Path, list[dict[str, Any]]], label_from: str) -> list[SourceFile]:
    sources: list[SourceFile] = []
    seen_labels: dict[str, int] = {}
    for path in paths:
        label = source_label(path, records_by_path[path], label_from)
        count = seen_labels.get(label, 0)
        seen_labels[label] = count + 1
        if count:
            label = f"{label}#{count + 1}"
        sources.append(SourceFile(label=label, path=path))
    return sources


def record_key(record: dict[str, Any], key_field: str) -> str:
    value = record.get(key_field)
    if value is None:
        raise ValueError(f"Record is missing key field {key_field!r}: {record}")
    return str(value)


def index_records(records: list[dict[str, Any]], key_field: str, source: Path) -> OrderedDict[str, dict[str, Any]]:
    indexed: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for record in records:
        key = record_key(record, key_field)
        if key in indexed:
            raise ValueError(f"{source}: duplicate {key_field} key: {key!r}")
        indexed[key] = record
    return indexed


def completion_text(record: dict[str, Any], max_chars: int) -> str:
    text = str(record.get("completion", ""))
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "\n\n...[truncated]..."
    return text


def bool_icon(value: Any) -> str:
    return "true" if value is True else "false" if value is False else "missing"


def markdown_escape_table(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def all_present_and_correct(records: list[dict[str, Any] | None]) -> bool:
    return all(record is not None and record.get("correct") is True for record in records)


def write_report(
    *,
    output_path: Path,
    sources: list[SourceFile],
    indexed_by_source: list[OrderedDict[str, dict[str, Any]]],
    key_order: list[str],
    key_field: str,
    max_completion_chars: int,
) -> None:
    total_questions = len(key_order)
    problematic_keys: list[str] = []
    for key in key_order:
        records = [indexed.get(key) for indexed in indexed_by_source]
        if not all_present_and_correct(records):
            problematic_keys.append(key)

    lines: list[str] = [
        "# GSM8K Detail Comparison",
        "",
        f"- Created at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Alignment key: `{key_field}`",
        f"- Source files: {len(sources)}",
        f"- Total aligned questions: {total_questions}",
        f"- Questions shown: {len(problematic_keys)}",
        "",
        "## Sources",
        "",
    ]

    for source in sources:
        lines.append(f"- `{source.label}`: `{source.path}`")

    lines.extend(["", "## Disagreements And Failures", ""])
    if not problematic_keys:
        lines.append("All compared records are present and correct.")
    for item_index, key in enumerate(problematic_keys, start=1):
        records = [indexed.get(key) for indexed in indexed_by_source]
        first_record = next((record for record in records if record is not None), None)
        question = first_record.get("question", key) if first_record is not None else key
        gold = first_record.get("gold_answer") if first_record is not None else None

        lines.extend(
            [
                f"### {item_index}. Question",
                "",
                f"- Key: `{key}`",
                f"- Gold answer: `{gold}`",
                "",
                "```text",
                str(question),
                "```",
                "",
                "| Source | Variant | Correct | Predicted | Latency(s) | Completion tokens |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )

        for source, record in zip(sources, records):
            if record is None:
                lines.append(f"| `{markdown_escape_table(source.label)}` |  | missing |  |  |  |")
                continue
            lines.append(
                "| "
                f"`{markdown_escape_table(source.label)}` | "
                f"{markdown_escape_table(record.get('variant'))} | "
                f"{bool_icon(record.get('correct'))} | "
                f"{markdown_escape_table(record.get('predicted_answer'))} | "
                f"{markdown_escape_table(record.get('latency_seconds'))} | "
                f"{markdown_escape_table(record.get('completion_tokens'))} |"
            )

        lines.append("")
        for source, record in zip(sources, records):
            lines.append(f"#### {source.label}")
            lines.append("")
            if record is None:
                lines.append("Missing record for this question.")
                lines.append("")
                continue
            lines.extend(
                [
                    f"- Correct: `{bool_icon(record.get('correct'))}`",
                    f"- Predicted answer: `{record.get('predicted_answer')}`",
                    f"- Variant: `{record.get('variant')}`",
                    "",
                    "```text",
                    completion_text(record, max_completion_chars),
                    "```",
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

    records_by_path = {path: load_jsonl(path) for path in paths}
    sources = make_sources(paths, records_by_path, args.label_from)
    indexed_by_source = [index_records(records_by_path[source.path], args.key, source.path) for source in sources]

    key_order: list[str] = []
    seen: set[str] = set()
    for indexed in indexed_by_source:
        for key in indexed:
            if key not in seen:
                seen.add(key)
                key_order.append(key)

    run_dir = timestamped_run_dir(Path(args.output_dir))
    output_path = run_dir / args.output_name
    write_report(
        output_path=output_path,
        sources=sources,
        indexed_by_source=indexed_by_source,
        key_order=key_order,
        key_field=args.key,
        max_completion_chars=args.max_completion_chars,
    )
    print(f"Wrote comparison report to {output_path}")


if __name__ == "__main__":
    main()
