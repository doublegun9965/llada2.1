from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare GSM8K results across threshold combinations and write a "
            "per-sample Markdown error-reason report."
        )
    )
    parser.add_argument(
        "results_dir",
        help="Directory containing details_threshold_*_edit_*.jsonl files.",
    )
    parser.add_argument(
        "--annotations-json",
        default=None,
        help="Optional JSON file that assigns wrong threshold pairs to semantic reasons.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gsm8k_threshold_reason_report",
        help="Base output directory. A fresh timestamped run directory is created.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
    return records


def combo_key(record: dict[str, Any]) -> str:
    return f"{float(record['threshold']):g},{float(record['edit_threshold']):g}"


def combo_label(key: str) -> str:
    threshold, edit_threshold = key.split(",", maxsplit=1)
    return f"({threshold},{edit_threshold})"


def combo_sort_key(key: str) -> tuple[float, float]:
    threshold, edit_threshold = key.split(",", maxsplit=1)
    return float(threshold), float(edit_threshold)


def timestamped_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    counter = 1
    while run_dir.exists():
        run_dir = base_dir / f"run_{timestamp}_{counter:02d}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_annotations(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(loaded, dict):
        raise ValueError("Annotations JSON must be an object keyed by sample_id")
    return {str(key): value for key, value in loaded.items()}


def short_answer(value: Any) -> str:
    if value is None:
        return "未解析到答案"
    rendered = str(value).replace("\n", r"\n")
    if len(rendered) > 80:
        return rendered[:77] + "..."
    return rendered


def markdown_text(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", "<br>")


def assign_reason_groups(
    *,
    sample_id: str,
    wrong_records: list[dict[str, Any]],
    annotations: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    by_combo = {combo_key(record): record for record in wrong_records}
    assigned: set[str] = set()
    groups: list[tuple[str, list[dict[str, Any]]]] = []

    for annotation in annotations.get(sample_id, []):
        reason = str(annotation["reason"])
        combos = [str(value) for value in annotation["combos"]]
        unknown = sorted(set(combos).difference(by_combo), key=combo_sort_key)
        duplicate = sorted(set(combos).intersection(assigned), key=combo_sort_key)
        if unknown:
            raise ValueError(
                f"sample_id={sample_id}: annotated combos are not wrong: {unknown}"
            )
        if duplicate:
            raise ValueError(
                f"sample_id={sample_id}: combos assigned more than once: {duplicate}"
            )
        assigned.update(combos)
        groups.append((reason, [by_combo[key] for key in combos]))

    remaining = [
        record for record in wrong_records if combo_key(record) not in assigned
    ]
    incomplete = [
        record for record in remaining if "####" not in str(record.get("completion", ""))
    ]
    if incomplete:
        groups.append(
            (
                "生成退化、截断或未按要求给出完整最终答案，无法形成可靠的数学推理链。",
                incomplete,
            )
        )

    complete = [record for record in remaining if record not in incomplete]
    complete_by_answer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in complete:
        complete_by_answer[short_answer(record.get("predicted_answer"))].append(record)
    for answer, records in complete_by_answer.items():
        groups.append(
            (
                f"模型得到错误答案 {answer}，但尚未提供人工标注的细粒度错因。",
                records,
            )
        )
    return groups


def validate_runs(runs: list[tuple[Path, list[dict[str, Any]]]]) -> list[str]:
    if not runs:
        raise ValueError("No details_threshold_*_edit_*.jsonl files found")
    expected_ids = [str(record["id"]) for record in runs[0][1]]
    for path, records in runs[1:]:
        ids = [str(record["id"]) for record in records]
        if ids != expected_ids:
            raise ValueError(f"{path}: sample IDs/order differ from the first run")
    return expected_ids


def write_report(
    *,
    output_path: Path,
    results_dir: Path,
    runs: list[tuple[Path, list[dict[str, Any]]]],
    annotations: dict[str, list[dict[str, Any]]],
) -> None:
    sample_ids = validate_runs(runs)
    records_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, records in runs:
        for record in records:
            records_by_sample[str(record["id"])].append(record)

    selected_ids = [
        sample_id
        for sample_id in sample_ids
        if not all(record.get("correct") is True for record in records_by_sample[sample_id])
    ]
    all_correct = len(sample_ids) - len(selected_ids)
    total_wrong = sum(
        record.get("correct") is not True
        for sample_id in selected_ids
        for record in records_by_sample[sample_id]
    )

    lines = [
        "# GSM8K 阈值组合错因对比",
        "",
        f"- 结果目录：`{results_dir}`",
        f"- 阈值组合数：`{len(runs)}`",
        f"- 样本数：`{len(sample_ids)}`",
        f"- 全组合均正确并跳过：`{all_correct}`",
        f"- 至少一个组合错误并纳入报告：`{len(selected_ids)}`",
        f"- 错误输出总数：`{total_wrong}`",
        "",
        "说明：正确/错误以 details 文件中的 `correct` 字段为准。相同错因的阈值组合合并展示；",
        "生成退化或没有完整答案的输出单独归类，不强行解释为具体数学错误。",
        "",
        "## 用例",
        "",
    ]

    for case_number, sample_id in enumerate(selected_ids, start=1):
        records = records_by_sample[sample_id]
        exemplar = records[0]
        correct_records = [record for record in records if record.get("correct") is True]
        wrong_records = [record for record in records if record.get("correct") is not True]
        correct_combos = sorted(
            (combo_key(record) for record in correct_records), key=combo_sort_key
        )
        reason_groups = assign_reason_groups(
            sample_id=sample_id,
            wrong_records=wrong_records,
            annotations=annotations,
        )

        lines.extend(
            [
                f"### 用例 {case_number}（sample_id={sample_id}）",
                "",
                f"- 题目：{markdown_text(exemplar.get('question', ''))}",
                f"- 标准答案：`{short_answer(exemplar.get('gold_answer'))}`",
                "- 正确案例："
                + ("、".join(combo_label(key) for key in correct_combos) or "无"),
                "",
            ]
        )

        for reason_index, (reason, group_records) in enumerate(reason_groups, start=1):
            sorted_records = sorted(
                group_records, key=lambda record: combo_sort_key(combo_key(record))
            )
            lines.extend(
                [
                    f"#### 错因 {reason_index}",
                    "",
                    reason,
                    "",
                    "| 阈值组合 | 输出答案 |",
                    "| --- | --- |",
                ]
            )
            for record in sorted_records:
                lines.append(
                    f"| {combo_label(combo_key(record))} | "
                    f"`{short_answer(record.get('predicted_answer'))}` |"
                )
            lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    paths = sorted(results_dir.glob("details_threshold_*_edit_*.jsonl"))
    runs = [(path, load_jsonl(path)) for path in paths]
    annotations = load_annotations(
        Path(args.annotations_json) if args.annotations_json else None
    )
    run_dir = timestamped_run_dir(Path(args.output_dir))
    output_path = run_dir / "threshold_error_reasons.md"
    write_report(
        output_path=output_path,
        results_dir=results_dir,
        runs=runs,
        annotations=annotations,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
