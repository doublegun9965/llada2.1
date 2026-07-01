from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
MATH_WORDS = {
    "add",
    "added",
    "altogether",
    "average",
    "difference",
    "divide",
    "divided",
    "each",
    "equal",
    "equals",
    "fewer",
    "left",
    "less",
    "minus",
    "more",
    "multiply",
    "product",
    "remain",
    "remaining",
    "subtract",
    "sum",
    "than",
    "therefore",
    "times",
    "total",
}
UNIT_WORDS = {
    "$",
    "cent",
    "cents",
    "dollar",
    "dollars",
    "hour",
    "hours",
    "minute",
    "minutes",
    "day",
    "days",
    "week",
    "weeks",
    "mile",
    "miles",
    "apple",
    "apples",
    "book",
    "books",
    "ticket",
    "tickets",
}
OPERATORS = {"+", "-", "*", "/", "=", "<", ">", "×", "÷"}
CRITICAL_TYPES = {
    "answer_marker",
    "answer_number",
    "number",
    "operator",
    "unit",
    "math_word",
}


@dataclass(frozen=True)
class AnalysisPaths:
    output_dir: Path
    token_events_path: Path
    token_proposals_path: Path
    edit_proposals_path: Path
    edit_annotation_path: Path
    token_summary_path: Path
    sample_summary_path: Path
    critical_token_stats_path: Path
    report_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SGLang dLLM trace JSONL into critical-token analysis tables."
    )
    parser.add_argument("trace_path", help="Trace JSONL written by SGLang dLLM trace patch.")
    parser.add_argument(
        "--details",
        required=True,
        help="GSM8K details JSONL from the same sequential evaluation run.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Model/tokenizer path used to decode trace token ids.",
    )
    parser.add_argument("--output-dir", default="outputs/critical_path_analysis")
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Write directly into --output-dir instead of creating run_<timestamp>.",
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=0.7,
        help="Fixed threshold for high_confidence_commit. Default: 0.7.",
    )
    parser.add_argument(
        "--early-fraction",
        type=float,
        default=0.3,
        help="Block-fraction cutoff for early_commit_within_block. Default: 0.3.",
    )
    parser.add_argument(
        "--late-fraction",
        type=float,
        default=0.7,
        help="Block-fraction cutoff for late_commit_within_block. Default: 0.7.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def timestamped_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    counter = 1
    while run_dir.exists():
        run_dir = base_dir / f"run_{timestamp}_{counter:02d}"
        counter += 1
    return run_dir


def make_paths(output_dir: Path) -> AnalysisPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    return AnalysisPaths(
        output_dir=output_dir,
        token_events_path=output_dir / "token_events.csv",
        token_proposals_path=output_dir / "token_proposals.csv",
        edit_proposals_path=output_dir / "edit_proposals.csv",
        edit_annotation_path=output_dir / "edit_annotation.md",
        token_summary_path=output_dir / "token_summary.csv",
        sample_summary_path=output_dir / "sample_summary.csv",
        critical_token_stats_path=output_dir / "critical_token_stats.csv",
        report_path=output_dir / "report.md",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def split_trace_into_block_segments(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_iteration: int | None = None
    for event in events:
        iteration = int(event.get("global_iteration", 0))
        if current and previous_iteration is not None and iteration <= previous_iteration:
            segments.append(current)
            current = []
        current.append(event)
        previous_iteration = iteration
    if current:
        segments.append(current)
    return segments


def decode_token(tokenizer: Any, token_id: int | None) -> str:
    if token_id is None:
        return ""
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def normalize_token_text(token: str) -> str:
    return token.replace("Ġ", " ").replace("▁", " ").strip()


def classify_token(token: str, token_id: int | None, special_ids: set[int]) -> str:
    normalized = normalize_token_text(token)
    lowered = normalized.lower()
    if token_id is not None and int(token_id) in special_ids:
        return "special"
    if "####" in normalized:
        return "answer_marker"
    if NUMBER_RE.search(normalized):
        return "number"
    if normalized in OPERATORS:
        return "operator"
    if any(operator in normalized for operator in OPERATORS) and len(normalized) <= 3:
        return "operator"
    if lowered in UNIT_WORDS:
        return "unit"
    if lowered in MATH_WORDS:
        return "math_word"
    if normalized and all(not char.isalnum() for char in normalized):
        return "punctuation"
    if not normalized:
        return "special"
    return "plain_text"


def is_critical_type(token_type: str) -> bool:
    return token_type in CRITICAL_TYPES


def as_bool_text(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return as_bool_text(value)
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def block_count_for_detail(record: dict[str, Any], block_size: int) -> int:
    completion_tokens = record.get("completion_tokens")
    if isinstance(completion_tokens, int) and completion_tokens > 0:
        return max(1, math.ceil(completion_tokens / block_size))
    return 1


def has_request_aligned_trace(events: list[dict[str, Any]]) -> bool:
    return any(
        event.get("request_id") is not None and event.get("dllm_block_offset") is not None
        for event in events
    )


def passes_threshold(event_type: str, prob: float | None, threshold: float, edit_threshold: float) -> bool | None:
    if prob is None:
        return None
    if event_type == "edit":
        return prob >= edit_threshold
    if event_type == "mask_fill":
        return prob >= threshold
    return None


def append_token_event(
    *,
    token_events: list[dict[str, Any]],
    token_event_counts: dict[tuple[str, int], int],
    tokenizer: Any,
    special_ids: set[int],
    sample_id: str,
    sample_index: int,
    generated_pos: int,
    global_iteration: int,
    block_index: int,
    block_iteration: int,
    block_offset: int,
    event_type: str,
    item: dict[str, Any],
    threshold: float,
    edit_threshold: float,
) -> None:
    old_token_id = int(item["old_token_id"])
    new_token_id = int(item["new_token_id"])
    old_token = decode_token(tokenizer, old_token_id)
    new_token = decode_token(tokenizer, new_token_id)
    token_type = classify_token(new_token, new_token_id, special_ids)
    event_key = (sample_id, generated_pos)
    event_index = token_event_counts[event_key]
    token_event_counts[event_key] += 1
    prob = float(item["prob"]) if item.get("prob") is not None else None
    token_events.append(
        {
            "event_id": f"{sample_id}:{generated_pos}:{event_index}",
            "token_id": f"{sample_id}:{generated_pos}",
            "sample_id": sample_id,
            "sample_index": sample_index,
            "generated_pos": generated_pos,
            "event_index": event_index,
            "global_iteration": global_iteration,
            "block_index": block_index,
            "block_iteration": block_iteration,
            "block_offset": block_offset,
            "event_type": event_type,
            "old_token": old_token,
            "old_token_id": old_token_id,
            "new_token": new_token,
            "new_token_id": new_token_id,
            "prob": prob,
            "token_type": token_type,
            "is_critical": is_critical_type(token_type),
            "is_commit_event": False,
            "changed_token": old_token_id != new_token_id,
            "accepted_by": item.get("accepted_by", ""),
            "A": item.get("A"),
            "fallback_rank": item.get("fallback_rank"),
            "passes_run_threshold": passes_threshold(
                event_type, prob, threshold, edit_threshold
            ),
        }
    )


def update_answer_span_types(token_rows: list[dict[str, Any]]) -> None:
    rows_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in token_rows:
        rows_by_sample[str(row["sample_id"])].append(row)

    for rows in rows_by_sample.values():
        rows.sort(key=lambda item: int(item["generated_pos"]))
        marker_positions = [
            int(row["generated_pos"])
            for row in rows
            if row.get("token_type") == "answer_marker" or "####" in str(row.get("final_token", ""))
        ]
        if not marker_positions:
            continue
        marker_pos = marker_positions[-1]
        answer_window = set(range(marker_pos, marker_pos + 6))
        for row in rows:
            generated_pos = int(row["generated_pos"])
            row["final_answer_span"] = generated_pos in answer_window
            if generated_pos in answer_window and row.get("token_type") == "number":
                row["token_type"] = "answer_number"
                row["is_critical"] = True


def analyze_trace(
    *,
    trace_path: Path,
    details_path: Path,
    model_path: str,
    output_dir: Path,
    high_confidence_threshold: float = 0.7,
    early_fraction: float = 0.3,
    late_fraction: float = 0.7,
    trust_remote_code: bool = False,
) -> AnalysisPaths:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    special_ids = set(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) or [])
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is not None:
        special_ids.add(int(mask_token_id))

    trace_events = read_jsonl(trace_path)
    details = sorted(read_jsonl(details_path), key=lambda item: int(item.get("index", 0)))
    paths = make_paths(output_dir)

    if not trace_events:
        write_empty_outputs(paths)
        return paths

    block_size = int(trace_events[0].get("block_size", 0) or 0)
    if block_size <= 0:
        raise ValueError(f"{trace_path}: first trace event is missing a positive block_size")

    token_events: list[dict[str, Any]] = []
    token_event_counts: dict[tuple[str, int], int] = defaultdict(int)
    sample_block_totals: dict[tuple[str, int], int] = {}
    sample_global_totals: dict[str, int] = defaultdict(int)
    warnings: list[str] = []

    if has_request_aligned_trace(trace_events):
        build_events_from_request_aligned_trace(
            trace_events=trace_events,
            details=details,
            tokenizer=tokenizer,
            special_ids=special_ids,
            block_size=block_size,
            token_events=token_events,
            token_event_counts=token_event_counts,
            sample_block_totals=sample_block_totals,
            sample_global_totals=sample_global_totals,
            warnings=warnings,
        )
    else:
        build_events_from_legacy_trace(
            trace_events=trace_events,
            details=details,
            tokenizer=tokenizer,
            special_ids=special_ids,
            block_size=block_size,
            token_events=token_events,
            token_event_counts=token_event_counts,
            sample_block_totals=sample_block_totals,
            sample_global_totals=sample_global_totals,
            warnings=warnings,
        )

    token_summary = build_token_summary(
        token_events=token_events,
        sample_block_totals=sample_block_totals,
        high_confidence_threshold=high_confidence_threshold,
        early_fraction=early_fraction,
        late_fraction=late_fraction,
    )
    update_answer_span_types(token_summary)
    propagate_final_token_types(token_events, token_summary)
    sample_summary = build_sample_summary(details, token_summary, token_events, sample_global_totals)
    critical_stats = build_critical_token_stats(token_summary, details)
    token_proposals = build_token_proposals(
        trace_events=trace_events,
        details=details,
        tokenizer=tokenizer,
        special_ids=special_ids,
        block_size=block_size,
    )
    if not token_proposals:
        warnings.append(
            "Trace contains no per-position proposals. Reapply the current "
            "dllm_trace.patch and rerun generation to produce proposal tables."
        )
    duplicate_runner_ups = sum(
        1
        for row in token_proposals
        if row.get("proposed_token_id") == row.get("second_token_id")
    )
    if duplicate_runner_ups:
        warnings.append(
            f"Trace contains {duplicate_runner_ups} proposal(s) whose proposed and "
            "second token ids are identical. These traces were produced by the "
            "pre-fix runner-up selection and should be regenerated before using "
            "second_token for analysis."
        )
    edit_proposals = [row for row in token_proposals if row.get("proposed_edit") is True]
    preserve_manual_annotations(paths.edit_proposals_path, edit_proposals)

    write_csv(paths.token_events_path, token_events, TOKEN_EVENTS_FIELDS)
    write_csv(paths.token_proposals_path, token_proposals, TOKEN_PROPOSAL_FIELDS)
    write_csv(paths.edit_proposals_path, edit_proposals, EDIT_PROPOSAL_FIELDS)
    write_edit_annotation(paths.edit_annotation_path, edit_proposals)
    write_csv(paths.token_summary_path, token_summary, TOKEN_SUMMARY_FIELDS)
    write_csv(paths.sample_summary_path, sample_summary, SAMPLE_SUMMARY_FIELDS)
    write_csv(paths.critical_token_stats_path, critical_stats, CRITICAL_STATS_FIELDS)
    write_report(
        paths=paths,
        trace_path=trace_path,
        details_path=details_path,
        model_path=model_path,
        warnings=warnings,
        sample_summary=sample_summary,
        token_summary=token_summary,
        critical_stats=critical_stats,
        token_proposals=token_proposals,
        edit_proposals=edit_proposals,
    )
    return paths


def request_detail_mapping(
    trace_events: list[dict[str, Any]], details: list[dict[str, Any]]
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    request_order: list[str] = []
    seen: set[str] = set()
    for event in trace_events:
        request_id = event.get("request_id")
        if request_id is None:
            continue
        request_id = str(request_id)
        if request_id not in seen:
            seen.add(request_id)
            request_order.append(request_id)

    explicit_ids = [detail.get("trace_request_id") for detail in details]
    if all(request_id is not None for request_id in explicit_ids):
        return request_order, {
            str(request_id): detail
            for request_id, detail in zip(explicit_ids, details)
        }
    return request_order, dict(zip(request_order, details))


def build_token_proposals(
    *,
    trace_events: list[dict[str, Any]],
    details: list[dict[str, Any]],
    tokenizer: Any,
    special_ids: set[int],
    block_size: int,
) -> list[dict[str, Any]]:
    if not has_request_aligned_trace(trace_events):
        return []

    request_order, detail_by_request = request_detail_mapping(trace_events, details)
    block_iteration_totals: dict[tuple[str, int], int] = defaultdict(int)
    for event in trace_events:
        request_id = event.get("request_id")
        block_start = event.get("dllm_block_offset")
        if request_id is None or block_start is None:
            continue
        key = (str(request_id), int(block_start))
        block_iteration_totals[key] = max(
            block_iteration_totals[key], int(event.get("global_iteration", 0))
        )

    global_offsets: dict[tuple[str, int], int] = {}
    for request_id in request_order:
        cumulative = 0
        for block_start in sorted(
            start for rid, start in block_iteration_totals if rid == request_id
        ):
            global_offsets[(request_id, block_start)] = cumulative
            cumulative += block_iteration_totals[(request_id, block_start)]

    rows: list[dict[str, Any]] = []
    for event in trace_events:
        request_id_value = event.get("request_id")
        block_start_value = event.get("dllm_block_offset")
        proposals = event.get("proposals") or []
        if request_id_value is None or block_start_value is None or not proposals:
            continue
        request_id = str(request_id_value)
        detail = detail_by_request.get(request_id)
        if detail is None:
            continue

        block_start = int(block_start_value)
        block_index = int(event.get("dllm_block_index", block_start // block_size))
        block_iteration = int(event.get("global_iteration", 0))
        global_iteration = global_offsets.get((request_id, block_start), 0) + block_iteration
        sample_id = str(detail.get("id", detail.get("index")))
        before_ids = event.get("block_token_ids_before")
        after_ids = event.get("block_token_ids")
        before_text = (
            tokenizer.decode(before_ids, skip_special_tokens=False)
            if isinstance(before_ids, list)
            else ""
        )
        after_text = (
            tokenizer.decode(after_ids, skip_special_tokens=False)
            if isinstance(after_ids, list)
            else ""
        )

        for proposal in proposals:
            block_offset = int(proposal["block_offset"])
            current_token_id = int(proposal["current_token_id"])
            proposed_token_id = int(proposal["proposed_token_id"])
            second_token_id = int(proposal["second_token_id"])
            proposed_token = decode_token(tokenizer, proposed_token_id)
            advantage = proposal.get("A", proposal.get("replacement_advantage"))
            margin = proposal.get("D", proposal.get("candidate_margin"))
            proposal_id = (
                f"{request_id}:{block_start}:{block_iteration}:{block_offset}"
            )
            rows.append(
                {
                    "proposal_id": proposal_id,
                    "request_id": request_id,
                    "sample_id": sample_id,
                    "sample_index": detail.get("index"),
                    "sample_correct": detail.get("correct"),
                    "gold_answer": detail.get("gold_answer"),
                    "predicted_answer": detail.get("predicted_answer"),
                    "threshold": event.get("threshold", detail.get("threshold")),
                    "edit_threshold": event.get(
                        "edit_threshold", detail.get("edit_threshold")
                    ),
                    "block_index": block_index,
                    "block_iteration": block_iteration,
                    "global_iteration": global_iteration,
                    "block_offset": block_offset,
                    "generated_pos": block_start + block_offset,
                    "state_hash": event.get("state_hash"),
                    "proposal_type": proposal.get("proposal_type"),
                    "current_token": decode_token(tokenizer, current_token_id),
                    "proposed_token": proposed_token,
                    "second_token": decode_token(tokenizer, second_token_id),
                    "current_token_id": current_token_id,
                    "proposed_token_id": proposed_token_id,
                    "second_token_id": second_token_id,
                    "p_old": proposal.get("p_old"),
                    "p_new": proposal.get("p_new"),
                    "p_second": proposal.get("p_second"),
                    "old_logit": proposal.get("old_logit"),
                    "new_logit": proposal.get("new_logit"),
                    "second_logit": proposal.get("second_logit"),
                    "A": advantage,
                    "D": margin,
                    "replacement_advantage": advantage,
                    "candidate_margin": margin,
                    "entropy": proposal.get("entropy"),
                    "proposed_edit": proposal.get("proposed_edit"),
                    "accepted_edit": proposal.get("accepted_edit"),
                    "accepted_update": proposal.get("accepted_update"),
                    "edit_selected_by": event.get("edit_selected_by", ""),
                    "accepted_by": proposal.get("accepted_by", ""),
                    "fallback_rank": proposal.get("fallback_rank"),
                    "rejected_reason": proposal.get("rejected_reason"),
                    "token_age": proposal.get("token_age"),
                    "token_type": classify_token(
                        proposed_token, proposed_token_id, special_ids
                    ),
                    "manual_label": "",
                    "manual_reason": "",
                    "manual_notes": "",
                    "_question": detail.get("question", ""),
                    "_before_text": before_text,
                    "_after_text": after_text,
                }
            )
    return rows


def preserve_manual_annotations(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        existing = {
            row.get("proposal_id", ""): row for row in csv.DictReader(handle)
        }
    for row in rows:
        old = existing.get(str(row["proposal_id"]))
        if old is None:
            continue
        for field in MANUAL_ANNOTATION_FIELDS:
            row[field] = old.get(field, "")


def markdown_code(value: Any) -> str:
    text = json.dumps("" if value is None else str(value), ensure_ascii=False)
    return f"`{text.replace('|', '&#124;')}`"


def write_edit_annotation(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Edit Proposal Annotation",
        "",
        "Annotate `manual_label` in `edit_proposals.csv` as "
        "`beneficial`, `harmful`, `neutral`, or `uncertain`, then fill "
        "`manual_reason` and optional `manual_notes`.",
        "",
        f"Proposed edits: `{len(rows)}`",
        "",
    ]
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["sample_id"]),
                int(row["block_index"]),
                int(row["block_iteration"]),
            )
        ].append(row)

    for (sample_id, block_index, block_iteration), group in grouped.items():
        first = group[0]
        lines.extend(
            [
                f"## Sample {sample_id}: block {block_index}, iteration {block_iteration}",
                "",
                f"- Correct: `{as_bool_text(first.get('sample_correct'))}`",
                f"- State hash: `{first.get('state_hash', '')}`",
                f"- Gold answer: {markdown_code(first.get('gold_answer'))}",
                f"- Predicted answer: {markdown_code(first.get('predicted_answer'))}",
                "",
            ]
        )
        if first.get("_question"):
            lines.extend(["Question:", "", str(first["_question"]), ""])
        if first.get("_before_text"):
            lines.extend(
                ["Before:", "", "```text", str(first["_before_text"]), "```", ""]
            )
        if first.get("_after_text"):
            lines.extend(
                ["After:", "", "```text", str(first["_after_text"]), "```", ""]
            )
        lines.extend(
            [
                "| Offset | Current | Proposed | Second | p_old | p_new | p_second | A | D | Entropy | Accepted | Accepted by | Fallback rank | Rejected reason | Manual label | Manual reason |",
                "|---:|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---|---:|---|---|---|",
            ]
        )
        for row in sorted(group, key=lambda item: int(item["block_offset"])):
            lines.append(
                "| "
                f"{row['block_offset']} | {markdown_code(row['current_token'])} | "
                f"{markdown_code(row['proposed_token'])} | {markdown_code(row['second_token'])} | "
                f"{format_float(row.get('p_old'))} | {format_float(row.get('p_new'))} | "
                f"{format_float(row.get('p_second'))} | {format_float(row.get('A'))} | "
                f"{format_float(row.get('D'))} | {format_float(row.get('entropy'))} | "
                f"{as_bool_text(row.get('accepted_edit'))} | "
                f"{row.get('accepted_by', '')} | {row.get('fallback_rank') or ''} | "
                f"{row.get('rejected_reason', '')} | {row.get('manual_label', '')} | "
                f"{str(row.get('manual_reason', '')).replace('|', '&#124;')} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_events_from_request_aligned_trace(
    *,
    trace_events: list[dict[str, Any]],
    details: list[dict[str, Any]],
    tokenizer: Any,
    special_ids: set[int],
    block_size: int,
    token_events: list[dict[str, Any]],
    token_event_counts: dict[tuple[str, int], int],
    sample_block_totals: dict[tuple[str, int], int],
    sample_global_totals: dict[str, int],
    warnings: list[str],
) -> None:
    request_order: list[str] = []
    seen_requests: set[str] = set()
    for event in trace_events:
        request_id = event.get("request_id")
        if request_id is None:
            continue
        request_id = str(request_id)
        if request_id not in seen_requests:
            seen_requests.add(request_id)
            request_order.append(request_id)

    detail_request_ids = [detail.get("trace_request_id") for detail in details]
    has_explicit_detail_ids = all(request_id is not None for request_id in detail_request_ids)

    detail_by_request: dict[str, dict[str, Any]] = {}
    if has_explicit_detail_ids:
        for request_id, detail in zip(detail_request_ids, details):
            request_id = str(request_id)
            if request_id in detail_by_request:
                raise ValueError(f"Duplicate trace_request_id in details: {request_id}")
            detail_by_request[request_id] = detail

        trace_request_ids = set(request_order)
        missing_request_ids = sorted(set(detail_by_request) - trace_request_ids)
        if missing_request_ids:
            preview = ", ".join(missing_request_ids[:5])
            raise ValueError(
                f"Trace is missing {len(missing_request_ids)} request id(s) present in details: "
                f"{preview}"
            )

        extra_request_ids = sorted(trace_request_ids - set(detail_by_request))
        if extra_request_ids:
            warnings.append(
                f"Request-aligned trace has {len(extra_request_ids)} extra request id(s) "
                "not present in details; they are ignored."
            )
    else:
        if any(request_id is not None for request_id in detail_request_ids):
            raise ValueError(
                "Only some details records contain trace_request_id; exact trace alignment is impossible."
            )
        if len(request_order) != len(details):
            raise ValueError(
                f"Request-aligned trace has {len(request_order)} request id(s), but details has "
                f"{len(details)} sample(s). Details do not contain trace_request_id, so positional "
                "alignment would be unsafe."
            )
        warnings.append(
            "Details do not contain trace_request_id; request ids were matched by request order. "
            "Regenerate details with the current evaluator for exact alignment."
        )
        for request_id, detail in zip(request_order, details):
            detail_by_request[request_id] = detail

    block_iteration_totals: dict[tuple[str, int], int] = defaultdict(int)
    for event in trace_events:
        request_id = event.get("request_id")
        block_start = event.get("dllm_block_offset")
        if request_id is None or block_start is None:
            continue
        key = (str(request_id), int(block_start))
        block_iteration_totals[key] = max(
            block_iteration_totals[key],
            int(event.get("global_iteration", 0)),
        )

    block_global_offsets: dict[tuple[str, int], int] = {}
    for request_id in request_order:
        cumulative = 0
        block_starts = sorted(
            block_start
            for rid, block_start in block_iteration_totals
            if rid == request_id
        )
        for block_start in block_starts:
            block_global_offsets[(request_id, block_start)] = cumulative
            cumulative += block_iteration_totals[(request_id, block_start)]

        detail = detail_by_request.get(request_id)
        if detail is not None:
            sample_id = str(detail.get("id", detail.get("index")))
            sample_global_totals[sample_id] = cumulative

    for trace_event in trace_events:
        request_id = trace_event.get("request_id")
        block_start = trace_event.get("dllm_block_offset")
        if request_id is None or block_start is None:
            continue
        request_id = str(request_id)
        block_start = int(block_start)
        detail = detail_by_request.get(request_id)
        if detail is None:
            continue

        sample_id = str(detail.get("id", detail.get("index")))
        sample_index = int(detail.get("index", 0))
        threshold = float(detail.get("threshold", 0.0))
        edit_threshold = float(detail.get("edit_threshold", 0.0))
        block_index = block_start // block_size
        block_iteration = int(trace_event.get("global_iteration", 0))
        global_iteration = block_global_offsets.get((request_id, block_start), 0) + block_iteration
        sample_block_totals[(sample_id, block_index)] = block_iteration_totals[
            (request_id, block_start)
        ]

        for event_type, list_name in (
            ("mask_fill", "mask_fills"),
            ("edit", "edits"),
            ("remask", "remasks"),
        ):
            for item in trace_event.get(list_name) or []:
                block_offset = int(item["block_offset"])
                append_token_event(
                    token_events=token_events,
                    token_event_counts=token_event_counts,
                    tokenizer=tokenizer,
                    special_ids=special_ids,
                    sample_id=sample_id,
                    sample_index=sample_index,
                    generated_pos=block_start + block_offset,
                    global_iteration=global_iteration,
                    block_index=block_index,
                    block_iteration=block_iteration,
                    block_offset=block_offset,
                    event_type=event_type,
                    item=item,
                    threshold=threshold,
                    edit_threshold=edit_threshold,
                )


def build_events_from_legacy_trace(
    *,
    trace_events: list[dict[str, Any]],
    details: list[dict[str, Any]],
    tokenizer: Any,
    special_ids: set[int],
    block_size: int,
    token_events: list[dict[str, Any]],
    token_event_counts: dict[tuple[str, int], int],
    sample_block_totals: dict[tuple[str, int], int],
    sample_global_totals: dict[str, int],
    warnings: list[str],
) -> None:
    segments = split_trace_into_block_segments(trace_events)
    segment_index = 0

    for detail in details:
        sample_id = str(detail.get("id", detail.get("index")))
        sample_index = int(detail.get("index", len(sample_global_totals) + 1))
        threshold = float(detail.get("threshold", 0.0))
        edit_threshold = float(detail.get("edit_threshold", 0.0))
        blocks_needed = block_count_for_detail(detail, block_size)
        cumulative_iteration_offset = 0

        for block_index in range(blocks_needed):
            if segment_index >= len(segments):
                warnings.append(
                    f"Missing trace block for sample_id={sample_id} block_index={block_index}; "
                    "remaining details cannot be fully aligned."
                )
                break
            segment = segments[segment_index]
            segment_index += 1
            total_block_iterations = max(int(event.get("global_iteration", 0)) for event in segment)
            sample_block_totals[(sample_id, block_index)] = total_block_iterations
            sample_global_totals[sample_id] += total_block_iterations

            for trace_event in segment:
                block_iteration = int(trace_event.get("global_iteration", 0))
                global_iteration = cumulative_iteration_offset + block_iteration
                for event_type, list_name in (
                    ("mask_fill", "mask_fills"),
                    ("edit", "edits"),
                    ("remask", "remasks"),
                ):
                    for item in trace_event.get(list_name) or []:
                        block_offset = int(item["block_offset"])
                        append_token_event(
                            token_events=token_events,
                            token_event_counts=token_event_counts,
                            tokenizer=tokenizer,
                            special_ids=special_ids,
                            sample_id=sample_id,
                            sample_index=sample_index,
                            generated_pos=block_index * block_size + block_offset,
                            global_iteration=global_iteration,
                            block_index=block_index,
                            block_iteration=block_iteration,
                            block_offset=block_offset,
                            event_type=event_type,
                            item=item,
                            threshold=threshold,
                            edit_threshold=edit_threshold,
                        )
            cumulative_iteration_offset += total_block_iterations

    if segment_index < len(segments):
        warnings.append(
            f"{len(segments) - segment_index} unassigned trace block segment(s). "
            "This legacy trace has no request_id/dllm_block_offset metadata, so "
            "the analyzer had to estimate sample alignment from completion_tokens."
        )


def build_token_summary(
    *,
    token_events: list[dict[str, Any]],
    sample_block_totals: dict[tuple[str, int], int],
    high_confidence_threshold: float,
    early_fraction: float,
    late_fraction: float,
) -> list[dict[str, Any]]:
    events_by_token: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in token_events:
        events_by_token[str(event["token_id"])].append(event)

    rows: list[dict[str, Any]] = []
    for token_id, events in events_by_token.items():
        events.sort(key=lambda item: int(item["event_index"]))
        sample_id = str(events[0]["sample_id"])
        generated_pos = int(events[0]["generated_pos"])
        block_index = int(events[0]["block_index"])
        block_offset = int(events[0]["block_offset"])
        commit_event = next((event for event in events if event["event_type"] == "mask_fill"), events[0])
        final_event = events[-1]
        edit_events = [event for event in events if event["event_type"] == "edit"]
        remask_events = [event for event in events if event["event_type"] == "remask"]
        total_block_iterations = sample_block_totals.get((sample_id, block_index))
        commit_block_iteration = int(commit_event["block_iteration"])
        commit_block_fraction = (
            commit_block_iteration / total_block_iterations
            if total_block_iterations
            else None
        )
        commit_prob = commit_event.get("prob")
        rows.append(
            {
                "token_id": token_id,
                "sample_id": sample_id,
                "sample_index": events[0]["sample_index"],
                "generated_pos": generated_pos,
                "block_index": block_index,
                "block_offset": block_offset,
                "final_token": final_event["new_token"],
                "final_token_id": final_event["new_token_id"],
                "token_type": classify_token(
                    str(final_event["new_token"]),
                    int(final_event["new_token_id"]),
                    set(),
                ),
                "is_critical": is_critical_type(
                    classify_token(str(final_event["new_token"]), int(final_event["new_token_id"]), set())
                ),
                "commit_global_iteration": commit_event["global_iteration"],
                "commit_block_iteration": commit_block_iteration,
                "commit_prob": commit_prob,
                "commit_event_type": commit_event["event_type"],
                "last_event_global_iteration": final_event["global_iteration"],
                "last_event_block_iteration": final_event["block_iteration"],
                "last_edit_global_iteration": edit_events[-1]["global_iteration"] if edit_events else None,
                "last_edit_block_iteration": edit_events[-1]["block_iteration"] if edit_events else None,
                "num_events": len(events),
                "num_edits": len(edit_events),
                "num_remasks": len(remask_events),
                "was_edited_after_commit": bool(edit_events),
                "was_remasked": bool(remask_events),
                "total_block_iterations": total_block_iterations,
                "commit_block_fraction": commit_block_fraction,
                "early_commit_within_block": (
                    commit_block_fraction is not None and commit_block_fraction <= early_fraction
                ),
                "late_commit_within_block": (
                    commit_block_fraction is not None and commit_block_fraction >= late_fraction
                ),
                "high_confidence_commit": (
                    commit_prob is not None and float(commit_prob) >= high_confidence_threshold
                ),
                "final_answer_span": False,
                "passes_run_threshold": commit_event.get("passes_run_threshold"),
            }
        )

        commit_event["is_commit_event"] = True

    rows.sort(key=lambda item: (int(item["sample_index"]), int(item["generated_pos"])))
    return rows


def propagate_final_token_types(
    token_events: list[dict[str, Any]], token_summary: list[dict[str, Any]]
) -> None:
    type_by_token_id = {
        str(row["token_id"]): (row["token_type"], row["is_critical"])
        for row in token_summary
    }
    for event in token_events:
        final_type, is_critical = type_by_token_id.get(
            str(event["token_id"]), (event["token_type"], event["is_critical"])
        )
        event["token_type"] = final_type
        event["is_critical"] = is_critical


def build_sample_summary(
    details: list[dict[str, Any]],
    token_summary: list[dict[str, Any]],
    token_events: list[dict[str, Any]],
    sample_global_totals: dict[str, int],
) -> list[dict[str, Any]]:
    tokens_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    events_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in token_summary:
        tokens_by_sample[str(row["sample_id"])].append(row)
    for event in token_events:
        events_by_sample[str(event["sample_id"])].append(event)

    rows: list[dict[str, Any]] = []
    for detail in sorted(details, key=lambda item: int(item.get("index", 0))):
        sample_id = str(detail.get("id", detail.get("index")))
        tokens = tokens_by_sample.get(sample_id, [])
        events = events_by_sample.get(sample_id, [])
        critical_tokens = [token for token in tokens if token.get("is_critical") is True]
        plain_tokens = [token for token in tokens if token.get("token_type") == "plain_text"]
        critical_early = [token for token in critical_tokens if token.get("early_commit_within_block") is True]
        critical_high_conf = [
            token for token in critical_tokens if token.get("high_confidence_commit") is True
        ]
        critical_edits = sum(int(token.get("num_edits") or 0) for token in critical_tokens)
        plain_edits = sum(int(token.get("num_edits") or 0) for token in plain_tokens)
        num_blocks = len({int(token["block_index"]) for token in tokens})
        rows.append(
            {
                "sample_id": sample_id,
                "sample_index": detail.get("index"),
                "question": detail.get("question"),
                "gold_answer": detail.get("gold_answer"),
                "predicted_answer": detail.get("predicted_answer"),
                "correct": detail.get("correct"),
                "threshold": detail.get("threshold"),
                "edit_threshold": detail.get("edit_threshold"),
                "num_blocks": num_blocks,
                "total_global_iterations": sample_global_totals.get(sample_id),
                "num_token_events": len(events),
                "num_tokens": len(tokens),
                "num_critical_tokens": len(critical_tokens),
                "num_critical_early_commits": len(critical_early),
                "critical_early_commit_rate": (
                    len(critical_early) / len(critical_tokens) if critical_tokens else None
                ),
                "num_high_confidence_critical_commits": len(critical_high_conf),
                "high_confidence_critical_commit_rate": (
                    len(critical_high_conf) / len(critical_tokens) if critical_tokens else None
                ),
                "num_critical_edits": critical_edits,
                "num_plain_edits": plain_edits,
            }
        )
    return rows


def build_critical_token_stats(
    token_summary: list[dict[str, Any]], details: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    correct_by_sample = {
        str(detail.get("id", detail.get("index"))): detail.get("correct")
        for detail in details
    }
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in token_summary:
        correct = correct_by_sample.get(str(row["sample_id"]))
        correct_label = "correct" if correct is True else "wrong" if correct is False else "unknown"
        groups[(correct_label, str(row["token_type"]))].append(row)

    stats: list[dict[str, Any]] = []
    for (correct_label, token_type), rows in sorted(groups.items()):
        critical_rows = [row for row in rows if row.get("is_critical") is True]
        commit_fractions = [
            float(row["commit_block_fraction"])
            for row in rows
            if row.get("commit_block_fraction") not in (None, "")
        ]
        commit_probs = [
            float(row["commit_prob"])
            for row in rows
            if row.get("commit_prob") not in (None, "")
        ]
        stats.append(
            {
                "correct_group": correct_label,
                "token_type": token_type,
                "num_tokens": len(rows),
                "num_critical_tokens": len(critical_rows),
                "early_commit_rate": ratio(rows, "early_commit_within_block"),
                "high_confidence_commit_rate": ratio(rows, "high_confidence_commit"),
                "avg_commit_block_fraction": mean(commit_fractions) if commit_fractions else None,
                "avg_commit_prob": mean(commit_probs) if commit_probs else None,
                "avg_num_edits": mean([int(row.get("num_edits") or 0) for row in rows]) if rows else None,
            }
        )
    return stats


def ratio(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if row.get(key) is True) / len(rows)


def write_report(
    *,
    paths: AnalysisPaths,
    trace_path: Path,
    details_path: Path,
    model_path: str,
    warnings: list[str],
    sample_summary: list[dict[str, Any]],
    token_summary: list[dict[str, Any]],
    critical_stats: list[dict[str, Any]],
    token_proposals: list[dict[str, Any]],
    edit_proposals: list[dict[str, Any]],
) -> None:
    correct_samples = sum(1 for row in sample_summary if row.get("correct") is True)
    total_samples = len(sample_summary)
    critical_tokens = [row for row in token_summary if row.get("is_critical") is True]
    lines = [
        "# dLLM Critical Token Analysis",
        "",
        f"- Trace path: `{trace_path}`",
        f"- Details path: `{details_path}`",
        f"- Model path: `{model_path}`",
        f"- Samples: `{total_samples}`",
        f"- Correct samples: `{correct_samples}`",
        f"- Token slots: `{len(token_summary)}`",
        f"- Critical token slots: `{len(critical_tokens)}`",
        f"- Position proposals: `{len(token_proposals)}`",
        f"- Proposed edits: `{len(edit_proposals)}`",
        "",
        "## Outputs",
        "",
        f"- `sample_summary.csv`: `{paths.sample_summary_path}`",
        f"- `token_summary.csv`: `{paths.token_summary_path}`",
        f"- `token_events.csv`: `{paths.token_events_path}`",
        f"- `token_proposals.csv`: `{paths.token_proposals_path}`",
        f"- `edit_proposals.csv`: `{paths.edit_proposals_path}`",
        f"- `edit_annotation.md`: `{paths.edit_annotation_path}`",
        f"- `critical_token_stats.csv`: `{paths.critical_token_stats_path}`",
        "",
    ]
    if warnings:
        lines.extend(["## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.extend(
        [
            "## Critical Token Stats",
            "",
            "| Correct Group | Token Type | Tokens | Early Commit Rate | High Confidence Rate | Avg Commit Fraction | Avg Edits |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in critical_stats:
        lines.append(
            "| "
            f"{row['correct_group']} | "
            f"{row['token_type']} | "
            f"{row['num_tokens']} | "
            f"{format_float(row.get('early_commit_rate'))} | "
            f"{format_float(row.get('high_confidence_commit_rate'))} | "
            f"{format_float(row.get('avg_commit_block_fraction'))} | "
            f"{format_float(row.get('avg_num_edits'))} |"
        )
    lines.append("")
    paths.report_path.write_text("\n".join(lines), encoding="utf-8")


def format_float(value: Any) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.4f}"


def write_empty_outputs(paths: AnalysisPaths) -> None:
    write_csv(paths.token_events_path, [], TOKEN_EVENTS_FIELDS)
    write_csv(paths.token_proposals_path, [], TOKEN_PROPOSAL_FIELDS)
    write_csv(paths.edit_proposals_path, [], EDIT_PROPOSAL_FIELDS)
    write_edit_annotation(paths.edit_annotation_path, [])
    write_csv(paths.token_summary_path, [], TOKEN_SUMMARY_FIELDS)
    write_csv(paths.sample_summary_path, [], SAMPLE_SUMMARY_FIELDS)
    write_csv(paths.critical_token_stats_path, [], CRITICAL_STATS_FIELDS)
    paths.report_path.write_text("# dLLM Critical Token Analysis\n\nNo trace events found.\n", encoding="utf-8")


SAMPLE_SUMMARY_FIELDS = [
    "sample_id",
    "sample_index",
    "question",
    "gold_answer",
    "predicted_answer",
    "correct",
    "threshold",
    "edit_threshold",
    "num_blocks",
    "total_global_iterations",
    "num_token_events",
    "num_tokens",
    "num_critical_tokens",
    "num_critical_early_commits",
    "critical_early_commit_rate",
    "num_high_confidence_critical_commits",
    "high_confidence_critical_commit_rate",
    "num_critical_edits",
    "num_plain_edits",
]

TOKEN_SUMMARY_FIELDS = [
    "token_id",
    "sample_id",
    "sample_index",
    "generated_pos",
    "block_index",
    "block_offset",
    "final_token",
    "final_token_id",
    "token_type",
    "is_critical",
    "commit_global_iteration",
    "commit_block_iteration",
    "commit_prob",
    "commit_event_type",
    "last_event_global_iteration",
    "last_event_block_iteration",
    "last_edit_global_iteration",
    "last_edit_block_iteration",
    "num_events",
    "num_edits",
    "num_remasks",
    "was_edited_after_commit",
    "was_remasked",
    "total_block_iterations",
    "commit_block_fraction",
    "early_commit_within_block",
    "late_commit_within_block",
    "high_confidence_commit",
    "final_answer_span",
    "passes_run_threshold",
]

TOKEN_EVENTS_FIELDS = [
    "event_id",
    "token_id",
    "sample_id",
    "sample_index",
    "generated_pos",
    "event_index",
    "global_iteration",
    "block_index",
    "block_iteration",
    "block_offset",
    "event_type",
    "old_token",
    "old_token_id",
    "new_token",
    "new_token_id",
    "prob",
    "token_type",
    "is_critical",
    "is_commit_event",
    "changed_token",
    "accepted_by",
    "A",
    "fallback_rank",
    "passes_run_threshold",
]

TOKEN_PROPOSAL_FIELDS = [
    "proposal_id",
    "request_id",
    "sample_id",
    "sample_index",
    "sample_correct",
    "gold_answer",
    "predicted_answer",
    "threshold",
    "edit_threshold",
    "block_index",
    "block_iteration",
    "global_iteration",
    "block_offset",
    "generated_pos",
    "state_hash",
    "proposal_type",
    "current_token",
    "proposed_token",
    "second_token",
    "current_token_id",
    "proposed_token_id",
    "second_token_id",
    "p_old",
    "p_new",
    "p_second",
    "old_logit",
    "new_logit",
    "second_logit",
    "A",
    "D",
    "replacement_advantage",
    "candidate_margin",
    "entropy",
    "proposed_edit",
    "accepted_edit",
    "accepted_update",
    "edit_selected_by",
    "accepted_by",
    "fallback_rank",
    "rejected_reason",
    "token_age",
    "token_type",
]

MANUAL_ANNOTATION_FIELDS = ["manual_label", "manual_reason", "manual_notes"]
EDIT_PROPOSAL_FIELDS = TOKEN_PROPOSAL_FIELDS + MANUAL_ANNOTATION_FIELDS

CRITICAL_STATS_FIELDS = [
    "correct_group",
    "token_type",
    "num_tokens",
    "num_critical_tokens",
    "early_commit_rate",
    "high_confidence_commit_rate",
    "avg_commit_block_fraction",
    "avg_commit_prob",
    "avg_num_edits",
]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not args.no_timestamp:
        output_dir = timestamped_output_dir(output_dir)
    paths = analyze_trace(
        trace_path=Path(args.trace_path),
        details_path=Path(args.details),
        model_path=args.model_path,
        output_dir=output_dir,
        high_confidence_threshold=args.high_confidence_threshold,
        early_fraction=args.early_fraction,
        late_fraction=args.late_fraction,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Wrote critical-token analysis to {paths.output_dir}")


if __name__ == "__main__":
    main()
