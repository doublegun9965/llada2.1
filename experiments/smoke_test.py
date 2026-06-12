from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llada_experiments import SGLangClient, load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal SGLang smoke test.")
    parser.add_argument("--prompt", default=None, help="Single prompt to run.")
    parser.add_argument("--input-jsonl", default=None, help="JSONL file with id and prompt fields.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", default=None, help="Output JSONL path.")
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.prompt:
        return [{"id": "cli_prompt", "prompt": args.prompt}]

    input_jsonl = Path(args.input_jsonl or "prompts/smoke.jsonl")
    rows: list[dict[str, str]] = []
    with input_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append({"id": str(row["id"]), "prompt": str(row["prompt"])})
    return rows


def main() -> None:
    args = parse_args()
    settings = load_settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output) if args.output else settings.output_dir / "smoke_results.jsonl"
    client = SGLangClient(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout_seconds=settings.timeout_seconds,
    )

    prompts = load_prompts(args)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in prompts:
            result = client.chat_completion(
                model=settings.model,
                prompt=item["prompt"],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            record = {
                "id": item["id"],
                "prompt": item["prompt"],
                "completion": result.text,
                "model": settings.model,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{item['id']}] {result.text}")

    print(f"Wrote {len(prompts)} result(s) to {output_path}")


if __name__ == "__main__":
    main()
