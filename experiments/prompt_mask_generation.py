from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from gsm8k_mask_reconstruct import load_model_and_tokenizer, timestamped_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local LLaDA2.1 generation with optional mask tokens added before or "
            "after a command-line prompt."
        )
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Prompt text to send to the model.")
    prompt_group.add_argument("--prompt-file", help="UTF-8 text file containing the prompt.")
    parser.add_argument("--model-path", default="model/llada2.1")
    parser.add_argument("--output-dir", default="outputs/prompt_mask_generation")
    parser.add_argument("--mask-count", type=int, default=0)
    parser.add_argument("--mask-position", choices=["head", "tail"], default="tail")
    parser.add_argument(
        "--mask-separator",
        default=" ",
        help="String used between repeated mask tokens and between masks and prompt.",
    )
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--editing-threshold", type=float, default=0.0)
    parser.add_argument("--max-post-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--num-to-transfer", type=int, default=1)
    parser.add_argument("--minimal-topk", type=int, default=1)
    parser.add_argument("--no-eos-early-stop", action="store_true")
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Tokenize the masked prompt directly instead of using tokenizer.apply_chat_template.",
    )
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map. Use 'none' to call model.to(--device) instead.",
    )
    parser.add_argument("--device", default=None, help="Used only when --device-map none.")
    return parser.parse_args()


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    return Path(args.prompt_file).read_text(encoding="utf-8")


def with_added_masks(
    *,
    prompt: str,
    mask_token: str,
    mask_count: int,
    mask_position: str,
    separator: str,
) -> str:
    if mask_count < 0:
        raise ValueError("--mask-count must be non-negative")
    if mask_count == 0:
        return prompt

    masks = separator.join([mask_token] * mask_count)
    if mask_position == "head":
        return f"{masks}{separator}{prompt}"
    if mask_position == "tail":
        return f"{prompt}{separator}{masks}"
    raise ValueError(f"Unsupported --mask-position: {mask_position}")


def build_input_ids(tokenizer: Any, prompt: str, use_chat_template: bool) -> Any:
    if use_chat_template:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if hasattr(encoded, "shape"):
            return encoded
        try:
            return encoded["input_ids"]
        except (KeyError, TypeError):
            pass
        raise TypeError(f"Unexpected apply_chat_template return type: {type(encoded)!r}")
    return tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]


def main() -> None:
    args = parse_args()
    run_dir = timestamped_run_dir(Path(args.output_dir))
    result_json_path = run_dir / "result.json"
    result_txt_path = run_dir / "result.txt"

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

    started = time.perf_counter()
    generated_ids = model.generate(
        inputs=input_ids,
        eos_early_stop=not args.no_eos_early_stop,
        gen_length=args.gen_length,
        block_length=args.block_length,
        threshold=args.threshold,
        editing_threshold=args.editing_threshold,
        max_post_steps=args.max_post_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        minimal_topk=args.minimal_topk,
        num_to_transfer=args.num_to_transfer,
        mask_id=int(tokenizer.mask_token_id),
        eos_id=int(tokenizer.eos_token_id or tokenizer.pad_token_id),
    )
    latency = time.perf_counter() - started

    generated_continuation_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    generated_continuation_text_with_special_tokens = tokenizer.decode(
        generated_ids[0], skip_special_tokens=False
    )
    model_input_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)

    result = {
        "model_path": args.model_path,
        "prompt": prompt,
        "masked_prompt": masked_prompt,
        "mask_token": tokenizer.mask_token,
        "mask_token_id": int(tokenizer.mask_token_id),
        "mask_count": args.mask_count,
        "mask_position": args.mask_position,
        "mask_separator": args.mask_separator,
        "use_chat_template": not args.no_chat_template,
        "model_input_text": model_input_text,
        "input_tokens": int(input_ids.shape[1]),
        "generated_token_count": int(generated_ids.shape[1]),
        "generated_tokens": int(generated_ids.shape[1]),
        "generated_continuation_text": generated_continuation_text,
        "generated_continuation_text_with_special_tokens": (
            generated_continuation_text_with_special_tokens
        ),
        "generated_text": generated_continuation_text,
        "latency_seconds": latency,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "threshold": args.threshold,
        "editing_threshold": args.editing_threshold,
        "max_post_steps": args.max_post_steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_to_transfer": args.num_to_transfer,
        "minimal_topk": args.minimal_topk,
    }

    result_json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result_txt_path.write_text(
        "\n".join(
            [
                "# Prompt Mask Generation",
                "",
                "## Prompt",
                prompt,
                "",
                "## Masked Prompt",
                masked_prompt,
                "",
                "## Generated Text",
                generated_continuation_text,
                "",
                "## Generated Text With Special Tokens",
                generated_continuation_text_with_special_tokens,
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Results written to {run_dir}")


if __name__ == "__main__":
    main()
