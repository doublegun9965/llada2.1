# Project Instructions for Codex Agents

This repository is for LLaDA 2.1 experiments that are developed locally but run on a remote server.

## Operating Model

- The user edits and commits code locally, pushes to GitHub, then pulls and runs it on the server.
- After making useful code changes, commit and push them to GitHub unless the user says not to.
- Do not assume experiments can run locally. Local verification should focus on syntax, CLI help, config parsing, and small unit-style checks.
- Server commands in README/scripts should be copy-paste friendly for Linux shell usage.

## Server-Local Configuration

- Prefer committed example/default config files plus ignored local override files.
- Do not require the user to edit tracked config files for server-specific values.
- Use this pattern whenever practical:
  - `*.json` or `*.example.json`: tracked defaults/templates.
  - `*.local.json`: user/server-specific config, ignored by Git.
- Existing examples:
  - `sglang_server/server_config.json`
  - `sglang_server/server_config.local.example.json`
  - `sglang_server/server_config.local.json`
  - `sglang_server/generation_config.json`
  - `sglang_server/generation_config.local.example.json`
  - `sglang_server/generation_config.local.json`
- Scripts should prefer `.local` files when present, then fall back to tracked defaults.
- If adding environment variables for server runtime, prefer an ignored local env file plus a tracked example file instead of hardcoding server-only exports in tracked scripts.
- Existing SGLang runtime env pattern:
  - `sglang_server/server_env.example`: tracked template.
  - `sglang_server/server_env.local`: ignored server-local env file loaded before SGLang starts.
  - Current useful env var: `export SGLANG_DISABLE_VLLM_RMSNORM=1`.

## SGLang and LLaDA 2.1

- Inference is expected to go through an SGLang OpenAI-compatible API.
- SGLang chat/completion treats the prompt as fixed context and continues after it; it does not replace mask tokens inside the prompt.
- For experiments that require in-place reconstruction of masked ground-truth/reasoning tokens, use a local model forward/denoising script instead of SGLang.
- `experiments/gsm8k_mask_reconstruct.py` defaults to strict reconstruction: fill masked tokens only. Passing `--editing-threshold` enables reconstruction + editing of non-mask tokens inside the gold solution span.
- For SGLang `0.5.12.post1`, LLaDA 2.1 `JointThreshold` parameters are server-startup DLLM algorithm config values, not OpenAI request-body values.
- Threshold sweeps must generate a YAML file and restart SGLang for each threshold pair using `--dllm-algorithm JointThreshold --dllm-algorithm-config <path>`.
- Current SGLang `JointThreshold` YAML keys are:
  - `threshold`
  - `edit_threshold`
- Do not put LLaDA 2.1 threshold values in `generation_config.json`; that file is only for request-level extra body values.
- SGLang source checkouts are local-only and ignored by Git. Current expected paths:
  - local workspace: `third_party/sglang-v0.5.12.post1`
  - server workspace: `/mnt/workspace/third_party/sglang-v0.5.12.post1`
- If modifying SGLang for experiments, commit patch files under `sglang_patches/`, not the SGLang source tree.
- Use `scripts/save_sglang_patch.sh <sglang-src-dir> <name.patch>` to export source changes and `scripts/apply_sglang_patches.sh /mnt/workspace/third_party/sglang-v0.5.12.post1` to apply committed patches on the server.
- `sglang_patches/deterministic_dllm_compat.patch` is the current SGLang `0.5.12.post1` + LLaDA 2.1 dLLM deterministic compatibility patch.
- After applying that patch, `sglang_server/server_config.local.json` can set `"enable_deterministic_inference": true`; `launch_sglang.py` will pass `--enable-deterministic-inference`.
- `sglang_patches/dllm_trace.patch` records server-side JointThreshold M2T/T2T trace events when `trace_path` is set in the DLLM YAML config.
- `scripts/apply_sglang_patches.sh <sglang-src-dir> <patch-name.patch>` applies only selected patches; without patch names it applies all patches.
- Use `scripts/render_sglang_dllm_trace.py <trace.jsonl> --model-path <model>` to render SGLang dLLM trace JSONL into Markdown.

## Experiment Outputs

- Write experiment results under `outputs/` by default.
- Every experiment run must write results into a fresh timestamped run directory to avoid overwriting previous runs. Prefer names like `outputs/<experiment>/run_YYYYmmdd_HHMMSS/`.
- If a script accepts `--output-dir`, treat it as the base directory and create a timestamped run subdirectory under it unless the user explicitly asks for a fixed output path.
- Do not commit generated outputs, logs, datasets, model weights, or server-specific local config.
- For sweeps, write both machine-readable summaries and per-example details when useful:
  - `summary.csv`
  - `summary.json`
  - `details_*.jsonl`

## Experiment Command Quick Reference

Fixed-threshold local GSM8K baseline for comparing later dynamic/adaptive strategies:

```bash
python experiments/gsm8k_local_baseline.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --limit 100 \
  --threshold 0.5 \
  --editing-threshold off \
  --gen-length 128 \
  --block-length 32 \
  --trace-limit 2
```

SGLang GSM8K threshold sweep with four concurrent requests per threshold pair:

```bash
python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --thresholds 0.4,0.5,0.6 \
  --edit-thresholds 0.0,0.2,0.4 \
  --max-tokens 512 \
  --batch-size 4
```

SGLang GSM8K assistant-prefill experiment, where the gold solution prefix is placed after the assistant generation header and the request uses `/v1/completions`:

```bash
python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --thresholds 0.5 \
  --edit-thresholds 0.0 \
  --max-tokens 512 \
  --assistant-prefill-tokens 10 \
  --batch-size 4
```

Render SGLang dLLM trace JSONL:

```bash
python scripts/render_sglang_dllm_trace.py \
  outputs/sglang_dllm_trace/trace.jsonl \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --trust-remote-code
```

## Coding Conventions

- Keep scripts runnable directly from a fresh Git checkout without requiring editable install when practical. Add `src/` to `sys.path` in top-level experiment scripts if needed.
- Favor explicit CLI arguments and JSON config files over hardcoded constants.
- When adding a new experiment script, also add a copy-paste-friendly server command to the README experiment command quick reference.
- Use ASCII in code and docs unless there is a clear reason otherwise.
- Keep changes narrowly scoped to the experiment or workflow being requested.
