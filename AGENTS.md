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

## Experiment Outputs

- Write experiment results under `outputs/` by default.
- Every experiment run must write results into a fresh timestamped run directory to avoid overwriting previous runs. Prefer names like `outputs/<experiment>/run_YYYYmmdd_HHMMSS/`.
- If a script accepts `--output-dir`, treat it as the base directory and create a timestamped run subdirectory under it unless the user explicitly asks for a fixed output path.
- Do not commit generated outputs, logs, datasets, model weights, or server-specific local config.
- For sweeps, write both machine-readable summaries and per-example details when useful:
  - `summary.csv`
  - `summary.json`
  - `details_*.jsonl`

## Coding Conventions

- Keep scripts runnable directly from a fresh Git checkout without requiring editable install when practical. Add `src/` to `sys.path` in top-level experiment scripts if needed.
- Favor explicit CLI arguments and JSON config files over hardcoded constants.
- Use ASCII in code and docs unless there is a clear reason otherwise.
- Keep changes narrowly scoped to the experiment or workflow being requested.
