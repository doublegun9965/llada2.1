# SGLang server

This folder contains SGLang server launch config and DLLM algorithm config for LLaDA2.1.

## 1. Edit server config

Edit a local server config on the server:

```bash
cp sglang_server/server_config.local.example.json sglang_server/server_config.local.json
vim sglang_server/server_config.local.json
```

It has the same shape as `sglang_server/server_config.json`:

```json
{
  "model_path": "/mt/workspace/models/LLaDA2.1-Mini",
  "served_model_name": "llada2.1",
  "host": "0.0.0.0",
  "port": 30000,
  "tensor_parallel_size": 1,
  "trust_remote_code": true,
  "dtype": "auto",
  "context_length": null,
  "mem_fraction_static": null,
  "dllm_algorithm": "JointThreshold",
  "dllm_algorithm_config": "sglang_server/dllm_algorithm_config.local.yaml",
  "extra_launch_args": []
}
```

`server_config.local.json` is ignored by Git. `start_sglang.sh` will prefer the local file when it exists, otherwise it falls back to `server_config.json`.

## 2. Start SGLang

For LLaDA2.1 on SGLang 0.5.12.post1, thresholds are read at server startup from a YAML file passed through `--dllm-algorithm-config`.

Create a local DLLM algorithm config:

```bash
cp sglang_server/dllm_algorithm_config.local.example.yaml sglang_server/dllm_algorithm_config.local.yaml
vim sglang_server/dllm_algorithm_config.local.yaml
```

Example:

```yaml
threshold: 0.5
edit_threshold: 0.0
max_post_edit_steps: 16
penalty_lambda: 0
```

```bash
cd /mt/workspace/llada2.1
bash sglang_server/start_sglang.sh
```

The script runs:

```bash
python -m sglang.launch_server ...
```

Logs are written to `logs/sglang_server.log`.

Check whether the OpenAI-compatible API is alive:

```bash
bash sglang_server/check_sglang.sh
```

## 3. Request extra body config

`generation_config.json` is only for request-level extra body values. Do not put LLaDA2.1 `threshold` or `edit_threshold` here; SGLang 0.5.12.post1 reads those from `--dllm-algorithm-config` at server startup.

Edit a local request extra body config only when you need non-DLLM request fields:

```bash
cp sglang_server/generation_config.local.example.json sglang_server/generation_config.local.json
vim sglang_server/generation_config.local.json
```

It has the same shape as `sglang_server/generation_config.json`:

```json
{
  "extra_body": {}
}
```

Run a test with the config:

```bash
python experiments/smoke_test.py \
  --prompt "say hello from the latest commit"
```

`smoke_test.py` uses `generation_config.local.json` automatically when it exists. Pass `--generation-config <path>` only when you want to test a specific config file.
