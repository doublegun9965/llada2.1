# SGLang server

This folder contains the server launch config and the LLaDA2.1 request-time decoding config.

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
  "extra_launch_args": []
}
```

`server_config.local.json` is ignored by Git. `start_sglang.sh` will prefer the local file when it exists, otherwise it falls back to `server_config.json`.

## 2. Start SGLang

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

## 3. Edit LLaDA2.1 decoding config

Edit a local decoding config:

```bash
cp sglang_server/generation_config.local.example.json sglang_server/generation_config.local.json
vim sglang_server/generation_config.local.json
```

It has the same shape as `sglang_server/generation_config.json`:

```json
{
  "extra_body": {
    "confidence_threshold": 0.85,
    "edit_threshold": 0.85
  }
}
```

These fields are sent directly in the JSON body of `/v1/chat/completions`. If your SGLang LLaDA2.1 branch uses different names, change the keys in `generation_config.local.json`. For example, if the branch uses the misspelled names `confidence_thresold` and `edit_thresold`, put those exact keys in `extra_body`.

Run a test with the config:

```bash
python experiments/smoke_test.py \
  --prompt "say hello from the latest commit"
```

`smoke_test.py` uses `generation_config.local.json` automatically when it exists. Pass `--generation-config <path>` only when you want to test a specific config file.
