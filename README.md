# llada2.1 experiments

This repository contains LLaDA 2.1 experiments. Code is developed locally, pushed to GitHub, pulled on the server, and run against an SGLang OpenAI-compatible server.

## Layout

```text
experiments/              Experiment entry points
scripts/                  Git and server helper scripts
sglang_server/            SGLang launch/config files
src/llada_experiments/    Shared Python utilities
prompts/                  Prompt examples
outputs/                  Generated results, ignored by Git
```

## Server Setup

Pull the latest code and install dependencies:

```bash
cd /mnt/workspace/llada2.1
git pull --ff-only
pip install -e .
```

If you use a local GSM8K JSONL file, put it anywhere and pass the path with `--input-jsonl`. Recommended path:

```bash
/mnt/workspace/data/gsm8k_test.jsonl
```

Each JSONL line should contain:

```json
{"question": "...", "answer": "... #### 42"}
```

## Config Files

Use local config files for server-specific settings. Local config files are ignored by Git and will not block `git pull`.

### SGLang Server Config

Create a server-local config:

```bash
cp sglang_server/server_config.local.example.json sglang_server/server_config.local.json
vim sglang_server/server_config.local.json
```

Important fields:

```json
{
  "model_path": "/mnt/workspace/models/LLaDA2.1-Mini",
  "served_model_name": "llada2.1",
  "host": "0.0.0.0",
  "port": 30000,
  "tensor_parallel_size": 1,
  "mem_fraction_static": null,
  "dllm_algorithm": "JointThreshold",
  "dllm_algorithm_config": "sglang_server/dllm_algorithm_config.local.yaml"
}
```

### SGLang Environment Variables

Create a server-local env file:

```bash
cp sglang_server/server_env.example sglang_server/server_env.local
vim sglang_server/server_env.local
```

Current recommended content for the ROCm/SGLang RMSNorm issue:

```bash
export SGLANG_DISABLE_VLLM_RMSNORM=1
```

Both automatic GSM8K sweep and manual `bash sglang_server/start_sglang.sh` load `sglang_server/server_env.local` before starting SGLang.

### LLaDA 2.1 Threshold Config

For SGLang `0.5.12.post1`, LLaDA 2.1 thresholds are startup-time DLLM algorithm settings. They are not request-body parameters.

Manual config file:

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

Changing `threshold` or `edit_threshold` requires restarting SGLang.

## Run GSM8K Sweep Automatically

This is the recommended path. Do not start SGLang manually first.

```bash
python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --thresholds 0.4,0.5,0.6 \
  --edit-thresholds 0.0,0.2,0.4 \
  --max-tokens 512
```

The script does this for each threshold pair:

```text
write DLLM YAML config
start SGLang
wait for /v1/models
run GSM8K
stop SGLang
move to next threshold pair
```

Set generation length with:

```bash
--max-tokens 512
```

Use smaller values for faster debugging:

```bash
python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 5 \
  --thresholds 0.5 \
  --edit-thresholds 0.0 \
  --max-tokens 256
```

Outputs are written to a fresh timestamped directory:

```text
outputs/gsm8k/run_<timestamp>/
  summary.csv
  summary.json
  details_threshold_<value>_edit_<value>.jsonl
  dllm_configs/
  server_logs/
```

## Manual SGLang Mode

Use this only when you want to start SGLang yourself.

1. Edit server config:

```bash
vim sglang_server/server_config.local.json
```

2. Edit threshold config:

```bash
vim sglang_server/dllm_algorithm_config.local.yaml
```

3. Start SGLang:

```bash
bash sglang_server/start_sglang.sh
```

4. In another terminal, check the server:

```bash
bash sglang_server/check_sglang.sh
```

5. Run one evaluation against the already-running server:

```bash
python experiments/gsm8k_threshold_sweep.py \
  --use-running-server \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --thresholds 0.5 \
  --edit-thresholds 0.0 \
  --max-tokens 512
```

`--use-running-server` can only evaluate one threshold pair, because thresholds are fixed when SGLang starts.

## Smoke Test

After SGLang is running:

```bash
python experiments/smoke_test.py --prompt "say hello"
```

If the server is not running, this will fail with `Connection refused`.

## Troubleshooting

### `RemoteProtocolError: Server disconnected without sending a response`

This usually means SGLang accepted the connection but closed it before returning a response. Common causes:

- SGLang worker crashed during generation.
- GPU OOM or memory pressure.
- The output length is too large for the current memory settings.
- The port is connected to an old/stale SGLang process.
- The model path or DLLM config is incompatible with the server.
- `SGLANG_DISABLE_VLLM_RMSNORM=1` is missing, which can trigger `TypeError: fused_add_rms_norm() takes 4 positional arguments but 6 were given` on this server setup.

Check the log printed by the sweep script, for example:

```text
outputs/gsm8k/run_<timestamp>/server_logs/threshold_0p5_edit_0p0.log
```

Useful commands:

```bash
tail -n 200 outputs/gsm8k/run_<timestamp>/server_logs/<log-file>.log
nvidia-smi
curl http://127.0.0.1:30000/v1/models
```

If the log contains:

```text
TypeError: fused_add_rms_norm() takes 4 positional arguments but 6 were given
```

create or check:

```bash
cat sglang_server/server_env.local
```

It should include:

```bash
export SGLANG_DISABLE_VLLM_RMSNORM=1
```

For debugging, reduce workload first:

```bash
python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 1 \
  --thresholds 0.5 \
  --edit-thresholds 0.0 \
  --max-tokens 128
```

### `Connection refused`

No SGLang server is listening on the configured port. Either run the automatic sweep script, or start SGLang manually with:

```bash
bash sglang_server/start_sglang.sh
```

### Hugging Face dataset download errors

Use local JSONL instead of downloading on the server:

```bash
--input-jsonl /mnt/workspace/data/gsm8k_test.jsonl
```

## Git Workflow

After local edits:

```powershell
.\scripts\sync.ps1 "describe the change"
```

On the server:

```bash
cd /mnt/workspace/llada2.1
git pull --ff-only
pip install -e .
```
