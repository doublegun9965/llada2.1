# llada2.1 experiments

## 中文说明

这个仓库用于做 LLaDA 2.1 实验。主要工作流是：本地改代码并推到 GitHub，服务器上 `git pull` 后运行实验。服务器端推理通过 SGLang 的 OpenAI-compatible API 完成。

### 目录结构

```text
experiments/              实验入口脚本
scripts/                  Git 和服务器辅助脚本
sglang_server/            SGLang 启动脚本和配置文件
src/llada_experiments/    共享 Python 工具
prompts/                  Prompt 示例
outputs/                  实验输出目录，默认不提交 Git
```

### 服务器准备

服务器上拉取最新代码并安装：

```bash
cd /mnt/workspace/llada2.1
git pull --ff-only
pip install -e .
```

GSM8K 数据可以放在任意路径，只要运行时用 `--input-jsonl` 指定即可。推荐路径：

```bash
/mnt/workspace/data/gsm8k_test.jsonl
```

每一行 JSONL 格式：

```json
{"question": "...", "answer": "... #### 42"}
```

### 本地配置文件

服务器相关的配置尽量写在 `.local` 文件里。这些文件会被 Git 忽略，不会影响之后 `git pull`。

创建 SGLang 本地配置：

```bash
cp sglang_server/server_config.local.example.json sglang_server/server_config.local.json
vim sglang_server/server_config.local.json
```

常用字段：

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

创建服务器环境变量文件：

```bash
cp sglang_server/server_env.example sglang_server/server_env.local
vim sglang_server/server_env.local
```

当前服务器上建议包含：

```bash
export SGLANG_DISABLE_VLLM_RMSNORM=1
```

自动 GSM8K sweep 和手动 `bash sglang_server/start_sglang.sh` 都会在启动 SGLang 前加载 `sglang_server/server_env.local`。

### LLaDA 2.1 阈值配置

对 SGLang `0.5.12.post1` 来说，LLaDA 2.1 的 `threshold` 和 `edit_threshold` 是 SGLang 启动时的 DLLM algorithm 配置，不是每个请求里的参数。

手动配置文件：

```bash
cp sglang_server/dllm_algorithm_config.local.example.yaml sglang_server/dllm_algorithm_config.local.yaml
vim sglang_server/dllm_algorithm_config.local.yaml
```

示例：

```yaml
threshold: 0.5
edit_threshold: 0.0
max_post_edit_steps: 16
penalty_lambda: 0
```

修改 `threshold` 或 `edit_threshold` 后必须重启 SGLang。当前 GSM8K sweep 脚本会为每组阈值自动写 YAML、启动 SGLang、评测、停止 SGLang，再进入下一组阈值。

### 自动运行 GSM8K 阈值实验

推荐使用这个方式，不需要你提前手动启动 SGLang：

```bash
python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --thresholds 0.4,0.5,0.6 \
  --edit-thresholds 0.0,0.2,0.4 \
  --max-tokens 512
```

`--thresholds` 和 `--edit-thresholds` 是排列组合，不是一一对应。

生成长度在这里改：

```bash
--max-tokens 512
```

调试时先用小样本：

```bash
python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 5 \
  --thresholds 0.5 \
  --edit-thresholds 0.0 \
  --max-tokens 256
```

每次运行都会写入新的时间戳目录，避免覆盖旧结果：

```text
outputs/gsm8k/run_<timestamp>/
  summary.csv
  summary.json
  details_threshold_<value>_edit_<value>.jsonl
  dllm_configs/
  server_logs/
```

### 正确答案前缀实验

默认 prompt 只包含 GSM8K 问题。如果想测试“给模型一段正确推理开头是否能提升表现”，加：

```bash
--gold-prefix-tokens 40
```

完整例子：

```bash
python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --thresholds 0.5 \
  --edit-thresholds 0.0 \
  --max-tokens 512 \
  --gold-prefix-tokens 40
```

默认 `--gold-prefix-style instructed` 会加显式续写指令。如果想最直接地拼接：

```bash
--gold-prefix-style direct
```

`direct` 实际发送：

```text
{question}
{gold answer 的前 40 个按空白分割的 token}
```

### 加噪完整 Ground Truth 实验

如果想把完整 ground truth 直接拼到 question 后面，但按比例把其中一部分 token 替换成噪声，使用：

```bash
--gold-noise-ratio 0.3
```

完整例子：

```bash
python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --thresholds 0.5 \
  --edit-thresholds 0.0 \
  --max-tokens 512 \
  --gold-noise-ratio 0.3
```

实际发送：

```text
{question}
{完整 gold answer，其中 30% 按空白分割的 token 被替换成 [MASK]}
```

常用参数：

- `--gold-noise-ratio 0.0`：拼接干净完整 ground truth。
- `--gold-noise-ratio 1.0`：把完整 ground truth 的所有 token 都替换。
- `--gold-noise-token "[MASK]"`：修改替换用的噪声 token。
- `--gold-noise-seed 0`：控制噪声位置，保证可复现。
- `--gold-noise-ratio` 和 `--gold-prefix-tokens` 互斥，不能同时开。

### 手动启动 SGLang

只有你想自己控制 SGLang 进程时才用这个模式。

1. 修改服务器配置：

```bash
vim sglang_server/server_config.local.json
```

2. 修改阈值配置：

```bash
vim sglang_server/dllm_algorithm_config.local.yaml
```

3. 启动 SGLang：

```bash
bash sglang_server/start_sglang.sh
```

4. 另开终端检查服务：

```bash
bash sglang_server/check_sglang.sh
```

5. 对已启动服务跑一次评测：

```bash
python experiments/gsm8k_threshold_sweep.py \
  --use-running-server \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --thresholds 0.5 \
  --edit-thresholds 0.0 \
  --max-tokens 512
```

`--use-running-server` 只能评测一组阈值，因为阈值是在 SGLang 启动时固定的。

### Smoke Test

SGLang 已启动后可以跑：

```bash
python experiments/smoke_test.py --prompt "say hello"
```

如果服务没启动，会报 `Connection refused`。

### 常见问题

#### `RemoteProtocolError: Server disconnected without sending a response`

这通常表示 SGLang 收到了请求，但生成过程中服务端进程挂了或断开了。常见原因：

- SGLang worker 崩溃。
- GPU OOM 或显存压力太大。
- `--max-tokens` 太大。
- 端口上连到的是旧的 SGLang 进程。
- 模型路径或 DLLM 配置不兼容。
- 缺少 `SGLANG_DISABLE_VLLM_RMSNORM=1`。

先看本次运行目录里的日志：

```bash
tail -n 200 outputs/gsm8k/run_<timestamp>/server_logs/<log-file>.log
```

如果日志里有：

```text
TypeError: fused_add_rms_norm() takes 4 positional arguments but 6 were given
```

检查：

```bash
cat sglang_server/server_env.local
```

应该包含：

```bash
export SGLANG_DISABLE_VLLM_RMSNORM=1
```

调试时先缩小任务：

```bash
python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 1 \
  --thresholds 0.5 \
  --edit-thresholds 0.0 \
  --max-tokens 128
```

#### `Connection refused`

说明当前端口没有 SGLang 服务。可以直接跑自动 sweep，或者手动启动：

```bash
bash sglang_server/start_sglang.sh
```

#### Hugging Face 数据集下载失败

优先使用本地 JSONL：

```bash
--input-jsonl /mnt/workspace/data/gsm8k_test.jsonl
```

### Git 工作流

本地改完后：

```powershell
.\scripts\sync.ps1 "describe the change"
```

服务器上：

```bash
cd /mnt/workspace/llada2.1
git pull --ff-only
pip install -e .
```

---

## English

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

### Correct Context Prefix Experiment

By default, the prompt only contains the GSM8K question. To test whether a correct reasoning prefix improves generation, add `--gold-prefix-tokens`.

Example using the first 40 whitespace-separated tokens from the gold solution:

```bash
python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --thresholds 0.5 \
  --edit-thresholds 0.0 \
  --max-tokens 512 \
  --gold-prefix-tokens 40
```

By default this uses `--gold-prefix-style instructed`, which wraps the prefix with explicit continuation instructions.

For the minimal prompt style, use `direct`:

```bash
python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --thresholds 0.5 \
  --edit-thresholds 0.0 \
  --max-tokens 512 \
  --gold-prefix-tokens 40 \
  --gold-prefix-style direct
```

`direct` sends:

```text
{question}
{first 40 whitespace-separated tokens from gold answer}
```

Notes:

- `--gold-prefix-tokens 0` is the default and disables this feature.
- `--gold-prefix-style instructed` keeps the more explicit prompt; `--gold-prefix-style direct` only appends the gold prefix after the question.
- The script uses whitespace-separated tokens, not the model tokenizer, to avoid adding tokenizer dependencies to the evaluation path.
- Per-example details include `gold_solution`, `gold_prefix_style`, `gold_prefix`, and the final `prompt` sent to SGLang.

### Noisy Ground Truth Context Experiment

To directly append the full ground truth after the question, with a fixed fraction of its whitespace-separated tokens replaced by noise, use `--gold-noise-ratio`.

Example replacing 30% of ground-truth tokens with `[MASK]`:

```bash
python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --thresholds 0.5 \
  --edit-thresholds 0.0 \
  --max-tokens 512 \
  --gold-noise-ratio 0.3
```

This sends:

```text
{question}
{full gold answer with 30% of whitespace-separated tokens replaced by [MASK]}
```

Options:

- `--gold-noise-ratio 0.0` appends the clean full ground truth.
- `--gold-noise-ratio 1.0` masks every whitespace-separated ground-truth token.
- `--gold-noise-token "[MASK]"` changes the replacement token.
- `--gold-noise-seed 0` controls which token positions are noised, so runs are reproducible.
- `--gold-noise-ratio` and `--gold-prefix-tokens` are mutually exclusive.
- Per-example details include `gold_noised_solution`, `gold_noise_indices`, `gold_noise_ratio`, and the final `prompt`.

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
