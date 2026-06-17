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

### 实验命令速查

自动启动 SGLang，跑 GSM8K 阈值 sweep：

```bash
python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --thresholds 0.4,0.5,0.6 \
  --edit-thresholds 0.0,0.2,0.4 \
  --max-tokens 512
```

SGLang 续写模式，给正确答案前缀：

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

SGLang 续写模式，拼接加噪 ground truth：

```bash
python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --thresholds 0.5 \
  --edit-thresholds 0.0 \
  --max-tokens 512 \
  --gold-noise-ratio 0.3
```

本地模型 strict mask reconstruction，只填 `<|mask|>`：

```bash
python experiments/gsm8k_mask_reconstruct.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --limit 10 \
  --mask-ratio 0.3 \
  --threshold 0.5 \
  --block-length 32
```

本地模型 reconstruction + editing，填 mask 并允许高置信度编辑非 mask token：

```bash
python experiments/gsm8k_mask_reconstruct.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --limit 10 \
  --mask-ratio 0.3 \
  --threshold 0.5 \
  --editing-threshold 0.8 \
  --max-edit-steps 16
```

本地模型 prompt mask generation，在 prompt 尾部追加若干 mask 后生成：

```bash
python experiments/prompt_mask_generation.py \
  --prompt "Solve 16 - 3 - 4, then multiply the result by 2." \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --mask-count 8 \
  --mask-position tail \
  --gen-length 256 \
  --threshold 0.5 \
  --editing-threshold 0.0
```

SGLang GSM8K prompt mask 对比实验，固定 `threshold=0.5`、`edit_threshold=0.0`，默认比较不加 mask、头部 8/4 个 mask、尾部 4/8 个 mask：

```bash
python experiments/gsm8k_prompt_mask_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --threshold 0.5 \
  --edit-threshold 0.0 \
  --max-tokens 512
```

对比多份 `details_*.jsonl`，只输出至少有一个实验答错或缺失的题目，生成方便阅读的 Markdown：

```bash
python scripts/compare_gsm8k_details.py \
  outputs/gsm8k_prompt_mask/run_<timestamp>/details_no_mask.jsonl \
  outputs/gsm8k_prompt_mask/run_<timestamp>/details_head_4.jsonl \
  outputs/gsm8k_prompt_mask/run_<timestamp>/details_tail_4.jsonl \
  --label-from variant
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
{完整 gold answer，其中 30% 按空白分割的 token 被替换成 [MASK]，并且最终答案数字也会强制替换成 [MASK]}
```

注意：`--gold-noise-ratio` 控制的是随机加噪比例。为了避免模型直接抄最终答案，GSM8K 里的最后一行 `#### <number>` 中的 `<number>` 会额外强制 mask，即使它没有被随机采中。

常用参数：

- `--gold-noise-ratio 0.0`：不做随机加噪，但最终答案数字仍然会被强制替换。
- `--gold-noise-ratio 1.0`：把完整 ground truth 的所有 token 都替换。
- `--gold-noise-token "[MASK]"`：修改替换用的噪声 token。
- `--gold-noise-seed 0`：控制噪声位置，保证可复现。
- `--gold-noise-ratio` 和 `--gold-prefix-tokens` 互斥，不能同时开。
- 每条样本的 details 会记录 `gold_noised_solution`、`gold_noise_indices`、`gold_final_answer_noise_indices` 和最终发送给 SGLang 的 `prompt`。

注意：这个实验仍然走 SGLang chat/completion，所以模型只会在 prompt 后面续写，不会原地替换 prompt 里的 `[MASK]`。如果要真正测试“模型能否还原 ground truth 内部的 mask”，请使用下面的本地模型重建实验。

### 本地模型 Mask 重建实验

这个实验不走 SGLang，而是直接加载 `model/llada2.1` 里的模型代码和权重，对 ground truth solution 内部的 `<|mask|>` token 做原地重建。

运行环境需要能导入 `torch`、`transformers`、`accelerate` 和 `safetensors`；如果当前 SGLang/模型环境已经包含这些依赖，可以直接复用那个环境。

运行示例：

```bash
python experiments/gsm8k_mask_reconstruct.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --limit 10 \
  --mask-ratio 0.3 \
  --threshold 0.5 \
  --block-length 32
```

它会构造：

```text
Problem:
{question}

Ground truth solution:
{部分 token 被替换成 <|mask|> 的 gold answer}
```

然后只预测被 mask 的 token，默认不会改动未 mask 的 token。`--mask-ratio` 控制 ground truth solution 中随机 mask 的 tokenizer token 比例。默认会额外强制 mask GSM8K 最后一行 `#### <number>` 里的 `<number>`，避免最终答案直接泄露。

常用参数：

- `--mask-ratio 0.0`：不做随机 mask，但最终答案数字仍然会被强制 mask。
- `--mask-ratio 1.0`：mask 整个 ground truth solution。
- `--no-force-final-answer-mask`：关闭最终答案强制 mask。
- `--threshold 0.5`：控制 mask token 被接受的置信度阈值。
- `--editing-threshold 0.8`：开启 reconstruction + editing，允许高置信度改写 ground truth solution 里未 mask 的 token；不传时只填 mask，不编辑其它 token。
- `--max-edit-steps 16`：所有 mask 填完后，最多继续做多少轮 editing refinement。
- `--num-to-transfer 1`：每轮至少填入多少个 mask token。
- `--attention-mode full`：默认模式，让模型看完整 corrupted 文本来预测 mask；`block-causal` 可用于对照自带 generate 的 block 逻辑。
- `--device-map auto`：默认按 Transformers 的 device map 加载模型。
- `--device-map none --device cuda`：不用 device map，直接把模型放到指定设备。

默认模式是 strict reconstruction：只填 `<|mask|>`，不改其它 token。要跑更接近 LLaDA2.1 原始 `generate()` 的 reconstruction + editing：

```bash
python experiments/gsm8k_mask_reconstruct.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --limit 10 \
  --mask-ratio 0.3 \
  --threshold 0.5 \
  --editing-threshold 0.8 \
  --max-edit-steps 16
```

这里两个阈值含义不同：

- `--threshold`：控制 `<|mask|>` 位置的预测 token 何时被接受。
- `--editing-threshold`：控制非 mask token 是否允许被模型改写。脚本只允许编辑 `Ground truth solution` 区间，不会改 question/prompt 前缀。

输出目录同样是时间戳目录：

```text
outputs/gsm8k_mask_reconstruct/run_<timestamp>/
  summary.json
  summary.csv
  details.jsonl
  details_pretty.md
```

`details_pretty.md` 更适合人工查看；`details.jsonl` 适合后续统计。主要指标：

- `mask_token_accuracy`：被 mask 的 token 有多少被还原对。
- `exact_reconstruction_rate`：整条样本所有 mask token 是否全部还原对。
- `final_answer_accuracy`：重建后的 solution 中 `#### <number>` 是否正确。

### Prompt 头尾 Mask 生成实验

这个实验不走 SGLang，直接用本地 LLaDA2.1 `generate()`。它会把命令行给出的 prompt 加上若干 `<|mask|>`，然后观察这些 mask 对后续生成的影响。

尾部加 8 个 mask：

```bash
python experiments/prompt_mask_generation.py \
  --prompt "Solve 16 - 3 - 4, then multiply the result by 2." \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --mask-count 8 \
  --mask-position tail \
  --gen-length 256 \
  --threshold 0.5 \
  --editing-threshold 0.0
```

头部加 8 个 mask：

```bash
python experiments/prompt_mask_generation.py \
  --prompt "Solve 16 - 3 - 4, then multiply the result by 2." \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --mask-count 8 \
  --mask-position head \
  --gen-length 256 \
  --threshold 0.5 \
  --editing-threshold 0.0
```

长 prompt 可以放文件里：

```bash
python experiments/prompt_mask_generation.py \
  --prompt-file /mnt/workspace/data/my_prompt.txt \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --mask-count 16 \
  --mask-position tail
```

常用参数：

- `--mask-count 0`：默认不加 mask。
- `--mask-position head|tail`：指定 mask 加在 prompt 前面还是后面。
- `--gen-length 256`：控制生成长度。
- `--threshold` 和 `--editing-threshold`：直接传给本地模型 `generate()`。
- 默认使用 tokenizer chat template；如果想直接 token 化原始文本，传 `--no-chat-template`。

输出目录：

```text
outputs/prompt_mask_generation/run_<timestamp>/
  result.json
  result.txt
```

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
{full gold answer with 30% of whitespace-separated tokens replaced by [MASK], and the final answer number also forced to [MASK]}
```

`--gold-noise-ratio` controls the random noise fraction. To avoid leaking the label, the `<number>` in the final GSM8K line `#### <number>` is always masked as an extra forced mask, even if random sampling did not select it.

Options:

- `--gold-noise-ratio 0.0` adds no random noise, but still masks the final answer number.
- `--gold-noise-ratio 1.0` masks every whitespace-separated ground-truth token.
- `--gold-noise-token "[MASK]"` changes the replacement token.
- `--gold-noise-seed 0` controls which token positions are noised, so runs are reproducible.
- `--gold-noise-ratio` and `--gold-prefix-tokens` are mutually exclusive.
- Per-example details include `gold_noised_solution`, `gold_noise_indices`, `gold_final_answer_noise_indices`, `gold_noise_ratio`, and the final `prompt`.

Note: this experiment still uses SGLang chat/completion. The model continues after the prompt; it does not replace `[MASK]` tokens inside the prompt. To test true in-place masked reconstruction, use the local model reconstruction experiment below.

### Local Model Mask Reconstruction Experiment

This experiment does not use SGLang. It directly loads the local model under `model/llada2.1` and reconstructs `<|mask|>` tokens inside the GSM8K gold solution.

The runtime environment must be able to import `torch`, `transformers`, `accelerate`, and `safetensors`. If the current SGLang/model environment already has them, reuse that environment.

Example:

```bash
python experiments/gsm8k_mask_reconstruct.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --limit 10 \
  --mask-ratio 0.3 \
  --threshold 0.5 \
  --block-length 32
```

The script builds:

```text
Problem:
{question}

Ground truth solution:
{gold answer with some tokenizer tokens replaced by <|mask|>}
```

It predicts only masked tokens by default and leaves unmasked tokens unchanged. `--mask-ratio` controls the random tokenizer-token mask ratio inside the gold solution. By default, the final GSM8K answer number in `#### <number>` is also force-masked to avoid direct label leakage.

Useful options:

- `--mask-ratio 0.0` adds no random masks, but still force-masks the final answer number.
- `--mask-ratio 1.0` masks the full gold solution.
- `--no-force-final-answer-mask` disables the final-answer forced mask.
- `--threshold 0.5` controls the confidence threshold for accepting mask predictions.
- `--editing-threshold 0.8` enables reconstruction + editing, allowing high-confidence rewrites of unmasked tokens inside the gold solution. Omit it for strict mask-only reconstruction.
- `--max-edit-steps 16` controls how many extra editing refinement steps can run after all masks are filled.
- `--num-to-transfer 1` controls the minimum number of mask tokens filled per iteration.
- `--attention-mode full` is the default and lets the model attend to the whole corrupted text; `block-causal` can be used to compare with the local generation-style block logic.
- `--device-map auto` uses Transformers device mapping.
- `--device-map none --device cuda` moves the model to one explicit device.

The default mode is strict reconstruction: it fills `<|mask|>` tokens and leaves other tokens unchanged. To run reconstruction + editing, closer to LLaDA2.1's original `generate()` behavior:

```bash
python experiments/gsm8k_mask_reconstruct.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --limit 10 \
  --mask-ratio 0.3 \
  --threshold 0.5 \
  --editing-threshold 0.8 \
  --max-edit-steps 16
```

The two thresholds have different roles:

- `--threshold` controls when predicted tokens are accepted at `<|mask|>` positions.
- `--editing-threshold` controls whether non-mask tokens may be rewritten. The script only edits the `Ground truth solution` span, not the question/prompt prefix.

Outputs are timestamped:

```text
outputs/gsm8k_mask_reconstruct/run_<timestamp>/
  summary.json
  summary.csv
  details.jsonl
  details_pretty.md
```

Main metrics:

- `mask_token_accuracy`: token-level reconstruction accuracy on masked positions.
- `exact_reconstruction_rate`: whether every masked token in an example was reconstructed exactly.
- `final_answer_accuracy`: whether the reconstructed `#### <number>` answer is correct.

### Prompt Head/Tail Mask Generation Experiment

This experiment does not use SGLang. It calls the local LLaDA2.1 `generate()` method after adding `<|mask|>` tokens before or after a command-line prompt, so you can observe how head/tail masks affect generation.

Add 8 masks to the prompt tail:

```bash
python experiments/prompt_mask_generation.py \
  --prompt "Solve 16 - 3 - 4, then multiply the result by 2." \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --mask-count 8 \
  --mask-position tail \
  --gen-length 256 \
  --threshold 0.5 \
  --editing-threshold 0.0
```

Add 8 masks to the prompt head:

```bash
python experiments/prompt_mask_generation.py \
  --prompt "Solve 16 - 3 - 4, then multiply the result by 2." \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --mask-count 8 \
  --mask-position head \
  --gen-length 256 \
  --threshold 0.5 \
  --editing-threshold 0.0
```

For long prompts, use a file:

```bash
python experiments/prompt_mask_generation.py \
  --prompt-file /mnt/workspace/data/my_prompt.txt \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --mask-count 16 \
  --mask-position tail
```

Useful options:

- `--mask-count 0` adds no masks.
- `--mask-position head|tail` controls whether masks are added before or after the prompt.
- `--gen-length 256` controls generation length.
- `--threshold` and `--editing-threshold` are passed through to local model `generate()`.
- The script uses the tokenizer chat template by default. Pass `--no-chat-template` to tokenize raw text directly.

Outputs are timestamped:

```text
outputs/prompt_mask_generation/run_<timestamp>/
  result.json
  result.txt
```

### SGLang GSM8K Prompt Mask Sweep

This experiment uses SGLang, starts one server with `threshold=0.5` and `edit_threshold=0.0` by default, then compares:

```text
no_mask
head_8
head_4
tail_4
tail_8
```

Run:

```bash
python experiments/gsm8k_prompt_mask_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --threshold 0.5 \
  --edit-threshold 0.0 \
  --max-tokens 512
```

To change the mask variants:

```bash
--variants none,head:8,head:4,tail:4,tail:8
```

Outputs:

```text
outputs/gsm8k_prompt_mask/run_<timestamp>/
  summary.csv
  summary.json
  details_no_mask.jsonl
  details_head_8.jsonl
  details_head_4.jsonl
  details_tail_4.jsonl
  details_tail_8.jsonl
  dllm_configs/
  server_logs/
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
