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

### SGLang 源码补丁

SGLang 源码 checkout 不提交到这个仓库；后续实验如果需要改 SGLang，只提交 `sglang_patches/` 下的 patch 文件。当前约定路径：

```text
本地: third_party/sglang-v0.5.12.post1
服务器: /mnt/workspace/third_party/sglang-v0.5.12.post1
```

从 SGLang 工作树导出 patch：

```bash
scripts/save_sglang_patch.sh third_party/sglang-v0.5.12.post1 my_experiment.patch
git add sglang_patches/my_experiment.patch
git commit -m "Add SGLang patch for my experiment"
git push
```

服务器拉取本仓库后应用 patch：

```bash
git pull --ff-only
scripts/apply_sglang_patches.sh /mnt/workspace/third_party/sglang-v0.5.12.post1
```

当前已有一个确定性兼容补丁：

```text
sglang_patches/deterministic_dllm_compat.patch
```

这个补丁用于 SGLang `0.5.12.post1` + LLaDA 2.1 dLLM：开启 `--enable-deterministic-inference` 时，绕过 dLLM prefill 不支持的 `truncation_align_size` 路径。应用补丁后，在 `sglang_server/server_config.local.json` 里设置：

```json
"enable_deterministic_inference": true
```

启动日志里应能看到 deterministic inference 已开启，以及 sampling backend 被切到 pytorch。

如果只想应用某一个补丁，可以把补丁名放在命令最后：

```bash
scripts/apply_sglang_patches.sh /mnt/workspace/third_party/sglang-v0.5.12.post1 dllm_trace.patch
```

`sglang_patches/dllm_trace.patch` 用来记录 SGLang 内部 `JointThreshold` 的 M2T/T2T 轨迹。默认不写 trace；只有在 DLLM YAML 里设置 `trace_path` 后才启用。

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
  --max-tokens 512 \
  --batch-size 4
```

`--batch-size 4` 表示同一个阈值组合下同时发送 4 个 SGLang 请求；不同阈值组合仍会按顺序重启 SGLang，因为 LLaDA2.1 阈值是 server-startup 配置。

SGLang assistant-prefill 模式，给 assistant 一个正确答案开头再续写：

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

把单个或多个 `details_threshold_<value>_edit_<value>.jsonl` 里的错误样本写成 Markdown，方便逐题排查：

```bash
python scripts/write_gsm8k_error_report.py \
  outputs/gsm8k_threshold_sweep/run_<timestamp>/details_threshold_0p5_edit_0p0.jsonl
```

记录本地 LLaDA2.1 `generate()` 的逐轮轨迹，观察每轮填了哪些 mask、edit 了哪些 token：

```bash
python experiments/trace_llada_generation.py \
  --prompt-file /mnt/workspace/data/my_prompt.txt \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --gen-length 128 \
  --block-length 32 \
  --threshold 0.5 \
  --editing-threshold 0.9
```

固定阈值本地 GSM8K baseline，用于作为后续动态阈值/自适应策略的对比参照。默认 `threshold=0.5` 且关闭 edit：

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

动态阈值本地 GSM8K 评测：根据当前 block 内已生成 token 比例自动切换阈值。默认前期 `threshold=0.9` 且关闭 edit，中期 `threshold=0.0` 且 `editing_threshold=0.9`，后期 `threshold=0.0` 且关闭 edit。为了避免低阈值时一轮填满整个 block，默认还会限制每轮写入数量：early 最多填 1 个 mask，mid/late 最多填 4 个 mask，mid 最多 edit 1 个 token：

```bash
cp experiments/dynamic_threshold_config.local.example.json experiments/dynamic_threshold_config.local.json
vim experiments/dynamic_threshold_config.local.json

python experiments/dynamic_threshold_generation.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --limit 100 \
  --gen-length 128 \
  --block-length 32
```

脚本会优先读取 `experiments/dynamic_threshold_config.local.json`，否则读取 `experiments/dynamic_threshold_config.json`。命令行显式传入的动态阈值参数会覆盖配置文件。如果只想看单条 prompt，也可以把 `--input-jsonl ... --limit ...` 换成 `--prompt-file /mnt/workspace/data/my_prompt.txt`。GSM8K 模式默认不写逐轮 trace；需要观察前几条轨迹时加 `--trace-limit 3`，会在 `traces/` 下同时写 `example_0001.jsonl` 和更适合阅读的 `example_0001.md`。

动态阈值配置里可以额外控制每轮最多提交多少改动，避免 `threshold=0.0` 时把大量低置信度 token 一次性写死：

```json
{
  "early_max_mask_fills_per_step": 1,
  "mid_max_mask_fills_per_step": 4,
  "late_max_mask_fills_per_step": 4,
  "mid_max_edits_per_step": 1
}
```

`*_max_*_per_step` 填正整数表示上限，填 `"off"` 表示不限制。一般不要把 `mid_editing_threshold` 设成 `0.0`，它会允许很低置信度的非 mask token 被改写。

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
  "dllm_algorithm_config": "sglang_server/dllm_algorithm_config.local.yaml",
  "enable_deterministic_inference": false
}
```

如果已经在服务器上应用 `sglang_patches/deterministic_dllm_compat.patch`，可以把 `enable_deterministic_inference` 改成 `true`。这会让 `sglang_server/launch_sglang.py` 启动 SGLang 时追加 `--enable-deterministic-inference`。

创建服务器环境变量文件：

```bash
cp sglang_server/server_env.example sglang_server/server_env.local
vim sglang_server/server_env.local
```

当前服务器上建议包含：

```bash
export SGLANG_DISABLE_VLLM_RMSNORM=1
```

自动 GSM8K sweep 和手动 `bash sglang_server/start_sglang.sh` 都会在启动 SGLang 前加载 `sglang_server/server_env.local`。这个变量需要配合 `sglang_patches/rocm_disable_vllm_rmsnorm.patch`，服务器上重新应用所有 SGLang patches 后才会生效。

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
trace_path: null
trace_snapshot_every: 1
trace_max_events: null
```

`experiments/gsm8k_threshold_sweep.py` 会读取这个 YAML 作为模板，然后为每组命令行里的 `--thresholds/--edit-thresholds` 生成实际启动用的 DLLM YAML。也就是说，`max_post_edit_steps` 等本地配置会自动带进 sweep。

GSM8K sweep 默认会忽略模板里的 `trace_path`，避免普通 sweep 意外写大量 trace。需要记录主评测阶段的 dLLM token 轨迹并生成统计表时，使用 `--critical-token-analysis`，脚本会为每个阈值组合自动写本次运行专用的 `trace_path`。

修改 `threshold` 或 `edit_threshold` 后必须重启 SGLang。当前 GSM8K sweep 脚本会为每组阈值自动写 YAML、启动 SGLang、评测、停止 SGLang，再进入下一组阈值。

### SGLang dLLM Trace

这个模式用于观察 SGLang 内部 `JointThreshold` 每轮 M2T 和 T2T 是怎么发生的。

1. 在服务器应用 trace patch：

```bash
scripts/apply_sglang_patches.sh /mnt/workspace/third_party/sglang-v0.5.12.post1 dllm_trace.patch
```

2. 修改 `sglang_server/dllm_algorithm_config.local.yaml`：

```yaml
threshold: 0.5
edit_threshold: 0.0
max_post_edit_steps: 16
penalty_lambda: 0
trace_path: outputs/sglang_dllm_trace/trace.jsonl
trace_snapshot_every: 1
trace_max_events: 200
```

3. 启动 SGLang 并发一条请求：

```bash
bash sglang_server/start_sglang.sh
python experiments/smoke_test.py --prompt-file /mnt/workspace/data/my_prompt.txt
```

`smoke_test.py` 调用的是 OpenAI-compatible `/v1/chat/completions`，会以 `messages=[{"role": "user", "content": prompt}]` 的方式发给 SGLang，因此 chat template 在 SGLang 服务端应用。

4. 把 trace 渲染成 Markdown：

```bash
python scripts/render_sglang_dllm_trace.py \
  outputs/sglang_dllm_trace/trace.jsonl \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --trust-remote-code
```

输出文件默认是：

```text
outputs/sglang_dllm_trace/trace.md
```

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

如果要把主评测 trace 转成 critical-token 分析表，先在服务器应用 trace patch，然后用 `--critical-token-analysis`。新版 trace patch 会记录 `request_id` 和 `dllm_block_offset`，分析脚本会优先用这些字段做精确样本/block 对齐：

```bash
scripts/apply_sglang_patches.sh /mnt/workspace/third_party/sglang-v0.5.12.post1 dllm_trace.patch

python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --thresholds 0.5 \
  --edit-thresholds 0.0 \
  --max-tokens 512 \
  --batch-size 1 \
  --critical-token-analysis
```

当前评测脚本会为每个样本向 SGLang 发送唯一的 `rid`，并在 details JSONL 中保存同值的 `trace_request_id`。分析器使用这两个字段精确关联；SGLang 启动阶段残留的 warmup request 即使出现在 trace 中间，也会被识别为额外 request 并忽略。旧 details 没有 `trace_request_id` 时只允许 request 数量完全一致的顺序对齐，数量不一致会直接报错，避免生成错位统计。

每个阈值组合会额外写：

```text
outputs/gsm8k/run_<timestamp>/
  critical_token_traces/threshold_<value>_edit_<value>.jsonl
  critical_token_analysis/threshold_<value>_edit_<value>/
    sample_summary.csv
    token_summary.csv
    token_events.csv
    token_proposals.csv
    edit_proposals.csv
    edit_annotation.md
    critical_token_stats.csv
    report.md
```

要查看某个样本的单个 block 在各轮迭代中的文本变化，可以把
`token_events.csv` 渲染成 Markdown：

```bash
python scripts/render_token_events_block.py \
  outputs/gsm8k/run_<timestamp>/critical_token_analysis/threshold_0p5_edit_0p0/token_events.csv \
  --sample-id 37 \
  --block-index 7
```

报告包含初始状态和每个有事件记录的 block iteration：完整解码文本、带
block offset 的 token 序列，以及本轮事件类型和概率。当前轮发生变化的 token
会加粗。默认输出到 CSV 同目录，也可以用 `--output-md` 指定路径。这里必须传
`token_events.csv`；`token_summary.csv` 只有最终 token 和提交/编辑统计，不能
还原逐轮文本。

`token_proposals.csv` 每行对应一次 forward 中的一个非 prompt 位置，包含当前、top-1、top-2 token 及其概率/logit、`A`、`D`、entropy、决策结果、token age 和 state hash。`edit_proposals.csv` 只保留 top-1 与当前 token 不同的 T2T proposal，并预留 `manual_label`、`manual_reason`、`manual_notes` 三列；建议标签为 `beneficial`、`harmful`、`neutral`、`uncertain`。重复运行分析器时，会按 `proposal_id` 保留已有人工标注。`edit_annotation.md` 按 sample/block/iteration 展示 before/after 文本和 proposal 表，便于人工浏览。

开启该模式时，脚本会在本次生成的 DLLM YAML 中自动设置 `trace_max_events: null` 和 `trace_snapshot_every: 1`，避免分析 trace 被截断，并保存每轮 block 快照供人工判断上下文。该模式的数据量和显存/运行开销明显高于普通 sweep，人工采样阶段建议先使用较小的 `--limit`。

如果 report 里仍然出现 `unassigned trace block segment(s)`，说明分析的是旧 trace，或者服务器上的 `dllm_trace.patch` 还没有重新应用。旧 trace 没有 `request_id/dllm_block_offset`，分析脚本只能从 `completion_tokens` 估计 block 归属，无法完全修复；请重新应用 patch 后重跑 sweep。

### Assistant Prefill 正确开头实验

默认评测走 SGLang `/v1/chat/completions`，脚本把 GSM8K prompt 作为 user message 发送，由 SGLang 服务端套 chat template。

如果想测试“给 assistant 一段正确推理开头是否能提升表现”，使用：

```bash
--assistant-prefill-tokens 10
```

完整例子：

```bash
python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --thresholds 0.5 \
  --edit-thresholds 0.0 \
  --max-tokens 512 \
  --assistant-prefill-tokens 10
```

这个模式会在客户端用 tokenizer 构造：

```text
chat_template(
  user: "Solve the following grade-school math problem..."
  assistant generation header
)
{gold solution 的前 10 个 tokenizer token}
```

然后改走 `/v1/completions`，让 SGLang 从 assistant 的正确开头后继续生成。这样 gold prefix 不再被塞进 user prompt。

常用参数：

- `--assistant-prefill-tokens 0`：默认值，关闭 assistant prefill，使用普通 chat completion。
- `--assistant-prefill-tokens 10`：取 gold solution 前 10 个 tokenizer token 作为 assistant prefill。
- `--tokenizer-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini`：手动指定 tokenizer；默认使用 `server_config.local.json` 的 `model_path`。
- 每条样本的 details 会记录 `user_prompt`、`assistant_prefix`、`assistant_prefix_token_ids`、`templated_prompt`、`completion` 和 `completion_with_assistant_prefix`。

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
- 缺少 `SGLANG_DISABLE_VLLM_RMSNORM=1`，或服务器上的 SGLang 没有应用 `rocm_disable_vllm_rmsnorm.patch`。

先看本次运行目录里的日志：

```bash
tail -n 200 outputs/gsm8k/run_<timestamp>/server_logs/<log-file>.log
```

如果日志里有：

```text
TypeError: fused_add_rms_norm() takes 4 positional arguments but 6 were given
```

检查 env 和 patch：

```bash
cat sglang_server/server_env.local
ls sglang_patches/rocm_disable_vllm_rmsnorm.patch
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
  "dllm_algorithm_config": "sglang_server/dllm_algorithm_config.local.yaml",
  "enable_deterministic_inference": false
}
```

After applying `sglang_patches/deterministic_dllm_compat.patch` on the server, set `enable_deterministic_inference` to `true` to make `sglang_server/launch_sglang.py` pass `--enable-deterministic-inference`.

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

Both automatic GSM8K sweep and manual `bash sglang_server/start_sglang.sh` load `sglang_server/server_env.local` before starting SGLang. This variable requires `sglang_patches/rocm_disable_vllm_rmsnorm.patch`; reapply all SGLang patches on the server after pulling.

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
trace_path: null
trace_snapshot_every: 1
trace_max_events: null
```

`experiments/gsm8k_threshold_sweep.py` reads this YAML as the base template and then overrides `threshold` and `edit_threshold` for each command-line threshold pair. Local options such as `max_post_edit_steps` are carried into the sweep configs.

The GSM8K sweep ignores `trace_path` from this template by default, so normal sweeps do not accidentally write large traces. To record main-evaluation dLLM token trajectories and generate analysis tables, pass `--critical-token-analysis`; the script writes a run-specific `trace_path` for each threshold pair.

Changing `threshold` or `edit_threshold` requires restarting SGLang.

### SGLang dLLM Trace

Apply only the trace patch when you want to inspect server-side M2T/T2T events:

```bash
scripts/apply_sglang_patches.sh /mnt/workspace/third_party/sglang-v0.5.12.post1 dllm_trace.patch
```

Then set `trace_path` in `sglang_server/dllm_algorithm_config.local.yaml`, restart SGLang, and run one chat request from a text prompt file:

```bash
bash sglang_server/start_sglang.sh
python experiments/smoke_test.py --prompt-file /mnt/workspace/data/my_prompt.txt
```

`smoke_test.py` uses `/v1/chat/completions`, so SGLang applies the model chat template from the user message.

Render Markdown:

```bash
python scripts/render_sglang_dllm_trace.py \
  outputs/sglang_dllm_trace/trace.jsonl \
  --model-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini \
  --trust-remote-code
```

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

### Assistant Prefill Correct Prefix Experiment

By default, the GSM8K sweep sends a user message through `/v1/chat/completions`, and SGLang applies the chat template on the server side.

To test whether a correct reasoning prefix improves generation, put the gold prefix after the assistant generation header with `--assistant-prefill-tokens`.

Example using the first 10 tokenizer tokens from the gold solution:

```bash
python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --thresholds 0.5 \
  --edit-thresholds 0.0 \
  --max-tokens 512 \
  --assistant-prefill-tokens 10
```

This client-side path builds:

```text
chat_template(
  user: "Solve the following grade-school math problem..."
  assistant generation header
)
{first 10 tokenizer tokens from gold solution}
```

Then it calls `/v1/completions`, so SGLang continues after the assistant prefill. The gold prefix is no longer placed inside the user prompt.

Options:

- `--assistant-prefill-tokens 0` is the default and disables this feature.
- `--assistant-prefill-tokens 10` uses the first 10 tokenizer tokens from the gold solution as assistant prefill.
- `--tokenizer-path /mnt/workspace/models/inclusionAI/LLaDA2.1-mini` overrides the tokenizer path. By default, the script uses `model_path` from `server_config.local.json`.
- Per-example details include `user_prompt`, `assistant_prefix`, `assistant_prefix_token_ids`, `templated_prompt`, `completion`, and `completion_with_assistant_prefix`.

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

To convert main-evaluation traces into critical-token analysis tables, apply the trace patch first and pass `--critical-token-analysis`. The updated trace patch records `request_id` and `dllm_block_offset`, and the analyzer uses those fields for exact sample/block alignment:

```bash
scripts/apply_sglang_patches.sh /mnt/workspace/third_party/sglang-v0.5.12.post1 dllm_trace.patch

python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mnt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --thresholds 0.5 \
  --edit-thresholds 0.0 \
  --max-tokens 512 \
  --batch-size 1 \
  --critical-token-analysis
```

The evaluator sends a unique SGLang `rid` for every sample and stores the same value as `trace_request_id` in the details JSONL. The analyzer joins on these fields, so a leftover startup warmup request is ignored even if it is not the last trace request. For old details without `trace_request_id`, positional alignment is allowed only when the request counts match; a mismatch now fails instead of producing shifted statistics.

Each threshold pair additionally writes:

```text
outputs/gsm8k/run_<timestamp>/
  critical_token_traces/threshold_<value>_edit_<value>.jsonl
  critical_token_analysis/threshold_<value>_edit_<value>/
    sample_summary.csv
    token_summary.csv
    token_events.csv
    token_proposals.csv
    edit_proposals.csv
    edit_annotation.md
    critical_token_stats.csv
    report.md
```

`token_proposals.csv` has one row per non-prompt position and forward pass. It records the current, top-1, and top-2 tokens and their probabilities/logits, `A`, `D`, entropy, decision outcome, token age, and state hash. `edit_proposals.csv` keeps only T2T proposals whose top-1 differs from the current token and adds `manual_label`, `manual_reason`, and `manual_notes` columns. Recommended labels are `beneficial`, `harmful`, `neutral`, and `uncertain`. Rerunning the analyzer preserves existing annotations by `proposal_id`. `edit_annotation.md` groups before/after context and proposals by sample, block, and iteration.

When this mode is enabled, the generated DLLM YAML automatically sets `trace_max_events: null` and `trace_snapshot_every: 1`, so traces are not truncated and every iteration has block snapshots for manual review. This mode produces substantially more data and adds runtime/memory overhead; use a small `--limit` for the initial annotation study.

If the report still shows `unassigned trace block segment(s)`, the input is an old trace or the server-side `dllm_trace.patch` was not reapplied. Old traces do not contain `request_id/dllm_block_offset`, so the analyzer can only estimate block ownership from `completion_tokens`; rerun the sweep after applying the updated patch.

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

Render wrong records from one or more threshold-sweep details files into a Markdown report:

```bash
python scripts/write_gsm8k_error_report.py \
  outputs/gsm8k_threshold_sweep/run_<timestamp>/details_threshold_0p5_edit_0p0.jsonl
```

Compare all threshold-pair details files in one run, skip samples that are
correct under every pair, and group wrong pairs by error reason:

```bash
python scripts/write_gsm8k_threshold_reason_report.py \
  outputs/gsm8k/run_<timestamp> \
  --annotations-json outputs/gsm8k/run_<timestamp>/error_reason_annotations.json
```

The optional annotations JSON maps each `sample_id` to semantic error-reason
groups. Unannotated wrong outputs are still included as incomplete/degenerate
generations or grouped by predicted answer. Each invocation writes a fresh run
under `outputs/gsm8k_threshold_reason_report/`.

```json
{
  "3": [
    {
      "combos": ["0,0", "0.3,0"],
      "reason": "Misread three sprints three times per week as three sprints total."
    }
  ]
}
```

`--use-running-server` can only evaluate one threshold pair, because thresholds are fixed when SGLang starts.

## Smoke Test

After SGLang is running:

```bash
python experiments/smoke_test.py --prompt-file /mnt/workspace/data/my_prompt.txt
```

`smoke_test.py` uses `/v1/chat/completions`, so SGLang applies the model chat template from the user message.

If the server is not running, this will fail with `Connection refused`.

## Troubleshooting

### `RemoteProtocolError: Server disconnected without sending a response`

This usually means SGLang accepted the connection but closed it before returning a response. Common causes:

- SGLang worker crashed during generation.
- GPU OOM or memory pressure.
- The output length is too large for the current memory settings.
- The port is connected to an old/stale SGLang process.
- The model path or DLLM config is incompatible with the server.
- `SGLANG_DISABLE_VLLM_RMSNORM=1` is missing, or the server-side SGLang checkout has not applied `rocm_disable_vllm_rmsnorm.patch`, which can trigger `TypeError: fused_add_rms_norm() takes 4 positional arguments but 6 were given` on this server setup.

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

create/check the env file and make sure the patch exists:

```bash
cat sglang_server/server_env.local
ls sglang_patches/rocm_disable_vllm_rmsnorm.patch
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
