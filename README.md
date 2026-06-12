# llada2.1 experiments

这个仓库用于管理 LLaDA 2.1 相关实验代码。本地负责写代码和提交到 GitHub，服务器从 GitHub 拉取最新代码并运行实验。推理侧默认通过 SGLang 的 OpenAI-compatible HTTP API 调用模型。

## Repository layout

```text
.
├── experiments/              # 可直接运行的实验入口
├── scripts/                  # 本地同步、服务器运行脚本
├── src/llada_experiments/    # 复用代码
├── prompts/                  # 输入 prompt 样例
├── outputs/                  # 实验输出，本地忽略，不提交
└── .env.example              # 环境变量样例
```

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
```

编辑 `.env`，填入服务器上的 SGLang 地址、模型名和 API key。如果 SGLang 不需要鉴权，`SGLANG_API_KEY=EMPTY` 即可。

## Run a smoke test

```powershell
python experiments/smoke_test.py --prompt "Explain diffusion language models in one paragraph."
```

输出会写到 `outputs/`，这个目录默认不提交到 GitHub。

## GitHub workflow

首次创建 GitHub 仓库后，把远端地址加进来：

```powershell
git remote add origin git@github.com:<your-user>/<your-repo>.git
git push -u origin main
```

之后每次改完代码，可以用：

```powershell
.\scripts\sync.ps1 "describe this experiment change"
```

这个脚本会执行 `git add`、`git commit`、`git push`。如果没有改动，它会直接退出。

## Server workflow

服务器首次部署：

```bash
git clone git@github.com:<your-user>/<your-repo>.git ~/llada2.1
cd ~/llada2.1
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

编辑服务器上的 `.env`，让 `SGLANG_BASE_URL` 指向云端 SGLang 服务。

之后每次本地 push 后，在服务器运行：

```bash
cd ~/llada2.1
git pull --ff-only
source .venv/bin/activate
python experiments/smoke_test.py --prompt "Say hello from the latest commit."
```

也可以把命令封装成：

```bash
bash scripts/server_pull_run.sh experiments/smoke_test.py --prompt "Say hello"
```

## Start SGLang

Server launch files live in `sglang_server/`.

Edit the model path and server parameters:

```bash
cp sglang_server/server_config.local.example.json sglang_server/server_config.local.json
vim sglang_server/server_config.local.json
```

Start the SGLang OpenAI-compatible server:

```bash
bash sglang_server/start_sglang.sh
```

Check whether the server is alive:

```bash
bash sglang_server/check_sglang.sh
```

LLaDA2.1 decoding parameters are in:

```bash
cp sglang_server/dllm_algorithm_config.local.example.yaml sglang_server/dllm_algorithm_config.local.yaml
vim sglang_server/dllm_algorithm_config.local.yaml
```

SGLang 0.5.12.post1 reads LLaDA2.1 `threshold` and `edit_threshold` at server startup through `--dllm-algorithm-config`, so changing these values requires restarting SGLang.

Run a smoke test:

```bash
python experiments/smoke_test.py \
  --prompt "say hello from the latest commit"
```

## GSM8K Threshold Sweep

Evaluate GSM8K accuracy and generation speed across LLaDA2.1 threshold pairs:

```bash
python experiments/gsm8k_threshold_sweep.py \
  --input-jsonl /mt/workspace/data/gsm8k_test.jsonl \
  --limit 100 \
  --thresholds 0.4,0.5,0.6 \
  --edit-thresholds 0.0,0.2,0.4 \
  --max-tokens 512
```

The script writes a DLLM YAML config, starts SGLang, waits until `/v1/models` is ready, evaluates GSM8K, then stops SGLang for each threshold pair. It shows a progress bar for each threshold pair by default. Use
`--no-progress` if you want plain line-by-line logging.

Outputs:

```text
outputs/gsm8k/run_<timestamp>/summary.csv
outputs/gsm8k/run_<timestamp>/summary.json
outputs/gsm8k/run_<timestamp>/details_threshold_<value>_edit_<value>.jsonl
outputs/gsm8k/run_<timestamp>/dllm_configs/
outputs/gsm8k/run_<timestamp>/server_logs/
```
