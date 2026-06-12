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
