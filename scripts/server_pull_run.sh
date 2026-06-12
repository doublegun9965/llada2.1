#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/server_pull_run.sh <python-script> [args...]"
  exit 2
fi

REPO_DIR="${REPO_DIR:-$HOME/llada2.1}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"

cd "$REPO_DIR"
git pull --ff-only

if [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

python "$@"
