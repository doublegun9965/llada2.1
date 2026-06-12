#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG="${1:-}"
if [ -z "$CONFIG" ]; then
  if [ -f "$SCRIPT_DIR/server_config.local.json" ]; then
    CONFIG="$SCRIPT_DIR/server_config.local.json"
  else
    CONFIG="$SCRIPT_DIR/server_config.json"
  fi
fi

cd "$REPO_DIR"
mkdir -p logs

python sglang_server/launch_sglang.py --config "$CONFIG" 2>&1 | tee logs/sglang_server.log
