#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1:30000/v1}"
curl -fsS "$BASE_URL/models"
echo
