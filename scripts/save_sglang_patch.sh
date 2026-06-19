#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <sglang-src-dir> <patch-name.patch>" >&2
  exit 2
fi

sglang_src=$1
patch_name=$2

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
patch_dir="$repo_root/sglang_patches"
patch_path="$patch_dir/$patch_name"

if [[ $patch_name != *.patch ]]; then
  echo "Patch name must end with .patch: $patch_name" >&2
  exit 2
fi

if [[ ! -d "$sglang_src/.git" ]]; then
  echo "Not a git checkout: $sglang_src" >&2
  exit 1
fi

mkdir -p "$patch_dir"
git -C "$sglang_src" diff --binary > "$patch_path"

if [[ ! -s "$patch_path" ]]; then
  rm -f "$patch_path"
  echo "No changes found in $sglang_src" >&2
  exit 1
fi

echo "Wrote $patch_path"
