#!/usr/bin/env bash
set -euo pipefail

sglang_src=${1:-/mnt/workspace/third_party/sglang-v0.5.12.post1}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
patch_dir="$repo_root/sglang_patches"

if [[ ! -d "$sglang_src/.git" ]]; then
  echo "Not a git checkout: $sglang_src" >&2
  exit 1
fi

shopt -s nullglob
patches=("$patch_dir"/*.patch)
if [[ ${#patches[@]} -eq 0 ]]; then
  echo "No patches found in $patch_dir"
  exit 0
fi

for patch in "${patches[@]}"; do
  echo "Checking $patch"
  git -C "$sglang_src" apply --check "$patch"
done

for patch in "${patches[@]}"; do
  echo "Applying $patch"
  git -C "$sglang_src" apply "$patch"
done

echo "Applied ${#patches[@]} patch(es) to $sglang_src"
