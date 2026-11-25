#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

if rg -n '/data/202500025|/root/anaconda|ADITTY/' . \
  -g '!README.md' -g '!scripts/check_repository.sh'; then
  echo "Found machine-specific paths." >&2
  exit 1
fi

unexpected_weights=$(
  git ls-files --cached --others --exclude-standard \
    | rg '\.(ckpt|pth|pt)$' \
    | rg -v '^src/models/components/equiformer_v2/Jd\.pt$' \
    || true
)
if [[ -n "$unexpected_weights" ]]; then
  echo "Found checkpoint or weight files:" >&2
  echo "$unexpected_weights" >&2
  exit 1
fi

echo "Repository check passed."
