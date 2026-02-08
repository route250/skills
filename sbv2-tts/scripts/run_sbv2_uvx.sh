#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBV2_PY="${SCRIPT_DIR}/sbv2.py"

if [[ ! -f "${SBV2_PY}" ]]; then
  echo "sbv2.py が見つかりません: ${SBV2_PY}" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv が見つかりません。先に uv をインストールしてください。" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  set -- --help
fi

exec uv run "${SBV2_PY}" "$@"
