#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBV2_PY="${SCRIPT_DIR}/sbv2.py"
PYPROJECT_TOML="${SCRIPT_DIR}/pyproject.toml"

if [[ ! -f "${SBV2_PY}" ]]; then
  echo "sbv2.py が見つかりません: ${SBV2_PY}" >&2
  exit 1
fi

if [[ ! -f "${PYPROJECT_TOML}" ]]; then
  echo "pyproject.toml が見つかりません: ${PYPROJECT_TOML}" >&2
  exit 1
fi

find_uv_bin() {
  local cmd="$1"
  if command -v "${cmd}" >/dev/null 2>&1; then
    command -v "${cmd}"
    return 0
  fi
  for p in \
    "/opt/homebrew/bin/${cmd}" \
    "/usr/local/bin/${cmd}" \
    "$HOME/.local/bin/${cmd}" \
    "$HOME/.uv/bin/${cmd}" \
    "$HOME/bin/${cmd}"
  do
    if [[ -x "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

UVX_BIN="${UVX_BIN:-}"
if [[ -z "${UVX_BIN}" ]]; then
  UVX_BIN="$(find_uv_bin uvx || true)"
fi

if [[ -z "${UVX_BIN}" ]]; then
  echo "uvx が見つかりません。PATH を確認するか UVX_BIN で実行ファイルを指定してください。" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  set -- --help
fi

# uvx 専用: プロジェクト依存を解決した一時環境で sbv2.py を実行する。
exec "${UVX_BIN}" --project "${SCRIPT_DIR}" python "${SBV2_PY}" "$@"
