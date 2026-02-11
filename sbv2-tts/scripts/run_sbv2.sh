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

cmd=uvx

UVX_BIN="${UVX_BIN:-}"
if [ -n "${UVX_BIN}" ]; then
  if [ ! -x "${UVX_BIN}" ]; then
    echo "指定された UVX_BIN が実行可能ではありません: ${UVX_BIN}" >&2
    exit 1
  fi
elif command -v "${cmd}" >/dev/null 2>&1; then
  UVX_BIN=$(command -v "${cmd}")
else
  for p in \
    "/opt/homebrew/bin/${cmd}" \
    "/usr/local/bin/${cmd}" \
    "$HOME/.local/bin/${cmd}" \
    "$HOME/.uv/bin/${cmd}" \
    "$HOME/bin/${cmd}"
  do
    if [[ -x "${p}" ]]; then
      UVX_BIN="${p}"
      break
    fi
  done
fi

if [[ ! -x "${UVX_BIN}" ]]; then
  echo "uvx が見つかりません。PATH を確認するか UVX_BIN で実行ファイルを指定してください。" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  set -- --help
fi

# uvx 専用: プロジェクト依存を解決した一時環境で sbv2.py を実行する。
if command -v python3.11 >/dev/null 2>&1; then
  uvx_python_opt="--python python3.11"
elif command -v python3.10 >/dev/null 2>&1; then
  uvx_python_opt="--python python3.10"
elif command -v python3.12 >/dev/null 2>&1; then
  uvx_python_opt="--python python3.12"
else
  echo "Python 3.10 以降のバージョンが見つかりません。" >&2
  exit 1
fi

exec "${UVX_BIN}" $uvx_python_opt --from "${SCRIPT_DIR}" python "${SBV2_PY}" "$@"
