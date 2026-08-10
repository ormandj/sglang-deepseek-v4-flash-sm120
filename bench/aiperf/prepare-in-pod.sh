#!/bin/sh
set -eu

if [ -z "${KUBERNETES_SERVICE_HOST:-}" ]; then
  echo "error: this installer is intended to run inside a Kubernetes pod" >&2
  exit 2
fi

if [ "$#" -ne 2 ]; then
  echo "usage: $0 AIPERF_SOURCE_DIR INSTALL_DIR" >&2
  exit 2
fi

source_dir=$1
install_dir=$2
venv_dir="$install_dir/venv"
cache_dir="${AIPERF_UV_CACHE_DIR:-/models/.bench-tools/uv-cache/aiperf-0.12.0}"
uv_bin=${AIPERF_UV_BIN:-/models/.bench-tools/uv-0.12.3-linux-x86_64/uv}

test -f "$source_dir/pyproject.toml"
test -x "$uv_bin"
if [ -e "$install_dir" ]; then
  echo "error: immutable install target already exists: $install_dir" >&2
  exit 2
fi

mkdir -p "$install_dir" "$cache_dir"
export UV_CACHE_DIR=$cache_dir
"$uv_bin" venv --system-site-packages "$venv_dir"
"$uv_bin" pip install --python "$venv_dir/bin/python" "$source_dir"
"$uv_bin" run --no-project --python "$venv_dir/bin/python" "$venv_dir/bin/aiperf" --version
