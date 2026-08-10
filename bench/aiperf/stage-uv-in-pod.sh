#!/bin/sh
set -eu

if [ -z "${KUBERNETES_SERVICE_HOST:-}" ]; then
  echo "error: this installer is intended to run inside a Kubernetes pod" >&2
  exit 2
fi
if [ "$#" -ne 2 ]; then
  echo "usage: $0 UV_ARCHIVE INSTALL_DIR" >&2
  exit 2
fi

archive=$1
install_dir=$2
archive_sha256=600cf9a742aca00d292673b16b5acffaa7b8c269a364ad0c2e79498dcb1fe101
binary_sha256=729d27dbea534ee540a2d3ef43a62fa1a10af7fcbb6d57a70d5859509f624578

test -f "$archive"
if [ -e "$install_dir" ]; then
  echo "error: immutable install target already exists: $install_dir" >&2
  exit 2
fi
parent_dir=$(dirname -- "$install_dir")
test -d "$parent_dir"

actual_archive_sha256=$(sha256sum "$archive" | awk '{print $1}')
if [ "$actual_archive_sha256" != "$archive_sha256" ]; then
  echo "error: uv archive SHA-256 mismatch" >&2
  exit 2
fi

temp_dir=$(mktemp -d "$parent_dir/.uv-0.12.3-stage.XXXXXX")
cleanup() {
  if [ -n "${temp_dir:-}" ] && [ -d "$temp_dir" ]; then
    rm -rf -- "$temp_dir"
  fi
}
trap cleanup EXIT HUP INT TERM
tar -xzf "$archive" -C "$temp_dir" --strip-components=1
actual_binary_sha256=$(sha256sum "$temp_dir/uv" | awk '{print $1}')
if [ "$actual_binary_sha256" != "$binary_sha256" ]; then
  echo "error: uv binary SHA-256 mismatch" >&2
  exit 2
fi
test -x "$temp_dir/uv"
test -x "$temp_dir/uvx"
mv "$temp_dir" "$install_dir"
temp_dir=
trap - EXIT HUP INT TERM
"$install_dir/uv" --version
