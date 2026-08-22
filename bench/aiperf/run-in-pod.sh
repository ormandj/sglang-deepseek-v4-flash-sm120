#!/bin/sh
set -eu

if [ -z "${KUBERNETES_SERVICE_HOST:-}" ]; then
  echo "error: measured AIPerf traffic must originate inside the serving pod" >&2
  exit 2
fi

if [ "$#" -ne 2 ]; then
  echo "usage: $0 CONFIG_FILE RUN_ID" >&2
  exit 2
fi

config_file=$1
run_id=$2
case "$run_id" in
  *[!A-Za-z0-9._-]*|'')
    echo "error: RUN_ID may contain only letters, digits, dot, underscore, and hyphen" >&2
    exit 2
    ;;
esac

inference_url=${INFERENCE_URL:-http://127.0.0.1:8000}
metrics_url=${SERVER_METRICS_URL:-http://127.0.0.1:8000/metrics}
export BENCH_API_KEY=${BENCH_API_KEY:-${SGLANG_API_KEY:-${VLLM_API_KEY:-}}}
case "$inference_url" in
  http://127.0.0.1:*/*|http://localhost:*/*)
    echo "error: INFERENCE_URL must not include an endpoint path: $inference_url" >&2
    exit 2
    ;;
  http://127.0.0.1:*|http://localhost:*) ;;
  *) echo "error: INFERENCE_URL must be a pod-localhost base URL, got $inference_url" >&2; exit 2 ;;
esac
case "$metrics_url" in
  http://127.0.0.1:*|http://localhost:*) ;;
  *) echo "error: SERVER_METRICS_URL must be pod-localhost, got $metrics_url" >&2; exit 2 ;;
esac

aiperf_python=${AIPERF_PYTHON:-/models/.bench-tools/aiperf-0.12.0-6ed4823d/venv/bin/python}
aiperf_bin=${AIPERF_BIN:-/models/.bench-tools/aiperf-0.12.0-6ed4823d/venv/bin/aiperf}
uv_bin=${AIPERF_UV_BIN:-/models/.bench-tools/uv-0.12.3-linux-x86_64/uv}
artifact_root=${AIPERF_ARTIFACT_ROOT:-/models/bench/results/aiperf-greenfield}
artifact_dir="$artifact_root/$run_id"

test -f "$config_file"
test -x "$aiperf_python"
test -x "$aiperf_bin"
test -x "$uv_bin"
if [ -e "$artifact_dir" ]; then
  echo "error: immutable artifact directory already exists: $artifact_dir" >&2
  exit 2
fi

mkdir -p "$artifact_dir"
export AIPERF_PYTHON=$aiperf_python
export ARTIFACT_DIR=$artifact_dir
export INFERENCE_URL=$inference_url
export SERVER_METRICS_URL=$metrics_url

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"$script_dir/capture-environment.sh" "$config_file" "$artifact_dir"
"$uv_bin" run --no-project --python "$aiperf_python" "$aiperf_bin" config validate "$config_file"
"$uv_bin" run --no-project --python "$aiperf_python" "$aiperf_bin" profile -f "$config_file"
