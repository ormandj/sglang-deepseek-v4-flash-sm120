#!/bin/sh
set -eu

if [ -z "${KUBERNETES_SERVICE_HOST:-}" ]; then
  echo "error: turnover gates must execute inside the serving pod" >&2
  exit 2
fi
if [ "$#" -ne 3 ]; then
  echo "usage: $0 CAMPAIGN_ID BUILD_ID release-screen|screen|qualification|publication" >&2
  exit 2
fi

campaign=$1
build_id=$2
mode=$3
for value in "$campaign" "$build_id"; do
  case "$value" in
    *[!A-Za-z0-9._-]*|'') echo "error: invalid identifier: $value" >&2; exit 2 ;;
  esac
done
case "$mode" in
  release-screen)
    repetitions=3
    concurrencies='8'
    ;;
  screen)
    repetitions=3
    concurrencies='1 2 4 8'
    ;;
  qualification|publication)
    repetitions=5
    concurrencies='1 2 4 8'
    ;;
  *) echo "error: unsupported turnover-gate mode: $mode" >&2; exit 2 ;;
esac

: "${BENCH_IMAGE_REF:?BENCH_IMAGE_REF must identify the immutable image}"
: "${BENCH_GITOPS_REVISION:?BENCH_GITOPS_REVISION must be set}"
: "${BENCH_PROJECT_REVISION:?BENCH_PROJECT_REVISION must be set}"
: "${AIPERF_REVISION:?AIPERF_REVISION must be set}"
: "${BENCH_MODEL_REVISION:?BENCH_MODEL_REVISION must be set}"

export MODEL_NAME=${MODEL_NAME:-deepseek-v4-flash}
export TOKENIZER_PATH=${TOKENIZER_PATH:-/models/deepseek-ai/DeepSeek-V4-Flash-0731}
export INFERENCE_URL=${INFERENCE_URL:-http://127.0.0.1:8000}
export SERVER_METRICS_URL=${SERVER_METRICS_URL:-http://127.0.0.1:8000/metrics}
export BENCH_API_KEY=${BENCH_API_KEY:-${SGLANG_API_KEY:-${VLLM_API_KEY:-}}}
export AIPERF_WORKERS=1
export AIPERF_RECORD_PROCESSORS=1

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
config_dir="$script_dir/configs"
lock="$script_dir/aiperf.lock.json"
aiperf_python=${AIPERF_PYTHON:-/models/.bench-tools/aiperf-0.12.0-6ed4823d/venv/bin/python}
uv_bin=${AIPERF_UV_BIN:-/models/.bench-tools/uv-0.12.3-linux-x86_64/uv}
campaign_root=${AIPERF_TURNOVER_ROOT:-/models/bench/results/aiperf-greenfield/turnover-gates}
gate_root="$campaign_root/$campaign/$build_id-$mode"
turnover_requests_override=${TURNOVER_REQUESTS:-}
turnover_isl=${TURNOVER_ISL:-256}
turnover_osl=${TURNOVER_OSL:-256}

test -x "$aiperf_python"
test -x "$uv_bin"
grep -F "\"commit\": \"$AIPERF_REVISION\"" "$lock" >/dev/null
for value in "$turnover_isl" "$turnover_osl"; do
  case "$value" in
    *[!0-9]*|'') echo "error: turnover shape values must be positive integers" >&2; exit 2 ;;
  esac
  if [ "$value" -lt 1 ]; then
    echo "error: turnover shape values must be positive integers" >&2
    exit 2
  fi
done
if [ -n "$turnover_requests_override" ]; then
  case "$turnover_requests_override" in
    *[!0-9]*) echo "error: TURNOVER_REQUESTS must be a positive integer" >&2; exit 2 ;;
  esac
  if [ "$turnover_requests_override" -lt 1 ]; then
    echo "error: TURNOVER_REQUESTS must be a positive integer" >&2
    exit 2
  fi
  for concurrency in $concurrencies; do
    if [ "$turnover_requests_override" -le "$concurrency" ]; then
      echo "error: TURNOVER_REQUESTS must exceed every measured concurrency" >&2
      exit 2
    fi
  done
fi
if [ -e "$gate_root" ]; then
  echo "error: immutable turnover gate already exists: $gate_root" >&2
  exit 2
fi
mkdir -p "$gate_root"

turnover_requests_for_concurrency() {
  concurrency=$1
  if [ -n "$turnover_requests_override" ]; then
    printf '%s\n' "$turnover_requests_override"
  elif [ "$concurrency" -lt 8 ]; then
    printf '%s\n' 16
  else
    printf '%s\n' 32
  fi
}

for concurrency in $concurrencies; do
  turnover_requests=$(turnover_requests_for_concurrency "$concurrency")
  export AIPERF_ARTIFACT_ROOT="$gate_root/warmup"
  export WARMUP_ISL="$turnover_isl"
  export WARMUP_OSL="$turnover_osl"
  export WARMUP_CONCURRENCY="$concurrency"
  export WARMUP_REQUESTS="$turnover_requests"
  export WARMUP_TEMPERATURE=0.0
  export WARMUP_TOP_P=1.0
  export AIPERF_RANDOM_SEED=$((2026082200 + concurrency))
  export SAMPLING_SEED=$((2026082200 + concurrency))
  "$script_dir/run-in-pod.sh" "$config_dir/warmup-coverage.yaml" "turnover-c$concurrency"
done

for concurrency in $concurrencies; do
  turnover_requests=$(turnover_requests_for_concurrency "$concurrency")
  repetition=1
  while [ "$repetition" -le "$repetitions" ]; do
    run_id=$(printf 'r%02d' "$repetition")
    seed=$((2026082200 + concurrency * 10 + repetition))
    export AIPERF_RANDOM_SEED="$seed"
    export SAMPLING_SEED="$seed"
    export AIPERF_ARTIFACT_ROOT="$gate_root/c$concurrency"
    export TURNOVER_CONCURRENCY="$concurrency"
    export TURNOVER_REQUESTS="$turnover_requests"
    export TURNOVER_ISL="$turnover_isl"
    export TURNOVER_OSL="$turnover_osl"
    "$script_dir/run-in-pod.sh" "$config_dir/turnover.yaml" "$run_id"
    cell="$gate_root/c$concurrency/$run_id"
    "$uv_bin" run --no-project --python "$aiperf_python" \
      "$script_dir/analyze_turnover.py" \
      --summary "$cell/profile_export_aiperf.json" \
      --records "$cell/profile_export.jsonl" \
      --server-summary "$cell/server_metrics_export.json" \
      --server-jsonl "$cell/server_metrics_export.jsonl" \
      --target-concurrency "$concurrency" \
      --expected-requests "$turnover_requests" \
      --target-isl "$turnover_isl" \
      --target-osl "$turnover_osl" \
      --output "$cell/turnover-analysis.json"
    repetition=$((repetition + 1))
  done
done

"$uv_bin" run --no-project --python "$aiperf_python" \
  "$script_dir/summarize_turnover_gate.py" \
  --root "$gate_root" --mode "$mode" --build-id "$build_id" \
  --output "$gate_root/summary.json"
date -u +%Y-%m-%dT%H:%M:%SZ > "$gate_root/completed-at-utc.txt"
(
  cd "$gate_root"
  find . -type f ! -name SHA256SUMS -exec sha256sum '{}' \; | LC_ALL=C sort \
    > SHA256SUMS
)
echo "completed turnover gate: $gate_root"
