#!/bin/sh
set -eu

if [ -z "${KUBERNETES_SERVICE_HOST:-}" ]; then
  echo "error: performance blocks must execute inside the serving pod" >&2
  exit 2
fi
if [ "$#" -ne 4 ]; then
  echo "usage: $0 CAMPAIGN_ID pilot|final PAIR_ID baseline|candidate" >&2
  exit 2
fi

campaign=$1
phase=$2
pair=$3
role=$4
case "$campaign" in
  *[!A-Za-z0-9._-]*|'') echo "error: invalid campaign ID" >&2; exit 2 ;;
esac
case "$phase" in
  pilot|final) ;;
  *) echo "error: phase must be pilot or final" >&2; exit 2 ;;
esac
case "$pair" in
  *[!A-Za-z0-9._-]*|'') echo "error: invalid pair ID" >&2; exit 2 ;;
esac
case "$role" in
  baseline|candidate) ;;
  *) echo "error: role must be baseline or candidate" >&2; exit 2 ;;
esac

: "${BENCH_BUILD_ID:?BENCH_BUILD_ID must identify this build}"
: "${BENCH_IMAGE_REF:?BENCH_IMAGE_REF must contain an immutable image reference}"
: "${BENCH_GITOPS_REVISION:?BENCH_GITOPS_REVISION must be set}"
: "${BENCH_PROJECT_REVISION:?BENCH_PROJECT_REVISION must be set}"
: "${AIPERF_REVISION:?AIPERF_REVISION must be set}"
: "${BENCH_MODEL_REVISION:?BENCH_MODEL_REVISION must be set}"

export MODEL_NAME=${MODEL_NAME:-deepseek-v4-flash}
export TOKENIZER_PATH=${TOKENIZER_PATH:-/models/deepseek-ai/DeepSeek-V4-Flash-0731}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
config_dir="$script_dir/configs"
panel="$script_dir/seed-panel.json"
lock="$script_dir/aiperf.lock.json"
aiperf_python=${AIPERF_PYTHON:-/models/.bench-tools/aiperf-0.12.0-6ed4823d/venv/bin/python}
uv_bin=${AIPERF_UV_BIN:-/models/.bench-tools/uv-0.12.3-linux-x86_64/uv}
campaign_root=${AIPERF_CAMPAIGN_ROOT:-/models/bench/results/aiperf-greenfield/campaigns/$campaign}
block_root="$campaign_root/$phase/$pair-$role"

test -x "$aiperf_python"
test -x "$uv_bin"
if [ -e "$block_root" ]; then
  echo "error: immutable block already exists: $block_root" >&2
  exit 2
fi
mkdir -p "$block_root"

"$uv_bin" run --no-project --python "$aiperf_python" "$script_dir/block_manifest.py" \
  --panel "$panel" --lock "$lock" --phase "$phase" --pair "$pair" \
  --role "$role" --campaign "$campaign" \
  --config "$config_dir/warmup-coverage.yaml" \
  --config "$config_dir/decode-steady.yaml" \
  --config "$config_dir/prefill-cold.yaml" \
  --output "$block_root/block-manifest.json"

AIPERF_RANDOM_SEED=$("$uv_bin" run --no-project --python "$aiperf_python" \
  "$script_dir/block_manifest.py" --panel "$panel" --lock "$lock" \
  --phase "$phase" --pair "$pair" --role "$role" --campaign "$campaign" \
  --field aiperf_random_seed)
SAMPLING_SEED=$("$uv_bin" run --no-project --python "$aiperf_python" \
  "$script_dir/block_manifest.py" --panel "$panel" --lock "$lock" \
  --phase "$phase" --pair "$pair" --role "$role" --campaign "$campaign" \
  --field sampling_seed)
export AIPERF_RANDOM_SEED SAMPLING_SEED
export AIPERF_PYTHON="$aiperf_python"
export AIPERF_WORKERS=1
export AIPERF_RECORD_PROCESSORS=1
export DECODE_TEMPERATURE=1.0
export DECODE_TOP_P=0.95
export INFERENCE_URL=${INFERENCE_URL:-http://127.0.0.1:8000}
export SERVER_METRICS_URL=${SERVER_METRICS_URL:-http://127.0.0.1:8000/metrics}

run_warmup_decode() {
  concurrency=$1
  export AIPERF_ARTIFACT_ROOT="$block_root/warmup/decode"
  export WARMUP_ISL=256
  export WARMUP_OSL=512
  export WARMUP_CONCURRENCY="$concurrency"
  export WARMUP_REQUESTS="$concurrency"
  export WARMUP_TEMPERATURE=1.0
  export WARMUP_TOP_P=0.95
  "$script_dir/run-in-pod.sh" "$config_dir/warmup-coverage.yaml" "c$concurrency"
}

run_warmup_prefill() {
  label=$1
  input_length=$2
  concurrency=$3
  export AIPERF_ARTIFACT_ROOT="$block_root/warmup/prefill"
  export WARMUP_ISL="$input_length"
  export WARMUP_OSL=1
  export WARMUP_CONCURRENCY="$concurrency"
  export WARMUP_REQUESTS="$concurrency"
  export WARMUP_TEMPERATURE=0.0
  export WARMUP_TOP_P=1.0
  "$script_dir/run-in-pod.sh" "$config_dir/warmup-coverage.yaml" "$label"
}

for concurrency in 1 2 4 8 16 32; do
  run_warmup_decode "$concurrency"
done
run_warmup_prefill 8k-c1 8192 1
run_warmup_prefill 8k-c2 8192 2
run_warmup_prefill 8k-c4 8192 4
run_warmup_prefill 64k-c1 65536 1
run_warmup_prefill 128k-c1 131072 1

run_decode() {
  concurrency=$1
  output_length=$2
  export AIPERF_ARTIFACT_ROOT="$block_root/decode"
  export DECODE_CONCURRENCY="$concurrency"
  export DECODE_REQUESTS="$concurrency"
  export DECODE_ISL=256
  export DECODE_OSL="$output_length"
  "$script_dir/run-in-pod.sh" "$config_dir/decode-steady.yaml" "c$concurrency"
  cell="$block_root/decode/c$concurrency"
  "$uv_bin" run --no-project --python "$aiperf_python" \
    "$script_dir/analyze_server_metrics.py" \
    --summary "$cell/server_metrics_export.json" \
    --jsonl "$cell/server_metrics_export.jsonl" \
    --target-concurrency "$concurrency" \
    --output "$cell/decode-analysis.json"
}

# Legacy paired-block shapes produce a natural terminal occupancy drop after a
# minimum 30-second exact-occupancy analysis window. The canonical publication
# runner uses one fixed 16K-input/4K-output shape instead.
run_decode 1 18432
run_decode 2 16384
run_decode 4 12288
run_decode 8 8192
run_decode 16 6144
run_decode 32 4096

run_prefill() {
  label=$1
  input_length=$2
  concurrency=$3
  requests=$4
  export AIPERF_ARTIFACT_ROOT="$block_root/prefill"
  export PREFILL_ISL="$input_length"
  export PREFILL_CONCURRENCY="$concurrency"
  export PREFILL_REQUESTS="$requests"
  "$script_dir/run-in-pod.sh" "$config_dir/prefill-cold.yaml" "$label"
  cell="$block_root/prefill/$label"
  "$uv_bin" run --no-project --python "$aiperf_python" "$script_dir/analyze_prefill.py" \
    --summary "$cell/profile_export_aiperf.json" \
    --records "$cell/profile_export.jsonl" \
    --server-summary "$cell/server_metrics_export.json" \
    --target-isl "$input_length" \
    --target-concurrency "$concurrency" \
    --expected-requests "$requests" \
    --output "$cell/prefill-analysis.json"
}

run_prefill 8k-c1 8192 1 40
run_prefill 8k-c2 8192 2 40
run_prefill 8k-c4 8192 4 40
run_prefill 64k-c1 65536 1 10
run_prefill 128k-c1 131072 1 5

"$uv_bin" run --no-project --python "$aiperf_python" \
  "$script_dir/build_block_results.py" --block "$block_root" \
  --output "$block_root/block-results.json"
date -u +%Y-%m-%dT%H:%M:%SZ > "$block_root/completed-at-utc.txt"
(
  cd "$block_root"
  find . -type f ! -name SHA256SUMS -exec sha256sum '{}' \; | LC_ALL=C sort \
    > SHA256SUMS
)
echo "completed performance block: $block_root"
