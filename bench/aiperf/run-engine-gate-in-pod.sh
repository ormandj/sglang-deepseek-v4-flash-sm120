#!/bin/sh
set -eu

if [ -z "${KUBERNETES_SERVICE_HOST:-}" ]; then
  echo "error: engine gates must execute inside the serving pod" >&2
  exit 2
fi
if [ "$#" -ne 3 ]; then
  echo "usage: $0 CAMPAIGN_ID BUILD_ID quick|prefill-quick|decode-supplement|qualification|publication" >&2
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
  quick|prefill-quick|decode-supplement|qualification|publication) ;;
  *) echo "error: mode must be quick, prefill-quick, decode-supplement, qualification, or publication" >&2; exit 2 ;;
esac

: "${BENCH_IMAGE_REF:?BENCH_IMAGE_REF must identify the immutable image}"
: "${BENCH_GITOPS_REVISION:?BENCH_GITOPS_REVISION must be set}"
: "${BENCH_PROJECT_REVISION:?BENCH_PROJECT_REVISION must be set}"
: "${AIPERF_REVISION:?AIPERF_REVISION must be set}"
: "${BENCH_MODEL_REVISION:?BENCH_MODEL_REVISION must be set}"
bench_dp_size=${BENCH_DP_SIZE:-1}
decode_minimum_window_seconds=${DECODE_MINIMUM_WINDOW_SECONDS:-6.5}
case "$bench_dp_size" in
  *[!0-9]*|'') echo "error: BENCH_DP_SIZE must be a positive integer" >&2; exit 2 ;;
esac
if [ "$bench_dp_size" -lt 1 ]; then
  echo "error: BENCH_DP_SIZE must be a positive integer" >&2
  exit 2
fi

export MODEL_NAME=${MODEL_NAME:-deepseek-v4-flash}
export TOKENIZER_PATH=${TOKENIZER_PATH:-/models/deepseek-ai/DeepSeek-V4-Flash-0731}
export INFERENCE_URL=${INFERENCE_URL:-http://127.0.0.1:8000}
export SERVER_METRICS_URL=${SERVER_METRICS_URL:-http://127.0.0.1:8000/metrics}
export AIPERF_WORKERS=1
export AIPERF_RECORD_PROCESSORS=1
bench_engine=${BENCH_ENGINE:-sglang}
case "$bench_engine" in
  sglang|vllm) ;;
  *) echo "error: BENCH_ENGINE must be sglang or vllm" >&2; exit 2 ;;
esac
# A caller may provide BENCH_API_KEY explicitly.  vLLM deployments commonly
# expose VLLM_API_KEY; use it when available without requiring authentication
# for keyless endpoints.  Neither variable is captured in artifacts.
export BENCH_API_KEY=${BENCH_API_KEY:-${VLLM_API_KEY:-}}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
config_dir="$script_dir/configs"
lock="$script_dir/aiperf.lock.json"
aiperf_python=${AIPERF_PYTHON:-/models/.bench-tools/aiperf-0.12.0-6ed4823d/venv/bin/python}
uv_bin=${AIPERF_UV_BIN:-/models/.bench-tools/uv-0.12.3-linux-x86_64/uv}
campaign_root=${AIPERF_CAMPAIGN_ROOT:-/models/bench/results/aiperf-greenfield/engine-gates}
gate_root="$campaign_root/$campaign/$build_id-$mode"

test -x "$aiperf_python"
test -x "$uv_bin"
grep -F "\"commit\": \"$AIPERF_REVISION\"" "$lock" >/dev/null
if [ -e "$gate_root" ]; then
  echo "error: immutable engine gate already exists: $gate_root" >&2
  exit 2
fi
mkdir -p "$gate_root"

run_warmup() {
  label=$1
  input_length=$2
  output_length=$3
  concurrency=$4
  export AIPERF_ARTIFACT_ROOT="$gate_root/warmup"
  export WARMUP_ISL="$input_length"
  export WARMUP_OSL="$output_length"
  if [ "$concurrency" -lt "$bench_dp_size" ]; then
    concurrency=$bench_dp_size
  fi
  export WARMUP_CONCURRENCY="$concurrency"
  export WARMUP_REQUESTS="$concurrency"
  export WARMUP_TEMPERATURE=0.0
  export WARMUP_TOP_P=1.0
  export AIPERF_RANDOM_SEED=2026081200
  export SAMPLING_SEED=2026081200
  "$script_dir/run-in-pod.sh" "$config_dir/warmup-coverage.yaml" "$label"
}

run_decode() {
  concurrency=$1
  repetitions=$2
  output_length=$3
  repetition=1
  while [ "$repetition" -le "$repetitions" ]; do
    seed=$((2026081200 + repetition))
    run_id=$(printf 'r%02d' "$repetition")
    export AIPERF_RANDOM_SEED="$seed"
    export SAMPLING_SEED="$seed"
    export AIPERF_ARTIFACT_ROOT="$gate_root/decode/c$concurrency"
    export DECODE_CONCURRENCY="$concurrency"
    export DECODE_ISL=16384
    export DECODE_OSL="$output_length"
    "$script_dir/run-in-pod.sh" "$config_dir/decode-engine.yaml" "$run_id"
    cell="$gate_root/decode/c$concurrency/$run_id"
    if [ "$bench_engine" = vllm ]; then
      "$uv_bin" run --no-project --python "$aiperf_python" \
        "$script_dir/analyze_vllm_server_metrics.py" \
        --summary "$cell/server_metrics_export.json" \
        --jsonl "$cell/server_metrics_export.jsonl" \
        --target-concurrency "$concurrency" \
        --average-context-lower 17408 \
        --average-context-upper 20480 \
        --minimum-window-seconds "$decode_minimum_window_seconds" \
        --minimum-samples 20 \
        --output "$cell/decode-analysis.json"
    else
      "$uv_bin" run --no-project --python "$aiperf_python" \
        "$script_dir/analyze_server_metrics.py" \
        --summary "$cell/server_metrics_export.json" \
        --jsonl "$cell/server_metrics_export.jsonl" \
        --target-concurrency "$concurrency" \
        --average-context-lower 17408 \
        --average-context-upper 20480 \
        --minimum-window-seconds "$decode_minimum_window_seconds" \
        --minimum-samples 20 \
        --output "$cell/decode-analysis.json"
    fi
    repetition=$((repetition + 1))
  done
}

run_prefill() {
  label=$1
  input_length=$2
  concurrency=$3
  requests=$4
  export AIPERF_RANDOM_SEED=2026081201
  export SAMPLING_SEED=2026081201
  export AIPERF_ARTIFACT_ROOT="$gate_root/prefill"
  export PREFILL_ISL="$input_length"
  export PREFILL_CONCURRENCY="$concurrency"
  export PREFILL_REQUESTS="$requests"
  if [ "$bench_engine" = vllm ]; then
    prefill_config="$config_dir/prefill-cold-vllm.yaml"
    isl_tolerance=${PREFILL_ISL_TOLERANCE:-128}
  else
    prefill_config="$config_dir/prefill-cold.yaml"
    isl_tolerance=${PREFILL_ISL_TOLERANCE:-16}
  fi
  "$script_dir/run-in-pod.sh" "$prefill_config" "$label"
  cell="$gate_root/prefill/$label"
  "$uv_bin" run --no-project --python "$aiperf_python" "$script_dir/analyze_prefill.py" \
    --summary "$cell/profile_export_aiperf.json" \
    --records "$cell/profile_export.jsonl" \
    --server-summary "$cell/server_metrics_export.json" \
    --target-isl "$input_length" \
    --target-concurrency "$concurrency" \
    --expected-requests "$requests" \
    --isl-tolerance "$isl_tolerance" \
    --engine "$bench_engine" \
    --output "$cell/prefill-analysis.json"
}

case "$mode" in
  quick)
    decode_shapes='1:3:4096 4:3:4096 8:3:4096'
    prefill_shapes='8k-c1:8192:1:3 32k-c1:32768:1:3 64k-c1:65536:1:3 128k-c1:130816:1:3'
    ;;
  prefill-quick)
    decode_shapes=''
    prefill_shapes='8k-c1:8192:1:3 32k-c1:32768:1:3 64k-c1:65536:1:3 128k-c1:130816:1:3'
    ;;
  decode-supplement)
    decode_shapes='2:3:4096 16:3:4096'
    prefill_shapes=''
    ;;
  qualification)
    if [ "$bench_engine" = vllm ]; then
      decode_shapes='1:5:4096 2:5:4096 4:5:4096 8:5:4096 16:3:4096'
    else
      decode_shapes='1:5:4096 2:5:4096 4:5:4096 8:5:4096 16:3:4096 32:3:4096'
    fi
    prefill_shapes='8k-c1:8192:1:5 32k-c1:32768:1:5 64k-c1:65536:1:5 128k-c1:130816:1:5'
    ;;
  publication)
    if [ "$bench_engine" = vllm ]; then
      decode_shapes='1:5:4096 2:5:4096 4:5:4096 8:5:4096 16:5:4096'
    else
      decode_shapes='1:5:4096 2:5:4096 4:5:4096 8:5:4096 16:5:4096 32:5:4096'
    fi
    prefill_shapes='8k-c1:8192:1:5 32k-c1:32768:1:5 64k-c1:65536:1:5 128k-c1:130816:1:5'
    ;;
esac

for shape in $decode_shapes; do
  old_ifs=$IFS
  IFS=:
  set -- $shape
  IFS=$old_ifs
  run_warmup "decode-c$1" 16384 256 "$1"
done
for shape in $prefill_shapes; do
  old_ifs=$IFS
  IFS=:
  set -- $shape
  IFS=$old_ifs
  run_warmup "prefill-$1" "$2" 1 "$3"
done

for shape in $decode_shapes; do
  old_ifs=$IFS
  IFS=:
  set -- $shape
  IFS=$old_ifs
  run_decode "$1" "$2" "$3"
done
for shape in $prefill_shapes; do
  old_ifs=$IFS
  IFS=:
  set -- $shape
  IFS=$old_ifs
  run_prefill "$1" "$2" "$3" "$4"
done

"$uv_bin" run --no-project --python "$aiperf_python" \
  "$script_dir/summarize_engine_gate.py" \
  --root "$gate_root" --mode "$mode" --engine "$bench_engine" \
  --build-id "$build_id" \
  --output "$gate_root/summary.json"
date -u +%Y-%m-%dT%H:%M:%SZ > "$gate_root/completed-at-utc.txt"
(
  cd "$gate_root"
  find . -type f ! -name SHA256SUMS -exec sha256sum '{}' \; | LC_ALL=C sort \
    > SHA256SUMS
)
echo "completed engine gate: $gate_root"
