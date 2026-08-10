#!/bin/sh
set -eu

if [ -z "${KUBERNETES_SERVICE_HOST:-}" ]; then
  echo "error: engine gates must execute inside the serving pod" >&2
  exit 2
fi
if [ "$#" -ne 3 ]; then
  echo "usage: $0 CAMPAIGN_ID BUILD_ID quick|decode-supplement|qualification" >&2
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
  quick|decode-supplement|qualification) ;;
  *) echo "error: mode must be quick, decode-supplement, or qualification" >&2; exit 2 ;;
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
aiperf_python=${AIPERF_PYTHON:-/models/.bench-tools/aiperf-0.12.0-03c9c6dd/venv/bin/python}
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
  lower_context=$4
  upper_context=$5
  repetition=1
  while [ "$repetition" -le "$repetitions" ]; do
    seed=$((2026081200 + repetition))
    run_id=$(printf 'r%02d' "$repetition")
    export AIPERF_RANDOM_SEED="$seed"
    export SAMPLING_SEED="$seed"
    export AIPERF_ARTIFACT_ROOT="$gate_root/decode/c$concurrency"
    export DECODE_CONCURRENCY="$concurrency"
    export DECODE_ISL=256
    export DECODE_OSL="$output_length"
    "$script_dir/run-in-pod.sh" "$config_dir/decode-engine.yaml" "$run_id"
    cell="$gate_root/decode/c$concurrency/$run_id"
    if [ "$bench_engine" = vllm ]; then
      "$uv_bin" run --no-project --python "$aiperf_python" \
        "$script_dir/analyze_vllm_server_metrics.py" \
        --summary "$cell/server_metrics_export.json" \
        --jsonl "$cell/server_metrics_export.jsonl" \
        --target-concurrency "$concurrency" \
        --average-context-lower "$lower_context" \
        --average-context-upper "$upper_context" \
        --minimum-window-seconds 10 \
        --output "$cell/decode-analysis.json"
    else
      "$uv_bin" run --no-project --python "$aiperf_python" \
        "$script_dir/analyze_server_metrics.py" \
        --summary "$cell/server_metrics_export.json" \
        --jsonl "$cell/server_metrics_export.jsonl" \
        --target-concurrency "$concurrency" \
        --average-context-lower "$lower_context" \
        --average-context-upper "$upper_context" \
        --minimum-window-seconds 10 \
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
    if [ "$bench_engine" = vllm ]; then
      decode_shapes='1:3:12288:2048:8192 8:2:6144:1280:4352'
    else
      decode_shapes='1:3:12288:2048:8192 8:2:6144:1280:4352 32:1:3072:768:2304'
    fi
    prefill_shapes='8k-c1:8192:1:8 64k-c1:65536:1:3 128k-c1:130816:1:2'
    ;;
  decode-supplement)
    decode_shapes='2:6:10240:2048:7168 4:4:8192:1536:5632 16:2:4096:1024:3072'
    prefill_shapes=''
    ;;
  qualification)
    if [ "$bench_engine" = vllm ]; then
      decode_shapes='1:8:12288:2048:8192 2:6:10240:2048:7168 4:4:8192:1536:5632 8:3:6144:1280:4352 16:2:4096:1024:3072'
    else
      decode_shapes='1:8:12288:2048:8192 2:6:10240:2048:7168 4:4:8192:1536:5632 8:3:6144:1280:4352 16:2:4096:1024:3072 32:2:3072:768:2304'
    fi
    prefill_shapes='8k-c1:8192:1:20 8k-c2:8192:2:20 8k-c4:8192:4:20 64k-c1:65536:1:5 128k-c1:130816:1:3'
    ;;
esac

for shape in $decode_shapes; do
  old_ifs=$IFS
  IFS=:
  set -- $shape
  IFS=$old_ifs
  run_warmup "decode-c$1" 256 512 "$1"
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
  run_decode "$1" "$2" "$3" "$4" "$5"
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
