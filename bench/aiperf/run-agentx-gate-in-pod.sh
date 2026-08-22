#!/bin/sh
set -eu

if [ -z "${KUBERNETES_SERVICE_HOST:-}" ]; then
  echo "error: AgentX gates must execute inside the serving pod" >&2
  exit 2
fi
if [ "$#" -ne 2 ]; then
  echo "usage: $0 CAMPAIGN_ID BUILD_ID" >&2
  exit 2
fi

campaign=$1
build_id=$2
for value in "$campaign" "$build_id"; do
  case "$value" in
    *[!A-Za-z0-9._-]*|'') echo "error: invalid identifier: $value" >&2; exit 2 ;;
  esac
done

: "${BENCH_IMAGE_REF:?BENCH_IMAGE_REF must identify the immutable image}"
: "${BENCH_GITOPS_REVISION:?BENCH_GITOPS_REVISION must be set}"
: "${BENCH_PROJECT_REVISION:?BENCH_PROJECT_REVISION must be set}"
: "${AIPERF_REVISION:?AIPERF_REVISION must be set}"
: "${BENCH_MODEL_REVISION:?BENCH_MODEL_REVISION must be set}"

export MODEL_NAME=${MODEL_NAME:-deepseek-v4-flash}
export TOKENIZER_PATH=${TOKENIZER_PATH:-/models/deepseek-ai/DeepSeek-V4-Flash-0731}
export INFERENCE_URL=${INFERENCE_URL:-http://127.0.0.1:8000}
export SERVER_METRICS_URL=${SERVER_METRICS_URL:-http://127.0.0.1:8000/metrics}
export MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH:-774656}
export AGENTX_DURATION_SECONDS=${AGENTX_DURATION_SECONDS:-900}
export AIPERF_WORKERS=${AIPERF_WORKERS:-8}
export AIPERF_RECORD_PROCESSORS=${AIPERF_RECORD_PROCESSORS:-2}
export BENCH_API_KEY=${BENCH_API_KEY:-${SGLANG_API_KEY:-${VLLM_API_KEY:-}}}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
lock="$script_dir/aiperf.lock.json"
uv_bin=${AIPERF_UV_BIN:-/models/.bench-tools/uv-0.12.3-linux-x86_64/uv}
aiperf_python=${AIPERF_PYTHON:-/models/.bench-tools/aiperf-0.12.0-6ed4823d/venv/bin/python}
campaign_root=${AIPERF_CAMPAIGN_ROOT:-/models/bench/results/aiperf-greenfield/agentx-gates}
gate_root="$campaign_root/$campaign/$build_id"

test -x "$uv_bin"
test -x "$aiperf_python"
grep -F "\"commit\": \"$AIPERF_REVISION\"" "$lock" >/dev/null

validate_cell() {
  artifact_dir=$1
  summary="$artifact_dir/profile_export_aiperf.json"
  environment="$artifact_dir/environment.txt"

  test -f "$summary"
  test -f "$environment"
  cmp -s "$script_dir/configs/agentx-mvp.yaml" "$artifact_dir/benchmark-config.yaml"
  grep -Fqx "image_ref=$BENCH_IMAGE_REF" "$environment"
  grep -Fqx "gitops_revision=$BENCH_GITOPS_REVISION" "$environment"
  grep -Fqx "project_revision=$BENCH_PROJECT_REVISION" "$environment"
  grep -Fqx "aiperf_revision=$AIPERF_REVISION" "$environment"
  grep -Fqx "model_revision=$BENCH_MODEL_REVISION" "$environment"
  grep -Fqx "max_context_length=$MAX_CONTEXT_LENGTH" "$environment"
  grep -Fqx "agentx_duration_seconds=$AGENTX_DURATION_SECONDS" "$environment"
  grep -Fqx "agentx_concurrency=$AGENTX_CONCURRENCY" "$environment"
  grep -Fqx "aiperf_workers=$AIPERF_WORKERS" "$environment"
  grep -Fqx "aiperf_record_processors=$AIPERF_RECORD_PROCESSORS" "$environment"
  grep -Fqx "aiperf_random_seed=$AIPERF_RANDOM_SEED" "$environment"
  grep -Fqx "sampling_seed=$SAMPLING_SEED" "$environment"
  "$uv_bin" run --no-project --python "$aiperf_python" python -c '
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text())
metadata = d.get("metadata", {})
if metadata.get("submission_valid") is not True:
    reasons = metadata.get("submission_invalid_reasons")
    raise SystemExit(f"invalid AgentX result: {reasons}")
if d.get("request_error_rate", {}).get("avg") != 0:
    raise SystemExit("AgentX result contains request errors")
' "$summary"
}

if [ -e "$gate_root/completed-at-utc.txt" ] || [ -e "$gate_root/SHA256SUMS" ]; then
  echo "error: immutable completed AgentX gate already exists: $gate_root" >&2
  exit 2
fi
mkdir -p "$gate_root"

for concurrency in 1 8; do
  export AIPERF_RANDOM_SEED=2026081201
  export SAMPLING_SEED=2026081201
  export AGENTX_CONCURRENCY=$concurrency
  export AIPERF_ARTIFACT_ROOT="$gate_root/c$concurrency"
  artifact_dir="$AIPERF_ARTIFACT_ROOT/profile"
  if [ -f "$artifact_dir/profile_export_aiperf.json" ]; then
    validate_cell "$artifact_dir"
    echo "retained completed AgentX C$concurrency cell: $artifact_dir"
    continue
  fi
  if [ -e "$AIPERF_ARTIFACT_ROOT" ]; then
    echo "error: incomplete AgentX C$concurrency cell exists: $AIPERF_ARTIFACT_ROOT" >&2
    exit 2
  fi
  "$script_dir/run-in-pod.sh" "$script_dir/configs/agentx-mvp.yaml" profile
  validate_cell "$artifact_dir"
done

date -u +%Y-%m-%dT%H:%M:%SZ > "$gate_root/completed-at-utc.txt"
(
  cd "$gate_root"
  find . -type f ! -name SHA256SUMS -exec sha256sum '{}' \; | LC_ALL=C sort \
    > SHA256SUMS
)
echo "completed AgentX gate: $gate_root"
