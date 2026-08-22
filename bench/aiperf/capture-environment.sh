#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: $0 CONFIG_FILE OUTPUT_DIR" >&2
  exit 2
fi

config_file=$1
output_dir=$2
mkdir -p "$output_dir"

{
  echo "captured_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "pod_name=${HOSTNAME:-unknown}"
  echo "namespace=$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace 2>/dev/null || echo unknown)"
  echo "image_ref=${BENCH_IMAGE_REF:-unknown}"
  echo "gitops_revision=${BENCH_GITOPS_REVISION:-unknown}"
  echo "project_revision=${BENCH_PROJECT_REVISION:-unknown}"
  echo "aiperf_revision=${AIPERF_REVISION:-unknown}"
  echo "model_revision=${BENCH_MODEL_REVISION:-unknown}"
  echo "benchmark_dp_size=${BENCH_DP_SIZE:-1}"
  echo "model_name=${MODEL_NAME:-unknown}"
  echo "tokenizer_path=${TOKENIZER_PATH:-unknown}"
  echo "max_context_length=${MAX_CONTEXT_LENGTH:-unknown}"
  echo "agentx_duration_seconds=${AGENTX_DURATION_SECONDS:-unknown}"
  echo "agentx_concurrency=${AGENTX_CONCURRENCY:-unknown}"
  echo "aiperf_workers=${AIPERF_WORKERS:-unknown}"
  echo "aiperf_record_processors=${AIPERF_RECORD_PROCESSORS:-unknown}"
  echo "aiperf_random_seed=${AIPERF_RANDOM_SEED:-unknown}"
  echo "sampling_seed=${SAMPLING_SEED:-unknown}"
} > "$output_dir/environment.txt"

server_command=$(tr '\000' ' ' < /proc/1/cmdline)
printf '%s' "$server_command" \
  | sed -E 's/(--(api-key|admin-api-key|ssl-keyfile-password|hf-token|access-token|auth-token|password|secret))([=[:space:]]+)[^[:space:]]+/\1\3<redacted>/g' \
  > "$output_dir/server-command.txt"
unset server_command
printf '\n' >> "$output_dir/server-command.txt"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,pstate,power.limit,memory.total,driver_version --format=csv,noheader > "$output_dir/gpus.csv"
lscpu > "$output_dir/lscpu.txt"
cp /etc/os-release "$output_dir/os-release.txt"
if [ -n "${BENCH_API_KEY:-}" ]; then
  curl -fsS -H "Authorization: Bearer ${BENCH_API_KEY}" \
    http://127.0.0.1:8000/v1/models > "$output_dir/models.json"
else
  curl -fsS http://127.0.0.1:8000/v1/models > "$output_dir/models.json"
fi
cp "$config_file" "$output_dir/benchmark-config.yaml"
sha256sum "$config_file" > "$output_dir/config.sha256"

if [ -n "${AIPERF_PYTHON:-}" ] && [ -x "${AIPERF_PYTHON}" ]; then
  uv_bin=${AIPERF_UV_BIN:-/models/.bench-tools/uv-0.12.3-linux-x86_64/uv}
  test -x "$uv_bin"
  "$uv_bin" pip freeze --python "$AIPERF_PYTHON" > "$output_dir/aiperf-environment.txt"
fi
