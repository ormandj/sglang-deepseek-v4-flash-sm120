#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest="$repo/release.json"

for tool in jq grep find wc tr; do
  command -v "$tool" >/dev/null || {
    echo "missing required tool: $tool" >&2
    exit 1
  }
done

candidate_tag=$(jq -er '.candidate_tag' "$manifest")
cache_schema=$(jq -er '.cache_schema' "$manifest")
product=$(jq -er '.product' "$manifest")
image="ghcr.io/ormandj/${product}:${candidate_tag}"

readme="$repo/README.md"
run_guide="$repo/RUN.md"
benchmarks="$repo/BENCHMARKS.md"
changelog="$repo/CHANGELOG.md"
agents="$repo/AGENTS.md"
launcher="$repo/examples/serve-dsv4-0731.sh"

for file in "$readme" "$run_guide" "$benchmarks" "$changelog" "$agents"; do
  [[ -s "$file" ]] || {
    echo "required documentation is missing or empty: ${file#$repo/}" >&2
    exit 1
  }
done

require_text() {
  local file=$1 expected=$2
  grep -F -- "$expected" "$file" >/dev/null || {
    echo "${file#$repo/} is missing current value: $expected" >&2
    exit 1
  }
}

require_text "$readme" "$image"
require_text "$readme" "/srv/cache/sglang-dsv4-0731-${cache_schema}"
require_text "$run_guide" "/srv/cache/sglang-dsv4-0731-${cache_schema}"
require_text "$changelog" "## ${candidate_tag}"

[[ $(grep -Fc -- 'docker run --rm' "$readme") -eq 1 ]] || {
  echo "README.md must contain exactly one canonical direct-Docker command" >&2
  exit 1
}
if grep -F -- 'docker run --rm' "$run_guide" >/dev/null; then
  echo "RUN.md must link to, not duplicate, the canonical Docker command" >&2
  exit 1
fi

critical_launcher_values=(
  'CONTEXT_LENGTH=${CONTEXT_LENGTH:-786432}'
  'SGLANG_MAX_NEW_TOKENS_LIMIT=${SGLANG_MAX_NEW_TOKENS_LIMIT:-393216}'
  'SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=${SGLANG_ENABLE_PCIE_IPC_ALLREDUCE:-1}'
  'SGLANG_PCIE_IPC_MAX_NUMEL=${SGLANG_PCIE_IPC_MAX_NUMEL:-786432}'
  'SGLANG_PCIE_IPC_AUTOTUNE=${SGLANG_PCIE_IPC_AUTOTUNE:-1}'
  '--mem-fraction-static 0.93'
  '--chunked-prefill-size 8192'
  '--cuda-graph-max-bs-decode 32'
  '--max-running-requests 48'
  '--speculative-dspark-block-size 5'
  '--warmups prefill_shapes,decode_paths'
)
for value in "${critical_launcher_values[@]}"; do
  require_text "$launcher" "$value"
done

critical_readme_values=(
  '--env SGLANG_MAX_NEW_TOKENS_LIMIT=393216'
  '--env SGLANG_OPT_DEEPGEMM_HC_PRENORM=1'
  '--env SGLANG_OPT_FUSE_MHC_POST_PRE=1'
  '--env SGLANG_OPT_FP8_WO_A_GEMM=1'
  '--env SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1'
  '--env SGLANG_PCIE_IPC_MAX_NUMEL=786432'
  '--env SGLANG_PCIE_IPC_AUTOTUNE=1'
  '--tensor-parallel-size 2'
  '--mem-fraction-static 0.93'
  '--context-length 786432'
  '--chunked-prefill-size 8192'
  '--cuda-graph-max-bs-decode 32'
  '--max-running-requests 48'
  '--speculative-dspark-block-size 5'
  '--warmups prefill_shapes,decode_paths'
)
for value in "${critical_readme_values[@]}"; do
  require_text "$readme" "$value"
done

results_dir="$repo/bench/results"
sglang_summary="$results_dir/sglang-${candidate_tag}-publication-summary.json"
sglang_quality="$results_dir/sglang-${candidate_tag}-gsm8k.json"
vllm_summary="$results_dir/vllm-r33-publication-summary.json"
vllm_quality="$results_dir/vllm-r33-gsm8k.json"

for file in "$sglang_summary" "$sglang_quality" "$vllm_summary" "$vllm_quality"; do
  [[ -s "$file" ]] || {
    echo "current result artifact is missing: ${file#$repo/}" >&2
    exit 1
  }
done

result_count=$(find "$results_dir" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d '[:space:]')
[[ "$result_count" == 4 ]] || {
  echo "current main must contain exactly four current comparison artifacts; found $result_count" >&2
  exit 1
}

[[ ! -e "$repo/PERFORMANCE-HISTORY.md" ]] || {
  echo "historical public panels belong in immutable tags, not PERFORMANCE-HISTORY.md" >&2
  exit 1
}

require_rounded() {
  local file=$1 json=$2 expression=$3 digits=$4 label=$5 raw rounded
  raw=$(jq -er "$expression" "$json")
  printf -v rounded "%.*f" "$digits" "$raw"
  require_text "$file" "$rounded"
  printf '%s=%s\n' "$label" "$rounded" >/dev/null
}

for concurrency in 1 2 4 8 16; do
  key="c${concurrency}"
  require_rounded "$readme" "$sglang_summary" ".decode.${key}.engine_forward_passes_per_second.median" 3 "sglang-c${concurrency}-forward"
  require_rounded "$readme" "$vllm_summary" ".decode.${key}.engine_forward_passes_per_second.median" 3 "vllm-c${concurrency}-forward"
  require_rounded "$readme" "$sglang_summary" ".decode.${key}.useful_tokens_per_second.median" 1 "sglang-c${concurrency}-synthetic"
  require_rounded "$readme" "$vllm_summary" ".decode.${key}.synthetic_decode_tokens_per_second.median" 1 "vllm-c${concurrency}-synthetic"

  require_rounded "$benchmarks" "$sglang_summary" ".decode.${key}.engine_forward_passes_per_second.median" 3 "sglang-c${concurrency}-forward"
  require_rounded "$benchmarks" "$vllm_summary" ".decode.${key}.engine_forward_passes_per_second.median" 3 "vllm-c${concurrency}-forward"
  require_rounded "$benchmarks" "$sglang_summary" ".decode.${key}.useful_tokens_per_second.median" 1 "sglang-c${concurrency}-synthetic"
  require_rounded "$benchmarks" "$vllm_summary" ".decode.${key}.synthetic_decode_tokens_per_second.median" 1 "vllm-c${concurrency}-synthetic"
done

require_rounded "$readme" "$sglang_summary" '.decode.c32.engine_forward_passes_per_second.median' 3 'sglang-c32-forward'
require_rounded "$readme" "$sglang_summary" '.decode.c32.useful_tokens_per_second.median' 1 'sglang-c32-synthetic'
require_rounded "$benchmarks" "$sglang_summary" '.decode.c32.engine_forward_passes_per_second.median' 3 'sglang-c32-forward'
require_rounded "$benchmarks" "$sglang_summary" '.decode.c32.useful_tokens_per_second.median' 1 'sglang-c32-synthetic'

for target in 8 32 64 128; do
  key="${target}k-c1"
  require_rounded "$readme" "$sglang_summary" ".prefill[\"${key}\"].prompt_tokens_per_second" 1 "sglang-${target}k-prefill"
  require_rounded "$readme" "$vllm_summary" ".prefill[\"${key}\"].prompt_tokens_per_second" 1 "vllm-${target}k-prefill"
  require_rounded "$benchmarks" "$sglang_summary" ".prefill[\"${key}\"].prompt_tokens_per_second" 1 "sglang-${target}k-prefill"
  require_rounded "$benchmarks" "$vllm_summary" ".prefill[\"${key}\"].prompt_tokens_per_second" 1 "vllm-${target}k-prefill"
done

for file in "$readme" "$benchmarks"; do
  sglang_correct=$(jq -er '.correct' "$sglang_quality")
  vllm_correct=$(jq -er '.correct' "$vllm_quality")
  sglang_pct=$(jq -er '.accuracy * 100' "$sglang_quality")
  vllm_pct=$(jq -er '.accuracy * 100' "$vllm_quality")
  printf -v sglang_pct_rounded '%.2f' "$sglang_pct"
  printf -v vllm_pct_rounded '%.2f' "$vllm_pct"
  normalized=$(tr -d ',' <"$file")
  [[ "$normalized" == *"$sglang_correct"* ]] || {
    echo "${file#$repo/} is missing SGLang correct count: $sglang_correct" >&2
    exit 1
  }
  [[ "$normalized" == *"$vllm_correct"* ]] || {
    echo "${file#$repo/} is missing vLLM correct count: $vllm_correct" >&2
    exit 1
  }
  require_text "$file" "${sglang_pct_rounded}%"
  require_text "$file" "${vllm_pct_rounded}%"
done

echo "documentation contract valid: ${candidate_tag}, cache ${cache_schema}, current result panel"
