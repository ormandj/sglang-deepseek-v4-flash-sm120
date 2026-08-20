#!/usr/bin/env bash
# Validated serving configuration for DeepSeek-V4-Flash-0731 on two RTX PRO 6000
# Blackwell GPUs (SM120, TP=2). Every environment variable and server flag below
# is part of the configuration this image was validated with; change them only
# deliberately.
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to a local DeepSeek-V4-Flash-0731 snapshot directory}"
: "${CACHE_DIR:?set CACHE_DIR to a persistent, image-specific cache directory}"

IMAGE=${IMAGE:-ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.6.0-rc.3}
PORT=${PORT:-8000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
TP_SIZE=${TP_SIZE:-2}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-786432}
CONTAINER_NAME=${CONTAINER_NAME:-dsv4-flash-sglang}
SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=${SGLANG_ENABLE_PCIE_IPC_ALLREDUCE:-1}
SGLANG_PCIE_IPC_MAX_NUMEL=${SGLANG_PCIE_IPC_MAX_NUMEL:-786432}
SGLANG_PCIE_IPC_AUTOTUNE=${SGLANG_PCIE_IPC_AUTOTUNE:-1}
# A request that omits max_tokens is admitted as unbounded and clamped only by
# the remaining context, so one client can hold the KV pool for hours. This
# ceiling is the model card's recommended 384K maximum output for the high and
# max reasoning effort levels; it never truncates a conforming request.
SGLANG_MAX_NEW_TOKENS_LIMIT=${SGLANG_MAX_NEW_TOKENS_LIMIT:-393216}

# Hierarchical KV cache (HiCache). Off by default: it is only worth its cost
# when several long-context conversations share the deployment. See RUN.md.
#
#   HICACHE=1                     enable the host (L2) tier
#   HICACHE_RATIO=N               host pool as a multiple of the device pool
#   HICACHE_STORAGE_DIR=/path     also enable the on-disk (L3) tier at this path
#
# HICACHE_RATIO is the sizing knob: the host pool holds RATIO x the device
# pool's tokens. Divide your intended concurrent-session count's total context
# by the device pool (reported as max_total_num_tokens at startup) to pick it.
# Host memory is pinned and therefore not reclaimable by the kernel.
HICACHE=${HICACHE:-0}
HICACHE_RATIO=${HICACHE_RATIO:-10}
HICACHE_WRITE_POLICY=${HICACHE_WRITE_POLICY:-write_through_selective}
HICACHE_STORAGE_DIR=${HICACHE_STORAGE_DIR:-}
HICACHE_STORAGE_PREFETCH_POLICY=${HICACHE_STORAGE_PREFETCH_POLICY:-timeout}

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "MODEL_DIR is not a directory: $MODEL_DIR" >&2
  exit 2
fi
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "MODEL_DIR does not look like a model snapshot (no config.json): $MODEL_DIR" >&2
  exit 2
fi
if [[ -e "$CACHE_DIR" && ! -d "$CACHE_DIR" ]]; then
  echo "CACHE_DIR exists but is not a directory: $CACHE_DIR" >&2
  exit 2
fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "PORT must be an integer from 1 through 65535" >&2
  exit 2
fi
if [[ -z "$IMAGE" || "$IMAGE" =~ [[:space:]] ]]; then
  echo "IMAGE must be a non-empty image reference without whitespace" >&2
  exit 2
fi
if ! [[ "$TP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "TP_SIZE must be a positive integer" >&2
  exit 2
fi
if ! [[ "$CONTEXT_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
  echo "CONTEXT_LENGTH must be a positive integer" >&2
  exit 2
fi
if [[ "$HICACHE" != "0" && "$HICACHE" != "1" ]]; then
  echo "HICACHE must be 0 or 1" >&2
  exit 2
fi
if [[ "$HICACHE" == "1" ]]; then
  if ! [[ "$HICACHE_RATIO" =~ ^[0-9]+([.][0-9]+)?$ ]] \
     || [[ "$HICACHE_RATIO" == 0 ]]; then
    echo "HICACHE_RATIO must be a positive number" >&2
    exit 2
  fi
  case "$HICACHE_WRITE_POLICY" in
    write_through|write_through_selective|write_back) ;;
    *) echo "HICACHE_WRITE_POLICY must be write_through, write_through_selective, or write_back" >&2
       exit 2 ;;
  esac
  case "$HICACHE_STORAGE_PREFETCH_POLICY" in
    best_effort|wait_complete|timeout) ;;
    *) echo "HICACHE_STORAGE_PREFETCH_POLICY must be best_effort, wait_complete, or timeout" >&2
       exit 2 ;;
  esac
  if [[ -n "$HICACHE_STORAGE_DIR" ]]; then
    if [[ -e "$HICACHE_STORAGE_DIR" && ! -d "$HICACHE_STORAGE_DIR" ]]; then
      echo "HICACHE_STORAGE_DIR exists but is not a directory: $HICACHE_STORAGE_DIR" >&2
      exit 2
    fi
    mkdir -p "$HICACHE_STORAGE_DIR"
  fi
elif [[ -n "$HICACHE_STORAGE_DIR" ]]; then
  echo "HICACHE_STORAGE_DIR is set but HICACHE=0; the storage tier requires HICACHE=1" >&2
  exit 2
fi

mkdir -p "$CACHE_DIR"
model_dir=$(cd "$MODEL_DIR" && pwd)
cache_dir=$(cd "$CACHE_DIR" && pwd)

# The in-container model path is fixed so the compiled-kernel cache in
# CACHE_DIR stays valid regardless of where the snapshot lives on the host.
# The FP4 indexer flag below requires #29927's DeepGEMM paged-MQA path, so the
# two indexer selectors explicitly opt out of the upstream SM120 fallbacks.
container_model_path=/models/deepseek-ai/DeepSeek-V4-Flash-0731
# Fixed in-container path for the optional on-disk KV tier, so the host
# location is free to move without changing the recorded server arguments.
container_hicache_path=/hicache

# HiCache contributes both docker and server arguments, so they are collected
# here rather than inlined into the command below.
hicache_docker_args=()
hicache_serve_args=()
if [[ "$HICACHE" == "1" ]]; then
  hicache_serve_args+=(
    --enable-hierarchical-cache
    --hicache-ratio "$HICACHE_RATIO"
    --hicache-write-policy "$HICACHE_WRITE_POLICY"
  )
  if [[ -n "$HICACHE_STORAGE_DIR" ]]; then
    hicache_storage_dir=$(cd "$HICACHE_STORAGE_DIR" && pwd)
    hicache_docker_args+=(
      --volume "${hicache_storage_dir}:${container_hicache_path}"
      --env SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$container_hicache_path"
    )
    hicache_serve_args+=(
      --hicache-storage-backend file
      --hicache-storage-prefetch-policy "$HICACHE_STORAGE_PREFETCH_POLICY"
    )
  fi
fi

exec docker run --rm \
  --name "$CONTAINER_NAME" \
  --entrypoint sglang \
  --gpus all \
  --shm-size 64g \
  --ulimit memlock=-1 \
  --publish "${PORT}:8000" \
  --volume "${model_dir}:${container_model_path}:ro" \
  --volume "${cache_dir}:/root/.cache" \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  --env TILELANG_CACHE_DIR=/root/.cache/tilelang \
  --env TRITON_CACHE_DIR=/root/.cache/triton \
  --env SGLANG_MAX_NEW_TOKENS_LIMIT="$SGLANG_MAX_NEW_TOKENS_LIMIT" \
  --env SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0 \
  --env SGLANG_OPT_USE_TILELANG_INDEXER=0 \
  --env SGLANG_OPT_DEEPGEMM_HC_PRENORM=1 \
  --env SGLANG_OPT_FUSE_MHC_POST_PRE=1 \
  --env SGLANG_OPT_FP8_WO_A_GEMM=1 \
  --env SGLANG_ENABLE_PCIE_IPC_ALLREDUCE="$SGLANG_ENABLE_PCIE_IPC_ALLREDUCE" \
  --env SGLANG_PCIE_IPC_MAX_NUMEL="$SGLANG_PCIE_IPC_MAX_NUMEL" \
  --env SGLANG_PCIE_IPC_AUTOTUNE="$SGLANG_PCIE_IPC_AUTOTUNE" \
  "${hicache_docker_args[@]+"${hicache_docker_args[@]}"}" \
  "$IMAGE" \
  serve \
  --model-path "$container_model_path" \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code \
  --tensor-parallel-size "$TP_SIZE" \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.93 \
  --context-length "$CONTEXT_LENGTH" \
  --chunked-prefill-size 8192 \
  --cuda-graph-max-bs-decode 32 \
  --max-running-requests 48 \
  --disable-custom-all-reduce \
  --fp8-gemm-backend auto \
  --enable-deepseek-v4-fp4-indexer \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --enable-metrics \
  --enable-cache-report \
  --warmups prefill_shapes,decode_paths \
  --sleep-on-idle \
  "${hicache_serve_args[@]+"${hicache_serve_args[@]}"}" \
  --host 0.0.0.0 \
  --port 8000
