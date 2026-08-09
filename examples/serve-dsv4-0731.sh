#!/usr/bin/env bash
# Validated serving configuration for DeepSeek-V4-Flash-0731 on two RTX PRO 6000
# Blackwell GPUs (SM120, TP=2). Every environment variable and server flag below
# is part of the configuration this image was validated with; change them only
# deliberately.
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to a local DeepSeek-V4-Flash-0731 snapshot directory}"
: "${CACHE_DIR:?set CACHE_DIR to a persistent, image-specific cache directory}"

IMAGE=${IMAGE:-ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.1.0-rc.2}
PORT=${PORT:-8000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
CONTAINER_NAME=${CONTAINER_NAME:-dsv4-flash-sglang}

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

mkdir -p "$CACHE_DIR"
model_dir=$(cd "$MODEL_DIR" && pwd)
cache_dir=$(cd "$CACHE_DIR" && pwd)

# The in-container model path is fixed so the compiled-kernel cache in
# CACHE_DIR stays valid regardless of where the snapshot lives on the host.
container_model_path=/models/deepseek-ai/DeepSeek-V4-Flash-0731

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
  --env SGLANG_OPT_USE_TILELANG_INDEXER=0 \
  --env SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0 \
  --env SGLANG_OPT_DEEPGEMM_HC_PRENORM=1 \
  --env TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  --env TILELANG_CACHE_DIR=/root/.cache/tilelang \
  --env TVM_CACHE_DIR=/root/.cache/tvm \
  "$IMAGE" \
  serve \
  --model-path "$container_model_path" \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.93 \
  --context-length 774656 \
  --chunked-prefill-size 8192 \
  --cuda-graph-max-bs-decode 32 \
  --max-running-requests 48 \
  --fp8-gemm-backend auto \
  --enable-deepseek-v4-fp4-indexer \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --enable-metrics \
  --enable-cache-report \
  --sleep-on-idle \
  --host 0.0.0.0 \
  --port 8000
