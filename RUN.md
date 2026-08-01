# Running the image

This is the exact procedure for the validated platform: Linux x86_64 with two NVIDIA RTX PRO 6000 Blackwell GPUs (compute capability SM120), running the model with TP=2. SM120 is the only platform this image has been validated on; SM121 has not been validated.

## Requirements

- Linux x86_64 with a driver that supports CUDA 13.
- Two RTX PRO 6000 Blackwell GPUs visible to the container.
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) configured, so `--gpus all` works.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for the model download command below.
- Persistent disk for the model snapshot and for the compiled-kernel cache.

Check the toolkit before going further:

```bash
docker run --rm --gpus all \
  --entrypoint nvidia-smi \
  ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:dsv4-0731
```

## Download the model

Keep the snapshot on persistent storage; the run script mounts it read-only.

```bash
export MODEL_DIR=/srv/models/DeepSeek-V4-Flash-0731
uvx --from huggingface-hub hf download \
  deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir "$MODEL_DIR"
```

## Prepare the cache directory

The container's `/root/.cache` holds the FlashInfer JIT modules, TileLang and TVM kernels, TorchInductor output, and DeepGEMM artifacts. Mount it from persistent storage, and use a directory dedicated to this image — cache entries from a different build are not reusable.

```bash
export CACHE_DIR=/srv/cache/sglang-dsv4-0731
mkdir -p "$CACHE_DIR"
```

## Start the server

```bash
MODEL_DIR="$MODEL_DIR" CACHE_DIR="$CACHE_DIR" ./examples/serve-dsv4-0731.sh
```

The script sets the validated environment and server flags. `PORT` (host port, default `8000`), `IMAGE`, `CUDA_VISIBLE_DEVICES` (default `0,1`), and `CONTAINER_NAME` can be overridden; the container always listens on port 8000 internally.

It runs the container with `--shm-size 64g`. SGLang's TP workers exchange tensors through `/dev/shm`, and Docker's 64 MiB default is far too small for this model — the server fails during startup without it. `--ulimit memlock=-1` is required for pinned host memory.

The first start compiles the patched SM120 FlashInfer modules and the TileLang and DeepGEMM kernels into `$CACHE_DIR`, so it takes considerably longer than later starts. Subsequent starts with the same image and the same cache directory reuse those artifacts.

## Health check

`SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0` makes `/health` a plain liveness check rather than a generation request:

```bash
curl -fsS http://localhost:8000/health
```

The server is ready once this returns successfully.

## OpenAI-compatible endpoint

The OpenAI-compatible API is at `http://localhost:8000/v1`, and the served model name is `deepseek-v4-flash`.

```bash
curl -fsS http://localhost:8000/v1/models | jq -r '.data[].id'
```

Smoke test:

```bash
curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
    "max_tokens": 32,
    "temperature": 0
  }' | jq -r '.choices[0].message.content'
```
