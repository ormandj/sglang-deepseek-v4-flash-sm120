# Running the image

This candidate targets Linux x86_64 and NVIDIA RTX PRO 6000 Blackwell
(SM120). The published measurements cover TP2. The same image and configurable
launcher support larger tensor-parallel topologies, but this repository does
not yet publish TP4 or TP8 performance results.

## Requirements

- A driver compatible with CUDA 13.
- Two or more SM120 GPUs visible to the container.
- Docker and the NVIDIA Container Toolkit.
- `uv` for downloading the model.
- Persistent storage for the model and compiled-kernel cache.

Check GPU passthrough:

```bash
docker run --rm --gpus all \
  --entrypoint nvidia-smi \
  ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.5.0-rc.1
```

## Model and cache

```bash
export MODEL_DIR=/srv/models/DeepSeek-V4-Flash-0731
uvx --from huggingface-hub hf download \
  deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  --local-dir "$MODEL_DIR"

export CACHE_DIR=/srv/cache/sglang-dsv4-0731-v20
mkdir -p "$CACHE_DIR"
```

`v20` is the cache schema for this source composition. Do not reuse a cache
directory from another image source tree.

## Start TP2

```bash
MODEL_DIR="$MODEL_DIR" CACHE_DIR="$CACHE_DIR" \
  ./examples/serve-dsv4-0731.sh
```

The script defaults to the candidate tag, GPUs 0 and 1, TP2, and a 786,432
context length. `IMAGE`, `PORT`, `CUDA_VISIBLE_DEVICES`, `TP_SIZE`,
`CONTEXT_LENGTH`, and `CONTAINER_NAME` may be overridden. The number of devices
listed in `CUDA_VISIBLE_DEVICES` must equal `TP_SIZE`. Record all overrides with
results.

The qualified runtime enables FlashInfer PCIe-IPC all-reduce for eligible
decode reductions. `--disable-custom-all-reduce` disables SGLang's legacy
custom all-reduce so those reductions reach the optional FlashInfer consumer.
Unsupported and prefill-sized reductions use NCCL. PCIe-IPC all-gather is
absent, and TRT/MNNVL fusion remains disabled.

The launch script sets `SGLANG_OPT_DEEPGEMM_HC_PRENORM=1`. The carried SM120
implementation selects DeepGEMM for large-token prefill batches and retains
the existing fallback below its 1,024-token dispatch threshold.

The launch script also sets `SGLANG_OPT_FUSE_MHC_POST_PRE=1`. The fused
implementation and runtime selector are already present in the image; open
[SGLang PR #34019](https://github.com/sgl-project/sglang/pull/34019) changes the
upstream SM120 default and is tracked rather than applied to this image.

The script additionally sets `SGLANG_OPT_FP8_WO_A_GEMM=1` to select the
SM120/SM121 target-model FP8 W_o_A path carried from upstream PR #34018.

The first start compiles SM120 kernels into `$CACHE_DIR`. The server performs
the qualified prefill and decode-path warmups before it reports ready. Reuse
the same v20 cache for subsequent starts of the identical image; do not time
compilation as serving startup or inference.

## Health and API

```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/v1/models | jq -r '.data[].id'
```

A deterministic smoke request:

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

Agentic requests should follow the model card: temperature `1.0` and top-p
`0.95`. The deterministic engine gate uses temperature `0` to isolate engine
behavior.

## Capacity

The TP2 script uses `--context-length 786432`. Confirm the actual pool after
every image, topology, memory-fraction, or graph change:

```bash
curl -fsS http://localhost:8000/get_server_info \
  | jq '{max_total_num_tokens, max_req_input_len}'
```

Do not copy the TP2 context limit to TP4 without checking the TP4 server
information and completing a near-limit request.

## Benchmarking

The executable harness lives in [`bench/aiperf`](bench/aiperf). Clients run
inside the selected serving pod against localhost. The frozen protocol uses:

- identical 16,384-token input and 4,096-token output shapes at C1–C32;
- five sequential fixed-seed repetitions at every published supported
  concurrency;
- separate cache-cold prefill diagnostics;
- full GSM8K correctness.

The published `v0.5.0-rc.1` panel uses five measured repetitions at every
supported concurrency and prefill length. All measured cells run sequentially
on one unchanged process after one warmup per distinct shape.
