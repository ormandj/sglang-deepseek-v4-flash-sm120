# Running the image

This candidate targets Linux x86_64 and NVIDIA RTX PRO 6000 Blackwell
(SM120). TP2 is the primary qualification topology. TP4 must pass the release
regression gate before public promotion.

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
  ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.2.0-rc.0
```

## Model and cache

```bash
export MODEL_DIR=/srv/models/DeepSeek-V4-Flash-0731
uvx --from huggingface-hub hf download \
  deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  --local-dir "$MODEL_DIR"

export CACHE_DIR=/srv/cache/sglang-dsv4-0731-v10
mkdir -p "$CACHE_DIR"
```

`v10` is a fresh cache schema. Do not reuse a `v0.1.0` cache directory for
this reimage.

## Start TP2

```bash
MODEL_DIR="$MODEL_DIR" CACHE_DIR="$CACHE_DIR" \
  ./examples/serve-dsv4-0731.sh
```

The script defaults to the candidate tag and two GPUs. `IMAGE`, `PORT`,
`CUDA_VISIBLE_DEVICES`, and `CONTAINER_NAME` may be overridden. Record all
overrides with results.

The runtime intentionally does not enable FlashInfer PCIe-IPC all-reduce,
PCIe-IPC all-gather, or TRT/MNNVL fusion. The reimage establishes an upstream-
default NCCL baseline before communication experiments are reconsidered.

The first start compiles SM120 kernels into `$CACHE_DIR`. Reuse the same v10
cache for subsequent starts of the identical image; do not time compilation as
serving startup or inference.

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
`0.95`. The deterministic performance gate uses temperature `0` to isolate
engine behavior; the AgentX trace replay covers the agentic request shape.

## Capacity

The TP2 script uses `--context-length 774656`. Confirm the actual pool after
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
- five repetitions at each published supported concurrency;
- separate cache-cold prefill diagnostics;
- C1 and C8 AgentX programming-trace replay;
- correctness and near-context-limit gates.

Results from the previous benchmark shape are historical and are not mixed
with `v0.2.0` measurements.
