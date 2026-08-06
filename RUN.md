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

The script pins `--chunked-prefill-size 8192`, the value the validated configuration resolves to. Prefill-throughput comparisons against other engines or builds are only meaningful at the same chunk size.

It also selects the DeepGEMM DSA indexer with `--enable-deepseek-v4-fp4-indexer`
and `SGLANG_OPT_USE_TILELANG_INDEXER=0` / `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0`.
On SM120 SGLang otherwise forces the TileLang indexer, which works against stock
DeepGEMM; this image ships the SM120-capable DeepGEMM that exposes
`fp8_fp4_paged_mqa_logits`, so it can use the DeepGEMM path instead. Measured on
the validated configuration with raw `input_ids`, warmup discarded, n=5 prefill
and n=10 decode at roughly 1k context:

| cell | TileLang | DeepGEMM | delta |
|---|---:|---:|---|
| prefill 128k | 6,310 | 6,581 | +4.3% |
| decode C1 | 296.4 | 303.5 | +2.4% |

Both indexers make the context-scaled allocation described under KV capacity
below, so this choice does not affect that requirement. Remove the three
settings to fall back to the TileLang default.

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

## Expected KV capacity at TP=2

On the validated two-GPU configuration, SGLang reported `max_total_num_tokens=774656` with FP8 KV cache and `mem-fraction-static=0.93` (0.94 leaves too little headroom for the FlashInfer MoE workspace on this composition and fails at the first real request). This is the total scheduler KV pool shared by all active and cached sequences; it is not 1,108,224 tokens per GPU or per request. The checkpoint's own window is 1,048,576 tokens, which is larger than that pool.

The script passes `--context-length 774656` to match that pool. Left unset, SGLang
uses the checkpoint's full 1,048,576-token window, and the DSA paged-MQA-logits
indexer sizes a float32 `(batch_size, max_seq_len)` buffer from it — roughly
1.4 GiB — irrespective of how long the actual request is. With
`mem-fraction-static 0.93` that allocation does not fit, and the scheduler exits
with a CUDA OOM on the first sufficiently large request:

```text
tilelang_kernel.py:1504 in tilelang_fp8_paged_mqa_logits
    logits = page_table.new_empty((batch_size, max_seq_len), dtype=torch.float32)
torch.OutOfMemoryError: Tried to allocate 1.38 GiB
```

The same allocation exists on the DeepGEMM indexer path, so the failure is not
specific to either implementation. Because a 1M window is larger than the KV pool
can serve anyway, bounding it to the pool size costs no usable context.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set for the same reason: the
OOM reports over 1.5 GiB reserved but unallocated, and expandable segments returns
that fragmentation to the allocator.

If you change `mem-fraction-static`, `kv-cache-dtype`, or the GPU count, re-read
`max_total_num_tokens` from the startup log and set `--context-length` to match it.

Confirm the values after startup:

```bash
curl -fsS http://localhost:8000/get_server_info \
  | jq '{max_total_num_tokens, max_req_input_len}'
```

Expect values near the measurement above when using the documented image, hardware, and launch settings. Available GPU memory, graph geometry, cache settings, or additional GPU consumers can change the calculated KV capacity.

## Recommended request settings

The [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731#how-to-run-locally) recommends `temperature=1.0` with `top_p=0.95` for agentic requests and `top_p=1.0` otherwise. Chat-completion requests may set `reasoning_effort` to `low`, `high`, or `max`.

Treat OpenCode and other tool-using coding harnesses as agentic workloads. Use `max` as the normal coding default:

```json
{
  "reasoning_effort": "max",
  "temperature": 1.0,
  "top_p": 0.95,
  "max_tokens": 393216
}
```

Use `high` when lower latency is more important than maximum deliberation. DeepSeek's published code-agent evaluation setup used `max`.

`393216` is the numeric API value for the 384K-token output budget the model card requires at `high`/`max`. Do not lower it: long reasoning chains consume large output budgets, and a smaller `max_tokens` truncates the model mid-reasoning. This is a client-side request limit; the launch command does not impose it.

## Known issue: single-request generation aborts near 78K generated tokens

This deployment currently cannot honor the full 384K output budget in a single
request. During decode, SGLang's hybrid-SWA accounting holds sliding-window KV
for every generated token (window eviction does not run on this path), so a
single request aborts at roughly `swa-full-tokens-ratio (0.1) x
max_total_num_tokens` — about **77,800 generated tokens** on the validated
configuration. The scheduler logs `KV cache pool is full. Retract requests.`
and the generation fails. Reproduced exactly at 77,824 tokens (the SWA pool
size). Prompt length does not count against this limit; prefill accounts SWA
correctly (a 131K prompt holds ~8.5K SWA tokens).

This is a server-side bug, not a model or client limit, and no launch
configuration currently works around it:

- Raising `--swa-full-tokens-ratio` shrinks the total pool nearly as fast as it
  raises the ceiling (0.3 gives ~123K per request but ~411K total; 1.0 gives
  ~143K both) and never approaches 384K.
- `--disable-hybrid-swa-memory` crashes at startup on this architecture: the
  DSV4 memory pool constructs its SWA pool unconditionally and receives
  `size=None` (`deepseek_v4_memory_pool.py:602`).

Until it is fixed, generations that exceed the ceiling abort rather than
degrade. Most turns — including long agentic coding turns — finish well under
it; the abort affects only single turns generating beyond ~78K tokens.

The following is only a deterministic health smoke test. Its `temperature=0` setting is not the model card's recommendation for normal inference:

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
