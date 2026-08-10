# Running the image

This is the validated procedure for Linux x86_64 with two NVIDIA RTX PRO 6000
Blackwell GPUs (SM120) at TP=2. SM121 has not been validated.

## Requirements

- A driver that supports CUDA 13.
- Two RTX PRO 6000 Blackwell GPUs visible to the container.
- Docker with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for the model
  download.
- Persistent storage for the model and compiled-kernel cache.

Check GPU passthrough first:

```bash
docker run --rm --gpus all \
  --entrypoint nvidia-smi \
  ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.1.0-rc.3
```

## Download the model

```bash
export MODEL_DIR=/srv/models/DeepSeek-V4-Flash-0731
uvx --from huggingface-hub hf download \
  deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  --local-dir "$MODEL_DIR"
```

That is the exact model revision used for the published comparison. Keep the
snapshot on persistent storage. The serving script mounts it read-only.

## Prepare the cache

The container's `/root/.cache` stores FlashInfer JIT modules, TileLang and TVM
kernels, TorchInductor output, and DeepGEMM artifacts. Cache contents are build
specific; do not share a directory with an older image.

```bash
export CACHE_DIR=/srv/cache/sglang-dsv4-0731-v2
mkdir -p "$CACHE_DIR"
```

`v2` is the cache schema recorded by `release.json`. A later schema must use a
new directory even when the image tag is similar.

## Start the server

```bash
MODEL_DIR="$MODEL_DIR" CACHE_DIR="$CACHE_DIR" ./examples/serve-dsv4-0731.sh
```

The script defaults to the immutable candidate in `release.json` and supplies
the complete validated environment and server flags. These values may be
overridden:

- `PORT` — host port, default `8000`;
- `IMAGE` — exact image reference;
- `CUDA_VISIBLE_DEVICES` — default `0,1`;
- `CONTAINER_NAME` — default `dsv4-flash-sglang`.

Record every override with benchmark results. An image override is a different
runtime composition even if the serving arguments are unchanged.

Inspect the live container before reporting a result:

```bash
docker inspect dsv4-flash-sglang --format '{{.Config.Image}}'
docker inspect dsv4-flash-sglang \
  --format '{{range .Config.Env}}{{println .}}{{end}}' | sort
docker inspect dsv4-flash-sglang --format '{{json .Config.Cmd}}'
```

The documented release is:

```text
ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.1.0-rc.3
```

The older moving `dsv4-0731` tag is not this release.

## Required container resources

The serving script uses `--shm-size 64g`. SGLang TP workers exchange tensors
through `/dev/shm`; Docker's 64 MiB default is insufficient. It also uses
`--ulimit memlock=-1` for pinned host memory.

The first start compiles the patched SM120 FlashInfer, TileLang, and DeepGEMM
kernels into `$CACHE_DIR`. Later starts with the same image and cache reuse
those artifacts. Do not compare a cold compiler startup with a warmed server.

## Why the validated settings differ from upstream defaults

### Chunked prefill

`--chunked-prefill-size 8192` matches the tested SGLang and vLLM batch setting.
Prefill comparisons are meaningful only when the engines use the same chunk
size.

### DeepGEMM DSA indexer

The script selects the DeepGEMM DSA indexer with
`--enable-deepseek-v4-fp4-indexer` and disables the forced TileLang and Torch
fallbacks. The image ships an SM120-capable DeepGEMM with
`fp8_fp4_paged_mqa_logits`; stock compositions without that symbol must use the
fallback instead.

### DSpark shared experts

RC3 carries SGLang PR #34139, which keeps the DSpark draft on separate
shared-expert modules after fusion decisions became runner-local. The change is
part of the release patch rather than a benchmark-only server option.

### Fused MHC pre-norm

`SGLANG_OPT_DEEPGEMM_HC_PRENORM=1` selects the SM120-capable DeepGEMM fused MHC
pre-norm. `SGLANG_OPT_USE_FLASHINFER_MHC=1` enables the FlashInfer path; the RC3
dispatch gate selects it only for token batches of at least 1,024. The TileLang
MHC pre-norm override remains disabled because that path fails CUDA graph
capture on SM120.

### All-reduce workspace and dispatch

The release patch sizes the FlashInfer all-reduce workspace before graph capture
and applies a token cap to the latency-optimized all-reduce path. This keeps
prefill-sized collectives on the appropriate backend while retaining the
decode-oriented path for small token counts.

### Prefix-cache reporting

`--enable-cache-report` exposes
`usage.prompt_tokens_details.cached_tokens` on OpenAI responses. The value
already exists in SGLang request metadata; the flag makes it visible to clients.

### Idle CPU usage

`--sleep-on-idle` blocks the rank-0 scheduler on its request sockets instead of
busy-waiting when the server is idle. The optional allocator flush remains off
unless `SGLANG_EMPTY_CACHE_INTERVAL` is set explicitly.

## Canonical benchmark procedure

Use the AIPerf harness and protocol in [BENCHMARKS.md](BENCHMARKS.md). The
executable source is in [`bench/aiperf`](bench/aiperf). The documentation
records:

- the pinned AIPerf commit;
- image identities and runtime settings;
- workload shape and sampling settings;
- warmup and cache controls;
- decode forward rate, useful throughput, acceptance, and TTFT;
- cold-prefill throughput and TTFT;
- commands for authenticated and keyless endpoints.

The repository publishes only the current RC3 and vLLM r33 measurements.

## Health and API checks

`SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0` makes `/health` a liveness check
rather than a generation request:

```bash
curl -fsS http://localhost:8000/health
```

The OpenAI-compatible API is at `http://localhost:8000/v1`, with served model
name `deepseek-v4-flash`:

```bash
curl -fsS http://localhost:8000/v1/models | jq -r '.data[].id'
```

This deterministic smoke test uses greedy sampling only to make the expected
answer stable. It is not the recommended production sampling configuration:

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

## KV capacity and context length

The serving script pins `--context-length 774656` so DSA indexer scratch
allocation is bounded below the checkpoint's full 1,048,576-token window.

Leaving the checkpoint window unbounded can force a large float32
`(batch_size, max_seq_len)` indexer allocation on the first sufficiently large
request. The same shape requirement exists on the TileLang and DeepGEMM indexer
paths, so switching indexers does not remove it.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces allocator
fragmentation around these large workspaces.

If GPU count, memory fraction, KV dtype, graph sizes, or another GPU consumer
changes, read the new capacity rather than copying the value above:

```bash
curl -fsS http://localhost:8000/get_server_info \
  | jq '{max_total_num_tokens, max_req_input_len}'
```

Set `--context-length` to a value the resulting pool can actually serve.

## Recommended request sampling

The
[model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731#how-to-run-locally)
recommends `temperature=1.0` with `top_p=0.95` for agentic requests and
`top_p=1.0` otherwise. Chat completions may set `reasoning_effort` to `low`,
`high`, or `max`.

Treat tool-using coding harnesses as agentic workloads. A maximum-deliberation
request is:

```json
{
  "reasoning_effort": "max",
  "temperature": 1.0,
  "top_p": 0.95,
  "max_tokens": 393216
}
```

Use `high` when latency matters more than maximum deliberation. `393216` is the
numeric API value for the model card's 384K output budget; reducing it can
truncate a long reasoning chain. This is a request limit, not a server launch
flag.

## Long-generation SWA eviction

This image carries SGLang PR #33805, which runs sliding-window KV eviction on
the DFLASH/DSPARK speculative path. Without that change, SWA KV accumulates
with generated tokens until the scheduler retracts a long request even though
the full-attention pool still has capacity. With the fix, long single-request
generation is bounded by the full scheduler KV pool rather than the smaller SWA
fraction.
