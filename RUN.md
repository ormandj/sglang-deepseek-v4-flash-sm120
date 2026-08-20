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
  ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.6.0-rc.3
```

## Model and cache

```bash
export MODEL_DIR=/srv/models/DeepSeek-V4-Flash-0731
uvx --from huggingface-hub hf download \
  deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  --local-dir "$MODEL_DIR"

export CACHE_DIR=/srv/cache/sglang-dsv4-0731-v24
mkdir -p "$CACHE_DIR"
```

`v24` is the cache schema for this source composition. Do not reuse a cache
directory from another image source tree.

## Start TP2

```bash
MODEL_DIR="$MODEL_DIR" CACHE_DIR="$CACHE_DIR" \
  ./examples/serve-dsv4-0731.sh
```

The script defaults to the candidate tag, GPUs 0 and 1, TP2, and a 774,656
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
the same v24 cache for subsequent starts of the identical image; do not time
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

## Hierarchical KV cache

Disabled by default. It addresses one specific failure: several long-context
conversations sharing a deployment whose device pool cannot hold them all at
once. In that state the prefix cache evicts between turns, every turn
re-prefills its entire context, and the GPU spends its time recomputing
prefixes it already had.

Decide by measuring. `--enable-cache-report` (already in the recorded
configuration) makes the server report reuse per request:

```bash
curl -fsS http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"input_ids": [1,2,3], "sampling_params": {"max_new_tokens": 8}}' \
  | jq '.meta_info | {prompt_tokens, cached_tokens}'
```

Send a conversation twice. If the second call's `cached_tokens` is close to
`prompt_tokens`, the cache is holding and HiCache gains you nothing. If it
collapses toward zero under real concurrency, prefixes are being evicted and
this is your problem. `sglang:cached_tokens_total` and
`sglang:evicted_tokens_total` on `/metrics` show the same thing in aggregate.

### Enabling it

| variable | default | meaning |
|---|---|---|
| `HICACHE` | `0` | `1` enables the host (L2) tier |
| `HICACHE_RATIO` | `10` | host pool as a multiple of the device pool |
| `HICACHE_WRITE_POLICY` | `write_through_selective` | `write_through`, `write_through_selective`, `write_back` |
| `HICACHE_STORAGE_DIR` | unset | host path for the on-disk (L3) tier; unset means L2 only |
| `HICACHE_STORAGE_PREFETCH_POLICY` | `timeout` | `best_effort`, `wait_complete`, `timeout` |

```bash
HICACHE=1 HICACHE_RATIO=10 \
MODEL_DIR="$MODEL_DIR" CACHE_DIR="$CACHE_DIR" \
  ./examples/serve-dsv4-0731.sh
```

### Sizing

`HICACHE_RATIO` multiplies the device pool. Take the total context of the
sessions you want covered and divide by `max_total_num_tokens`:

```text
ten sessions x 786,432 tokens = 7,864,320
7,864,320 / 801,536           = 9.8  ->  ratio 10
```

Confirm it at startup. The server logs each host pool as it allocates, and the
page counts should be your ratio times the device pool's:

```text
Allocating 24.62 GB host memory for V4 paged pool 'deepseek_v4_c4' (pages=31310, ...)
Tree cache initialized: ... hicache_attached=True
```

On this hardware ratio 10 costs about 61 GiB per rank, ~122 GiB across TP2.
**That memory is pinned and cannot be reclaimed by the kernel**, so it is
subtracted from page cache for the lifetime of the process. Size it against
free RAM, not total RAM.

The device-side envelope is untouched: `max_total_num_tokens`, `context_len`,
and free GPU memory are identical with HiCache on and off.

### The storage tier

The host tier is RAM and is lost on restart, so every active conversation pays
one full cold prefill afterwards. Those cold prefills serialize, which on a
busy deployment is minutes of queue. Pointing `HICACHE_STORAGE_DIR` at a local
disk persists prefixes across restarts:

```bash
HICACHE=1 HICACHE_RATIO=10 HICACHE_STORAGE_DIR=/srv/hicache \
MODEL_DIR="$MODEL_DIR" CACHE_DIR="$CACHE_DIR" \
  ./examples/serve-dsv4-0731.sh
```

Entries are content-addressed, so identical prefixes deduplicate and nothing
depends on process state. Budget roughly the same bytes per token as the host
tier. `SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE`,
`SGLANG_HICACHE_FILE_BACKEND_EVICTION_RATIO`, and
`SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE` bound its disk use.

`HICACHE_STORAGE_PREFETCH_POLICY` decides what happens when a fetch is slow:
`wait_complete` guarantees the hit but ties time-to-first-token to disk
latency, `best_effort` never stalls but may discard the fetch and re-prefill,
and `timeout` waits briefly and then gives up.

### What it does not do

It does not raise concurrency. The device pool is unchanged, so the same number
of sessions stay resident and `#running-req` does not move; only the cost of
swapping between them changes.

It does not help a first, uncached prefill -- that gets about 10% slower,
because the same pass also populates the host tier. Prompts whose uncached
extent exceeds `--chunked-prefill-size` are additionally processed one at a
time, so a burst of cold long contexts queues serially regardless of this
setting. HiCache's value is in the second and later turns of a conversation.

## Benchmarking

The executable harness lives in [`bench/aiperf`](bench/aiperf). Clients run
inside the selected serving pod against localhost. The frozen protocol uses:

- identical 16,384-token input and 4,096-token output shapes at C1–C32;
- five sequential fixed-seed repetitions at every published supported
  concurrency;
- separate cache-cold prefill diagnostics;
- full GSM8K correctness.

The published `v0.6.0-rc.3` panel uses five measured repetitions at every
supported concurrency and prefill length. All measured cells run sequentially
on one unchanged process after one warmup per distinct shape.
