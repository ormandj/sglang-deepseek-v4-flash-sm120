# Operating the image

Use the direct Docker command in [`README.md`](README.md) for the first start.
This guide covers the decisions that follow: using the optional wrapper,
changing topology or capacity, enabling hierarchical KV cache, and checking the
result. It intentionally does not duplicate the canonical launch command.

## Keep the launch contract together

This image is qualified as a complete source-and-runtime configuration. The
following groups are performance or capacity inputs, not cosmetic defaults:

- the image tag and `v24` compilation-cache namespace;
- FP8 KV cache, DSpARK block size 5, and the DeepSeek-V4 FP4 indexer;
- HC prenorm, fused MHC post+pre, and FP8 W_o_A selectors;
- PCIe-IPC all-reduce, its 786,432-element ceiling, and persisted autotuning;
- the `0.93` static-memory fraction, 786,432 context, 8,192-token prefill
  chunks, graph batch ceiling 32, and scheduler ceiling 48;
- prefill/decode warmups and the 393,216-token generation ceiling.

Begin with the recorded combination. If you change one of these inputs, use a
new cache directory when source or kernel shape changes, inspect the reported
KV pool, warm every served shape, and run a near-limit request before relying on
the deployment.

## Optional wrapper

The wrapper validates paths and basic topology inputs, then runs the same
configuration shown in the README:

```bash
MODEL_DIR=/srv/models/DeepSeek-V4-Flash-0731 \
CACHE_DIR=/srv/cache/sglang-dsv4-0731-v24 \
  ./examples/serve-dsv4-0731.sh
```

The required variables are `MODEL_DIR` and `CACHE_DIR`. Available overrides are:

| Variable | Default | Purpose |
|---|---|---|
| `IMAGE` | current GHCR candidate | Run another immutable image reference |
| `PORT` | `8000` | Host port mapped to container port 8000 |
| `CUDA_VISIBLE_DEVICES` | `0,1` | GPUs exposed to the server process |
| `TP_SIZE` | `2` | Tensor-parallel size; must match the device count |
| `CONTEXT_LENGTH` | `786432` | Total prompt-plus-generation budget per request |
| `CONTAINER_NAME` | `dsv4-flash-sglang` | Docker container name |
| `SGLANG_MAX_NEW_TOKENS_LIMIT` | `393216` | Server-wide generation ceiling |
| `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE` | `1` | Qualified PCIe-IPC reduction path |
| `SGLANG_PCIE_IPC_MAX_NUMEL` | `786432` | PCIe-IPC eligibility ceiling |
| `SGLANG_PCIE_IPC_AUTOTUNE` | `1` | Persist machine-local tuning choices |
| `HICACHE` | `0` | Enable the host-memory KV tier |
| `HICACHE_RATIO` | `10` | Host pool as a multiple of the device pool |
| `HICACHE_WRITE_POLICY` | `write_through_selective` | Host-tier write policy |
| `HICACHE_STORAGE_DIR` | unset | Host path for the optional disk tier |
| `HICACHE_STORAGE_PREFETCH_POLICY` | `timeout` | Disk-tier fetch policy |

Example TP4 invocation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TP_SIZE=4 \
MODEL_DIR=/srv/models/DeepSeek-V4-Flash-0731 \
CACHE_DIR=/srv/cache/sglang-dsv4-0731-v24-tp4 \
  ./examples/serve-dsv4-0731.sh
```

The image and launcher support TP4 and TP8, but the published capacity and
performance results are TP2. Do not copy the TP2 pool or context claims to
another topology without measuring it.

## Capacity and long outputs

`--mem-fraction-static` reserves device memory for the static KV pool and
trades capacity directly against request workspace. The following TP2 sweep
used `--context-length 786432`; pool and free-memory values are SGLang startup
reports.

| Static fraction | KV tokens | Free after startup | Near-limit request | Four 250K requests |
|---:|---:|---:|---|---|
| 0.90 | 474,624 | 5,488 MiB | 470K passes; 4,322 MiB floor | passes |
| **0.93 (shipped)** | **801,536** | **2,628 MiB** | **780K passes; 120 MiB floor** | **passes; 120 MiB floor** |
| 0.95 | 1,019,648 | 440 MiB | scheduler rank crashes | not reached |

The 0.95 server starts, but a 550,000-token request exhausts workspace. Choose
this setting by completed near-limit work, not by the startup pool alone. The
shipped 0.93 result is also tight: its 120 MiB floor is a measured pass, not a
large safety margin.

`--context-length` is the total prompt plus generated-token budget. A request
is rejected when `input + max_tokens` exceeds it. If `max_tokens` is omitted,
the server treats generation as unbounded, clamps it to the remaining context,
and then applies `SGLANG_MAX_NEW_TOKENS_LIMIT`.

For example, a session with 400K prompt/context tokens, 300K reasoning tokens,
and a 30K answer uses about 730K of the 786,432-token request budget. One such
request fits the 801,536-token pool; two do not remain resident together.

After every image, topology, memory-fraction, context, or graph change:

```bash
curl -fsS http://localhost:8000/get_server_info \
  | jq '{max_total_num_tokens, max_req_input_len}'
```

Then complete a request near the intended limit. A successful startup does not
exercise peak prefill workspace.

## Hierarchical KV cache

HiCache is off by default. Enable it when multiple long-running conversations
cycle through a deployment and returning turns lose their prefix-cache hits.
It adds a pinned host-memory tier (L2) and, optionally, a persistent disk tier
(L3). It does not enlarge the device pool or increase resident concurrency.

Check real reuse first. `--enable-cache-report` is already enabled, so a
response from `/generate` includes `prompt_tokens` and `cached_tokens` in
`meta_info`. The `/metrics` endpoint also exposes aggregate cached and evicted
token counters. If returning requests already retain nearly all prompt tokens,
HiCache has little to recover.

Enable the host tier with the wrapper:

```bash
HICACHE=1 HICACHE_RATIO=10 \
MODEL_DIR=/srv/models/DeepSeek-V4-Flash-0731 \
CACHE_DIR=/srv/cache/sglang-dsv4-0731-v24 \
  ./examples/serve-dsv4-0731.sh
```

The ratio multiplies the device pool. Ten full-context sessions require:

```text
10 x 786,432 / 801,536 = 9.8  ->  HICACHE_RATIO=10
```

On the measured TP2 host, ratio 10 pins about 61 GiB per rank, or 122 GiB in
total. Pinned memory is not reclaimable by the kernel, so size it against free
RAM. Confirm the allocated pages in startup logs.

For persistence across restarts, add a storage path:

```bash
HICACHE=1 \
HICACHE_RATIO=10 \
HICACHE_STORAGE_DIR=/srv/hicache \
MODEL_DIR=/srv/models/DeepSeek-V4-Flash-0731 \
CACHE_DIR=/srv/cache/sglang-dsv4-0731-v24 \
  ./examples/serve-dsv4-0731.sh
```

`HICACHE_STORAGE_PREFETCH_POLICY` may be `best_effort`, `wait_complete`, or
`timeout`. The default `timeout` waits briefly before falling back to prefill.
The file-backend size, eviction ratio, and minimum-free-space environment
variables exposed by SGLang can bound disk use.

Measured at 1.14x device-pool oversubscription with three 300K sessions:

| Returning-turn metric | HiCache off | HiCache on |
|---|---:|---:|
| Prefill | 18–20 s | about 0.5 s |
| Prefix reuse | 64% | 99.3% |
| Tokens recomputed over two turns | 657,216 | 13,632 |

The first uncached prefill was about 10% slower because it also populated the
host tier. HiCache is intended for repeated long conversations, not one-shot
traffic.

## Exposure and client settings

The quickstart publishes port 8000 on all host interfaces and does not add
authentication. Bind or firewall the port appropriately, or put the service
behind an authenticated proxy before exposing it to an untrusted network.

Agentic requests should follow the model card defaults: temperature `1.0` and
top-p `0.95`. The temperature-0 request in the README is only a deterministic
engine smoke test.

## Common checks

- **Startup is slow once:** the first start compiles kernels and performs shape
  warmups. Reuse the matching cache; do not treat cold compilation as serving
  latency.
- **A large request kills one TP rank:** lower the static memory fraction or
  request envelope. Free startup memory and a large KV count are not sufficient
  evidence.
- **Returning long turns prefill again:** inspect cached/evicted-token metrics,
  then size HiCache to the working set if prefixes are cycling out.
- **Changing TP:** use one listed GPU per TP rank, isolate the compiled cache by
  topology when kernel shapes change, and remeasure capacity.

For performance and quality qualification, use [`BENCHMARKS.md`](BENCHMARKS.md)
and the executable protocol in [`bench/aiperf/README.md`](bench/aiperf/README.md).
