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
  ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.1.0-rc.2
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
export CACHE_DIR=/srv/cache/sglang-dsv4-0731-v2
mkdir -p "$CACHE_DIR"
```

## Start the server

```bash
MODEL_DIR="$MODEL_DIR" CACHE_DIR="$CACHE_DIR" ./examples/serve-dsv4-0731.sh
```

The script defaults to the immutable candidate named by `release.json` and sets
the complete validated environment and server flags. `PORT` (host port, default
`8000`), `IMAGE`, `CUDA_VISIBLE_DEVICES` (default `0,1`), and `CONTAINER_NAME`
can be overridden; the container always listens on port 8000 internally. An
`IMAGE` override creates a different runtime configuration and must be recorded
when reporting results.

While the server is running, these commands show the exact image, environment,
and command line used by the container:

```bash
docker inspect dsv4-flash-sglang --format '{{.Config.Image}}'
docker inspect dsv4-flash-sglang \
  --format '{{range .Config.Env}}{{println .}}{{end}}' | sort
docker inspect dsv4-flash-sglang --format '{{json .Config.Cmd}}'
```

The image must be `v0.1.0-rc.2` for the results documented here. The old
`dsv4-0731` tag names an earlier pre-SemVer build and is not this release.

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

## Prefix cache reporting

The script passes `--enable-cache-report`, which populates
`usage.prompt_tokens_details.cached_tokens` on OpenAI responses -- the standard
field for how much of a prompt was served from the prefix cache.

It costs nothing. `cached_tokens` is already carried in the engine's per-request
`meta_info`; the flag only controls whether the response surfaces it. With a
774,656-token window and a radix cache, multi-turn and shared-prefix workloads
reuse a large fraction of their prompts, and without this flag a client has no
way to see it.

## Idle CPU usage

The script passes `--sleep-on-idle`. Without it SGLang's scheduler busy-waits
between requests, pinning one CPU thread per rank at 100% -- two cores at TP=2,
continuously, even with no traffic.

The flag replaces the spin with a `zmq.Poller` blocking wait on the sockets that
carry incoming requests, so it costs no wake-up latency: an arriving request
signals the socket immediately, and the poll timeout only bounds how long the
scheduler blocks when nothing arrives. It runs on the rank-0 scheduler only, so
the collective ranks are unaffected.

It matters more than it looks on power-limited cards. These are 300 W Max-Q
parts; two cores spinning at 100% is package power and heat spent for nothing,
and it eats the thermal headroom a burst of traffic would otherwise have.

The sleeper can also periodically call `empty_cache()`, but only when
`SGLANG_EMPTY_CACHE_INTERVAL` is set above zero. It defaults to `-1`, so with
this image's defaults no allocator flush is introduced.

## Speculative decoding depth

The script sets `--speculative-dspark-block-size` explicitly. The checkpoint's
`config.json` carries `dspark_block_size = 5`, and the server logs a gamma
mismatch when the flag and the checkpoint disagree.

We do not currently publish a depth recommendation. An earlier revision of this
section reported a corruption rate and a throughput comparison across depths;
neither is supported by the result artifacts it claimed to summarize, and both
have been removed rather than restated.

[sgl-project/sglang#33800](https://github.com/sgl-project/sglang/issues/33800)
reported DSpark depth 5 corrupting output on SM120 and was root-caused on
2026-08-07 to the SM120 mHC combine fallback — an einsum whose intermediates
were allocated inside the `use_symmetric_memory` pool, where they could collide
with in-flight collective buffers. Not DSpark, not the draft head, not the
depth. PR #29927 replaces that einsum with a kernel and fixes it as a side
effect; this image carries #29927.

## Long-generation corruption under DSpark

Before the PR #32183 pin, long single-file code generation degraded under
DSpark: the model leaked its own reasoning into the output, inserted `...`
placeholders the prompt forbade, left the code fence unclosed, and terminated
early with `finish_reason=stop` far below the token limit. Short answers were
unaffected, so eval scores did not show it.

The cause is in the DeepSeek-V4 compressed-state planner: the compressor-state
rewrite window was fixed at `kMaxMTPDraftTokens = 4`, while DSpark verifies
`block_size + 1` rows, which exceeds 4 at any depth above 3. The planner therefore
rewrote only part of the window and the compressed attention state accumulated
pollution every 4 tokens, which only becomes visible once a generation is long
enough for the indexer's 512-token top-k to start selecting from the polluted
region.

Measured on the validated configuration, scoring each generation by whether the
code fence closes and the extracted script passes `node --check`:

| cell | without #32183 | with #32183 |
|---|---:|---:|
| long write, temperature 1.0 | 0/8 | 14/14 |
| long write, greedy | 0/1 | 3/3 |
| decode throughput | 237.5 tok/s | 319.8 tok/s |

Throughput improves because the draft model is no longer verifying against a
degraded target.

## Fused MHC pre-norm on SM120

The script sets `SGLANG_OPT_DEEPGEMM_HC_PRENORM=1`. SGLang's SM120 branch
disables both fused MHC pre-norm paths, which forces an eager float32
fallback: a `pow` + float cast + CUDA-core `SGEMM` + `mean` sequence measured
at 2,893 ms, about 15% of a 128k prefill, attributed by a `with_stack` profile
to `deepseek_v4.py` `hc_pre_torch_impl`. That guard predates first-class SM120
support for the DeepGEMM `tf32_hc_prenorm_gemm` kernel, which this image ships.

Re-enabling it measured, on the validated configuration with warmup discarded
and n=5:

| cell | default | fused pre-norm | delta |
|---|---:|---:|---|
| prefill 128k | 6,717 | 8,034 | +19.6% |
| prefill 64k | 7,601 | 9,055 | +19.1% |
| decode C1 | 306.0 | 302.7 | within noise |

Activations are bfloat16 on both paths and bfloat16 is exactly representable
in tf32, so only the float32 mixing weights are truncated; the squared sum and
the GEMM accumulator stay float32 on both. GSM8K over five runs was
indistinguishable from the default path.

The other fused path, TileLang MHC pre-norm
(`SGLANG_OPT_USE_TILELANG_MHC_PRE=1`), fails CUDA graph capture on SM120
("invalid argument", both TP ranks). Leave it disabled.

## Measured performance and quality

All numbers below were produced on the validated configuration with
[llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
(`llm_decode_bench.py`), the same client and the same flags against both
engines, so the cells compare directly.

**Hardware matters for the absolute values.** These are RTX PRO 6000 Blackwell
**Max-Q** cards: a 300 W hard limit (not a configurable cap) and **PCIe Gen 4
x16** (`pcie.link.gen.max = 4`, measured 28.2 GB/s unidirectional peer-to-peer).
A 600 W card on PCIe Gen 5 has roughly twice the power budget and twice the
inter-GPU bandwidth, and should be expected to produce higher numbers than
these on both engines.

The vLLM column is `voipmonitor/vllm:gilded-gnosis-v20-...-r27` in its
documented DSpark configuration (K=5, b12x-a8 backend, `max_num_seqs 16`,
`max_num_batched_tokens 8192`, `gpu_memory_utilization 0.975`). Both engines
were given the same 8192-token batch size, which is the setting that makes
prefill comparable.

### Decode, aggregate tokens/second

| concurrency | SGLang agg | SGLang per-user | vLLM agg | vLLM per-user |
|---|---:|---:|---:|---:|
| 1 | 194.1 | 191.2 | **199.5** | 199.5 |
| 2 | **295.6** | 147.3 | 291.3 | 145.6 |
| 4 | **439.9** | 108.3 | 410.7 | 102.7 |
| 8 | **597.7** | 73.2 | 589.5 | 73.2 |
| 16 | **909.3** | 55.3 | 821.4 | 51.3 |
| 32 | **1,315.0** | 39.4 | not reachable (`max_num_seqs 16`) | — |

SGLang figures are medians of five runs. Run-to-run spread was 1.7-3.3% at every
concurrency except C1, where it was 7.2%, so read the C1 gap as a tie rather
than a 2.7% deficit.

### Time to first token, p50 seconds

| concurrency | SGLang (this image) | vLLM v20 r27 |
|---|---:|---:|
| 1 | **0.178** | 0.493 |
| 2 | **0.178** | 0.642 |
| 4 | **0.195** | 0.835 |
| 8 | **0.187** | 0.845 |
| 16 | **0.197** | 0.942 |
| 32 | **0.205** | — |

SGLang answers 2.8x sooner at concurrency 1 and 4.8x sooner at 16, and its
first-token latency is flat from 1 to 32 concurrent requests (0.178 to 0.205)
while vLLM's nearly doubles across 1 to 16.

### Prefill, tokens/second

| context | SGLang (this image) | vLLM v20 r27 |
|---|---:|---:|
| 8k | 7,317 | **7,540** |
| 64k | 8,312 | **8,945** |
| 128k | 7,712 | **8,238** |

### GSM8K, 1,319 questions, temperature 0, parallel 8

| | SGLang (this image) | vLLM v20 r27 |
|---|---:|---:|
| accuracy | 0.9375 (mean of 5; 0.9356-0.9401) | 0.9393 |
| wall clock | **315 s** (mean of 5; 312-319) | 749 s |
| completion tokens | 116,174-116,707 | 119,238 |

Accuracy is a tie: three questions apart, well inside this stack's run-to-run
variance. Scoring the same split five times on one engine spanned 0.9363 to
0.9401, and 64 of the 116 questions that were ever wrong changed answer between
runs, so differences below roughly 0.4 points are not resolvable without
replicates.

Throughput on this workload is not a tie. The suite is 1,319 short requests
averaging 88 output tokens, so total time is dominated by per-request latency
rather than sustained token rate, and SGLang finishes **2.2x sooner**. This is
the regime most agentic and coding traffic falls into, and it inverts the
sustained-decode ranking above.

### Long single-file code generation

Eight-to-four runs of one prompt asking for a complete single-file HTML5 game,
`reasoning_effort=max`, `max_tokens=131072` (the largest budget vLLM accepts).
A run counts as complete only if the code fence closes, the document reaches
`</html>`, no placeholder text appears, and the extracted `<script>` bodies pass
`node --check`.

| | SGLang (this image) | vLLM v20 r27 |
|---|---:|---:|
| complete | **4/4** | **4/4** |
| median completion tokens | 57,281 | **40,764** |
| median wall clock | 264 s | **180 s** |
| median reasoning characters | 148,605 | **94,215** |
| median content characters | 23,532 | 25,138 |

Both engines produce valid, complete, parseable files. vLLM reaches an
equivalent answer with roughly 30% fewer tokens and in about two-thirds the
wall clock, spending noticeably less of the budget in the reasoning phase for
the same amount of delivered code.

This cell is what caught the DSpark compressed-state corruption that PR #32183
fixes. Before that fix the same prompt produced 0 complete runs out of 8:
leaked reasoning in the output, forbidden `...` placeholders, unclosed fences,
and early termination at `finish_reason=stop`. GSM8K did not move at all across
the same change, because short answers never reach the length where the
corruption appears.

### Speculative decoding

Both engines draft 5 tokens per step. We do not publish an acceptance
comparison: acceptance is not a quality signal — speculative decoding is
distribution-preserving, so a rejected draft token is replaced by the target
model's own token and the output distribution is unchanged — and its effect on
speed is already captured by the decode figures above.

Acceptance is observable per request on this image. SGLang PR #33518 is carried,
so `"return_spec_tokens_details": true` on a chat completion returns
`sglext.spec_tokens_details` with `spec_accept_rate`, `spec_accept_length`,
`spec_verify_ct`, the correct/proposed draft counts, and a per-position
histogram.

The SGLang figures come from the engine's own `sglang:spec_accept_length` and
`sglang:spec_accept_rate` gauges, cross-checked against the accept length SGLang
prints per decode batch (median 2.70 over 264 samples). An earlier revision of
this document reported 92.9% and 6.5 accepted per draft. That was a measurement
error: the harness derived acceptance from `sglang:generation_tokens_total`
minus `sglang:spec_verify_calls_total`, and both counters only advance when a
request finishes naturally. The decode benchmark cancels its streams at each
timed boundary, so across a full sweep those counters moved by 60 tokens and 9
verifications -- one incidental request -- and the ratio described that request
rather than the benchmark. The vLLM figures were never affected: they come from
true per-draft counters (28,515 drafts, 142,575 draft tokens, 53,163 accepted).

### Context and KV capacity

| | SGLang (this image) | vLLM v20 r27 |
|---|---:|---:|
| KV cache | **778,496 tokens** | 143,439 tokens |
| usable context | **774,656** | 133,120 |
| concurrent requests at 32k each | **23.8** | 4.4 |
| concurrent requests at 64k each | **11.9** | 2.2 |
| concurrent requests at 128k each | **5.9** | 1.1 |
| accepts `max_tokens=393216` | yes | **no, HTTP 400** |

This is the largest difference between the two and it is a memory result, not a
configuration preference. vLLM reports `Available KV cache memory: 7.95 GiB` and
`GPU KV cache size: 143,439 tokens` at startup, and at `max_model_len 133120`
its own log states `Maximum concurrency for 133,120 tokens per request: 1.08x`
-- one full-length request very nearly exhausts the pool. The concurrency rows
above divide each engine's KV pool by a common per-request context so the two
are comparable; they are not scheduler limits. SGLang's scheduler is separately
capped at 48 running requests by `--max-running-requests`, and vLLM's at 16 by
`max_num_seqs`, so short-request concurrency is bounded by those settings rather
than by KV capacity. Requesting the 384K
output budget the model card recommends for `high`/`max` reasoning is rejected:

```
max_tokens=393216 cannot be greater than max_model_len=max_total_tokens=133120
```

SGLang reaches 5.4x the KV capacity on the same two cards through FP8 KV cache
(half the bytes per token) and hybrid sliding-window memory, which sizes the
SWA pool at a fraction of the full-attention pool rather than giving every layer
full-size KV.

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

## Fixed: single-request generation no longer aborts near 78K generated tokens

Images before this composition aborted single requests at roughly
`swa-full-tokens-ratio (0.1) x max_total_num_tokens` (~77,800) generated
tokens: the DFLASH/DSPARK speculative decode path never ran sliding-window KV
eviction, so SWA KV accumulated 1:1 with generated tokens until the scheduler
logged `KV cache pool is full. Retract requests.` and killed the generation.

This image carries the fix (see the dspark SWA eviction entry in the README).
Verified by generating 100,000 tokens in a single request with SWA usage
holding at ~1% throughout; on the prior image the same request died at 77,809
tokens. Single-request output is now bounded by the full KV pool, which
accommodates the 384K budget above alongside a prompt.

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
