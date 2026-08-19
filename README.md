# SGLang for DeepSeek-V4-Flash on SM120

This repository publishes a ready-to-run Linux x86_64 SGLang image for
`deepseek-ai/DeepSeek-V4-Flash-0731` on NVIDIA RTX PRO 6000 Blackwell (SM120).
The image is built directly from this repository's pinned source composition;
you do not need to build SGLang, FlashInfer, DeepGEMM, or the image locally.

## Fast path

The current candidate is:

```text
ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.5.0-rc.1
```

Install Docker, the NVIDIA Container Toolkit, and `uv`, then download the
pinned model snapshot and create a release-specific compiled-kernel cache:

```bash
export MODEL_DIR=/srv/models/DeepSeek-V4-Flash-0731
uvx --from huggingface-hub hf download \
  deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  --local-dir "$MODEL_DIR"

export CACHE_DIR=/srv/cache/sglang-dsv4-0731-v20
mkdir -p "$CACHE_DIR"
```

Start the prebuilt image directly at TP2:

```bash
docker run --rm \
  --name dsv4-flash-sglang \
  --entrypoint sglang \
  --gpus all \
  --shm-size 64g \
  --ulimit memlock=-1 \
  --publish 8000:8000 \
  --volume "$MODEL_DIR:/models/deepseek-ai/DeepSeek-V4-Flash-0731:ro" \
  --volume "$CACHE_DIR:/root/.cache" \
  --env CUDA_VISIBLE_DEVICES=0,1 \
  --env SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  --env TILELANG_CACHE_DIR=/root/.cache/tilelang \
  --env TVM_CACHE_DIR=/root/.cache/tvm \
  --env TRITON_CACHE_DIR=/root/.cache/triton \
  --env SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0 \
  --env SGLANG_OPT_USE_TILELANG_INDEXER=0 \
  --env SGLANG_OPT_DEEPGEMM_HC_PRENORM=1 \
  --env SGLANG_OPT_FUSE_MHC_POST_PRE=1 \
  --env SGLANG_OPT_FP8_WO_A_GEMM=1 \
  --env SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1 \
  --env SGLANG_PCIE_IPC_MAX_NUMEL=786432 \
  --env SGLANG_PCIE_IPC_AUTOTUNE=1 \
  ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.5.0-rc.1 \
  serve \
  --model-path /models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.94 \
  --context-length 786432 \
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
  --host 0.0.0.0 \
  --port 8000
```

### KV cache capacity

`--mem-fraction-static` sets how much device memory is reserved for the static
KV pool. It is the most consequential capacity knob and trades directly against
per-request workspace. `max_total_num_tokens` is SGLang's own self-reported
startup figure on 2x RTX PRO 6000 Max-Q at TP2. The validated v0.5.0-rc.1
configuration is 0.94 with a 786,432-token context; the row below was measured
on the live v0.5.0-rc.1 deployment:

| `--mem-fraction-static` | `--context-length` | `max_total_num_tokens` | Free after startup | 240k-token request |
|---|---:|---:|---:|---|
| 0.94 | 786,432 | 910,592 | ≈3.9 GB* | passes (256,038-token) |

* Free after startup is derived from the earlier 0.95/3.11 GB measurement plus
  the 109,056-token pool returned by the 0.95 -> 0.94 step; the live deployment
  was not restarted to re-measure at 0.94.

Earlier v0.3.x-era capacity sweeps of this image line showed the same knob
trading pool against per-request workspace: at 0.96/774,656 the pool measured
1,099,264 with only 2.21 GB free, and a single 240,269-token request killed the
server with `c10::OutOfMemoryError` even though the pool held it and startup
was clean, while 0.95/786,432 returned 200. v0.5.0-rc.1 carries the page-split
workspace allocation from SGLang #35116, so the workspace ceiling on this build
should be re-measured on a planned restart. Do not choose this knob by
maximising `max_total_num_tokens`.

`--context-length` is not only an input limit: it also sizes per-request
workspace, so declaring more context than you serve costs memory on every
request. A configuration can boot cleanly and still fail later on a large
request, so validate with a near-limit request rather than trusting startup:

```bash
curl -fsS http://localhost:8000/get_server_info \
  | jq '{max_total_num_tokens, max_req_input_len}'
```

Alternatively, clone this repository and use the checked wrapper, which
executes the same configuration and validates its inputs:

```bash
MODEL_DIR="$MODEL_DIR" CACHE_DIR="$CACHE_DIR" \
  ./examples/serve-dsv4-0731.sh
```

The wrapper validates the local paths and runs the Docker command recorded in
this repository. It defaults to GPUs `0,1`, TP2, port 8000, and the qualified
786,432-token context limit. Override those values explicitly when needed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TP_SIZE=4 \
CONTEXT_LENGTH=786432 \
MODEL_DIR="$MODEL_DIR" \
CACHE_DIR="$CACHE_DIR" \
  ./examples/serve-dsv4-0731.sh
```

The number of devices in `CUDA_VISIBLE_DEVICES` must equal `TP_SIZE`. TP2 is
the measured configuration, not an image limitation. Confirm capacity and
complete a near-limit request before advertising the TP2 context limit on a
different topology.

Verify the server:

```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/get_server_info \
  | jq '{max_total_num_tokens, max_req_input_len}'
curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

[RUN.md](RUN.md) includes the expanded Docker command, requirements, cache and
capacity guidance, and authenticated API examples.

## What the image contains

The GitHub Actions workflow builds the image from this repository's
`Containerfile`, `release.json`, `stack.lock.json`, and recorded patches. OCI
labels record the exact repository revision and source trees. Candidate and
stable SemVer tags are immutable; there is no mutable `latest` tag.

The qualified collective configuration is:

- FlashInfer PCIe-IPC all-reduce enabled for eligible decode reductions;
- SGLang legacy custom all-reduce disabled so it does not intercept them;
- NCCL used for unsupported and prefill-sized reductions;
- TRT/MNNVL all-reduce fusion disabled;
- PCIe-IPC all-gather absent.

The serving recipe also enables SM120 DeepGEMM HC prenorm, fused MHC
post+pre, and the FP8 W_o_A target-model path. Pre-readiness warmups compile
the qualified prefill and decode paths before the server reports ready.

### Source identity

| Component | Pinned identity |
|---|---|
| Base image | `lmsysorg/sglang:nightly-dev-cu13-20260818-c0b6474b@sha256:51e576f02368480c055c7aadb67590d82b172e2392123ce4cf4cc8251b2d8caf` |
| SGLang main | `87a09494fa3fbd685bd7c88d6a2dbdd3135de602` |
| SGLang effective tree | `dc71e9b9bb380e96bcb2c0cb4aed120f79478c3e` |
| FlashInfer main | `7aa0cd3b64f84c50c18ee958e24f708afb2103c1` |
| FlashInfer effective tree | `489e9318e4d21d3ecddc8d2ec8f138dde93784b5` |
| FlashInfer package | `0.6.18.dev20260818` |
| DeepGEMM base | `75f60622bc6d317306a41c1f38dc9d888b3ec841` |
| DeepGEMM effective tree | `ff371cf8cc7186c8dab8e07cc7cda6c28baa092f` |
| DeepGEMM package | `0.0.0+sm120jit3` |

The SGLang patch contains the audited source heads of
[`sgl-project/sglang#29927`](https://github.com/sgl-project/sglang/pull/29927),
[`sgl-project/sglang#35116`](https://github.com/sgl-project/sglang/pull/35116),
[`sgl-project/sglang#35118`](https://github.com/sgl-project/sglang/pull/35118),
[`sgl-project/sglang#33614`](https://github.com/sgl-project/sglang/pull/33614),
[`sgl-project/sglang#32686`](https://github.com/sgl-project/sglang/pull/32686),
[`sgl-project/sglang#33568`](https://github.com/sgl-project/sglang/pull/33568),
[`sgl-project/sglang#34018`](https://github.com/sgl-project/sglang/pull/34018),
and [`sgl-project/sglang#34528`](https://github.com/sgl-project/sglang/pull/34528).
The FlashInfer patch contains
[`flashinfer-ai/flashinfer#3930`](https://github.com/flashinfer-ai/flashinfer/pull/3930),
[`flashinfer-ai/flashinfer#4393`](https://github.com/flashinfer-ai/flashinfer/pull/4393),
the exact CUDA-runtime resolver follow-up submitted as
[`aryanputta/flashinfer#1`](https://github.com/aryanputta/flashinfer/pull/1),
and the outer-autotune preservation follow-up submitted as
[`qsang-nv/flashinfer#1`](https://github.com/qsang-nv/flashinfer/pull/1).
The DeepGEMM patch contains the held-constant SM120 single-token split-K change
submitted as
[`sgl-project/DeepGEMM#77`](https://github.com/sgl-project/DeepGEMM/pull/77)
and the production fix from
[`sgl-project/DeepGEMM#76`](https://github.com/sgl-project/DeepGEMM/pull/76)
at commit `865f8f202b62a0bc8a6f32513fc33d2789c87031`. The included PR head is
`4900cbd750b4fb10bf756bd1be1f4357b66eac74`.
Every source head, integration commit, patch checksum, and resulting tree is
recorded in [stack.lock.json](stack.lock.json). Run
[`scripts/verify-patches.sh`](scripts/verify-patches.sh) to reconstruct and
verify the three effective trees.

### Upstream validation

The CUDA-runtime follow-up reproduces the serving-container collision where a
mapped TileLang `libcudart_stub.so` preceded the actual runtime. The exact
matcher selected the real runtime and the same server configuration initialized
successfully. The change has 12 focused resolver cases; no performance change
is attributed to it.

The PCIe-IPC follow-up ran in the actual SGLang model-warmup autotune context
with a persisted machine-local tuning cache. Matched C1 and C8 Nsight captures
retained 127 complete paired target-plus-draft steps. The profiled step changed
from 15.681 to 14.965 milliseconds at C1 and from 35.399 to 33.421 milliseconds
at C8. All 11,938 captured C8 all-reduces used PCIe-IPC. These measurements
establish parent-backend execution; they are not isolated attribution for the
cache-preservation follow-up.

The DeepGEMM split-K pull request records an isolated SM120 projection profile
of approximately 22.4 microseconds with split-K plus reduction and 13.2
microseconds with `split_k=1` for `M=4, N=8192, K=1024`. All three changes were
also present in the source-equivalent stack that completed the v0.5.0-rc.1
n=5 decode and prefill panels and full GSM8K validation.

## Current measurements

The current SGLang candidate and the retained vLLM r33 measurements used the
same two RTX PRO 6000 Blackwell Max-Q GPUs at TP2 over PCIe Gen 4 x16, with a
300 W limit per GPU. One engine was active at a time, and clients ran in the
serving pod against localhost.

Decode used a 16,384-token input, 4,096 forced output tokens, temperature 0,
top-p 1, and five fixed-seed repetitions at every supported concurrency. Each
row is a same-process median over the fixed 17,408-20,480 average-context
window. Forward passes/s is the primary engine-execution metric. Synthetic
output tok/s also includes the generated path's speculative acceptance and is
therefore more variable. ITL is AIPerf's average post-first-token time per
generated token.

### Fixed-window decode

| Engine | C | n | Forward passes/s | Synthetic output tok/s | ITL ms/token |
|---|---:|---:|---:|---:|---:|
| SGLang v0.5.0-rc.1 | 1 | 5 | 65.476 | 325.7 | 3.160 |
| vLLM r33 | 1 | 5 | 66.580 | 255.2 | 3.890 |
| SGLang v0.5.0-rc.1 | 2 | 5 | 48.699 | 480.3 | 4.337 |
| vLLM r33 | 2 | 5 | 46.340 | 421.0 | 5.220 |
| SGLang v0.5.0-rc.1 | 4 | 5 | 33.538 | 657.0 | 6.291 |
| vLLM r33 | 4 | 5 | 33.070 | 616.0 | 7.060 |
| SGLang v0.5.0-rc.1 | 8 | 5 | 23.330 | 917.8 | 9.478 |
| vLLM r33 | 8 | 5 | 23.450 | 795.8 | 11.080 |
| SGLang v0.5.0-rc.1 | 16 | 5 | 17.518 | 1,373.5 | 14.583 |
| vLLM r33 | 16 | 5 | 15.860 | 1,097.2 | 17.590 |
| SGLang v0.5.0-rc.1 | 32 | 5 | 13.013 | 2,047.9 | 22.349 |
| vLLM r33 | 32 | 0 | not reachable: vLLM-reported KV 143,599 tok, `max_num_seqs=16` (upstream TP2 recipe) | — | — |

### DSpARK acceptance

Acceptance is recorded to explain variation in synthetic output throughput;
it is not used as an engine-clock metric.

| Engine | C | Median acceptance rate | Median output tokens/forward/request |
|---|---:|---:|---:|
| SGLang v0.5.0-rc.1 | 1 | 0.787 | 4.939 |
| vLLM r33 | 1 | 0.588 | 3.938 |
| SGLang v0.5.0-rc.1 | 2 | 0.813 | 4.977 |
| vLLM r33 | 2 | 0.731 | 4.655 |
| SGLang v0.5.0-rc.1 | 4 | 0.778 | 4.985 |
| vLLM r33 | 4 | 0.764 | 4.821 |
| SGLang v0.5.0-rc.1 | 8 | 0.798 | 5.016 |
| vLLM r33 | 8 | 0.649 | 4.245 |
| SGLang v0.5.0-rc.1 | 16 | 0.782 | 4.947 |
| vLLM r33 | 16 | 0.672 | 4.359 |
| SGLang v0.5.0-rc.1 | 32 | 0.765 | 4.881 |
| vLLM r33 | 32 | not reachable: vLLM-reported KV 143,599 tok, `max_num_seqs=16` (upstream TP2 recipe) | — |

> **On the missing vLLM C=32 rows.** The vLLM side runs the upstream-documented
> TP2 profile unmodified -- "Gilded Gnosis v20 r33, documented TP2 fixed-K5"
> from [`local-inference-lab/rtx6kpro`](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/ds4dspark-v20-r33.md).
> C=32 is unreachable under it for two independent reasons. The recipe sets
> `max_num_seqs=16`, capping concurrency at 16 outright. Independently, vLLM
> self-reported at startup:
>
> ```
> Available KV cache memory: 8.07 GiB
> GPU KV cache size: 143,599 tokens
> Maximum concurrency for 131,072 tokens per request: 1.10x
> ```
>
> Those are vLLM's own figures, not our measurement of it. At the benchmark's
> 16,384-token input plus 4,096 output, 32 streams need roughly 655,000 KV
> tokens against the 143,599 available.
>
> Both limits belong to that published profile, not to vLLM as an engine.
> Raising `max_num_seqs` or `--gpu-memory-utilization`, lowering
> `--max-model-len`, or shortening the input would all change the answer. The
> engines also ran different contexts (vLLM `max_model_len=131,072` vs SGLang
> 774,656), so the KV pools are not directly comparable.


### Cold prefill

Each cell used five cache-busted C1 requests with one output token. The table
reports observed input tokens divided by TTFT; TTFT is not repeated as a
separate headline metric.

| Engine | 8K prompt tok/s | 32K prompt tok/s | 64K prompt tok/s | 128K prompt tok/s |
|---|---:|---:|---:|---:|
| SGLang v0.5.0-rc.1 | 7,665.1 | 8,602.4 | 8,284.2 | 7,785.1 |
| vLLM r33 | 7,689.8 | 8,784.7 | 8,518.7 | 7,953.6 |

### Quality checks

| Engine | GSM8K questions | Correct | Accuracy | Request errors |
|---|---:|---:|---:|---:|
| SGLang v0.5.0-rc.1 | 1,319 | 1,261 | 95.60% | 0 |
| vLLM r33 | 1,319 | 1,243 | 94.24% | 0 |

[BENCHMARKS.md](BENCHMARKS.md) documents the exact method and every run-level
value. The executable harness, scoring code, graders, and machine-readable
summaries are under [`bench/`](bench/). [CHANGELOG.md](CHANGELOG.md) records
release composition; [PERFORMANCE-HISTORY.md](PERFORMANCE-HISTORY.md) retains
prior public snapshots separately from the current result view.

## License

Apache-2.0. The patches derive from SGLang, FlashInfer, and DeepGEMM, which are
also Apache-2.0 licensed.
