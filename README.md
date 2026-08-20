# SGLang for DeepSeek-V4-Flash on SM120

A ready-to-run Linux x86_64 image for serving
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
with SGLang on NVIDIA RTX PRO 6000 Blackwell (SM120). The image contains the
qualified SGLang, FlashInfer, and DeepGEMM composition; no local engine build is
required.

Current release: `0.7.0-rc1`

```text
ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.7.0-rc.1
```

## Why use this image

- **A measured SM120 starting point.** The launch recipe, compilation-cache
  schema, warmups, and engine selectors are qualified together. These settings
  materially affect speed, memory use, and first-request behavior.
- **Large TP2 KV capacity.** The shipped `0.93` memory fraction provides an
  801,536-token KV pool and a 786,432-token per-request context budget on two
  RTX PRO 6000 Max-Q GPUs.
- **Long-conversation reuse.** Optional hierarchical KV cache (HiCache) can
  keep evicted prefixes in host memory and a bounded file-storage tier. It is
  disabled by default so one-shot workloads do not pay its write cost.
- **Reproducible and inspectable.** Immutable SemVer tags, exact source locks,
  patch hashes, machine-readable results, and the full benchmark harness are
  published in this repository.

### Performance and capacity at a glance

The tables below compare the qualified TP2 SGLang configuration with the
retained vLLM r33 TP2 profile. Each performance cell is the median of five
same-process repetitions. Forward passes/s is the primary decode execution
metric; synthetic output tok/s also reflects speculative acceptance and is not
an estimate of application throughput.

| Concurrency | SGLang forward/s | vLLM forward/s | SGLang synthetic tok/s | vLLM synthetic tok/s |
|---:|---:|---:|---:|---:|
| 1 | 65.106 | 66.575 | 345.3 | 255.2 |
| 2 | 48.797 | 46.337 | 457.3 | 421.0 |
| 4 | 33.670 | 33.074 | 660.0 | 616.0 |
| 8 | 23.226 | 23.454 | 939.4 | 795.8 |
| 16 | 17.853 | 15.865 | 1397.0 | 1097.2 |
| 32 | 13.336 | not reachable in profile | 2101.7 | — |

| Cold-prefill target | SGLang prompt tok/s | vLLM prompt tok/s |
|---:|---:|---:|
| 8K | 7925.9 | 7689.8 |
| 32K | 8825.5 | 8784.7 |
| 64K | 8461.4 | 8518.7 |
| 128K | 7953.3 | 7953.6 |

| TP2 profile | SGLang 0.7.0-rc1 | vLLM r33 |
|---|---:|---:|
| Reported KV cache | 801,536 tokens | 143,599 tokens |
| Declared context limit | 786,432 | 131,072 |
| Scheduler sequence limit | 48 | 16 |
| Fixed-window C32 decode | measured | not reachable in profile |
| GSM8K, correct / 1,319 | 1,260 (95.53%) | 1,243 (94.24%) |
| GSM8K request errors | 0 | 0 |

The capacity rows describe these exact deployment profiles, not an
engine-independent memory-efficiency test. Hardware, request shape, acceptance,
quality method, latency, and complete run data are in
[`BENCHMARKS.md`](BENCHMARKS.md).

## Quickstart

Start with this configuration unchanged, confirm it on your system, and then
adjust one capacity or topology setting at a time. In particular, keep the
documented environment variables, warmups, memory fraction, cache schema, and
engine flags together unless you are deliberately requalifying the result.

### 1. Prepare the host

You need Linux x86_64, a CUDA 13-compatible NVIDIA driver, Docker with the
NVIDIA Container Toolkit, two visible SM120 GPUs, persistent model storage,
and [`uv`](https://docs.astral.sh/uv/).

Download the pinned model snapshot and create an image-specific compiled-kernel
cache:

```bash
export MODEL_DIR=/srv/models/DeepSeek-V4-Flash-0731
uvx --from huggingface-hub hf download \
  deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  --local-dir "$MODEL_DIR"

export CACHE_DIR=/srv/cache/sglang-dsv4-0731-v26
mkdir -p "$CACHE_DIR"
```

Do not reuse a compiled-kernel cache from another image source tree or cache
schema.

### 2. Run the qualified TP2 configuration

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
  --env TRITON_CACHE_DIR=/root/.cache/triton \
  --env SGLANG_MAX_NEW_TOKENS_LIMIT=393216 \
  --env SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0 \
  --env SGLANG_OPT_USE_TILELANG_INDEXER=0 \
  --env SGLANG_OPT_DEEPGEMM_HC_PRENORM=1 \
  --env SGLANG_OPT_FUSE_MHC_POST_PRE=1 \
  --env SGLANG_OPT_FP8_WO_A_GEMM=1 \
  --env SGLANG_DSV4_MQA_LOGITS_FREE_MEM_FRACTION=0.8 \
  --env SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1 \
  --env SGLANG_PCIE_IPC_MAX_NUMEL=786432 \
  --env SGLANG_PCIE_IPC_AUTOTUNE=1 \
  ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.7.0-rc.1 \
  serve \
  --model-path /models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.93 \
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

The first start compiles SM120 kernels before the service reports ready. Keep
the v26 cache for later starts of this exact image.

### 3. Verify health, capacity, and generation

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

A clean startup is not a capacity test. Before exposing a changed topology,
context limit, memory fraction, or graph configuration, complete a request near
the intended limit.

### Optional wrapper

The maintained wrapper executes the same recipe and validates its inputs:

```bash
MODEL_DIR="$MODEL_DIR" CACHE_DIR="$CACHE_DIR" \
  ./examples/serve-dsv4-0731.sh
```

It defaults to GPUs `0,1`, TP2, port 8000, and the qualified context. It also
exposes explicit TP, image, port, context, output-limit, and HiCache overrides.
See [`RUN.md`](RUN.md) before changing them.

## More information

- [`RUN.md`](RUN.md): topology changes, capacity sizing, HiCache, safe exposure,
  and operational checks.
- [`BENCHMARKS.md`](BENCHMARKS.md): paired results, exact configurations,
  metric definitions, and reproduction commands.
- [`bench/README.md`](bench/README.md): executable benchmark and quality tools.
- [`CHANGELOG.md`](CHANGELOG.md): current public release changes and validation.
- [`release.json`](release.json) and [`stack.lock.json`](stack.lock.json): release
  identity, source pins, effective trees, package versions, and patch hashes.
- [`LICENSE`](LICENSE): Apache-2.0 licensing for this bundle; source patches
  derive from Apache-2.0 SGLang, FlashInfer, and DeepGEMM projects.

The image is reconstructed by GitHub Actions from the locked files and patches
in this repository. Candidate and stable SemVer tags are immutable; no mutable
`latest` tag is published.
