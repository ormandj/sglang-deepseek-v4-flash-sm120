# SGLang for DeepSeek-V4-Flash on SM120

This repository publishes a ready-to-run SGLang container image for
`deepseek-ai/DeepSeek-V4-Flash-0731` on NVIDIA RTX PRO 6000 Blackwell (SM120)
GPUs. You do not need to build SGLang, FlashInfer, or the image locally.

## Fast path: run the prebuilt image

The current public release candidate is already built and available for
anonymous pulls from GitHub Container Registry:

```text
ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.2.1-rc.0
```

Pull it directly with Docker; no local image build is required:

```bash
docker pull ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.2.1-rc.0
```

The image is built and published by
[this repository's GitHub Actions workflow](.github/workflows/build-image.yml)
using this repository's `Containerfile`, `release.json`, `stack.lock.json`, and
recorded patches as its build inputs. The image records this GitHub repository
and the exact source commit in its OCI source and revision labels. It is not a
separately maintained or manually modified image.

It is a Linux x86_64 image validated with two SM120 GPUs at tensor parallel
size 2. Candidate and stable tags are immutable, and this project does not
publish a mutable `latest` tag. [`release.json`](release.json) is the version
source of truth.

### Start the server

Install Docker, the NVIDIA Container Toolkit, and `uv`. Download the pinned
model snapshot and create a persistent compiled-kernel cache:

```bash
export MODEL_DIR=/srv/models/DeepSeek-V4-Flash-0731
uvx --from huggingface-hub hf download \
  deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  --local-dir "$MODEL_DIR"

export CACHE_DIR=/srv/cache/sglang-dsv4-0731-v11
mkdir -p "$CACHE_DIR"

export GPU_IDS=0,1
export TP_SIZE=2
export CONTEXT_LENGTH=774656
```

Start the prebuilt image directly:

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
  --env CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  --env SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  --env TILELANG_CACHE_DIR=/root/.cache/tilelang \
  --env TVM_CACHE_DIR=/root/.cache/tvm \
  --env SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0 \
  --env SGLANG_OPT_USE_TILELANG_INDEXER=0 \
  --env SGLANG_OPT_DEEPGEMM_HC_PRENORM=1 \
  --env SGLANG_OPT_FUSE_MHC_POST_PRE=1 \
  --env SGLANG_OPT_FP8_WO_A_GEMM=1 \
  ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.2.1-rc.0 \
  serve \
  --model-path /models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code \
  --tensor-parallel-size "$TP_SIZE" \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.93 \
  --context-length "$CONTEXT_LENGTH" \
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
  --sleep-on-idle \
  --host 0.0.0.0 \
  --port 8000
```

This command mounts the model read-only, persists compiled kernels under
`CACHE_DIR`, and exposes the OpenAI-compatible API on port 8000. The first
launch compiles SM120 kernels; subsequent launches reuse the persistent `v11`
cache.

`GPU_IDS` selects the devices and `TP_SIZE` sets the tensor-parallel size. The
number of comma-separated devices must equal `TP_SIZE`. For example, change to
TP4 before running the same Docker command with:

```bash
export GPU_IDS=0,1,2,3
export TP_SIZE=4
```

TP2 is the published validated configuration, not an image limitation. TP4 or
TP8 uses the same image with a matching device list and `TP_SIZE`; published
performance and capacity numbers do not yet cover those topologies. Keep the
TP2 context limit as the initial value, then inspect the actual pool after
startup before increasing it:

```bash
curl -fsS http://localhost:8000/get_server_info \
  | jq '{max_total_num_tokens, max_req_input_len}'
```

Complete a near-limit request before advertising a larger context. No
topology-specific image is required.

In another shell, verify the server and send a request:

```bash
curl -fsS http://localhost:8000/health

curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

See [RUN.md](RUN.md) for requirements, image and port overrides, API examples,
capacity checks, and operational notes. For users who prefer a checked wrapper,
[examples/serve-dsv4-0731.sh](examples/serve-dsv4-0731.sh) validates the paths
and executes this same Docker configuration.

`v0.2.1-rc.0` adds the SM120/SM121 FP8 W_o_A target-model path from SGLang
#34018 to the clean `v0.2.0-rc.0` source stack. It does not carry the
experimental TRT/MNNVL all-reduce fusion, PCIe-IPC communication paths, TBO,
or other performance experiments from the previous release line.

The launch command enables the fused MHC post+pre implementation already in
the image with `SGLANG_OPT_FUSE_MHC_POST_PRE=1`. This follows the open upstream
[SGLang PR #34019](https://github.com/sgl-project/sglang/pull/34019), which
changes the SM120 default. That PR is tracked but is not part of this image's
source patch, and enabling the existing runtime selector does not rebuild or
change the image.

## Source composition

- Base image: `lmsysorg/sglang:nightly-dev-cu13-20260812-c7c03ec5`, pinned by
  digest.
- SGLang main: `dc5f6c488317645d96dc630b1f410e4dfb6f9667`.
- SGLang effective tree: `68de16e0e3ddc5b5d04b6a2bdfbabbbebefc5e03`.
- FlashInfer main: `065971254bca6ad0509d775e5806de53b64ac7b9`.
- FlashInfer effective tree: `09b10c6dc66ca0c96c62a13dfa5ea3b63f1018e4`.
- FlashInfer package version: `0.6.18.dev20260811`.

The SGLang patch contains the current reviewed heads of:

| Source | Scope |
|---|---|
| SGLang #29927 | SM120 DeepSeek-V4 model, DeepGEMM indexer, prefill, and FP4 MoE support |
| SGLang #33614 | Synchronize DSpark sampling and graph decisions across TP ranks |
| SGLang #32686 | Bound DeepGEMM warmup allocation by available memory |
| SGLang #33568 | Reference-compatible DeepSeek-V4 tool encoding |
| SGLang #33805 | Sliding-window KV eviction on the DFLASH/DSPARK path |
| SGLang #34018 | SM120/SM121 FP8 W_o_A target-model GEMM |

The FlashInfer patch contains PR #3930 and its exact CUDA-runtime filename
resolver follow-up. Every source head, integration commit, patch checksum, and
tree hash is recorded in [stack.lock.json](stack.lock.json).
[`scripts/verify-patches.sh`](scripts/verify-patches.sh) independently fetches
the pinned upstream commits, applies the recorded patches, and checks both
effective trees.

## Measurements

Both engines were measured on the same two RTX PRO 6000 Blackwell Max-Q GPUs
at TP2 over PCIe Gen 4 x16, with a 300 W limit per GPU. Only one engine was
active. Clients ran inside the serving pod against localhost. Decode used a
16,384-token input, 4,096 forced output tokens, temperature 0, and top-p 1.
SGLang C1 used ten fixed-seed prompt paths; vLLM C1 and every other measured
cell used five.

Each table entry below is a same-process median. C1 uses ten fixed-seed prompt
paths; every other supported concurrency uses five. ITL is AIPerf's average
post-first-token time per generated token. Synthetic fixed-window output tok/s
is a controlled engine-comparison metric, not expected production throughput.
It combines verifier-step rate with output tokens produced per step, so DSpARK
acceptance directly affects it. Different fixed prompt/seed paths have
different accepted-draft-length distributions, making this metric more variable
than verifier steps/s. Acceptance and output tokens/step are reported
separately below.

### Synthetic fixed-window decode

| Engine | C | n | Verifier steps/s | Synthetic output tok/s (acceptance-dependent) | ITL ms/token |
|---|---:|---:|---:|---:|---:|
| SGLang v0.2.1-rc.0 | 1 | 10 | 62.37 | 322.0 | 3.35 |
| vLLM r33 | 1 | 5 | 66.58 | 255.2 | 3.89 |
| SGLang v0.2.1-rc.0 | 2 | 5 | 46.85 | 462.1 | 4.57 |
| vLLM r33 | 2 | 5 | 46.34 | 421.0 | 5.22 |
| SGLang v0.2.1-rc.0 | 4 | 5 | 32.16 | 665.4 | 6.49 |
| vLLM r33 | 4 | 5 | 33.07 | 616.0 | 7.06 |
| SGLang v0.2.1-rc.0 | 8 | 5 | 22.33 | 871.0 | 10.63 |
| vLLM r33 | 8 | 5 | 23.45 | 795.8 | 11.08 |
| SGLang v0.2.1-rc.0 | 16 | 5 | 17.19 | 1,356.3 | 14.89 |
| vLLM r33 | 16 | 5 | 15.86 | 1,097.2 | 17.59 |
| SGLang v0.2.1-rc.0 | 32 | 5 | 12.64 | 1,951.5 | 22.81 |
| vLLM r33 | 32 | 0 | Not measured: insufficient KV capacity | Not measured | Not measured |

### DSpARK acceptance

Each entry is the median of the per-run means. Sample counts match the decode
table above.

| Engine | C | Acceptance rate | Output tokens/step/request |
|---|---:|---:|---:|
| SGLang v0.2.1-rc.0 | 1 | 0.841 | 5.206 |
| vLLM r33 | 1 | 0.588 | 3.938 |
| SGLang v0.2.1-rc.0 | 2 | 0.773 | 4.819 |
| vLLM r33 | 2 | 0.731 | 4.655 |
| SGLang v0.2.1-rc.0 | 4 | 0.827 | 5.080 |
| vLLM r33 | 4 | 0.764 | 4.821 |
| SGLang v0.2.1-rc.0 | 8 | 0.755 | 4.825 |
| vLLM r33 | 8 | 0.649 | 4.245 |
| SGLang v0.2.1-rc.0 | 16 | 0.769 | 4.938 |
| vLLM r33 | 16 | 0.672 | 4.359 |
| SGLang v0.2.1-rc.0 | 32 | 0.760 | 4.819 |
| vLLM r33 | 32 | Not measured | Not measured |

### Cold prefill

One output token and five cache-cold requests were used per cell.

| Engine | 8K prompt tok/s | 32K prompt tok/s | 64K prompt tok/s | 128K prompt tok/s |
|---|---:|---:|---:|---:|
| SGLang v0.2.1-rc.0 | 7,541.7 | 8,528.6 | 7,907.6 | 7,771.2 |
| vLLM r33 | 7,689.8 | 8,784.7 | 8,518.7 | 7,953.6 |

### Completed quality result

The SGLang GSM8K run used the same RC0 image with HC prenorm enabled and the
fused-MHC selector unset. It is retained as a separate quality result; the
performance sweep above does not replace it.

| Engine | Gate | Questions | Correct | Accuracy | Request errors |
|---|---|---:|---:|---:|---:|
| SGLang v0.2.1-rc.0 | GSM8K | 1,319 | 1,262 | 95.68% | 0 |
| vLLM r33 | GSM8K | 1,319 | 1,243 | 94.24% | 0 |

Near-context and AgentX checks are not represented as completed results.

[BENCHMARKS.md](BENCHMARKS.md) documents the frozen method, exact revisions,
metric definitions, and commands. Immutable per-release snapshots are in
[PERFORMANCE-HISTORY.md](PERFORMANCE-HISTORY.md). The executable harness,
scoring code, graders, and machine-readable summaries are under
[`bench/`](bench/).

## Validated configuration

- Hardware: 2× NVIDIA RTX PRO 6000 Blackwell Max-Q, SM120, TP2.
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731` at
  `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`.
- KV cache: FP8 E4M3.
- Context length: 774,656.
- Chunked prefill: 8,192 tokens.
- HC prenorm: SM120 DeepGEMM for large-token prefill batches; the existing
  fallback remains selected below the 1,024-token dispatch threshold.
- Fused MHC post+pre: `SGLANG_OPT_FUSE_MHC_POST_PRE=1`.
- FP8 W_o_A target path: `SGLANG_OPT_FP8_WO_A_GEMM=1`.
- DSpARK block size: 5.
- Communication: upstream NCCL; custom all-reduce disabled.

TP4 has not been validated on this hardware.

## License

Apache-2.0. The patches derive from SGLang and FlashInfer, which are also
Apache-2.0 licensed.
