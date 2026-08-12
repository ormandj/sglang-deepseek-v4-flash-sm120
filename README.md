# SGLang for DeepSeek-V4-Flash on SM120

This repository publishes a ready-to-run SGLang container image for
`deepseek-ai/DeepSeek-V4-Flash-0731` on NVIDIA RTX PRO 6000 Blackwell (SM120)
GPUs. You do not need to build SGLang, FlashInfer, or the image locally.

## Fast path: run the prebuilt image

The current public release candidate is already built and available for
anonymous pulls from GitHub Container Registry:

```text
ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.2.0-rc.0
```

Pull it directly with Docker; no local image build is required:

```bash
docker pull ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.2.0-rc.0
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

export CACHE_DIR=/srv/cache/sglang-dsv4-0731-v10
mkdir -p "$CACHE_DIR"
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
  --env CUDA_VISIBLE_DEVICES=0,1 \
  --env SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  --env TILELANG_CACHE_DIR=/root/.cache/tilelang \
  --env TVM_CACHE_DIR=/root/.cache/tvm \
  --env SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0 \
  --env SGLANG_OPT_USE_TILELANG_INDEXER=0 \
  --env SGLANG_OPT_DEEPGEMM_HC_PRENORM=1 \
  ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.2.0-rc.0 \
  serve \
  --model-path /models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.93 \
  --context-length 774656 \
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

This command mounts the model read-only, uses GPUs 0 and 1, persists compiled
kernels under `CACHE_DIR`, and exposes the OpenAI-compatible API on port 8000.
The first launch compiles SM120 kernels; subsequent launches reuse the
persistent `v10` cache.

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

`v0.2.0-rc.0` is a clean reimage. It does not carry the experimental
TRT/MNNVL all-reduce fusion, PCIe-IPC communication paths, TBO, or other
performance experiments from the previous release line.

## Source composition

- Base image: `lmsysorg/sglang:nightly-dev-cu13-20260812-c7c03ec5`, pinned by
  digest.
- SGLang main: `dc5f6c488317645d96dc630b1f410e4dfb6f9667`.
- SGLang effective tree: `8c336f4426844b2028938f7c542c0a403d37b804`.
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
16,384-token input, 4,096 forced output tokens, temperature 0, top-p 1, and
five fixed prompt paths at every supported concurrency.

Each table entry below is the median of five same-process repetitions. ITL is
AIPerf's average post-first-token time per generated token.

### Controlled decode

| Engine | C | Verifier steps/s | Useful decode tok/s | ITL ms/token |
|---|---:|---:|---:|---:|
| SGLang v0.2.0-rc.0 | 1 | 55.54 | 301.6 | 3.58 |
| vLLM r33 | 1 | 66.58 | 255.2 | 3.89 |
| SGLang v0.2.0-rc.0 | 2 | 44.22 | 480.3 | 4.35 |
| vLLM r33 | 2 | 46.34 | 421.0 | 5.22 |
| SGLang v0.2.0-rc.0 | 4 | 32.19 | 698.5 | 6.34 |
| vLLM r33 | 4 | 33.07 | 616.0 | 7.06 |
| SGLang v0.2.0-rc.0 | 8 | 21.39 | 865.3 | 10.66 |
| vLLM r33 | 8 | 23.45 | 795.8 | 11.08 |
| SGLang v0.2.0-rc.0 | 16 | 16.44 | 1,332.0 | 15.88 |
| vLLM r33 | 16 | 15.86 | 1,097.2 | 17.59 |
| SGLang v0.2.0-rc.0 | 32 | 12.15 | 1,881.0 | 24.77 |
| vLLM r33 | 32 | Not measured: insufficient KV capacity | Not measured | Not measured |

### DSpARK acceptance

Each entry is the median of the five per-run means.

| Engine | C | Acceptance rate | Accepted tokens/step/request |
|---|---:|---:|---:|
| SGLang v0.2.0-rc.0 | 1 | 0.889 | 5.444 |
| vLLM r33 | 1 | 0.588 | 3.938 |
| SGLang v0.2.0-rc.0 | 2 | 0.864 | 5.321 |
| vLLM r33 | 2 | 0.731 | 4.655 |
| SGLang v0.2.0-rc.0 | 4 | 0.886 | 5.429 |
| vLLM r33 | 4 | 0.764 | 4.821 |
| SGLang v0.2.0-rc.0 | 8 | 0.793 | 4.963 |
| vLLM r33 | 8 | 0.649 | 4.245 |
| SGLang v0.2.0-rc.0 | 16 | 0.796 | 4.981 |
| vLLM r33 | 16 | 0.672 | 4.359 |
| SGLang v0.2.0-rc.0 | 32 | 0.754 | 4.768 |
| vLLM r33 | 32 | Not measured | Not measured |

### Cold prefill

One output token and five cache-cold requests were used per cell.

| Engine | 8K prompt tok/s | 32K prompt tok/s | 64K prompt tok/s | 128K prompt tok/s |
|---|---:|---:|---:|---:|
| SGLang v0.2.0-rc.0 | 6,424.4 | 7,075.6 | 6,835.5 | 6,546.2 |
| vLLM r33 | 7,689.8 | 8,784.7 | 8,518.7 | 7,953.6 |

### Completed quality result

| Engine | Gate | Questions | Correct | Accuracy | Request errors |
|---|---|---:|---:|---:|---:|
| SGLang v0.2.0-rc.0 | GSM8K | 1,319 | 1,258 | 95.38% | 0 |
| vLLM r33 | GSM8K | 1,319 | 1,243 | 94.24% | 0 |

Near-context and AgentX checks are not represented as completed results.

[BENCHMARKS.md](BENCHMARKS.md) documents the frozen method, exact revisions,
metric definitions, and commands. The executable harness, scoring code,
graders, and machine-readable summaries are under [`bench/`](bench/).

## Validated configuration

- Hardware: 2× NVIDIA RTX PRO 6000 Blackwell Max-Q, SM120, TP2.
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731` at
  `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`.
- KV cache: FP8 E4M3.
- Context length: 774,656.
- Chunked prefill: 8,192 tokens.
- HC prenorm: SM120 DeepGEMM for large-token prefill batches; the existing
  fallback remains selected below the 1,024-token dispatch threshold.
- DSpARK block size: 5.
- Communication: upstream NCCL; custom all-reduce disabled.

TP4 has not been validated on this hardware.

## License

Apache-2.0. The patches derive from SGLang and FlashInfer, which are also
Apache-2.0 licensed.
