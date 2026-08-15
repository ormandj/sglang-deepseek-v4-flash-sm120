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
  --env TRITON_CACHE_DIR=/root/.cache/triton \
  --env SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0 \
  --env SGLANG_OPT_USE_TILELANG_INDEXER=0 \
  --env SGLANG_OPT_DEEPGEMM_HC_PRENORM=1 \
  ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.2.0-rc.0 \
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
launch compiles SM120 kernels; subsequent launches reuse the persistent `v10`
cache.

### KV cache capacity

`--mem-fraction-static` sets how much device memory is reserved for the static
KV pool. It is the single most consequential capacity knob, and it trades
directly against per-request workspace:

| `--mem-fraction-static` | `--context-length` | `max_total_num_tokens` | Free after startup | 240k-token request |
|---|---:|---:|---:|---|
| 0.95 | 786,432 | 990,208 | 3.11 GB | passes |
| 0.96 | 524,288 | 1,099,264 | 2.38 GB | not tested |
| 0.96 | 774,656 | 1,099,264 | 2.21 GB | **crashes the server** |

`max_total_num_tokens` is SGLang's own self-reported figure at startup, as
is the free-memory column. Measured on 2x RTX PRO 6000 Max-Q at TP2.
Raising the fraction buys KV pool and
spends the headroom that long single requests need for activations and
workspace; lowering it does the reverse.

The third row is the important one. At 0.96/774,656 a single 240,269-token
request killed the server with `c10::OutOfMemoryError` even though the KV pool
was large enough to hold it and startup was clean. The same request returns 200
at 0.95/786,432 -- a *larger* context with a *smaller* pool -- because the extra
headroom covers the workspace. Do not choose this knob by maximising
`max_total_num_tokens`.

A configuration can boot cleanly and still fail later on a large request, so
validate with a near-limit request rather than trusting startup alone:

```bash
curl -fsS http://localhost:8000/get_server_info \
  | jq '{max_total_num_tokens, max_req_input_len}'
```

Note that `--context-length` is not only an input limit: it also sizes
per-request workspace. Declaring a context far larger than you serve costs
memory on every request.

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
| SGLang v0.2.0-rc.0 | 1 | 56.09 | 269.3 | 3.83 |
| vLLM r33 | 1 | 66.58 | 255.2 | 3.89 |
| SGLang v0.2.0-rc.0 | 2 | 45.37 | 493.5 | 4.24 |
| vLLM r33 | 2 | 46.34 | 421.0 | 5.22 |
| SGLang v0.2.0-rc.0 | 4 | 32.32 | 713.7 | 6.06 |
| vLLM r33 | 4 | 33.07 | 616.0 | 7.06 |
| SGLang v0.2.0-rc.0 | 8 | 21.65 | 870.1 | 10.60 |
| vLLM r33 | 8 | 23.45 | 795.8 | 11.08 |
| SGLang v0.2.0-rc.0 | 16 | 16.45 | 1,359.0 | 15.11 |
| vLLM r33 | 16 | 15.86 | 1,097.2 | 17.59 |
| SGLang v0.2.0-rc.0 | 32 | 12.29 | 1,949.2 | 23.11 |
| vLLM r33 | 32 | not reachable: vLLM-reported KV 143,599 tok, `max_num_seqs=16` (upstream TP2 recipe) | — | — |

### DSpARK acceptance

Each entry is the median of the five per-run means.

| Engine | C | Acceptance rate | Accepted tokens/step/request |
|---|---:|---:|---:|
| SGLang v0.2.0-rc.0 | 1 | 0.764 | 4.821 |
| vLLM r33 | 1 | 0.588 | 3.938 |
| SGLang v0.2.0-rc.0 | 2 | 0.888 | 5.439 |
| vLLM r33 | 2 | 0.731 | 4.655 |
| SGLang v0.2.0-rc.0 | 4 | 0.924 | 5.618 |
| vLLM r33 | 4 | 0.764 | 4.821 |
| SGLang v0.2.0-rc.0 | 8 | 0.791 | 4.954 |
| vLLM r33 | 8 | 0.649 | 4.245 |
| SGLang v0.2.0-rc.0 | 16 | 0.795 | 4.976 |
| vLLM r33 | 16 | 0.672 | 4.359 |
| SGLang v0.2.0-rc.0 | 32 | 0.778 | 4.889 |
| vLLM r33 | 32 | not reachable: vLLM-reported KV 143,599 tok, `max_num_seqs=16` (upstream TP2 recipe) | — |

> **On the missing vLLM C=32 rows.** The vLLM side was not hand-tuned by us: it
> runs the upstream-documented TP2 profile for this model, "Gilded Gnosis v20
> r33, documented TP2 fixed-K5", from
> [`local-inference-lab/rtx6kpro`](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/ds4dspark-v20-r33.md).
> Its concurrency and context settings are that recipe's, unmodified.
>
> That configuration cannot reach C=32, for two independent reasons. It sets
> `max_num_seqs=16`, which caps concurrency at 16 outright. Independently,
> vLLM reported at startup:
>
> ```
> Available KV cache memory: 8.07 GiB
> GPU KV cache size: 143,599 tokens
> Maximum concurrency for 131,072 tokens per request: 1.10x
> ```
>
> Those are vLLM's own self-reported figures, not our measurement of it. At the
> benchmark's 16,384-token input plus 4,096 output, 32 streams need roughly
> 655,000 KV tokens against the 143,599 available.
>
> Both limits are properties of *that published profile*, not capability
> ceilings of vLLM. Raising `max_num_seqs`, raising `--gpu-memory-utilization`,
> lowering `--max-model-len`, or shortening the per-request input would all
> change the answer. Do not read these blank cells as vLLM being unable to serve
> this concurrency -- they reflect a recipe tuned for a different operating
> point, run unaltered.
>
> Note also that the two engines were not configured for the same context:
> vLLM ran `max_model_len=131,072` against SGLang's 774,656. The KV pool figures
> are therefore not directly comparable between the two.

### Cold prefill

One output token and five cache-cold requests were used per cell.

| Engine | 8K prompt tok/s | 32K prompt tok/s | 64K prompt tok/s | 128K prompt tok/s |
|---|---:|---:|---:|---:|
| SGLang v0.2.0-rc.0 | 7,484.1 | 8,521.2 | 8,299.8 | 7,824.5 |
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
