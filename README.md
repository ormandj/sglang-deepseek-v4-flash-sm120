# SGLang for DeepSeek-V4-Flash on SM120

This repository publishes a ready-to-run Linux x86_64 SGLang image for
`deepseek-ai/DeepSeek-V4-Flash-0731` on NVIDIA RTX PRO 6000 Blackwell (SM120).
The image is built directly from this repository's pinned source composition;
you do not need to build SGLang, FlashInfer, DeepGEMM, or the image locally.

## Fast path

The current candidate is:

```text
ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.3.3-rc.0
```

Install Docker, the NVIDIA Container Toolkit, and `uv`, then download the
pinned model snapshot and create a release-specific compiled-kernel cache:

```bash
export MODEL_DIR=/srv/models/DeepSeek-V4-Flash-0731
uvx --from huggingface-hub hf download \
  deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  --local-dir "$MODEL_DIR"

export CACHE_DIR=/srv/cache/sglang-dsv4-0731-v15
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
  --env SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0 \
  --env SGLANG_OPT_USE_TILELANG_INDEXER=0 \
  --env SGLANG_OPT_DEEPGEMM_HC_PRENORM=1 \
  --env SGLANG_OPT_FUSE_MHC_POST_PRE=1 \
  --env SGLANG_OPT_FP8_WO_A_GEMM=1 \
  --env SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1 \
  --env SGLANG_PCIE_IPC_MAX_NUMEL=786432 \
  ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.3.3-rc.0 \
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

Alternatively, clone this repository and use the checked wrapper, which
executes the same configuration and validates its inputs:

```bash
MODEL_DIR="$MODEL_DIR" CACHE_DIR="$CACHE_DIR" \
  ./examples/serve-dsv4-0731.sh
```

The wrapper validates the local paths and runs the Docker command recorded in
this repository. It defaults to GPUs `0,1`, TP2, port 8000, and the qualified
774,656-token context limit. Override those values explicitly when needed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TP_SIZE=4 \
CONTEXT_LENGTH=774656 \
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
post+pre, and the FP8 W_o_A target-model path. The new v0.3.3 correction keeps
the logical runtime token dimension dynamic when SM120 GEMMs swap operands,
preventing residual token counts from producing distinct W_o_A JIT kernels.

### Source identity

| Component | Pinned identity |
|---|---|
| Base image | `lmsysorg/sglang:nightly-dev-cu13-20260813-273d978b@sha256:5e012cc3cfe06fd7718bab6f7b8183fad56df28a6b934058edb4d59afc42d440` |
| SGLang main | `9d34c2809f58f3d84ef5dd343733e3f5e86395d5` |
| SGLang effective tree | `2833e60bfdb9c17820095e2e1aa478bb2eb041ef` |
| FlashInfer main | `ed6c709849fe1c02d4545b4e743a436405f6ca5b` |
| FlashInfer effective tree | `b68d15ef203d0e0e11b3734ade4ccf6dca1b6b4d` |
| FlashInfer package | `0.6.18.dev20260813` |
| DeepGEMM base | `7509acb3e261b5acba06087e91c70c409a43419c` |
| DeepGEMM effective tree | `b166d085065d39155a8f745126d6db88597d268c` |
| DeepGEMM package | `0.1.5.post2+sm120jit2` |

The SGLang patch contains the audited source heads of #29927, #33614, #32686,
#33568, #33805, #34018, and #34528. The FlashInfer patch contains #3930,
#4393, the exact CUDA-runtime filename resolver follow-up, and the focused
outer-autotune preservation fix. The DeepGEMM patch contains the held-constant
SM120 single-token split-K change and the reviewed runtime-dimension fix.
Every source head, integration commit, patch checksum, and resulting tree is
recorded in [stack.lock.json](stack.lock.json). Run
[`scripts/verify-patches.sh`](scripts/verify-patches.sh) to reconstruct and
verify the three effective trees.

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
| SGLang v0.3.3-rc.0 | 1 | 5 | 63.864 | 331.7 | 3.110 |
| vLLM r33 | 1 | 5 | 66.580 | 255.2 | 3.890 |
| SGLang v0.3.3-rc.0 | 2 | 5 | 47.737 | 488.6 | 4.321 |
| vLLM r33 | 2 | 5 | 46.340 | 421.0 | 5.220 |
| SGLang v0.3.3-rc.0 | 4 | 5 | 34.322 | 653.9 | 6.481 |
| vLLM r33 | 4 | 5 | 33.070 | 616.0 | 7.060 |
| SGLang v0.3.3-rc.0 | 8 | 5 | 22.968 | 912.3 | 9.911 |
| vLLM r33 | 8 | 5 | 23.450 | 795.8 | 11.080 |
| SGLang v0.3.3-rc.0 | 16 | 5 | 17.304 | 1,392.0 | 14.440 |
| vLLM r33 | 16 | 5 | 15.860 | 1,097.2 | 17.590 |
| SGLang v0.3.3-rc.0 | 32 | 5 | 12.722 | 2,036.8 | 22.085 |
| vLLM r33 | 32 | 0 | Not measured: insufficient KV capacity | Not measured | Not measured |

### DSpARK acceptance

Acceptance is recorded to explain variation in synthetic output throughput;
it is not used as an engine-clock metric.

| Engine | C | Median acceptance rate | Median output tokens/forward/request |
|---|---:|---:|---:|
| SGLang v0.3.3-rc.0 | 1 | 0.851 | 5.215 |
| vLLM r33 | 1 | 0.588 | 3.938 |
| SGLang v0.3.3-rc.0 | 2 | 0.807 | 5.001 |
| vLLM r33 | 2 | 0.731 | 4.655 |
| SGLang v0.3.3-rc.0 | 4 | 0.779 | 4.891 |
| vLLM r33 | 4 | 0.764 | 4.821 |
| SGLang v0.3.3-rc.0 | 8 | 0.775 | 4.872 |
| vLLM r33 | 8 | 0.649 | 4.245 |
| SGLang v0.3.3-rc.0 | 16 | 0.793 | 5.027 |
| vLLM r33 | 16 | 0.672 | 4.359 |
| SGLang v0.3.3-rc.0 | 32 | 0.790 | 4.984 |
| vLLM r33 | 32 | Not measured | Not measured |

### Cold prefill

Each cell used five cache-busted C1 requests with one output token. The table
reports observed input tokens divided by TTFT; TTFT is not repeated as a
separate headline metric.

| Engine | 8K prompt tok/s | 32K prompt tok/s | 64K prompt tok/s | 128K prompt tok/s |
|---|---:|---:|---:|---:|
| SGLang v0.3.3-rc.0 | 7,776.8 | 8,743.1 | 8,464.5 | 7,966.6 |
| vLLM r33 | 7,689.8 | 8,784.7 | 8,518.7 | 7,953.6 |

### Quality checks

| Engine | GSM8K questions | Correct | Accuracy | Request errors |
|---|---:|---:|---:|---:|
| SGLang v0.3.3-rc.0 | 1,319 | 1,261 | 95.60% | 0 |
| vLLM r33 | 1,319 | 1,243 | 94.24% | 0 |

The requested release-specific long-output diagnostic completed 8/8 requests
with `finish_reason=stop`; all eight passed the code-fence, closing-tag,
placeholder, and JavaScript-parse checks. It used temperature 1.0, top-p 0.95,
`reasoning_effort=max`, and a 131,072-token response cap. It is not part of the
routine recurring benchmark protocol.

[BENCHMARKS.md](BENCHMARKS.md) documents the exact method and every run-level
value. The executable harness, scoring code, graders, and machine-readable
summaries are under [`bench/`](bench/). [CHANGELOG.md](CHANGELOG.md) records
release composition; [PERFORMANCE-HISTORY.md](PERFORMANCE-HISTORY.md) retains
prior public snapshots separately from the current result view.

## License

Apache-2.0. The patches derive from SGLang, FlashInfer, and DeepGEMM, which are
also Apache-2.0 licensed.
