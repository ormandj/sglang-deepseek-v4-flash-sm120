# sglang-deepseek-v4-flash-sm120

An SGLang container image for serving DeepSeek-V4-Flash-0731 on RTX PRO 6000 Blackwell (SM120) GPUs, built from pinned upstream sources plus a small set of patches that are not yet in an upstream release.

Current release candidate:

```
ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.1.0-rc.3
```

After the candidate passes the project acceptance gates, its exact manifest is
promoted without rebuilding to `v0.1.0`. Both tags are immutable. The workflow
also emits an immutable `build-N` diagnostic tag; there is deliberately no
`latest` or model-name moving alias. [release.json](release.json) is the version
source of truth and uses cache namespace `v2`, which is versioned independently
from the image.

See [RUN.md](RUN.md) for the run guide and [examples/serve-dsv4-0731.sh](examples/serve-dsv4-0731.sh) for the validated serving configuration.

## Source composition

- Base: the official CUDA 13 / Torch 2.13 SGLang nightly `lmsysorg/sglang:nightly-dev-cu13-20260810-7c90840b`, pinned by digest.
- SGLang: release-pinned main commit `5a8e360e`, plus one source patch containing the listed carries and remaining local integration changes. The final tree is `70fa46c3`.
- FlashInfer: release-pinned main commit `4fbac49f` (version `0.6.18.dev20260810`), rebuilt from source with PRs #4308 and #4393, giving tree `616094d4`. Merged PRs #4315, #4329 and #4380 are supplied by that main commit. The matching cubin wheel is installed by URL and pinned by SHA256. The prebuilt JIT-cache wheel is removed so the patched SM120 modules compile once into the persistent runtime cache.

Every pin is recorded in [stack.lock.json](stack.lock.json). [scripts/verify-patches.sh](scripts/verify-patches.sh) checks that the lock, the `Containerfile`, and the patch bytes agree, then clones the pinned commits and confirms that applying the patches in build order reproduces the recorded tree hashes.

## Included changes

| Source | Head | Scope in this image |
|---|---|---|
| SGLang PR #29927 | `21a5bc8e` | Complete SM120 DSV4 stack: DeepGEMM paged-MQA indexer, batched sparse-MLA prefill, and FP4 MoE |
| SGLang PR #33614 | `fca0998f` | Keeps DSpark speculative sampling, verify budgets, graph-capture decisions, and capture-safe broadcasts identical across TP ranks |
| SGLang PR #32194 | `f8fac391` | Skip unused DSV4 draft metadata in speculative decoding |
| SGLang PR #30700 | `aead319d` | FlashInfer all-reduce-only dispatch and auto-enable |
| SGLang PR #32330 | `f330c748` | FlashInfer TRT-LLM all-reduce on SM12X; rebased with an mnnvl preflight multicast-granularity fix (trtllm still skips the multicast query, which fails on SM120) |
| SGLang PR #32686 | `15c0902e` | DeepGEMM warmup: pick the largest warmup `M` that fits the memory budget |
| SGLang PR #32815 | `1dbf09f6` | Enable FP8 W_o_A GEMM when the installed DeepGEMM supports it |
| FlashInfer PR #4308 | `4ec7f230` | Makes the fused-MoE profiler's MXFP8 × MXFP4 quantization state match the runtime path, so autotuning stays enabled |
| FlashInfer PR #4393 | `6573c652` | `pcie_ipc` all-reduce for intra-node PCIe machines with no NVLink or multicast. Present in the image; off by default |
| SGLang PR #33518 | `c3249eb9` | Per-request speculative statistics on the OpenAI endpoints, behind `return_spec_tokens_details`; the current head's unique implementation commit is `1c3f7b70` |
| SGLang PR #33568 | `cab45a29` | Match DSV4/DSV32 tool prompts to DeepSeek's reference encoding |
| SGLang PR #33732 | `d5c25357` | Preserve an explicitly selected SM120 DSV4 indexer backend instead of overwriting it with the TileLang default |
| SGLang PR #33805 | `26a2a398` | Run sliding-window KV eviction on the DFLASH/DSPARK speculative path |
| SGLang PR #34139 | `16797c8c` | Keep DSpark draft shared experts separate after #33889 made fusion decisions runner-local |

Every carried pull request, its owner, and when we can drop it:

| Source | Pinned head | Owner | Drops when |
|---|---|---|---|
| SGLang PR #29927 | `21a5bc8e` | AliceChenyy | merged |
| SGLang PR #33614 | `fca0998f` | JackZeng0208 | merged |
| SGLang PR #32194 | `f8fac391` | mattteochen | merged |
| SGLang PR #30700 | `aead319d` | wenscarl | merged |
| SGLang PR #32330 | `f330c748` | ormandj | merged |
| SGLang PR #32686 | `15c0902e` | ormandj | merged |
| SGLang PR #32815 | `1dbf09f6` | ormandj | merged |
| FlashInfer PR #4308 | `4ec7f230` | — | merged |
| FlashInfer PR #4393 | `6573c652` | — | merged |
| SGLang PR #33518 | `c3249eb9` | Muqi1029 | merged |
| SGLang PR #33568 | `cab45a29` | ormandj | merged |
| SGLang PR #33732 | `d5c25357` | ormandj | merged |
| SGLang PR #33805 | `26a2a398` | ormandj | merged |
| SGLang PR #34139 | `16797c8c` | ormandj | merged |

The current-main refresh removed SGLang #32183 because merged SGLang #34189
replaced its hardcoded four-token retention fix with ring-derived sizing and
broader CPU/GPU coverage. SGLang #33616 and FlashInfer #4315, #4329, and #4380
are also supplied by main. FlashInfer #4380 replaces the former #4309 carry.
SGLang #33140 was removed by an earlier refresh after it merged as
`059269594c`.

Image-local fixes:

| Fix | Head | Scope in this image |
|---|---|---|
| SGLang SM120 DeepGEMM capability probe | `d4dc7502` | Enables the SM120 DeepGEMM capability when the required grouped FP4 symbol is installed |
| SGLang all-reduce prefill workspace | `73d125d0` | Sizes the FlashInfer all-reduce workspace for the largest configured prefill forward before CUDA graph capture |
| SGLang SM120 all-reduce execution gate | `861b99ca` | Admits SM120 through the all-reduce execution gate for the backend enabled by PR #32330 |
| SGLang all-reduce token cap | `f99f4a0b` | Bounds the all-reduce-only kAllReduce path by the same token cap as the fused path. Without it the only ceiling is the workspace allocation, so sizing that workspace for the prefill forward can route prefill-sized all-reduces onto a min-latency kernel instead of the NCCL ring |
| SGLang pcie_ipc consumer | `790a72c2` | Makes FlashInfer PR #4393's backend selectable from SGLang; the FlashInfer PR provides kernels and policy but no SGLang integration |
| SGLang FlashInfer mHC dispatch gate | `eb9004a0` | On SM120, enables FlashInfer mHC only for token batches of at least 1024. This keeps the prefill path available without routing decode through the fixed-cost DeepGEMM TF32 prenorm that regressed the earlier global opt-in |

The intent is upstream-first: everything here is either an open upstream pull request carried at a pinned head, or a narrow local fix intended to be replaced by an upstream change. As those land, the corresponding patch content is dropped rather than maintained.

### Reported upstream, patches not yet filed

The runtime settings in [RUN.md](RUN.md) work around three SGLang defaults that
look wrong on SM120. Upstream changes for these are **TBD**; until they land the
workarounds live in [examples/serve-dsv4-0731.sh](examples/serve-dsv4-0731.sh).

| Finding | Effect here | Upstream status |
|---|---|---|
| The SM120 branch disables both fused MHC pre-norm paths, forcing an eager float32 fallback. The guard predates first-class SM120 support for the DeepGEMM `tf32_hc_prenorm_gemm` kernel | The validated runtime re-enables the DeepGEMM path | TBD — the TileLang half of the same guard is justified and should stay (it fails CUDA graph capture on SM120) |
| `SGLANG_OPT_USE_TOPK_V2` is force-disabled on SM120 without an `is_set()` guard, unlike its neighbours, so an operator-set value is overridden with no escape hatch | An explicit operator selection cannot be honored | TBD — gating alone changes no default |
| DSpark speculative decoding never accumulates presence/frequency/repetition penalties: `eagle_prepare_for_decode` calls `cumulate_penalty_output_tokens`, the dflash-family branch does not | Those sampling penalties are silently inert under DSpark | TBD |

## Current measurements

SGLang `v0.1.0-rc.3` and vLLM r33 were measured separately on the same two RTX
PRO 6000 Blackwell Max-Q GPUs at TP=2 and 300 W per GPU. Clients ran inside each
serving container against localhost. The fixed decode panel used temperature
0, top-p 1, ignored EOS, and one unchanged server process per engine.

### Decode engine rate

| C | repetitions | vLLM r33 median forward/s | SGLang rc.3 median forward/s | rc.3 paired change |
|---:|---:|---:|---:|---:|
| 1 | 3 | 65.622 | 59.614 | -9.28% |
| 2 | 6 | 47.801 | 46.783 | -0.65% |
| 4 | 4 | 34.072 | 34.129 | -0.35% |
| 8 | 2 | 23.862 | 22.828 | -4.34% |
| 16 | 2 | 16.618 | 17.322 | +4.24% |
| 32 | 1 SGLang only | not run | 12.854 | — |

### Useful decode throughput and acceptance

| C | vLLM r33 median useful tok/s | SGLang rc.3 median useful tok/s | vLLM accepted tokens/forward/request | rc.3 accepted tokens/forward/request |
|---:|---:|---:|---:|---:|
| 1 | 353.0 | 303.4 | 5.405 | 5.082 |
| 2 | 484.6 | 534.4 | 5.128 | 5.702 |
| 4 | 632.6 | 706.0 | 4.610 | 5.266 |
| 8 | 938.1 | 985.5 | 4.872 | 5.378 |
| 16 | 1,098.3 | 1,358.9 | 4.118 | 4.871 |
| 32 | not run | 1,890.0 | not run | 4.573 |

### Decode TTFT

Each entry is the median of the per-run p50 TTFT values. Brackets contain the
minimum and maximum per-run p50 values.

| C | repetitions | vLLM r33 median TTFT [min, max] | SGLang rc.3 median TTFT [min, max] |
|---:|---:|---:|---:|
| 1 | 3 | 506.5 ms [501.2, 511.5] | 231.8 ms [223.7, 248.9] |
| 2 | 6 | 774.9 ms [769.1, 808.9] | 391.8 ms [379.6, 404.7] |
| 4 | 4 | 1,139.2 ms [1,128.9, 1,369.9] | 407.7 ms [399.1, 427.9] |
| 8 | 2 | 1,533.3 ms [1,255.6, 1,811.0] | 640.0 ms [447.2, 832.8] |
| 16 | 2 | 1,559.0 ms [1,555.1, 1,562.9] | 1,365.2 ms [570.9, 2,159.5] |
| 32 | 1 SGLang only | not run | 965.4 ms [965.4, 965.4] |

### Cold prefill

| target | vLLM r33 prompt tok/s | SGLang rc.3 prompt tok/s | rc.3 change | vLLM median TTFT | rc.3 median TTFT |
|---:|---:|---:|---:|---:|---:|
| 8K C1 | 7,722.7 | 7,516.2 | -2.67% | 1,068.4 ms | 1,086.3 ms |
| 64K C1 | 8,647.4 | 8,343.9 | -3.51% | 7,564.6 ms | 7,828.1 ms |
| near-128K C1 | 8,001.0 | 7,780.1 | -2.76% | 16,358.0 ms | 16,811.6 ms |

[BENCHMARKS.md](BENCHMARKS.md) documents the metric definitions, fixed panel,
commands, and source versions. The executable harness is in
[`bench/aiperf`](bench/aiperf).

## Validated configuration

- Hardware: 2× NVIDIA RTX PRO 6000 Blackwell (SM120), TP=2. SM121 is not validated.
- Model: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), served as `deepseek-v4-flash`.
- Runtime settings: [examples/serve-dsv4-0731.sh](examples/serve-dsv4-0731.sh).
- Speculative decoding depth is set explicitly in the serving script; the checkpoint's `config.json` carries `dspark_block_size = 5`. We publish no depth recommendation — see [RUN.md](RUN.md). The validated runtime re-enables the fused DeepGEMM MHC pre-norm path on SM120.

## License

Apache-2.0. See [LICENSE](LICENSE). The patches under `patches/` are derived from SGLang and FlashInfer, which are Apache-2.0 licensed.
