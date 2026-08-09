# sglang-deepseek-v4-flash-sm120

An SGLang container image for serving DeepSeek-V4-Flash-0731 on RTX PRO 6000 Blackwell (SM120) GPUs, built from pinned upstream sources plus a small set of patches that are not yet in an upstream release.

Current release candidate:

```
ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.1.0-rc.2
```

After the candidate passes the project acceptance gates, its exact manifest is
promoted without rebuilding to `v0.1.0`. Both tags are immutable. The workflow
also emits an immutable `build-N` diagnostic tag; there is deliberately no
`latest` or model-name moving alias. [release.json](release.json) is the version
source of truth and uses cache namespace `v2`, which is versioned independently
from the image.

See [RUN.md](RUN.md) for the run guide and [examples/serve-dsv4-0731.sh](examples/serve-dsv4-0731.sh) for the validated serving configuration.

## Source composition

- Base: the official CUDA 13 / Torch 2.13 SGLang nightly `lmsysorg/sglang:nightly-dev-cu13-20260808-ce84df0f`, pinned by digest.
- SGLang: release-pinned main commit `dc9624de`, plus one source patch containing the listed carries and remaining local integration changes. The final tree is `cee50a9f`.
- FlashInfer: release-pinned main commit `29196cf4` (version `0.6.18.dev20260808`), rebuilt from source with PRs #4308 and #4393, giving tree `a089f6c4`. Merged PRs #4329 and #4380 are supplied by that main commit. The matching cubin wheel is installed by URL and pinned by SHA256. The prebuilt JIT-cache wheel is removed so the patched SM120 modules compile once into the persistent runtime cache.

Every pin is recorded in [stack.lock.json](stack.lock.json). [scripts/verify-patches.sh](scripts/verify-patches.sh) checks that the lock, the `Containerfile`, and the patch bytes agree, then clones the pinned commits and confirms that applying the patches in build order reproduces the recorded tree hashes.

## Included changes

| Source | Head | Scope in this image |
|---|---|---|
| SGLang PR #29927 | `21a5bc8e` | Complete SM120 DSV4 stack: DeepGEMM paged-MQA indexer, batched sparse-MLA prefill, and FP4 MoE |
| SGLang PR #32183 | `22ef4312` | Propagates the runtime DSpark verifier width into the DeepSeek-V4 compressed-state planner instead of the hardcoded `kMaxMTPDraftTokens = 4` rewrite window. Without it, verifier widths above four rewrite only part of the window and eventually pollute compressed attention state during long generation |
| SGLang PR #33614 | `56eae704` | Keeps DSpark speculative sampling state identical across TP ranks |
| SGLang PR #32194 | `f8fac391` | Skip unused DSV4 draft metadata in speculative decoding |
| SGLang PR #30700 | `aead319d` | FlashInfer all-reduce-only dispatch and auto-enable |
| SGLang PR #32330 | `f330c748` | FlashInfer TRT-LLM all-reduce on SM12X; rebased with an mnnvl preflight multicast-granularity fix (trtllm still skips the multicast query, which fails on SM120) |
| SGLang PR #32686 | `15c0902e` | DeepGEMM warmup: pick the largest warmup `M` that fits the memory budget |
| SGLang PR #32815 | `1dbf09f6` | Enable FP8 W_o_A GEMM when the installed DeepGEMM supports it |
| FlashInfer PR #4308 | `4ec7f230` | Makes the fused-MoE profiler's MXFP8 × MXFP4 quantization state match the runtime path, so autotuning stays enabled |
| FlashInfer PR #4393 | `6573c652` | `pcie_ipc` all-reduce for intra-node PCIe machines with no NVLink or multicast. Present in the image; off by default |
| SGLang PR #33518 | `c3249eb9` | Per-request speculative statistics on the OpenAI endpoints, behind `return_spec_tokens_details`; the current head's unique implementation commit is `1c3f7b70` |
| SGLang PR #33568 | `cab45a29` | Match DSV4/DSV32 tool prompts to DeepSeek's reference encoding |
| SGLang PR #33805 | `26a2a398` | Run sliding-window KV eviction on the DFLASH/DSPARK speculative path |

Every carried pull request, its owner, and when we can drop it:

| Source | Pinned head | Owner | Drops when |
|---|---|---|---|
| SGLang PR #29927 | `21a5bc8e` | AliceChenyy | merged |
| SGLang PR #33614 | `56eae704` | JackZeng0208 | merged |
| SGLang PR #32194 | `f8fac391` | mattteochen | merged |
| SGLang PR #30700 | `aead319d` | wenscarl | merged |
| SGLang PR #32330 | `f330c748` | ormandj | merged |
| SGLang PR #32686 | `15c0902e` | ormandj | merged |
| SGLang PR #32815 | `1dbf09f6` | ormandj | merged |
| SGLang PR #32183 | `22ef4312` | slchenchn | merged |
| FlashInfer PR #4308 | `4ec7f230` | — | merged |
| FlashInfer PR #4393 | `6573c652` | — | merged |
| SGLang PR #33518 | `c3249eb9` | Muqi1029 | merged |
| SGLang PR #33568 | `cab45a29` | ormandj | merged |
| SGLang PR #33805 | `26a2a398` | ormandj | merged |

The current-main refresh removed SGLang #33616 and FlashInfer #4380 because
they merged. It also removed the previously unlisted FlashInfer #4329 patch
content because that change is now in main. FlashInfer #4380 is the upstream
replacement for the former #4309 carry. SGLang #33140 was removed by an
earlier refresh after it merged as `059269594c`.

Image-local fixes:

| Fix | Head | Scope in this image |
|---|---|---|
| SGLang SM120 DeepGEMM capability probe | `d4dc7502` | Enables the SM120 DeepGEMM capability when the required grouped FP4 symbol is installed |
| SGLang all-reduce prefill workspace | `73d125d0` | Sizes the FlashInfer all-reduce workspace for the largest configured prefill forward before CUDA graph capture |
| SGLang SM120 all-reduce execution gate | `861b99ca` | Admits SM120 through the all-reduce execution gate for the backend enabled by PR #32330 |
| SGLang all-reduce token cap | `f99f4a0b` | Bounds the all-reduce-only kAllReduce path by the same token cap as the fused path. Without it the only ceiling is the workspace allocation, so sizing that workspace for the prefill forward can route prefill-sized all-reduces onto a min-latency kernel instead of the NCCL ring |
| SGLang pcie_ipc consumer | `790a72c2` | Makes FlashInfer PR #4393's backend selectable from SGLang; the FlashInfer PR provides kernels and policy but no SGLang integration |
| SGLang DSpark shared-expert gate | `16797c8c` | Keeps the DSpark draft on separate shared-expert modules after #33889 made fusion decisions runner-local. Without this gate the draft builds an extra fused expert slot and rejects all bundled shared-expert weights |

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

## Current v0.1.0-rc.2 validation

The source-identical private deployment of rc.2 was compared with the documented
vLLM DSpark r33 image on the same two RTX PRO 6000 Blackwell Max-Q GPUs, one
engine at a time. Both used TP=2, DSpark K=5, the same model snapshot, the same
unmodified `llm_decode_bench.py`, temperature 1.0, and five measured
repetitions. The public image corresponding to the tested release lock is
`ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.1.0-rc.2@sha256:71766e76fd1ffd2bcac4ba79cbeb2317b326144558fc2e7bb49e2bbea05e8cbb`.

### Sustained decode

Aggregate throughput is the median of five independent 20-second runs. TTFT is
the median of the five per-run p50 values. Every cell reached the requested
concurrency with zero queueing and zero request errors.

| C | SGLang rc.2 tok/s | vLLM r33 tok/s | SGLang vs vLLM | SGLang p50 TTFT | vLLM p50 TTFT |
|---:|---:|---:|---:|---:|---:|
| 1 | 178.4 | 185.5 | -3.8% | 0.190 s | 0.495 s |
| 2 | 275.9 | 271.3 | +1.7% | 0.375 s | 0.772 s |
| 4 | 405.8 | 392.1 | +3.5% | 0.366 s | 1.122 s |
| 8 | 566.2 | 573.7 | -1.3% | 0.382 s | 1.287 s |
| 16 | 846.1 | 814.2 | +3.9% | 0.399 s | 1.548 s |
| 32 | 1,234.9 | — | — | 0.447 s | — |

vLLM r33 was intentionally stopped at C=16: that deployment is configured for
16 sequences and its KV envelope does not support the C=32 cell. SGLang supports
and was measured at C=32.

### Exact cold prefill

Client throughput is prompt tokens divided by TTFT. Each reported value is the
median of five independently calibrated exact-token runs.

| target | reported prompt tokens | SGLang rc.2 tok/s | vLLM r33 tok/s | SGLang vs vLLM | SGLang TTFT | vLLM TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 8,194 | 7,395 | 7,659 | -3.4% | 1.108 s | 1.070 s |
| 65,536 | 65,538 | 8,367 | 8,755 | -4.4% | 7.833 s | 7.486 s |
| 131,008 | 131,010 | 7,797 | 8,062 | -3.3% | 16.802 s | 16.250 s |

The engines are within 4% across the shared decode cells. vLLM is 3.4-4.6%
faster on these exact cold-prefill cells, while SGLang has materially lower
first-token latency in the sustained-decode setup.

See [BENCHMARKS.md](BENCHMARKS.md) for the pinned benchmark commit and hash,
exact commands, warmup and admission rules, all per-run values, image
configuration, the client-side KV guard override used for vLLM, and sanitized
raw reports. These are the only cross-engine performance figures published by
this repository; results from older image revisions have been removed.

## Validated configuration

- Hardware: 2× NVIDIA RTX PRO 6000 Blackwell (SM120), TP=2. SM121 is not validated.
- Model: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), served as `deepseek-v4-flash`.
- Runtime settings: [examples/serve-dsv4-0731.sh](examples/serve-dsv4-0731.sh).
- Speculative decoding depth is set explicitly in the serving script; the checkpoint's `config.json` carries `dspark_block_size = 5`. We publish no depth recommendation — see [RUN.md](RUN.md). The validated runtime re-enables the fused DeepGEMM MHC pre-norm path on SM120.

## License

Apache-2.0. See [LICENSE](LICENSE). The patches under `patches/` are derived from SGLang and FlashInfer, which are Apache-2.0 licensed.
