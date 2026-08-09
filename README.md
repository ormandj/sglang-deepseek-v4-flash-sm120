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
- SGLang: current main `dc9624de`, plus one source patch containing only the still-open carries and remaining local integration changes. The final tree is `cee50a9f`.
- FlashInfer: current main `29196cf4` (version `0.6.18.dev20260808`), rebuilt from source with only open PRs #4308 and #4393, giving tree `a089f6c4`. Merged PRs #4329 and #4380 are supplied by main. The matching cubin wheel is installed by URL and pinned by SHA256. The prebuilt JIT-cache wheel is removed so the patched SM120 modules compile once into the persistent runtime cache.

Every pin is recorded in [stack.lock.json](stack.lock.json). [scripts/verify-patches.sh](scripts/verify-patches.sh) checks that the lock, the `Containerfile`, and the patch bytes agree, then clones the pinned commits and confirms that applying the patches in build order reproduces the recorded tree hashes.

## Included changes

| Source | Head | Scope in this image |
|---|---|---|
| SGLang PR #29927 | `21a5bc8e` | Complete SM120 DSV4 stack: DeepGEMM paged-MQA indexer, batched sparse-MLA prefill, and FP4 MoE. On this hardware it raises both prefill throughput and speculative acceptance length |
| SGLang PR #32183 | `22ef4312` | Propagates the runtime DSpark verifier width into the DeepSeek-V4 compressed-state planner instead of the hardcoded `kMaxMTPDraftTokens = 4` rewrite window. At block size 7 the verify width is 8, so the planner rewrote only part of the window and compressed attention state was polluted every 4 tokens. Long single-file code generation went from 0/8 to 14/14 complete (closed code fence plus `node --check`), greedy 0/1 to 3/3, and decode rose from 237.5 to 319.8 tok/s |
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
| SGLang all-reduce token cap | `f99f4a0b` | Bounds the all-reduce-only kAllReduce path by the same token cap as the fused path. Without it the only ceiling is the workspace allocation, so sizing that workspace for the prefill forward routes prefill-sized all-reduces onto a min-latency kernel instead of the NCCL ring. Worth +3.7% prefill at 64k and +2.0% at 128k here, while decode keeps the FlashInfer kernel |
| SGLang pcie_ipc consumer | `790a72c2` | Makes FlashInfer PR #4393's backend selectable from SGLang; the FlashInfer PR provides kernels and policy but no SGLang integration |
| SGLang DSpark shared-expert gate | `16797c8c` | Keeps the DSpark draft on separate shared-expert modules after #33889 made fusion decisions runner-local. Without this gate the draft built an extra fused expert slot and rejected all bundled shared-expert weights. Corrected counters measured 13.59% draft-token acceptance on rc.1 and 39.69% after the gate on rc.2 |

The intent is upstream-first: everything here is either an open upstream pull request carried at a pinned head, or a narrow local fix intended to be replaced by an upstream change. As those land, the corresponding patch content is dropped rather than maintained.

### Reported upstream, patches not yet filed

The runtime settings in [RUN.md](RUN.md) work around three SGLang defaults that
look wrong on SM120. Upstream changes for these are **TBD**; until they land the
workarounds live in [examples/serve-dsv4-0731.sh](examples/serve-dsv4-0731.sh).

| Finding | Effect here | Upstream status |
|---|---|---|
| The SM120 branch disables both fused MHC pre-norm paths, forcing an eager float32 fallback measured at ~15% of a 128k prefill. The guard predates first-class SM120 support for the DeepGEMM `tf32_hc_prenorm_gemm` kernel | +19.6% prefill at 128k once re-enabled | TBD — the TileLang half of the same guard is justified and should stay (it fails CUDA graph capture on SM120) |
| `SGLANG_OPT_USE_TOPK_V2` is force-disabled on SM120 without an `is_set()` guard, unlike its neighbours, so an operator-set value is overridden with no escape hatch | Runs the slower top-k transform; no override possible | TBD — gating alone changes no default |
| DSpark speculative decoding never accumulates presence/frequency/repetition penalties: `eagle_prepare_for_decode` calls `cumulate_penalty_output_tokens`, the dflash-family branch does not | Those sampling penalties are silently inert under DSpark | TBD |

## v0.1.0-rc.2 validation

The source-identical private rc.2 image was tested on the production 2x RTX PRO
6000 Blackwell Max-Q TP=2 deployment after a clean-cache startup. The public
image is
`ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.1.0-rc.2@sha256:71766e76fd1ffd2bcac4ba79cbeb2317b326144558fc2e7bb49e2bbea05e8cbb`.

### rc.1 regression and rc.2 fix

The rc.1 before-fix diagnostic is compared with post-fix rc.2 medians from five
accepted repetitions per concurrency. rc.1 had rejected every bundled draft
shared-expert tensor; rc.2 loaded the separate draft experts normally.

| C | rc.1 before fix | rc.2 after fix n=5 median | delta |
|---:|---:|---:|---:|
| 1 | 104.7 | 187.9 | +79.5% |
| 2 | 164.1 | 403.0 | +145.6% |
| 4 | 244.7 | 438.1 | +79.0% |
| 8 | 344.1 | 606.1 | +76.1% |
| 16 | 535.7 | 913.1 | +70.5% |
| 32 | 777.1 | 1,316.5 | +69.4% |

Corrected SGLang counters moved from 13.59% to 39.69% draft-token acceptance
and from 1.680 to 2.984 accepted tokens per verification including the target
bonus. This establishes that the rc.1 regression is fixed.

### Repeated performance and quality campaign

Decode values below are medians of five accepted repetitions per concurrency.
The five full runs also include warmups, five coding requests, exact cold
8K/64K/128K prefill, and all 1,319 GSM8K questions. Every accepted measurement
passed admission, timing, speculative-counter, prefill, coding, and quality
validation.

The comparison values are medians recomputed directly from all five saved
`r9-armD-w5-r1..r5` artifacts used for the published r11 table.

| C | r11 median tok/s | rc.2 n=5 median tok/s | delta |
|---:|---:|---:|---:|
| 1 | 194.1 | 187.9 | -3.16% |
| 2 | 295.6 | 403.0 | +36.33% |
| 4 | 439.9 | 438.1 | -0.40% |
| 8 | 597.7 | 606.1 | +1.39% |
| 16 | 909.3 | 913.1 | +0.42% |
| 32 | 1,315.0 | 1,316.5 | +0.11% |

| Exact cold prefill | r11 median tok/s | rc.2 n=5 median tok/s | delta |
|---:|---:|---:|---:|
| 8K | 7,342 | 7,378 | +0.49% |
| 64K | 8,320 | 8,530 | +2.52% |
| 128K | 7,732 | 7,934 | +2.61% |

The five-run coding median was 267.9 tok/s versus 264.8 (+1.19%). Corrected
speculative counters reported a 42.17% median draft-token acceptance rate and
3.109 accepted tokens per verification including the target bonus.

GSM8K averaged 93.72% accuracy (6,181/6,595 correct) versus 93.75%
(6,183/6,595) on r11. All 6,595 rc.2 responses were extractable and had valid
finish reasons. Median aggregate throughput was 369.9 tok/s versus 372.4, and
mean wall time was 318.3 seconds versus 314.8.

The matched long-write check used eight sequential requests with
`reasoning_effort=max`, temperature 1.0, top-p 0.95, and a 131,072-token cap.
rc.2 completed 8/8 with normal stop reasons, closed HTML fences and documents,
no placeholders, and JavaScript accepted by `node --check`. The true median
completion length was 57,956.5 tokens (17,110-67,884), versus r11's 57,029.5
(47,298-66,123).

## Current rc.2 comparison with vLLM

On the validated rc.2 configuration (2x RTX PRO 6000 Blackwell **Max-Q**, 300 W
hard limit, **PCIe Gen 4 x16**, TP=2), benchmarked with
[llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
against `voipmonitor/vllm` v20 r27 in its documented DSpark configuration,
using the same client and flags:

### Decode: aggregate tokens/second

| concurrency | SGLang rc.2 n=5 median | vLLM v20 r27 | SGLang delta |
|---|---:|---:|---:|
| 1 | 187.9 | **199.5** | -5.81% |
| 2 | **403.0** | 291.3 | +38.35% |
| 4 | **438.1** | 410.7 | +6.67% |
| 8 | **606.1** | 589.5 | +2.82% |
| 16 | **913.1** | 821.4 | +11.16% |
| 32 | **1,316.5** | — | — |

SGLang figures are medians of five accepted repetitions per concurrency.

vLLM caps at concurrency 16 (`max_num_seqs 16`); 32 is SGLang only.

### Everything else

| | SGLang rc.2 | vLLM v20 r27 |
|---|---:|---:|
| prefill @ 8k | 7,378 tok/s | **7,540 tok/s** |
| prefill @ 64k | 8,530 tok/s | **8,945 tok/s** |
| prefill @ 128k | 7,934 tok/s | **8,238 tok/s** |
| GSM8K accuracy | 0.9372 | 0.9393 |
| **GSM8K wall clock** | **318.3 s** | 749 s |
| GSM8K aggregate | **369.9 tok/s** | 159.3 tok/s |
| long-write complete (131k budget) | **8/8** | 4/4 |
| long-write median tokens | **57,956.5** | 40,764 |
| **usable context** | **774,656 tokens** | 133,120 tokens |
| KV cache | **778,496 tokens** | 143,439 tokens |
| concurrent 128k requests in KV | **5.9** | 1.1 |

SGLang trails vLLM by 5.8% at C1 and by 2.1-4.6% on prefill. SGLang leads
sustained decode at C2-C16 by 2.8-38.3%, finishes the 1,319-question GSM8K suite
in **less than half the wall clock** at comparable accuracy, and serves **5.4x
the KV capacity** -- it
accepts the 384K output budget the model card recommends for `high`/`max`
reasoning, which vLLM rejects outright at `max_model_len 133120`.

Absolute values are specific to Max-Q cards on PCIe Gen 4; a 600 W card on Gen 5
has twice the power budget and twice the inter-GPU bandwidth. Full tables,
methodology, and the KV-capacity analysis are in [RUN.md](RUN.md).

## Validated configuration

- Hardware: 2× NVIDIA RTX PRO 6000 Blackwell (SM120), TP=2. SM121 is not validated.
- Model: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), served as `deepseek-v4-flash`.
- Runtime settings: [examples/serve-dsv4-0731.sh](examples/serve-dsv4-0731.sh).
- Speculative decoding depth is set explicitly in the serving script; the checkpoint's `config.json` carries `dspark_block_size = 5`. We publish no depth recommendation — see [RUN.md](RUN.md). The fused DeepGEMM MHC pre-norm is re-enabled, worth +19.6% prefill at 128k (measured n=5: 6,717 → 8,034 tok/s).

## License

Apache-2.0. See [LICENSE](LICENSE). The patches under `patches/` are derived from SGLang and FlashInfer, which are Apache-2.0 licensed.
