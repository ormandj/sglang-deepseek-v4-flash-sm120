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
| SGLang DSpark shared-expert gate | `16797c8c` | Keeps the DSpark draft on separate shared-expert modules after #33889 made fusion decisions runner-local. Without this gate the draft built an extra fused expert slot, rejected all bundled shared-expert weights, and cut speculative acceptance from the r11 ~0.32 median to 0.136 |

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

## Previous r11 evidence

`v0.1.0-rc.1` is not promotable: its DSpark draft skipped the shared-expert
weights and regressed decode by 41-46%. The corrected `v0.1.0-rc.2` has no
performance or quality claim until it passes the same hardware gates. The
figures below describe its validated r11 predecessor and remain here as the
comparison baseline.

On the validated r11 configuration (2x RTX PRO 6000 Blackwell **Max-Q**, 300 W hard
limit, **PCIe Gen 4 x16**, TP=2), benchmarked with
[llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
against both r11 and `voipmonitor/vllm` v20 r27 in its documented DSpark
configuration, same client and same flags:

### Decode: aggregate tokens/second, per-user tokens/second, TTFT p50

| concurrency | SGLang agg | SGLang/user | SGLang TTFT | vLLM agg | vLLM/user | vLLM TTFT |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 194.1 | 191.2 | **0.179 s** | **199.5** | 199.5 | 0.493 s |
| 2 | **295.6** | 147.3 | **0.177 s** | 291.3 | 145.6 | 0.642 s |
| 4 | **439.9** | 108.3 | **0.185 s** | 410.7 | 102.7 | 0.835 s |
| 8 | **597.7** | 73.2 | **0.195 s** | 589.5 | 73.2 | 0.845 s |
| 16 | **909.3** | 55.3 | **0.194 s** | 821.4 | 51.3 | 0.942 s |
| 32 | **1,315.0** | 39.4 | **0.201 s** | — | — | — |

SGLang figures are medians of five runs; run-to-run spread was 1.7-3.3% at
every concurrency except C1, where it was 7.2%.

vLLM caps at concurrency 16 (`max_num_seqs 16`); 32 is SGLang only.

### Everything else

| | SGLang (r11 predecessor) | vLLM v20 r27 |
|---|---:|---:|
| prefill @ 8k | 7,342 tok/s | **7,540 tok/s** |
| prefill @ 64k | 8,320 tok/s | **8,945 tok/s** |
| prefill @ 128k | 7,732 tok/s | **8,238 tok/s** |
| GSM8K accuracy | 0.9375 | 0.9393 |
| **GSM8K wall clock** | **315 s** | 749 s |
| GSM8K aggregate | **372.4 tok/s** | 159.3 tok/s |
| long-write complete (131k budget) | **8/8** | 4/4 |
| long-write median tokens | **57,029** | 40,764 |
| long-write generation rate | 221.2 tok/s | **232.4 tok/s** |
| **usable context** | **774,656 tokens** | 133,120 tokens |
| KV cache | **778,496 tokens** | 143,439 tokens |
| concurrent 128k requests in KV | **5.9** | 1.1 |

vLLM leads prefill by 3-8% and single-stream decode by 2.7%. SGLang leads
sustained decode at every concurrency from 2 upward (+1.5% to +10.7%), answers
2.8x to 4.8x sooner, finishes the 1,319-question GSM8K suite in **less than half
the wall clock** at comparable accuracy, and serves **5.4x the KV capacity** -- it
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
