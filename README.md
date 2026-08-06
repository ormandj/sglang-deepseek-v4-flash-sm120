# sglang-deepseek-v4-flash-sm120

An SGLang container image for serving DeepSeek-V4-Flash-0731 on RTX PRO 6000 Blackwell (SM120) GPUs, built from pinned upstream sources plus a small set of patches that are not yet in an upstream release.

Canonical image:

```
ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:dsv4-0731
```

One `linux/amd64` image is published under that single tag. There is deliberately no `latest` tag.

See [RUN.md](RUN.md) for the run guide and [examples/serve-dsv4-0731.sh](examples/serve-dsv4-0731.sh) for the validated serving configuration.

## Source composition

- Base: the official CUDA 13 SGLang nightly `lmsysorg/sglang:nightly-dev-cu13-20260803-12eadf86`, pinned by digest.
- SGLang: main `d2c405f1`, plus a single source patch carrying the pinned heads of eight upstream pull requests, five local fixes, and a local tool-schema encoding fix. The final tree in the image is `0491c1f4`.
- FlashInfer: main `67f76379` (version `0.6.17.dev20260804`), rebuilt from source with a single patch carrying upstream PRs #4308 and #4309, giving tree `d8567dec`. PR #3903 is already in main. The matching cubin wheel is installed by URL and pinned by SHA256. The prebuilt JIT-cache wheel is removed so the patched SM120 modules compile once into the persistent runtime cache.

Every pin is recorded in [stack.lock.json](stack.lock.json). [scripts/verify-patches.sh](scripts/verify-patches.sh) checks that the lock, the `Containerfile`, and the patch bytes agree, then clones the pinned commits and confirms that applying the patches in build order reproduces the recorded tree hashes.

## Included changes

| Source | Head | Scope in this image |
|---|---|---|
| SGLang PR #29927 | `21a5bc8e` | Complete SM120 DSV4 stack: DeepGEMM paged-MQA indexer, batched sparse-MLA prefill, and FP4 MoE. On this hardware it raises both prefill throughput and speculative acceptance length |
| SGLang PR #33614 | `56eae704` | Keeps DSpark speculative sampling state identical across TP ranks |
| SGLang PR #33140 | `a580251a` | Official DSV4 reasoning-effort support |
| SGLang PR #32194 | `f8fac391` | Skip unused DSV4 draft metadata in speculative decoding |
| SGLang PR #30700 | `c2439af9` | FlashInfer all-reduce-only dispatch and auto-enable |
| SGLang PR #32330 | `f330c748` | FlashInfer TRT-LLM all-reduce on SM12X; rebased with an mnnvl preflight multicast-granularity fix (trtllm still skips the multicast query, which fails on SM120) |
| SGLang PR #32686 | `15c0902e` | DeepGEMM warmup: pick the largest warmup `M` that fits the memory budget |
| SGLang PR #32815 | `1dbf09f6` | Enable FP8 W_o_A GEMM when the installed DeepGEMM supports it |
| FlashInfer PR #4308 | `4ec7f230` | Makes the fused-MoE profiler's MXFP8 × MXFP4 quantization state match the runtime path, so autotuning stays enabled |
| FlashInfer PR #4309 | `bf136350` | Top-k-192 decode and prefill dispatch for the SM120 sparse MLA kernels |

Image-local fixes:

| Fix | Head | Scope in this image |
|---|---|---|
| SGLang SM120 DeepGEMM capability probe | `d4dc7502` | Enables the SM120 DeepGEMM capability when the required grouped FP4 symbol is installed |
| SGLang all-reduce prefill workspace | `73d125d0` | Sizes the FlashInfer all-reduce workspace for the largest configured prefill forward before CUDA graph capture |
| SGLang SM120 all-reduce execution gate | `861b99ca` | Admits SM120 through the all-reduce execution gate for the backend enabled by PR #32330 |
| SGLang all-reduce token cap | `f99f4a0b` | Bounds the all-reduce-only kAllReduce path by the same token cap as the fused path. Without it the only ceiling is the workspace allocation, so sizing that workspace for the prefill forward routes prefill-sized all-reduces onto a min-latency kernel instead of the NCCL ring. Worth +3.7% prefill at 64k and +2.0% at 128k here, while decode keeps the FlashInfer kernel |
| SGLang dspark SWA eviction | `bcc988b9` | Runs sliding-window KV eviction on the DFLASH/DSPARK speculative decode path. Without it, hybrid-SWA models retain SWA KV for every generated token and a single request is retracted at `swa-full-tokens-ratio x max_total_num_tokens` generated tokens (reproduced at exactly 77,824 on this hardware); upstream's own pool sizing assumes this eviction runs |
| SGLang DSV4/DSV32 tool-schema encoding | — | Keeps unset protocol-model defaults (most visibly `strict: false`) out of the rendered DSV4/DSV32 tool schemas, so served prompts match the checkpoint's reference encoder exactly; proposed upstream in SGLang PR #33568 |

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

Separately, [sgl-project/sglang#33805](https://github.com/sgl-project/sglang/pull/33805)
(sliding-window KV eviction on the dflash-family speculative decode path) is
open; it is already carried in this image as the dspark SWA eviction fix above.

## Validated configuration

- Hardware: 2× NVIDIA RTX PRO 6000 Blackwell (SM120), TP=2. SM121 is not validated.
- Model: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), served as `deepseek-v4-flash`.
- Runtime settings: [examples/serve-dsv4-0731.sh](examples/serve-dsv4-0731.sh).
- Speculative decoding runs at DSpark depth 7, the model card's value; the checkpoint default of 5 corrupts output on SM120 ([#33800](https://github.com/sgl-project/sglang/issues/33800)). The fused DeepGEMM MHC pre-norm is re-enabled, worth +19.6% prefill at 128k. Both are explained in [RUN.md](RUN.md).

## License

Apache-2.0. See [LICENSE](LICENSE). The patches under `patches/` are derived from SGLang and FlashInfer, which are Apache-2.0 licensed.
