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
- SGLang: main `d2c405f1`, plus a single source patch carrying the pinned heads of eight upstream pull requests, four local SM120 fixes, and a local tool-schema encoding fix. The final tree in the image is `b845e816`.
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
| SGLang DSV4/DSV32 tool-schema encoding | — | Keeps unset protocol-model defaults (most visibly `strict: false`) out of the rendered DSV4/DSV32 tool schemas, so served prompts match the checkpoint's reference encoder exactly; proposed upstream in SGLang PR #33568 |

The intent is upstream-first: everything here is either an open upstream pull request carried at a pinned head, or a narrow local fix intended to be replaced by an upstream change. As those land, the corresponding patch content is dropped rather than maintained.

## Validated configuration

- Hardware: 2× NVIDIA RTX PRO 6000 Blackwell (SM120), TP=2. SM121 is not validated.
- Model: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), served as `deepseek-v4-flash`.
- Runtime settings: [examples/serve-dsv4-0731.sh](examples/serve-dsv4-0731.sh).

## License

Apache-2.0. See [LICENSE](LICENSE). The patches under `patches/` are derived from SGLang and FlashInfer, which are Apache-2.0 licensed.
