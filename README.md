# sglang-deepseek-v4-flash-sm120

An SGLang container image for serving DeepSeek-V4-Flash-0731 on RTX PRO 6000 Blackwell (SM120) GPUs, built from pinned upstream sources plus a small set of patches that are not yet in an upstream release.

Canonical image:

```
ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:dsv4-0731
```

One `linux/amd64` image is published under that single tag. There is deliberately no `latest` tag.

See [RUN.md](RUN.md) for the run guide and [examples/serve-dsv4-0731.sh](examples/serve-dsv4-0731.sh) for the validated serving configuration.

## Source composition

- Base: the official CUDA 13 SGLang nightly `lmsysorg/sglang:nightly-dev-cu13-20260731-68d44294`, pinned by digest.
- SGLang: main `9dcaf6bf`, plus the current heads of five upstream pull requests and two local follow-ups. The tree after the upstream patches is `5aa76976`; the final tree in the image is `98411f8b`.
- FlashInfer: main `668a1ba1` (version `0.6.17.dev20260731`), rebuilt from source with upstream PR #3930 and two local patches, giving tree `164ff8cf`. The matching cubin wheel is installed by URL and pinned by SHA256. The prebuilt JIT-cache wheel is removed so the patched SM120 modules compile once into the persistent runtime cache.

Every pin is recorded in [stack.lock.json](stack.lock.json). [scripts/verify-patches.sh](scripts/verify-patches.sh) checks that the lock, the `Containerfile`, and the patch bytes agree, then clones the pinned commits and confirms that applying the patches in build order reproduces the recorded tree hashes.

## Included changes

| Source | Head | Scope in this image |
|---|---|---|
| SGLang PR #29927 | `bfc395a8` | [SM120] DeepSeek-V4: DeepGEMM paged-MQA indexer, FP4 MoE, and page-split |
| SGLang PR #32815 | `1dbf09f6` | Layer communicator fusion gate |
| SGLang PR #30700 | `2960b751` | FlashInfer all-reduce-only dispatch and auto-enable |
| SGLang PR #32330 | `34c9d596` | FlashInfer TRT-LLM all-reduce on SM120, without the multicast preflight |
| SGLang PR #32686 | `15c0902e` | DeepGEMM warmup: pick the largest warmup `M` that fits the memory budget |
| FlashInfer PR #3930 | `e855cc25` | Exact CUDA runtime library match when resolving the loaded library |

Local follow-ups, not upstream pull requests:

| Follow-up | Head | Scope in this image |
|---|---|---|
| SGLang all-reduce prefill workspace | — | Sizes the FlashInfer all-reduce workspace for the largest configured prefill forward before CUDA graph capture |
| SGLang DSV4-0731 reasoning effort | `5912c5d3` | Selects the 0731 checkpoint's low/high/max reasoning-effort prompts only when the model config carries the DSpark fields; earlier DeepSeek-V4 checkpoints keep their existing mapping |
| FlashInfer SM120 DSV4 top-k-192 | `4d42fdbb` | Top-k-192 decode and prefill dispatch for the SM120 sparse MLA kernels |
| FlashInfer MXFP8 × MXFP4 profiler | `c8fb671d` | Makes the fused-MoE profiler's quantization state match the runtime path, so autotuning stays enabled |

The intent is upstream-first: everything here is either an open upstream pull request carried at a pinned head, or a narrow local fix intended to be replaced by an upstream change. As those land, the corresponding patch is dropped rather than maintained.

## Validated configuration

- Hardware: 2× NVIDIA RTX PRO 6000 Blackwell (SM120), TP=2. SM121 is not validated.
- Model: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), served as `deepseek-v4-flash`.
- Runtime settings: [examples/serve-dsv4-0731.sh](examples/serve-dsv4-0731.sh).

## License

Apache-2.0. See [LICENSE](LICENSE). The patches under `patches/` are derived from SGLang and FlashInfer, which are Apache-2.0 licensed.
