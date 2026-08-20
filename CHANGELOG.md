# Changelog

This changelog records user-visible composition, runtime, and benchmark-tooling
changes. Exact source revisions, effective trees, patch hashes, and image
digests are recorded in each release bundle and its OCI labels.

## v0.6.0-rc.3 - 2026-08-20

### Changed

- Refreshed SGLang, FlashInfer, and DeepGEMM to the 2026-08-19 source
  composition recorded in `stack.lock.json`, and re-derived every carried pull
  request against those new bases rather than re-applying the previous patch
  set.
- Raised the qualified capacity envelope to `--mem-fraction-static 0.93` with
  `--context-length 786432`.
- Started the `v24` runtime compilation-cache namespace.
- Removed `TVM_CACHE_DIR` from the documented recipe; it was never read by this
  image.
- Added `SGLANG_MAX_NEW_TOKENS_LIMIT=393216` to the documented recipe, matching
  the model card's recommended maximum output length for the `high` and `max`
  reasoning effort levels.

### Fixed

- Bounded the DeepSeek-V4 paged indexer's MQA logits allocation against free
  device memory, composed with #29927's SM120 metadata row cap. The unbounded
  path allocated a dense FP32 `[query_rows x max_c4_seq_len]` tensor and could
  request more memory than remained, killing both tensor-parallel ranks on a
  long-context prefill.
- Restored SGLang #34018's FP8 W_o_A GEMM on SM120. The SM120 post-process
  block in `server_args.py` unconditionally disabled
  `SGLANG_OPT_FP8_WO_A_GEMM`, so an explicit setting could not survive it and
  the carried pull request had no effect. The setting is now honoured when it
  was set explicitly.

### Documented

- Stated that `--context-length` is a total budget of prompt plus generated
  tokens, not an input limit, and worked the arithmetic for a long-context
  reasoning session against the KV pool.
- Replaced the KV capacity table with measurements at 0.90, 0.93, and 0.95,
  including near-limit and concurrent-request outcomes and the observed
  free-memory floors. The previous table described 0.95 as passing; at the
  current context setting it exhausts per-request workspace and kills a
  scheduler rank.

### Validation

- Completed five decode repetitions at C1, C2, C4, C8, C16, and C32, and five
  cache-cold prefill requests at 8K, 32K, 64K, and 128K, all at the shipped
  0.93 capacity envelope.
- Completed the full 1,319-question GSM8K set with 1,263 correct and zero
  request errors.
- Confirmed a 780,000-token request and four concurrent 250,000-token requests
  both complete at the shipped envelope.
- Confirmed a 400,000-token prompt generating 32,768 tokens completes at
  375 tok/s, and that a batch mixing it with three short requests completes
  with no failures.

## v0.5.0-rc.1 - 2026-08-18

### Changed

- Refreshed SGLang, FlashInfer, and DeepGEMM to the pinned 2026-08-18 source
  composition recorded in `stack.lock.json`.
- Added the inference-safe page-split workspace allocation from SGLang #35116
  and fused HC-prenorm combine from SGLang #35118.
- Added pre-readiness prefill-shape and decode-path warmups and enabled the
  persisted PCIe-IPC autotune cache in the documented launch recipe.
- Included the current-main environment-registry compatibility correction for
  the DeepSeek-V4 integration.
- Started the `v20` runtime compilation-cache namespace.
- Reduced the decode measurement-window duration check to a one-second sanity
  floor while retaining the exact occupancy, empty-queue, no-prefill, monotonic
  counter, fixed-context, and 20-scrape requirements.

### Validation

- Completed five decode repetitions at C1, C2, C4, C8, C16, and C32.
- Completed five cache-cold prefill requests at 8K, 32K, 64K, and 128K.
- Completed the full 1,319-question GSM8K set with 1,261 correct and zero
  request errors.

## v0.3.3-rc.0 - 2026-08-14

### Changed

- Retained the v0.3.2 SGLang, FlashInfer, PCIe-IPC, model, and launch
  configuration.
- Corrected DeepGEMM's compiled-dimension mapping for SM120 operand-swapped
  GEMMs so residual token counts do not create distinct W_o_A kernels; the
  production change is tracked in
  [`sgl-project/DeepGEMM#76`](https://github.com/sgl-project/DeepGEMM/pull/76).
- Added focused BMM/einsum and ordinary-GEMM regression coverage for multiple
  runtime token counts within one compiled tile bucket.
- Submitted the retained SM120 single-token split-K heuristic as
  [`sgl-project/DeepGEMM#77`](https://github.com/sgl-project/DeepGEMM/pull/77),
  including its isolated projection profile and complete serving qualification.
- Started the `v15` runtime compilation-cache namespace.
- Made the documented Docker wrapper configurable through `TP_SIZE` and
  `CONTEXT_LENGTH` while retaining the qualified TP2 defaults.

### Validation

- Completed five decode repetitions at C1, C2, C4, C8, C16, and C32.
- Completed five cache-cold prefill requests at 8K, 32K, 64K, and 128K.
- Completed the full 1,319-question GSM8K set with 1,261 correct and zero
  request errors.
- Completed the requested eight-request long-output structural diagnostic with
  eight passes.
- Replayed the original residual-shape sequence; request 29 completed without
  the former first-use JIT stall.

## v0.3.2-rc.0 - 2026-08-14

Status: internal evaluation candidate; not published to GHCR.

- Rebased the reviewed DeepSeek-V4/SM120 stack onto SGLang `9d34c2809f` and
  FlashInfer `ed6c709849`.
- Removed the SGLang #34759 carry after its DSpark EP1 allocation fix merged to
  main.
- Retained PCIe-IPC all-reduce with a 786,432-element limit and excluded
  TRT/MNNVL fusion.
- Started the `v14` runtime compilation-cache namespace.

## v0.3.1-rc.1 - 2026-08-13

Status: internal evaluation candidate; not published to GHCR.

- Preserved explicit machine-local PCIe-IPC launch selections when an
  enclosing FlashInfer autotune context had no matching distributed tune
  group; the focused follow-up is
  [`qsang-nv/flashinfer#1`](https://github.com/qsang-nv/flashinfer/pull/1)
  against
  [`flashinfer-ai/flashinfer#4393`](https://github.com/flashinfer-ai/flashinfer/pull/4393).

## v0.3.1-rc.0 - 2026-08-13

Status: internal evaluation candidate; not published to GHCR.

- Enabled the optional PCIe-IPC all-reduce consumer from
  [`sgl-project/sglang#34528`](https://github.com/sgl-project/sglang/pull/34528)
  and
  [`flashinfer-ai/flashinfer#4393`](https://github.com/flashinfer-ai/flashinfer/pull/4393)
  for the qualified TP2 decode shape.
- Excluded TRT/MNNVL fusion and PCIe-IPC all-gather.

## v0.2.1-rc.0 - 2026-08-12

- Added the SM120/SM121 FP8 W_o_A target-model path from
  [`sgl-project/sglang#34018`](https://github.com/sgl-project/sglang/pull/34018).
- Added `SGLANG_OPT_FP8_WO_A_GEMM=1` to the serving recipe while retaining HC
  prenorm and fused MHC post+pre.
- Moved the runtime compilation cache to schema `v11`.
- Increased the measured TP2 maximum KV-token pool from 694,528 to 772,096
  under the otherwise matched serving configuration.
- Renamed the synthetic output-rate metric so it is not presented as expected
  production throughput.

## v0.2.0-rc.0 - 2026-08-12

- Reimaged from audited SGLang and FlashInfer main revisions rather than
  continuing the `0.1.x` patch stack.
- Removed the SM12X TRT/MNNVL all-reduce-fusion carry and returned PCIe-only
  tensor-parallel configurations to upstream NCCL.
- Removed the experimental PCIe-IPC, TBO, one-token-output,
  prefill-workspace, and local MHC-default patches.
- Removed carries already merged, replaced, or supplied by upstream main,
  including
  [`sgl-project/sglang#30700`](https://github.com/sgl-project/sglang/pull/30700)
  and
  [`flashinfer-ai/flashinfer#4308`](https://github.com/flashinfer-ai/flashinfer/pull/4308).
- Retained the audited
  [`sgl-project/sglang#29927`](https://github.com/sgl-project/sglang/pull/29927),
  [`sgl-project/sglang#33614`](https://github.com/sgl-project/sglang/pull/33614),
  [`sgl-project/sglang#32686`](https://github.com/sgl-project/sglang/pull/32686),
  [`sgl-project/sglang#33568`](https://github.com/sgl-project/sglang/pull/33568),
  and [`sgl-project/sglang#33805`](https://github.com/sgl-project/sglang/pull/33805)
  carries plus [`flashinfer-ai/flashinfer#3930`](https://github.com/flashinfer-ai/flashinfer/pull/3930)
  and its exact-runtime-filename follow-up,
  [`aryanputta/flashinfer#1`](https://github.com/aryanputta/flashinfer/pull/1).
- Started the `v10` runtime compilation-cache namespace.
