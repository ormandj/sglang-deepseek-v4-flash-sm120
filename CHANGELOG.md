# Changelog

This changelog records user-visible composition, runtime, and benchmark-tooling
changes. Exact source revisions, effective trees, patch hashes, and
image digests are recorded in each release bundle and OCI labels.

## Unreleased

### Benchmark tooling

- Standardized future publication panels at five repetitions for every
  supported decode concurrency and cold-prefill length.
- Preserved request settings and provenance in optional long-write validation
  summaries without retaining generated response bodies.

## v0.2.1-rc.0 - 2026-08-12

### Changed

- Added the SM120/SM121 FP8 W_o_A target-model path from SGLang pull request
  #34018 at head `c3ffe8cfd3cf6cf9c30fc470cf7b76754954f3f0`.
- Added `SGLANG_OPT_FP8_WO_A_GEMM=1` to the serving recipe. The HC-prenorm and
  fused-MHC runtime settings remain enabled.
- Moved the compiled-kernel cache namespace from `v10` to `v11`.
- Increased the measured TP2 maximum KV-token pool from 694,528 to 772,096
  under the otherwise matched serving configuration.
- Expanded publication measurement to ten fixed-seed C1 prompt paths, five
  repetitions at every other supported decode concurrency, and five cold
  prefill requests per length.
- Renamed the synthetic output-rate metric so it is not presented as expected
  production throughput.

## v0.2.0-rc.0 - 2026-08-12

### Changed

- Reimaged from audited SGLang and FlashInfer main revisions rather than
  continuing the `0.1.x` patch stack.
- Removed the SM12X TRT/MNNVL all-reduce-fusion carry and returned PCIe-only
  tensor-parallel configurations to upstream NCCL.
- Removed the experimental PCIe-IPC, TBO, one-token-output,
  prefill-workspace, and local mHC-default patches.
- Removed carries already merged, replaced, or supplied by upstream main,
  including SGLang #30700 and FlashInfer #4308.
- Retained only the audited SGLang #29927, #33614, #32686, #33568, and #33805
  carries plus FlashInfer #3930 and its exact-runtime-filename follow-up.
- Started the fresh `v10` compiled-kernel cache namespace.

### Runtime recipe

- Uses TP2, FP8 KV cache, DSpARK block size 5, upstream NCCL, and disabled
  custom all-reduce for the published SM120 configuration.
- Enables `SGLANG_OPT_DEEPGEMM_HC_PRENORM=1` and
  `SGLANG_OPT_FUSE_MHC_POST_PRE=1`.
- Keeps long-output quality testing optional; GSM8K is the routine release
  quality gate.
