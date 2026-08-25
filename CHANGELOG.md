# Changelog

This file describes the current public release. Prior release notes and results
remain available in repository history. Exact source revisions, effective
trees, package versions, patch hashes, and image identity are recorded in
[`stack.lock.json`](stack.lock.json) and the image's OCI labels.

## 0.8.1-rc10 — 2026-08-25

The immutable image tag is `v0.8.1-rc.10`.

### Runtime

- Retain the exact qualified `0.8.1-rc4` SGLang, FlashInfer, DeepGEMM, refill,
  and runtime composition, and add SGLang PR #36003 at
  `64f51d9a95efb84dba1291b62250360db5eb300f`.
- Preserve reserved physical MLA KV slot 0 in the SGLang-owned BF16, FP8,
  scale, and CUDA TMA/JIT writers when graph-padding rows are present. The
  focused probe establishes this invariant on the CUDA FP8 path; it does not
  reproduce the reported user-visible output symptom or establish a causal
  link to it.
- Start compilation-cache schema `v38` because the correction changes compiled
  MLA KV-writer source and its JIT ABI.
- Retain the qualified TP2 runtime envelope: FP8 KV, DSpARK block size 5, a
  786,432-token context limit, an 801,536-token KV pool, and 48 scheduler
  request slots. HiCache remained disabled for all published measurements.

### Benchmark harness

- Name fixed-window generated-token slope as synthetic output rate so it is
  not presented as expected interactive or application throughput.
- Add a bounded five-repetition C8 benchmark mode.

### Validation

- Completed five decode repetitions at C1, C2, C4, C8, C16, and C32 and five
  cache-cold prefill requests at 8K, 32K, 64K, and 128K with zero request
  errors. Relative to `0.8.1-rc4`, paired forward-rate changes range from
  -1.08% to +0.73%; cold-prefill changes range from -0.58% to -0.01%.
- Completed the three-repetition C8 turnover screen with zero request errors
  and an exact match between expected and observed server POST counts.
- Completed full GSM8K with 1,266 of 1,319 answers correct (95.98%) and zero
  request errors.
- Completed a cold 780,000-token request in 237.8 seconds and four concurrent
  cold 250,000-token requests in 138.1 seconds, with all prompt tokens
  recomputed and no failures.
- Completed the public Weka coding-trace AgentX gate for 900 seconds each at C1
  and C8. Both cells were submission-valid with zero request errors.
- Passed eight focused upstream GPU tests, the reserved-slot probe, the
  concurrent long-context output-integrity screen, and the grounded-identifier
  integrity workload.
