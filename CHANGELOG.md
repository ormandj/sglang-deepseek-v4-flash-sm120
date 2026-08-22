# Changelog

This file describes the current public release surface. Prior release notes and
results remain available in their immutable Git tags. Exact source revisions,
effective trees, package versions, patch hashes, and image identity are recorded
in [`stack.lock.json`](stack.lock.json) and the image's OCI labels.

## 0.8.1-rc1 — 2026-08-21

The immutable image tag is `v0.8.1-rc.1`.

### Runtime and capacity

- Refreshed SGLang and FlashInfer from audited current upstream heads while
  retaining the reviewed DeepGEMM composition.
- Added occupancy-aware refill batching so completed requests do not fragment
  replacement prefills during dynamic concurrency, while genuine demand growth
  is still admitted immediately.
- Kept the policy model-independent and explicitly opt-in outside the existing
  DeepSeek-V4 default path.
- Qualified `--mem-fraction-static 0.93`, a 786,432-token total context budget,
  an 801,536-token TP2 KV pool, and a 393,216-token server generation ceiling.
- Started compilation-cache schema `v34` and retained pre-readiness prefill and
  decode-path warmups plus persisted PCIe-IPC autotuning.

### Serving documentation

- Made the direct Docker recipe the primary quickstart and retained the
  validated wrapper as an optional path.
- Documented which environment variables and flags form the measured launch
  contract and how to recheck capacity after changing them.
- Kept hierarchical KV-cache host and bounded file-storage configuration
  available through the optional wrapper. HiCache remains disabled by default.
- Reorganized current performance, prefill, TP2 capacity, and quality evidence
  into paired SGLang/vLLM columns.
- Published the closed-loop turnover harness, analyzer, comparison tool, tests,
  and risk-based release policy.

### Validation

- Completed five decode repetitions at C1, C2, C4, C8, C16, and C32 and five
  cache-cold prefill requests at 8K, 32K, 64K, and 128K.
- Completed five turnover repetitions at C1, C2, C4, and C8 without request
  failures or server restarts.
- Completed GSM8K: 1,263 of 1,319 correct with zero request errors.
- Completed a 780,000-token request, four concurrent 250,000-token requests,
  and zero process restarts at the shipped TP2 envelope.
