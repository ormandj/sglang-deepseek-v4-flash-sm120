# Changelog

This file describes the current public release surface. Prior release notes and
results remain available in their immutable Git tags. Exact source revisions,
effective trees, package versions, patch hashes, and image identity are recorded
in [`stack.lock.json`](stack.lock.json) and the image's OCI labels.

## v0.6.0-rc.3 — 2026-08-20

### Runtime and capacity

- Refreshed the qualified SGLang, FlashInfer, and DeepGEMM source composition.
- Restored the explicit SM120 FP8 W_o_A path and bounded the DeepSeek-V4 paged
  indexer's large MQA-logits allocation against available device memory.
- Qualified `--mem-fraction-static 0.93`, a 786,432-token total context budget,
  an 801,536-token TP2 KV pool, and a 393,216-token server generation ceiling.
- Started compilation-cache schema `v24` and retained pre-readiness prefill and
  decode-path warmups plus persisted PCIe-IPC autotuning.

### Serving documentation

- Made the direct Docker recipe the primary quickstart and retained the
  validated wrapper as an optional path.
- Documented which environment variables and flags form the measured launch
  contract and how to recheck capacity after changing them.
- Added hierarchical KV-cache sizing, host-memory, disk-persistence, and
  workload-fit guidance. HiCache remains disabled by default.
- Reorganized current performance, prefill, TP2 capacity, and quality evidence
  into paired SGLang/vLLM columns.

### Validation

- Completed five decode repetitions at C1, C2, C4, C8, C16, and C32 and five
  cache-cold prefill requests at 8K, 32K, 64K, and 128K.
- Completed GSM8K: 1,263 of 1,319 correct with zero request errors.
- Completed a 780,000-token request, four concurrent 250,000-token requests,
  and a mixed long/short generation batch at the shipped TP2 envelope.
- Measured HiCache with three 300K sessions at 1.14x device-pool
  oversubscription: returning-turn prefill fell from 18–20 seconds to about
  0.5 seconds, while device-pool capacity remained unchanged.
