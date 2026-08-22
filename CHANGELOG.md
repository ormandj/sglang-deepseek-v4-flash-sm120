# Changelog

This file describes the current public release. Prior release notes and results remain available in repository history. Exact source revisions, effective trees, package versions, patch hashes, and image identity are recorded in [`stack.lock.json`](stack.lock.json) and the image's OCI labels.

## 0.8.1-rc4 — 2026-08-22

The immutable image tag is `v0.8.1-rc.4`.

### Runtime

- Rebased the qualified composition to SGLang main `cce0a1244b5a3ee60411fb403f65f856a1d012f0` and FlashInfer main `fb28d7242b3506a2348265962041acc1fb56cca4`, and started compilation-cache schema `v37`.
- Changed DFlash/DSpark replacement-prefill batching to use observed request demand instead of unused configured request capacity. Workload growth is admitted immediately, partial refills retain their target, contractions adapt, and incomplete refills have a configurable scheduler-pass bound.
- Restored the SM120 architecture-guard import required by the current SGLang main integration.
- Retained the qualified TP2 runtime envelope: a 786,432-token context limit, an 801,536-token KV pool, and 48 scheduler request slots.

### Benchmark harness

- Added the full closed-loop C1/C2/C4/C8 turnover gate and the shorter risk-based C8 release screen.
- Added exact server-side chat-request counting so unrelated traffic invalidates a turnover cell without rejecting legitimate finished-request cleanup overlap.
- Kept authenticated and keyless endpoints supported without writing credentials to artifacts.

### Validation

- Completed five decode repetitions at C1, C2, C4, C8, C16, and C32 and five cache-cold prefill requests at 8K, 32K, 64K, and 128K with zero request errors.
- Relative to `0.8.1-rc1`, decode forward-rate geometric changes range from -1.14% to +1.53%; cold-prefill throughput changes range from -0.30% to +0.34%.
- Completed five turnover repetitions at C1, C2, C4, and C8 with zero request errors. In the matched C8 control, automatic replacement batching increased mean output throughput from 591.29 to 610.91 tokens/s and requests per prefill pass from 1.3256 to 2.0603, while mean TTFT increased from 241.54 to 352.72 ms.
- Completed GSM8K with 1,260 of 1,319 answers correct (95.53%) and zero request errors, matching `0.7.0-rc1`.
- Completed a cold 780,000-token request and four concurrent cold 250,000-token requests with every submitted prompt token recomputed, zero request failures, and zero process restarts.
- Completed the public Weka coding-trace AgentX gate for 900 seconds each at C1 and C8. Both cells were submission-valid with zero request errors, and the serving process remained ready with zero restarts.
