# Release performance history

This file preserves the measured performance snapshot for each published
`0.2.x` release candidate. [BENCHMARKS.md](BENCHMARKS.md) remains the
current-result view; this file is historical and never substitutes results
across releases or benchmark methods.

“Synthetic fixed-window output tok/s” is the aggregate slope of the engine's
generated-output-token counter over the fixed 17,408–20,480 average-context
decode interval in the synthetic benchmark. It is not client end-to-end,
production, interactive, or application throughput. Values below are median
`[min, max]` across the stated same-process prompt-path repetitions. The rate
combines verifier-step rate with output tokens produced per step and is
therefore driven in part by DSpARK acceptance. Different fixed prompt/seed
paths produce different accepted-draft-length distributions, so this rate is
more variable than verifier steps/s.

## v0.2.0-rc.0

| Field | Value |
|---|---|
| Measured image digest | `sha256:5d13c9315089156a8c57deb272fcd5171b0e45c090343b7a8b9c497aa4d90703` |
| SGLang effective tree | `8c336f4426844b2028938f7c542c0a403d37b804` |
| FlashInfer effective tree | `09b10c6dc66ca0c96c62a13dfa5ea3b63f1018e4` |
| AIPerf revision | `6ed4823d127b3a6d12c63fb8c2ca5eff13f9ba23` |
| Harness revision | `2075063b36ee03d8ce69857917329e514b65d6a9` |
| Decode samples | five at each supported concurrency |
| Prefill samples | five cache-cold requests per length |
| Runtime selectors | HC prenorm and fused MHC post+pre enabled; upstream NCCL; custom all-reduce disabled |

Hardware was two RTX PRO 6000 Blackwell Max-Q GPUs at TP2 over PCIe Gen 4 x16
with a 300 W limit per GPU. Decode used 16,384 input tokens, 4,096 forced output
tokens, temperature 0, top-p 1, and fixed seeds.

| C | n | Verifier steps/s | Synthetic output tok/s (acceptance-dependent) | ITL ms/token | Acceptance rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 59.61 `[58.22, 62.17]` | 278.8 `[229.2, 349.4]` | 3.65 `[2.90, 4.13]` | 0.737 `[0.581, 1.000]` |
| 2 | 5 | 44.63 `[44.12, 45.70]` | 438.3 `[428.4, 486.0]` | 4.52 `[4.28, 4.63]` | 0.796 `[0.779, 0.851]` |
| 4 | 5 | 31.55 `[31.40, 34.37]` | 595.9 `[485.0, 743.5]` | 6.96 `[5.98, 8.24]` | 0.716 `[0.568, 0.869]` |
| 8 | 5 | 22.27 `[21.94, 22.32]` | 855.9 `[806.6, 904.0]` | 10.55 `[10.22, 10.92]` | 0.753 `[0.723, 0.802]` |
| 16 | 5 | 16.97 `[16.88, 17.14]` | 1,342.0 `[1,274.3, 1,412.0]` | 15.12 `[14.59, 15.38]` | 0.783 `[0.731, 0.816]` |
| 32 | 5 | 12.61 `[12.59, 12.64]` | 1,974.2 `[1,885.9, 2,028.9]` | 22.89 `[22.49, 23.07]` | 0.771 `[0.733, 0.790]` |

| 8K prompt tok/s | 32K prompt tok/s | 64K prompt tok/s | 128K prompt tok/s |
|---:|---:|---:|---:|
| 7,519.9 | 8,425.2 | 8,146.2 | 7,689.8 |

| Gate | Questions | Correct | Accuracy | Request errors |
|---|---:|---:|---:|---:|
| GSM8K | 1,319 | 1,258 | 95.38% | 0 |

## v0.2.1-rc.0

| Field | Value |
|---|---|
| Measured image digest | `sha256:2ee5ad4d83863ac5c3b46a2db921f0afdcce1544f69915721a6271df705d1286` |
| SGLang effective tree | `68de16e0e3ddc5b5d04b6a2bdfbabbbebefc5e03` |
| FlashInfer effective tree | `09b10c6dc66ca0c96c62a13dfa5ea3b63f1018e4` |
| AIPerf revision | `6ed4823d127b3a6d12c63fb8c2ca5eff13f9ba23` |
| Harness revision | `6fc08101a6c309b998a2f393b30065942e98a0b4` |
| Decode samples | ten at C1; five at every other supported concurrency |
| Prefill samples | five cache-cold requests per length |
| Runtime selectors | HC prenorm, fused MHC post+pre, and FP8 W_o_A enabled; upstream NCCL; custom all-reduce disabled |

Hardware and decode shape matched v0.2.0-rc.0.

| C | n | Verifier steps/s | Synthetic output tok/s (acceptance-dependent) | ITL ms/token | Acceptance rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 62.37 `[59.73, 63.86]` | 322.0 `[292.7, 362.9]` | 3.35 `[3.07, 3.53]` | 0.841 `[0.711, 0.999]` |
| 2 | 5 | 46.85 `[46.24, 49.28]` | 462.1 `[428.5, 573.0]` | 4.57 `[3.80, 4.72]` | 0.773 `[0.711, 0.947]` |
| 4 | 5 | 32.16 `[31.82, 32.70]` | 665.4 `[614.4, 699.2]` | 6.49 `[6.20, 7.27]` | 0.827 `[0.754, 0.868]` |
| 8 | 5 | 22.33 `[22.19, 22.89]` | 871.0 `[781.9, 950.1]` | 10.63 `[9.93, 10.86]` | 0.755 `[0.693, 0.818]` |
| 16 | 5 | 17.19 `[16.91, 17.25]` | 1,356.3 `[1,331.4, 1,386.4]` | 14.89 `[14.65, 15.09]` | 0.769 `[0.748, 0.812]` |
| 32 | 5 | 12.64 `[12.60, 12.66]` | 1,951.5 `[1,945.3, 2,031.1]` | 22.81 `[22.44, 23.04]` | 0.760 `[0.753, 0.793]` |

| 8K prompt tok/s | 32K prompt tok/s | 64K prompt tok/s | 128K prompt tok/s |
|---:|---:|---:|---:|
| 7,541.7 | 8,528.6 | 7,907.6 | 7,771.2 |

| Gate | Questions | Correct | Accuracy | Request errors |
|---|---:|---:|---:|---:|
| GSM8K | 1,319 | 1,262 | 95.68% | 0 |
