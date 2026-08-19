# Release performance history

This file preserves the measured performance snapshot for each published
release candidate. [BENCHMARKS.md](BENCHMARKS.md) remains the
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

## v0.5.0-rc.1

| Field | Value |
|---|---|
| Public image tag | `ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.5.0-rc.1` |
| SGLang effective tree | `dc71e9b9bb380e96bcb2c0cb4aed120f79478c3e` |
| FlashInfer effective tree | `489e9318e4d21d3ecddc8d2ec8f138dde93784b5` |
| DeepGEMM effective tree | `ff371cf8cc7186c8dab8e07cc7cda6c28baa092f` |
| AIPerf revision | `6ed4823d127b3a6d12c63fb8c2ca5eff13f9ba23` |
| Harness revision | `15f5dcc169f64b629523803a9c0d8ded1728aabf` |
| Decode samples | five at every supported concurrency |
| Prefill samples | five cache-cold requests per length |
| Runtime selectors | HC prenorm, fused MHC post+pre, FP8 W_o_A, and PCIe-IPC all-reduce enabled; legacy custom all-reduce and TRT/MNNVL disabled |

Hardware and decode shape matched v0.3.3-rc.0.

| C | n | Forward passes/s | Synthetic output tok/s | ITL ms/token | Acceptance rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 65.476 `[64.723, 67.402]` | 325.7 `[291.9, 404.4]` | 3.160 `[2.482, 3.415]` | 0.787 `[0.700, 1.000]` |
| 2 | 5 | 48.699 `[47.477, 48.852]` | 480.3 `[469.5, 546.3]` | 4.337 `[4.103, 4.575]` | 0.813 `[0.739, 0.911]` |
| 4 | 5 | 33.538 `[33.292, 34.900]` | 657.0 `[645.4, 749.5]` | 6.291 `[5.988, 6.617]` | 0.778 `[0.756, 0.829]` |
| 8 | 5 | 23.330 `[22.879, 23.515]` | 917.8 `[879.6, 968.0]` | 9.478 `[9.186, 10.347]` | 0.798 `[0.736, 0.825]` |
| 16 | 5 | 17.518 `[17.316, 17.728]` | 1,373.5 `[1,365.9, 1,460.6]` | 14.583 `[13.288, 14.803]` | 0.782 `[0.763, 0.820]` |
| 32 | 5 | 13.013 `[12.919, 13.071]` | 2,047.9 `[1,970.7, 2,132.6]` | 22.349 `[21.959, 22.385]` | 0.765 `[0.747, 0.806]` |

| 8K prompt tok/s | 32K prompt tok/s | 64K prompt tok/s | 128K prompt tok/s |
|---:|---:|---:|---:|
| 7,665.1 | 8,602.4 | 8,284.2 | 7,785.1 |

| Gate | Questions | Correct | Accuracy | Request errors |
|---|---:|---:|---:|---:|
| GSM8K | 1,319 | 1,261 | 95.60% | 0 |

## v0.3.3-rc.0

| Field | Value |
|---|---|
| Public image tag | `ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.3.3-rc.0` |
| SGLang effective tree | `2833e60bfdb9c17820095e2e1aa478bb2eb041ef` |
| FlashInfer effective tree | `b68d15ef203d0e0e11b3734ade4ccf6dca1b6b4d` |
| DeepGEMM effective tree | `b166d085065d39155a8f745126d6db88597d268c` |
| AIPerf revision | `6ed4823d127b3a6d12c63fb8c2ca5eff13f9ba23` |
| Harness revision | `5146a50047fd896fc2b81fb34df4d4efabcddfdf` |
| Decode samples | five at every supported concurrency |
| Prefill samples | five cache-cold requests per length |
| Runtime selectors | HC prenorm, fused MHC post+pre, FP8 W_o_A, and PCIe-IPC all-reduce enabled; legacy custom all-reduce and TRT/MNNVL disabled |

Hardware was two RTX PRO 6000 Blackwell Max-Q GPUs at TP2 over PCIe Gen 4 x16
with a 300 W limit per GPU. Decode used 16,384 input tokens, 4,096 forced output
tokens, temperature 0, top-p 1, and fixed seeds.

| C | n | Forward passes/s | Synthetic output tok/s | ITL ms/token | Acceptance rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 63.864 `[63.016, 66.412]` | 331.7 `[321.5, 398.5]` | 3.110 `[2.745, 3.241]` | 0.851 `[0.777, 1.000]` |
| 2 | 5 | 47.737 `[46.815, 48.797]` | 488.6 `[456.9, 514.4]` | 4.321 `[4.096, 4.783]` | 0.807 `[0.715, 0.886]` |
| 4 | 5 | 34.322 `[32.950, 37.119]` | 653.9 `[652.1, 783.3]` | 6.481 `[5.839, 9.767]` | 0.779 `[0.732, 0.812]` |
| 8 | 5 | 22.968 `[22.795, 23.197]` | 912.3 `[864.6, 936.0]` | 9.911 `[9.488, 13.423]` | 0.775 `[0.740, 0.794]` |
| 16 | 5 | 17.304 `[17.066, 17.533]` | 1,392.0 `[1,388.6, 1,412.5]` | 14.440 `[14.202, 14.548]` | 0.793 `[0.786, 0.817]` |
| 32 | 5 | 12.722 `[12.624, 12.763]` | 2,036.8 `[1,968.7, 2,102.9]` | 22.085 `[21.874, 22.587]` | 0.790 `[0.769, 0.821]` |

| 8K prompt tok/s | 32K prompt tok/s | 64K prompt tok/s | 128K prompt tok/s |
|---:|---:|---:|---:|
| 7,776.8 | 8,743.1 | 8,464.5 | 7,966.6 |

| Gate | Questions | Correct | Accuracy | Request errors |
|---|---:|---:|---:|---:|
| GSM8K | 1,319 | 1,261 | 95.60% | 0 |

The requested long-output diagnostic completed eight of eight requests and all
eight passed its structural and JavaScript-parse checks. It is not a routine
recurring performance gate.

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
