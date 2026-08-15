# v0.3.3-rc.0 and vLLM r33 measurements

Last updated: 2026-08-14 CDT.

This document presents the current SGLang `v0.3.3-rc.0` publication panel and
the retained vLLM r33 panel. It reports the two engines' values without a
winner/loser interpretation. The executable harness, analyzers, scoring logic,
quality graders, and machine-readable summaries are under [`bench/`](bench/).

## System and software

- two NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs;
- SM120, TP2 over PCIe Gen 4 x16;
- 300 W power limit per GPU;
- one engine active at a time;
- clients executed inside the serving pod against `127.0.0.1:8000`;
- model `deepseek-ai/DeepSeek-V4-Flash-0731` at
  `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`;
- speculative decoding: DSpARK block size 5;
- SGLang HC prenorm, fused MHC post+pre, and FP8 W_o_A enabled;
- SGLang FlashInfer PCIe-IPC all-reduce enabled for eligible decode
  reductions, legacy custom all-reduce disabled, and NCCL fallback retained;
- SGLang TRT/MNNVL fusion disabled and PCIe-IPC all-gather absent.

The qualified SGLang source identity is:

```text
SGLang main                  9d34c2809f58f3d84ef5dd343733e3f5e86395d5
SGLang effective tree        2833e60bfdb9c17820095e2e1aa478bb2eb041ef
FlashInfer main              ed6c709849fe1c02d4545b4e743a436405f6ca5b
FlashInfer effective tree    b68d15ef203d0e0e11b3734ade4ccf6dca1b6b4d
FlashInfer version           0.6.18.dev20260813
DeepGEMM base                7509acb3e261b5acba06087e91c70c409a43419c
DeepGEMM effective tree      b166d085065d39155a8f745126d6db88597d268c
DeepGEMM version             0.1.5.post2+sm120jit2
cache schema                 v15
```

The performance and quality measurements used a pre-release build with those
exact three effective trees and the same serving options. The SemVer build
changes only release metadata and the cache namespace.

vLLM used local-inference-lab r33:

```text
docker.io/voipmonitor/vllm@sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82
```

Its configuration used TP2, DSpARK block size 5, `max_num_seqs=16`,
`max_num_batched_tokens=8192`, FP8 KV cache, `max_model_len=131072`, and
`gpu_memory_utilization=0.975`.

## Benchmark implementation

AIPerf 0.12.0 is pinned at:

```text
6ed4823d127b3a6d12c63fb8c2ca5eff13f9ba23
```

The SGLang panel used benchmark project revision:

```text
5146a50047fd896fc2b81fb34df4d4efabcddfdf
```

The retained vLLM r33 panel used benchmark project revision:

```text
6fc08101a6c309b998a2f393b30065942e98a0b4
```

Both revisions use the same request shapes, fixed context interval, five
repetitions per supported cell, and 20-scrape minimum. The current SGLang
analyzer calibration accepts an interval of at least 6.5 seconds; the retained
vLLM panel used the earlier 7-second minimum.

Each engine used one unchanged process. All cells ran sequentially in the
selected serving pod against localhost. Every distinct decode concurrency and
prefill length was warmed once before measurement; there was no restart or
per-repetition warmup.

Decode used OpenAI chat completions with streaming, 16,384 requested input
tokens, 4,096 output tokens with EOS ignored, temperature 0, top-p 1, and five
fixed-seed paths at every supported concurrency. The analyzer fits the OLS
slope of server-side counters over the same 17,408-20,480 average-context
interval. A valid interval requires exact occupancy, an empty queue, no
prefill work, monotonic counters, at least 6.5 seconds, and at least 20 metric
scrapes.

Forward passes/s is the primary engine-execution statistic. Synthetic output
tok/s is the generated-output-token counter slope over the same interval. It
combines verifier rate with speculative acceptance and is not expected
production, interactive, or application throughput. Acceptance is reported
separately so a generated-path change is not represented as an engine-clock
change.

AIPerf calculates per-request ITL as:

```text
(request latency - TTFT) / (output tokens - 1)
```

ITL is average post-first-token time per generated token, including
speculative acceptance and scheduling. TTFT remains in raw output but is not a
separate headline because the long-prompt value primarily restates prefill.

Cold prefill used one output token, temperature 0, top-p 1, explicit cache
busting, and 8K, 32K, 64K, and 130,816-token input targets. Prompt throughput
is observed input tokens divided by TTFT. Every cell used five requests.

The repetitions are same-process prompt-path subsamples, not independent
deployment replicates. The machine-readable summary retains every run, its
median, min/max, sample standard deviation, and coefficient of variation. No
valid result was excluded because its performance was surprising.

vLLM r33 was not measured at C32 because its deployment lacked enough KV
capacity to admit that workload. No value is imputed.

## Fixed-window decode

### SGLang run values

| C | Forward passes/s, five runs | Median | Sample CV | Synthetic output tok/s, five runs | Median | Median ITL |
|---:|---|---:|---:|---|---:|---:|
| 1 | 63.016, 66.412, 64.423, 63.864, 63.478 | 63.864 | 2.05% | 343.0, 398.5, 321.5, 324.1, 331.7 | 331.7 | 3.110 ms |
| 2 | 46.815, 47.737, 48.060, 48.797, 47.453 | 47.737 | 1.54% | 514.4, 492.9, 488.6, 456.9, 482.0 | 488.6 | 4.321 ms |
| 4 | 34.999, 34.322, 33.595, 37.119, 32.950 | 34.322 | 4.64% | 652.3, 677.8, 653.9, 783.3, 652.1 | 653.9 | 6.481 ms |
| 8 | 23.197, 23.155, 22.795, 22.968, 22.918 | 22.968 | 0.73% | 936.0, 912.3, 864.6, 913.0, 888.7 | 912.3 | 9.911 ms |
| 16 | 17.230, 17.533, 17.304, 17.317, 17.066 | 17.304 | 0.97% | 1,388.6, 1,392.0, 1,412.5, 1,390.1, 1,400.7 | 1,392.0 | 14.440 ms |
| 32 | 12.676, 12.729, 12.722, 12.763, 12.624 | 12.722 | 0.43% | 2,017.5, 2,069.5, 2,036.8, 2,102.9, 1,968.7 | 2,036.8 | 22.085 ms |

### Current engine table

| Engine | C | n | Forward passes/s | Synthetic output tok/s | ITL ms/token |
|---|---:|---:|---:|---:|---:|
| SGLang v0.3.3-rc.0 | 1 | 5 | 63.864 | 331.7 | 3.110 |
| vLLM r33 | 1 | 5 | 66.580 | 255.2 | 3.890 |
| SGLang v0.3.3-rc.0 | 2 | 5 | 47.737 | 488.6 | 4.321 |
| vLLM r33 | 2 | 5 | 46.340 | 421.0 | 5.220 |
| SGLang v0.3.3-rc.0 | 4 | 5 | 34.322 | 653.9 | 6.481 |
| vLLM r33 | 4 | 5 | 33.070 | 616.0 | 7.060 |
| SGLang v0.3.3-rc.0 | 8 | 5 | 22.968 | 912.3 | 9.911 |
| vLLM r33 | 8 | 5 | 23.450 | 795.8 | 11.080 |
| SGLang v0.3.3-rc.0 | 16 | 5 | 17.304 | 1,392.0 | 14.440 |
| vLLM r33 | 16 | 5 | 15.860 | 1,097.2 | 17.590 |
| SGLang v0.3.3-rc.0 | 32 | 5 | 12.722 | 2,036.8 | 22.085 |
| vLLM r33 | 32 | 0 | not reachable: vLLM-reported KV 143,599 tok, `max_num_seqs=16` (upstream TP2 recipe) | — | — |

## DSpARK acceptance

Each SGLang value is the median of the five per-run means. Acceptance explains
synthetic output-rate variation; it is not used as an engine-speed result.

| Engine | C | Acceptance rate | Output tokens/forward/request |
|---|---:|---:|---:|
| SGLang v0.3.3-rc.0 | 1 | 0.851 | 5.215 |
| vLLM r33 | 1 | 0.588 | 3.938 |
| SGLang v0.3.3-rc.0 | 2 | 0.807 | 5.001 |
| vLLM r33 | 2 | 0.731 | 4.655 |
| SGLang v0.3.3-rc.0 | 4 | 0.779 | 4.891 |
| vLLM r33 | 4 | 0.764 | 4.821 |
| SGLang v0.3.3-rc.0 | 8 | 0.775 | 4.872 |
| vLLM r33 | 8 | 0.649 | 4.245 |
| SGLang v0.3.3-rc.0 | 16 | 0.793 | 5.027 |
| vLLM r33 | 16 | 0.672 | 4.359 |
| SGLang v0.3.3-rc.0 | 32 | 0.790 | 4.984 |
| vLLM r33 | 32 | not reachable: vLLM-reported KV 143,599 tok, `max_num_seqs=16` (upstream TP2 recipe) | — |

> **vLLM C=32 capacity.** The vLLM side runs the upstream-documented TP2 profile
> unmodified ("Gilded Gnosis v20 r33, documented TP2 fixed-K5",
> [`local-inference-lab/rtx6kpro`](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/ds4dspark-v20-r33.md)),
> which sets `max_num_seqs=16`. Independently, vLLM self-reported at startup:
> `Available KV cache memory: 8.07 GiB`, `GPU KV cache size: 143,599 tokens`,
> `Maximum concurrency for 131,072 tokens per request: 1.10x`. At 16,384-token
> input plus 4,096 output, 32 streams need roughly 655,000 KV tokens against the
> 143,599 available. Both limits belong to that profile, not to vLLM as an
> engine. Contexts also differed (131,072 vs 774,656), so the pools are not
> directly comparable.


## Cold prefill

### SGLang run values

| Target | Prompt tok/s, five requests | Aggregate prompt tok/s |
|---:|---|---:|
| 8K | 7,426.6, 7,859.1, 7,896.3, 7,930.5, 7,849.7 | 7,776.8 |
| 32K | 8,765.1, 8,793.9, 8,753.7, 8,704.1, 8,725.6 | 8,743.1 |
| 64K | 8,575.0, 8,545.8, 8,463.0, 8,391.6, 8,366.8 | 8,464.5 |
| 128K | 8,137.5, 8,018.2, 7,909.6, 7,895.3, 7,889.3 | 7,966.6 |

### Current engine table

| Engine | 8K prompt tok/s | 32K prompt tok/s | 64K prompt tok/s | 128K prompt tok/s |
|---|---:|---:|---:|---:|
| SGLang v0.3.3-rc.0 | 7,776.8 | 8,743.1 | 8,464.5 | 7,966.6 |
| vLLM r33 | 7,689.8 | 8,784.7 | 8,518.7 | 7,953.6 |

## Quality results

Each engine ran the full GSM8K set once at concurrency 16, temperature 0,
seed 42, and a 16,384-token response cap.

| Engine | Questions | Correct | Accuracy | Request errors |
|---|---:|---:|---:|---:|
| SGLang v0.3.3-rc.0 | 1,319 | 1,261 | 95.60% | 0 |
| vLLM r33 | 1,319 | 1,243 | 94.24% | 0 |

All 1,319 SGLang responses used the grader's documented last-number fallback
because the model did not emit the dataset's `####` marker. Correctness and
fallback counts are separate facts.

The requested SGLang long-output diagnostic ran eight sequential requests at
temperature 1.0, top-p 0.95, `reasoning_effort=max`, and a 131,072-token cap.
All eight returned HTTP success with `finish_reason=stop` and passed the
code-fence, closing-tag, placeholder, and JavaScript-parse checks. Completion
token counts were 48,581; 55,660; 61,188; 29,583; 22,212; 59,825; 34,869; and
30,778 (median 41,725). This requested diagnostic is not part of the routine
recurring benchmark protocol.

## Machine-readable results

- [`sglang-v0.3.3-rc.0-publication-summary.json`](bench/results/sglang-v0.3.3-rc.0-publication-summary.json)
- [`sglang-v0.3.3-rc.0-gsm8k.json`](bench/results/sglang-v0.3.3-rc.0-gsm8k.json)
- [`sglang-v0.3.3-rc.0-long-output.json`](bench/results/sglang-v0.3.3-rc.0-long-output.json)
- [`vllm-r33-publication-summary.json`](bench/results/vllm-r33-publication-summary.json)
- [`vllm-r33-gsm8k.json`](bench/results/vllm-r33-gsm8k.json)

The summaries retain TTFT and request latency because AIPerf records them, but
neither is presented as an independent engine-performance result.

## Reproducing the engine gate

Install the exact AIPerf revision recorded by
[`bench/aiperf/aiperf.lock.json`](bench/aiperf/aiperf.lock.json), then stage
this repository's `bench/` directory inside the serving pod. Run against the
pod's localhost endpoint:

```bash
BENCH_IMAGE_REF='image@sha256:...' \
BENCH_GITOPS_REVISION='<deployment revision>' \
BENCH_PROJECT_REVISION='<this repository revision>' \
AIPERF_REVISION='6ed4823d127b3a6d12c63fb8c2ca5eff13f9ba23' \
BENCH_MODEL_REVISION='9e165c30e2704aec5d9d593cce3eebd58bbef1cb' \
BENCH_ENGINE=sglang \
./bench/aiperf/run-engine-gate-in-pod.sh campaign build publication
```

Use `BENCH_ENGINE=vllm` for vLLM. Set `BENCH_API_KEY` when authentication is
required; omit it for a keyless endpoint. The key is not written to output.

The runner rejects execution outside Kubernetes, a non-loopback endpoint,
wrong token shape, incomplete or failed request sets, wrong occupancy, queued
work, prefill in a decode interval, counter resets, cached tokens in a cold
prefill cell, and an insufficient equal-context interval.

See [`bench/aiperf/README.md`](bench/aiperf/README.md) and
[`bench/aiperf/STATISTICAL-DESIGN.md`](bench/aiperf/STATISTICAL-DESIGN.md) for
the complete method and gate commands.
