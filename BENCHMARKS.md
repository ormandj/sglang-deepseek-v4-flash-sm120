# v0.2.1-rc.0 and vLLM r33 measurements

Last updated: 2026-08-12 CDT.

This document contains only the current SGLang `v0.2.1-rc.0` and vLLM r33
publication measurements. The executable harness, analyzers, scoring logic,
quality graders, and tests are under [`bench/`](bench/).

## System and software

- two NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs;
- SM120, TP2 over PCIe Gen 4 x16;
- 300 W power limit per GPU;
- one engine active at a time;
- clients executed inside the serving pod against `127.0.0.1:8000`;
- model `deepseek-ai/DeepSeek-V4-Flash-0731` at
  `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`;
- speculative decoding: DSpARK block size 5;
- HC prenorm: `SGLANG_OPT_DEEPGEMM_HC_PRENORM=1`;
- fused MHC post+pre: `SGLANG_OPT_FUSE_MHC_POST_PRE=1`;
- FP8 W_o_A target path: `SGLANG_OPT_FP8_WO_A_GEMM=1`;
- SGLang communication: NCCL with custom all-reduce disabled.

SGLang used this repository's `v0.2.1-rc.0` source composition. The measured
private candidate digest was
`sha256:2ee5ad4d83863ac5c3b46a2db921f0afdcce1544f69915721a6271df705d1286`:

```text
SGLang main                  dc5f6c488317645d96dc630b1f410e4dfb6f9667
SGLang effective tree        68de16e0e3ddc5b5d04b6a2bdfbabbbebefc5e03
FlashInfer main              065971254bca6ad0509d775e5806de53b64ac7b9
FlashInfer effective tree    09b10c6dc66ca0c96c62a13dfa5ea3b63f1018e4
FlashInfer version           0.6.18.dev20260811
cache schema                 v11
```

vLLM used local-inference-lab r33:

```text
docker.io/voipmonitor/vllm@sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82
```

The r33 configuration used TP2, DSpARK block size 5,
`max_num_seqs=16`, `max_num_batched_tokens=8192`, FP8 KV cache,
`max_model_len=131072`, and `gpu_memory_utilization=0.975`.

## Benchmark implementation

AIPerf 0.12.0 is pinned at:

```text
6ed4823d127b3a6d12c63fb8c2ca5eff13f9ba23
```

The engine measurement was collected with project harness revision:

```text
6fc08101a6c309b998a2f393b30065942e98a0b4
```

The results below were collected with that revision. The current checked
harness standardizes future publication panels at five repetitions for every
supported concurrency; it does not retroactively alter this release's recorded
sample counts. TTFT remains in raw AIPerf output but is not scored or presented
as an independent result.

## Publication method

Each engine used one unchanged server process. All clients ran inside the
selected serving pod against localhost, and all cells ran sequentially. Every
supported decode concurrency and every cold-prefill length was warmed once
before measurement. There was no restart or per-repetition warmup.

Decode used:

- OpenAI chat completions with streaming;
- 16,384 requested input tokens;
- 4,096 output tokens with EOS ignored and the full length forced;
- temperature 0, top-p 1, ten fixed-seed paths at C1, and five repetitions at
  every other measured concurrency;
- exact C1, C2, C4, C8, C16, and C32 occupancy where supported;
- the same 17,408–20,480 average-context analysis interval;
- an empty request queue and no prefill work during the accepted interval.

The primary engine rate is the OLS slope of the engine's server-side decode
step counter. Synthetic fixed-window output rate comes from the server's
generation-token counter over the same interval. It is not client end-to-end,
production, interactive, or application throughput. It combines verifier-step
rate with output tokens produced per step, so DSpARK acceptance directly
affects it. Different fixed prompt/seed paths have different
accepted-draft-length distributions, making this metric more variable than
verifier steps/s. Acceptance statistics remain separate so a different
generated path cannot be represented as an engine-clock change.

AIPerf calculates per-request ITL as:

```text
(request latency - TTFT) / (output tokens - 1)
```

It is average post-first-token time per generated token, including speculative
acceptance and scheduling effects. It is not a distribution of literal chunk
arrival gaps. TTFT is omitted from the published tables because, with these
long prompts, it primarily restates prefill speed. Request latency is retained
in raw results but is not a headline metric because it combines prefill,
decode, and scheduling.

Cold prefill used one output token, temperature 0, top-p 1, explicit cache
busting, and 8K, 32K, 64K, and 130,816-token input targets. Prompt throughput
uses the observed input token count divided by TTFT. Only prompt tok/s is
presented.

Every table entry is a same-process median. SGLang C1 uses ten repetitions;
vLLM C1 and every other measured cell use five. Those repetitions are fixed
prompt paths, not independent process deployments. The machine-readable
summaries retain every run value, min/max, sample standard deviation, and
sample coefficient of variation.

vLLM r33 was not measured at C32 because that deployment did not have enough KV
capacity to admit the workload. No value is imputed.

## Synthetic fixed-window decode

| Engine | C | n | Verifier steps/s | Synthetic output tok/s (acceptance-dependent) | ITL ms/token |
|---|---:|---:|---:|---:|---:|
| SGLang v0.2.1-rc.0 | 1 | 10 | 62.37 | 322.0 | 3.35 |
| vLLM r33 | 1 | 5 | 66.58 | 255.2 | 3.89 |
| SGLang v0.2.1-rc.0 | 2 | 5 | 46.85 | 462.1 | 4.57 |
| vLLM r33 | 2 | 5 | 46.34 | 421.0 | 5.22 |
| SGLang v0.2.1-rc.0 | 4 | 5 | 32.16 | 665.4 | 6.49 |
| vLLM r33 | 4 | 5 | 33.07 | 616.0 | 7.06 |
| SGLang v0.2.1-rc.0 | 8 | 5 | 22.33 | 871.0 | 10.63 |
| vLLM r33 | 8 | 5 | 23.45 | 795.8 | 11.08 |
| SGLang v0.2.1-rc.0 | 16 | 5 | 17.19 | 1,356.3 | 14.89 |
| vLLM r33 | 16 | 5 | 15.86 | 1,097.2 | 17.59 |
| SGLang v0.2.1-rc.0 | 32 | 5 | 12.64 | 1,951.5 | 22.81 |
| vLLM r33 | 32 | 0 | Not measured: insufficient KV capacity | Not measured | Not measured |

## DSpARK acceptance

Each entry is the median of the per-run means. Sample counts match the decode
table above.

| Engine | C | Acceptance rate | Output tokens/step/request |
|---|---:|---:|---:|
| SGLang v0.2.1-rc.0 | 1 | 0.841 | 5.206 |
| vLLM r33 | 1 | 0.588 | 3.938 |
| SGLang v0.2.1-rc.0 | 2 | 0.773 | 4.819 |
| vLLM r33 | 2 | 0.731 | 4.655 |
| SGLang v0.2.1-rc.0 | 4 | 0.827 | 5.080 |
| vLLM r33 | 4 | 0.764 | 4.821 |
| SGLang v0.2.1-rc.0 | 8 | 0.755 | 4.825 |
| vLLM r33 | 8 | 0.649 | 4.245 |
| SGLang v0.2.1-rc.0 | 16 | 0.769 | 4.938 |
| vLLM r33 | 16 | 0.672 | 4.359 |
| SGLang v0.2.1-rc.0 | 32 | 0.760 | 4.819 |
| vLLM r33 | 32 | Not measured | Not measured |

## Cold prefill

| Engine | 8K prompt tok/s | 32K prompt tok/s | 64K prompt tok/s | 128K prompt tok/s |
|---|---:|---:|---:|---:|
| SGLang v0.2.1-rc.0 | 7,541.7 | 8,528.6 | 7,907.6 | 7,771.2 |
| vLLM r33 | 7,689.8 | 8,784.7 | 8,518.7 | 7,953.6 |

## Completed quality result

Each engine ran the full GSM8K set once at concurrency 16, temperature 0, seed
42, and a 16,384-token completion cap.

The SGLang GSM8K run used the same unchanged `v0.2.1-rc.0` process as the
performance panel. It is retained as a separate quality result; the
performance sweep above does not replace it.

| Engine | Questions | Correct | Accuracy | Request errors |
|---|---:|---:|---:|---:|
| SGLang v0.2.1-rc.0 | 1,319 | 1,262 | 95.68% | 0 |
| vLLM r33 | 1,319 | 1,243 | 94.24% | 0 |

Near-context and AgentX checks are not represented as completed results.

## Machine-readable results

- [`sglang-v0.2.1-rc.0-publication-summary.json`](bench/results/sglang-v0.2.1-rc.0-publication-summary.json)
- [`vllm-r33-publication-summary.json`](bench/results/vllm-r33-publication-summary.json)
- [`sglang-v0.2.1-rc.0-gsm8k.json`](bench/results/sglang-v0.2.1-rc.0-gsm8k.json)
- [`vllm-r33-gsm8k.json`](bench/results/vllm-r33-gsm8k.json)

The summaries retain TTFT and request latency because AIPerf records them, but
neither is scored or interpreted as an independent engine-performance result.

## Running the benchmark

Install the exact AIPerf revision recorded by
[`bench/aiperf/aiperf.lock.json`](bench/aiperf/aiperf.lock.json), then stage
this repository's `bench/` directory in the serving pod. Run against the pod's
localhost endpoint:

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
required; omit it for a keyless endpoint. Neither key is written to output.

The runner rejects execution outside Kubernetes, a non-loopback endpoint,
wrong token shape, incomplete request sets, failed requests, wrong occupancy,
queued work, prefill in a decode interval, counter resets, cached tokens in a
cold-prefill cell, and an insufficient equal-context interval.

See [`bench/aiperf/README.md`](bench/aiperf/README.md) and
[`bench/aiperf/STATISTICAL-DESIGN.md`](bench/aiperf/STATISTICAL-DESIGN.md) for
the complete method and quality-gate commands.
