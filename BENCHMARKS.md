# v0.2.0-rc.0 and vLLM r33 measurements

Last updated: 2026-08-12 CDT.

This document contains only the current SGLang `v0.2.0-rc.0` and vLLM r33
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
- SGLang communication: NCCL with custom all-reduce disabled.

SGLang used this repository's `v0.2.0-rc.0` source composition:

```text
SGLang main                  dc5f6c488317645d96dc630b1f410e4dfb6f9667
SGLang effective tree        8c336f4426844b2028938f7c542c0a403d37b804
FlashInfer main              065971254bca6ad0509d775e5806de53b64ac7b9
FlashInfer effective tree    09b10c6dc66ca0c96c62a13dfa5ea3b63f1018e4
FlashInfer version           0.6.18.dev20260811
cache schema                 v10
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
2a9f3b30806bd920c1a256bddbe6e9d5476ee2fc
```

The engine runner and analyzers copied into this public repository are
unchanged from that revision. Later changes add quality gates and clarify
reporting: TTFT remains in raw AIPerf output but is not scored or presented as
an independent result.

## Publication method

Each engine used one unchanged server process. All clients ran inside the
selected serving pod against localhost, and all cells ran sequentially. Every
supported decode concurrency and every cold-prefill length was warmed once
before measurement. There was no restart or per-repetition warmup.

Decode used:

- OpenAI chat completions with streaming;
- 16,384 requested input tokens;
- 4,096 output tokens with EOS ignored and the full length forced;
- temperature 0, top-p 1, and five fixed prompt paths;
- exact C1, C2, C4, C8, C16, and C32 occupancy where supported;
- the same 17,408–20,480 average-context analysis interval;
- an empty request queue and no prefill work during the accepted interval.

The primary engine rate is the OLS slope of the engine's server-side decode
step counter. Useful output-token rate comes from the server's generation-token
counter over the same interval. Acceptance statistics remain separate so a
different generated path cannot be represented as an engine-clock change.

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

Every table entry is the median of five same-process repetitions. Those
repetitions are fixed prompt paths, not independent process deployments. The
machine-readable summaries retain all five values, min/max, sample standard
deviation, and sample coefficient of variation.

vLLM r33 was not measured at C32 because that deployment did not have enough KV
capacity to admit the workload. No value is imputed.

## Controlled decode

| Engine | C | Verifier steps/s | Useful decode tok/s | ITL ms/token |
|---|---:|---:|---:|---:|
| SGLang v0.2.0-rc.0 | 1 | 55.54 | 301.6 | 3.58 |
| vLLM r33 | 1 | 66.58 | 255.2 | 3.89 |
| SGLang v0.2.0-rc.0 | 2 | 44.22 | 480.3 | 4.35 |
| vLLM r33 | 2 | 46.34 | 421.0 | 5.22 |
| SGLang v0.2.0-rc.0 | 4 | 32.19 | 698.5 | 6.34 |
| vLLM r33 | 4 | 33.07 | 616.0 | 7.06 |
| SGLang v0.2.0-rc.0 | 8 | 21.39 | 865.3 | 10.66 |
| vLLM r33 | 8 | 23.45 | 795.8 | 11.08 |
| SGLang v0.2.0-rc.0 | 16 | 16.44 | 1,332.0 | 15.88 |
| vLLM r33 | 16 | 15.86 | 1,097.2 | 17.59 |
| SGLang v0.2.0-rc.0 | 32 | 12.15 | 1,881.0 | 24.77 |
| vLLM r33 | 32 | Not measured: insufficient KV capacity | Not measured | Not measured |

## DSpARK acceptance

Each entry is the median of the five per-run means.

| Engine | C | Acceptance rate | Accepted tokens/step/request |
|---|---:|---:|---:|
| SGLang v0.2.0-rc.0 | 1 | 0.889 | 5.444 |
| vLLM r33 | 1 | 0.588 | 3.938 |
| SGLang v0.2.0-rc.0 | 2 | 0.864 | 5.321 |
| vLLM r33 | 2 | 0.731 | 4.655 |
| SGLang v0.2.0-rc.0 | 4 | 0.886 | 5.429 |
| vLLM r33 | 4 | 0.764 | 4.821 |
| SGLang v0.2.0-rc.0 | 8 | 0.793 | 4.963 |
| vLLM r33 | 8 | 0.649 | 4.245 |
| SGLang v0.2.0-rc.0 | 16 | 0.796 | 4.981 |
| vLLM r33 | 16 | 0.672 | 4.359 |
| SGLang v0.2.0-rc.0 | 32 | 0.754 | 4.768 |
| vLLM r33 | 32 | Not measured | Not measured |

## Cold prefill

| Engine | 8K prompt tok/s | 32K prompt tok/s | 64K prompt tok/s | 128K prompt tok/s |
|---|---:|---:|---:|---:|
| SGLang v0.2.0-rc.0 | 6,424.4 | 7,075.6 | 6,835.5 | 6,546.2 |
| vLLM r33 | 7,689.8 | 8,784.7 | 8,518.7 | 7,953.6 |

## Completed quality result

The full GSM8K set ran once at concurrency 16, temperature 0, seed 42, and a
16,384-token completion cap.

| Engine | Questions | Correct | Accuracy | Request errors |
|---|---:|---:|---:|---:|
| SGLang v0.2.0-rc.0 | 1,319 | 1,258 | 95.38% | 0 |

The long-format n=5, near-context, and AgentX checks remain pending and are not
represented as completed results.

## Machine-readable results

- [`sglang-v0.2.0-rc.0-publication-summary.json`](bench/results/sglang-v0.2.0-rc.0-publication-summary.json)
- [`vllm-r33-publication-summary.json`](bench/results/vllm-r33-publication-summary.json)
- [`sglang-v0.2.0-rc.0-gsm8k.json`](bench/results/sglang-v0.2.0-rc.0-gsm8k.json)

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
