# Engine-gate measurement design

## Decision question

The routine question is whether an engine, kernel, dependency, or server
configuration change altered decode execution or cold-prefill performance on
the fixed SM120 deployment.

Generated text is an input to later decode work. With FP8 MoE inference and
speculative decoding, even a greedy request can follow a different token and
expert-routing path on repeated execution. Useful tokens per second therefore
mixes two quantities:

```text
useful tokens / second
  = speculative decode steps / second
  x useful tokens / step
```

The benchmark records both factors. It never treats speculative acceptance as
proof of a kernel speedup, and it never hides acceptance when reporting useful
throughput.

## Same-process repeatability

All repetitions hold the requested concurrency, require an empty queue, admit
no prefill during the decode window, and use the same in-pod AIPerf placement.
The fixed prompt and seed panel exposes path-dependent variation without adding
server-restart variation between panel members.

## Workloads and estimands

### Decode engine gate

The gate uses temperature zero and top-p one to avoid adding sampling RNG to a
test whose purpose is engine execution. The prompt corpus and seeds are fixed
before comparison. Each request has exact input and minimum output lengths and
ignores EOS.

For each prompt-panel member, the analyzer selects a fixed average
context-length interval and estimates:

- OLS slope of speculative decode steps: primary engine-work rate;
- OLS slope of useful output tokens: deployed useful throughput;
- useful tokens per forward per request: acceptance/path control.

The slopes are accepted only when exact occupancy holds for at least 98% of the
window, the request queue remains empty, prefill counters do not change,
counters remain monotonic, and at least 10 seconds and 30 scrapes are present.
The scrape-count requirement is independent of elapsed time; the duration
floor must not reject a candidate merely because it traverses the fixed context
interval faster.

### Cold prefill

Prefill uses temperature zero and one output token. Each request is cache-busted.
SGLang is explicitly flushed at the cell boundary. vLLM r33 does not expose its
development-only reset API, so its analyzer instead requires the live cached
prompt-token counter to remain zero. The primary rate is observed prompt tokens
divided by the first request start through the last first-token time. Median
TTFT is reported separately. Server counters are controls rather than the
primary clock. The common near-limit target is 130,816 synthetic content tokens;
the retained headroom accounts for engine chat-template expansion below the
131,072-token context limit. Throughput uses each request's observed input-token
count, not the nominal target.

### Production workload and quality

Temperature 1.0 and top-p 0.95 are appropriate for the agentic workload that
characterizes deployed behavior. That workload answers a different question
and remains separate from the engine regression gate. GSM8K and long-output
completion are correctness and stability checks, not decode-capacity
replicates.

## Prompt-panel allocation

A batch at concurrency C already averages C token paths. The fixed panel gives
more repetitions to low concurrency, where one path otherwise dominates:

| Mode | C1 | C2 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|---:|
| quick | 3 | - | - | 2 | - | 1 |
| decode supplement | - | 6 | 4 | - | 2 | - |
| qualification | 8 | 6 | 4 | 3 | 2 | 2 |
| publication | 5 | 5 | 5 | 5 | 5 | 5 where supported |

The decode supplement is used only to extend a matched quick comparison with
the missing intermediate concurrency cells. Keeping it separate avoids
repeating valid quick-gate work while preserving immutable artifacts and the
qualification repetition budget for C2/C4/C16.

C32 is omitted, not imputed, for a deployment whose configured sequence or KV
capacity cannot admit it. Comparisons publish the resulting capacity mismatch.

The publication panel is deliberately uniform: every reported decode
concurrency contains five fresh prompt-path repetitions from one unchanged
server process. A publication comparison never splices cells from quick,
supplement, qualification, or an earlier run. Each of the three published cold
prefill lengths likewise contains five cache-cold requests. This fixed rule
keeps the public method simple and repeatable; it does not turn same-process
prompt paths into independent deployment replicates.

These counts are engineering budgets, not claims of a universal statistically
correct sample size. The summary publishes every value, median, mean, range,
sample standard deviation, and sample CV. A high CV is evidence that the cell
needs more paths or a targeted diagnostic; it is not a reason to discard a
valid result.

## Comparison and independence

Baseline and candidate use the same committed seeds and workload shapes. A
single gate runs all panel members sequentially on one unchanged process. This
removes restart churn from routine comparisons and makes prompt-path variation
visible quickly.

Panel members within one process are subsamples. They do not estimate
deployment-to-deployment variation. Routine decisions therefore report paired
effect sizes and controls without inferential language.

When a candidate is close enough to the decision boundary that formal
uncertainty matters, independent serving-process blocks become the experimental
unit. Collect matched baseline/candidate blocks, alternate order, estimate the
paired log-ratio variance from those blocks, and choose additional block count
for the specific smallest effect that matters. Do not automatically run a
fixed five-to-ten-block campaign for every code change, and do not pretend a
maximum count achieved a target power when it did not.

## Validity and exclusions

Objective analyzer failures include wrong token shape, cancellation, missing
records, failure to reach target concurrency, nonempty queue, occupancy loss,
counter reset, prefill in a decode window, cached tokens in cold prefill, or an
insufficient equal-context window. Performance value itself is never an
exclusion reason.

Public result packages contain the configs, fixed seeds, run-level values,
source revisions, image digest, and calculation code while removing internal
cluster names, registry locations, endpoints, and credentials.
