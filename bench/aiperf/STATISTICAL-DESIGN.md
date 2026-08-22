# Engine measurement design

## Decision target

The routine decision is whether a source, dependency, kernel, or server change
improves DeepSeek-V4-Flash-0731 engine performance on the fixed SM120 system
without a meaningful regression in the user-facing or scale guardrails.

The priorities are explicit:

1. C1 single-user programming performance;
2. C2/C4/C8 sub-agent fan-out;
3. C16/C32 scale behavior;
4. cold-prefill rate;
5. production-shaped AgentX behavior and correctness/stability gates.

No weighted composite hides a regression in one dimension.

## Why engine work and useful throughput are separate

With speculative decoding:

```text
useful output tokens / second
  = decode forward passes / second
  x useful tokens / forward
```

Generated token paths can change speculative acceptance, expert routing, and
later work even with greedy sampling. The controlled gate therefore records
the server's forward-pass rate as the primary engine execution measure and
reports useful tokens/sec and tokens/forward beside it. A change is not called
a kernel or scheduler speedup solely because one generated path accepted more
draft tokens.

Temperature 0 and top-p 1 remove intentional sampling randomness from this
engine test. Temperature 1 and top-p 0.95 are retained in the separate AgentX
workload because that test asks about deployed agentic behavior.

## Fixed shape and comparable window

All decode concurrencies use 16K requested input and 4K forced output. Holding
shape constant avoids changing the amount of prefill, decode context, or work
per request as concurrency changes. A 4K output is long enough to expose a
stable server-side interval without making routine gates multi-hour campaigns.

The analyzer fits counter slopes only over the same 17,408–20,480 average
context interval. It requires:

- exact target occupancy for at least 98% of the interval;
- an empty request queue;
- no prefill counter change;
- monotonic counters;
- at least 1 second and 20 metric scrapes.

The scrape-count requirement is authoritative for sampling density, so the
duration floor is a backstop only. Successive floors of 7.0, 6.5, and 6.0
seconds each rejected otherwise valid equal-context intervals that contained
the required 20 scrapes, and each rejection cost a campaign restart without
changing any measured value. The floor is now 1 second: the 20-scrape
requirement, the fixed context interval, and the occupancy, queue, and
prefill controls decide admission.

Loosening this threshold does not invalidate results measured under a
stricter one. The floor only admits or rejects a window; it does not change
the slope computed for an admitted window. Every retained interval that
passed a 6.0-, 6.5-, or 7.0-second floor also passes this one, so previously
published tables remain directly comparable.

Client throughput includes fill and drain and is therefore not used as the
fixed-window engine clock. ITL is retained as the user-facing decode measure.
Request latency is a secondary integrated measurement of the fixed request.
TTFT remains in the raw AIPerf output because it is needed to calculate prefill
rate, but it is not scored or published separately: for long prompts it is the
same prefill observation expressed as elapsed time.

## Repetitions and uncertainty

An exploratory decode panel uses three fixed prompt paths at C1/C2/C4/C8. A
quick panel uses three paths at C1/C4/C8 plus its matched prefill panel. A
qualification panel uses five paths at the priority C1/C2/C4/C8 cells and three
at C16/C32. Public tables use five at every supported concurrency for a simple
uniform contract.

These are same-process prompt-path repetitions, not independent machine or
deployment replicates. They provide:

- the median effect;
- every individual run value;
- min/max, sample standard deviation, and sample coefficient of variation;
- acceptance/path controls for each decode run.

They do not justify a p-value or a claim that a sub-noise percentage is real.
If an otherwise qualified candidate is close enough that the decision depends
on a small effect, the next experiment uses independent matched process blocks:
alternate baseline/candidate order, measure the paired log ratios, estimate
variance from those blocks, and add blocks only for the stated smallest effect
that matters. Routine changes do not pay that cold-start cost preemptively.

## Warmup and process lifetime

Warmup is coverage, not a measured repetition. After a server start, image or
configuration change, or profiling attachment, each decode concurrency and
prefill length is exercised once. The measured panel then runs sequentially
without restarting the process or warming between repetitions.

This removes avoidable cold-compile and process-start variability while keeping
the comparison representative of a warm serving engine. A cold-cache prefill
cell still gets a distinct request marker and the engine-specific cache control
described in the harness README.

## Production workload

AgentX is not used to diagnose an individual kernel. It is a separate external
validity check using AIPerf's date-pinned public trace scenario and locked replay
rules. C1 represents the primary coding-agent session and C8 represents fan-out.
The scenario must run at least 900 seconds and must report
`submission_valid=true`.

Fixed synthetic and AgentX results are both factual. They answer different
questions and are never merged into one score.

## Request turnover and refill batching

The clean decode panel intentionally rejects any interval containing prefill
work. It therefore cannot detect a scheduler regression that admits each
replacement request as a separate prefill while a fixed number of other
requests continue decoding.

The turnover panel closes that gap without contaminating the decode metric. It
submits 16 unique requests at C1/C2/C4 and 32 at C8, providing at least four
closed-loop replacement waves with 256-input/256-output shapes. The routine
engine-release screen uses three same-process prompt-path repetitions at C8.
The broader screen uses three at C1/C2/C4/C8, and full qualification/publication
uses five at every concurrency. Scheduler, admission, batching, or refill
changes, a new upstream-main integration, and an anomalous C8 release screen
require the full panel. The analyzer validates request shape and client/server
occupancy, then reports output rate and latency beside the number of requests
served per prefill forward pass. That batching ratio is diagnostic evidence for
the frozen short-prompt/chunking method, not a portable engine score.

Turnover results remain a separate decision dimension. They are never averaged
with fixed decode, cold prefill, AgentX, capacity, or quality.

## Validity and reporting

Objective invalidation conditions are wrong token shape, failed requests,
cancellation, failure to reach target concurrency, nonempty queue, prefill in a
decode window, counter reset, cached tokens in a cold-prefill cell, missing
records, or an insufficient equal-context window. A result is never discarded
because its performance is surprising or unfavorable.

Public results state the image digest, source revisions, model and hardware,
server command/configuration, AIPerf revision, workload shape, sampling
settings, supported concurrency, and all run-level engine, acceptance, prefill,
and ITL values. Request latency may be included as a separately labeled
end-to-end workload result. TTFT is omitted because it is not an independent
measurement. Internal endpoints, credentials, cluster names, and registry
locations are omitted because they do not affect reproduction.
