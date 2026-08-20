# Public image repository instructions

## Purpose

This repository publishes the ready-to-run DeepSeek-V4-Flash SM120 image, its
exact source lock, the supported launch contract, and selected reproducible
results. Optimize documentation for a new operator choosing and starting this
image. Keep internal infrastructure, release-candidate deliberation, failed
experiments, and private diagnostic detail out of the public repository.

## Start here

- Inspect branch, status, remotes, revision, and `git worktree list` before
  changing files. Preserve unrelated work and use an isolated worktree when the
  active checkout is not clean.
- Read `README.md`, `RUN.md`, `BENCHMARKS.md`, `release.json`,
  `stack.lock.json`, and `scripts/validate-release.sh` before release or
  documentation changes.
- Use `uv` or `uvx` for Python execution and tooling. Do not invoke `python`,
  `python3`, `pip`, or `pip3` directly.

## Documentation ownership

Each fact has one detailed owner. Other files may summarize it briefly and must
link to the owner instead of copying the full explanation.

| File | Owns | Must not become |
|---|---|---|
| `README.md` | what the image is, measured reasons to use it, current image reference, exact direct-Docker quickstart, core paired result tables, next links | source/PR inventory, experiment log, or exhaustive operations guide |
| `RUN.md` | wrapper options, launch-contract cautions, topology/capacity changes, HiCache, exposure, operational checks | a second copy of the full Docker command or benchmark method |
| `BENCHMARKS.md` | current published performance, capacity, quality, exact configurations, metric definitions, and reproduction entry point | release history or raw profiler notebook |
| `bench/README.md` | benchmark-tool map and which runner to use | a duplicate current results page |
| `bench/aiperf/README.md` | executable frozen benchmark contract | product overview or release notes |
| `CHANGELOG.md` | concise current public release changes and validation | cumulative private candidate history |
| `release.json` / `stack.lock.json` | release identity and exact composition | prose guidance |

Historical public documentation and result panels belong in immutable release
tags. Current main keeps only the current SGLang result pair and the retained
comparison result pair in `bench/results/`. Do not recreate
`PERFORMANCE-HISTORY.md` or copy old release artifacts forward.

## Release synchronization

Treat the following as one contract and update them in the same change:

- candidate tag and cache schema in `release.json`, `stack.lock.json`, the
  launcher default, README quickstart, RUN examples, and current result names;
- runtime environment variables and server flags in
  `examples/serve-dsv4-0731.sh` and the README's canonical Docker command;
- measured context, static-memory fraction, KV capacity, scheduler limit,
  near-limit outcomes, and HiCache claims in README/RUN/BENCHMARKS;
- performance and quality summaries in the JSON artifacts and paired Markdown
  tables;
- user-visible changes and completed qualification in `CHANGELOG.md`.

Never update a result table from memory, a partial run, or a different method.
Published cells require the frozen method's complete matched sample set. Keep
decode, synthetic acceptance-weighted output rate, prefill, capacity, and
quality distinct. Do not use winner/loser language or collapse them into an
overall score.

The direct Docker command is the primary public startup path. The wrapper is
maintained and optional. Preserve the qualified startup environment and flags
as the recommended starting point; when a setting is user-adjustable, explain
its consequence and the verification required after changing it.

## Public-content boundaries

- Publish measured facts and enough configuration to reproduce them, not
  private endpoints, credentials, secret names, internal image references, or
  irrelevant diagnostic data.
- Deep source/carry audit detail belongs in the private project repository. The
  public lock and OCI labels define the shipped identity.
- Candidate and stable SemVer tags are immutable. Do not introduce or document
  a mutable `latest` tag.
- Do not claim TP4/TP8 performance or capacity from TP2 evidence.

## Required checks

Before handing off a documentation or release change, run:

```bash
bash -n scripts/*.sh examples/*.sh
jq -e . release.json stack.lock.json bench/results/*.json >/dev/null
./scripts/validate-release.sh
./scripts/validate-docs.sh
./scripts/verify-patches.sh
git diff --check
```

Also inspect every changed Markdown link and review the rendered tables for
side-by-side SGLang/vLLM columns rather than alternating engine rows.
