#!/usr/bin/env python3
"""Compare matched baseline and candidate turnover-gate summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from compare_engine_gates import ComparisonError, _paired_change


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ComparisonError(f"{path} does not contain a JSON object")
    return document


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if baseline.get("mode") != candidate.get("mode"):
        raise ComparisonError("gate modes differ")
    if baseline.get("engine") != candidate.get("engine"):
        raise ComparisonError("gate engines differ")
    baseline_cells = baseline.get("turnover", {})
    candidate_cells = candidate.get("turnover", {})
    if set(baseline_cells) != set(candidate_cells):
        raise ComparisonError("turnover cell sets differ")
    if not baseline_cells:
        raise ComparisonError("turnover gates have no cells")

    metrics = (
        "output_tokens_per_second",
        "median_ttft_ms",
        "median_itl_ms",
        "requests_per_prefill_pass",
    )
    cells: dict[str, Any] = {}
    for cell in sorted(baseline_cells):
        if baseline_cells[cell].get("shape") != candidate_cells[cell].get("shape"):
            raise ComparisonError(f"turnover shapes differ for {cell}")
        baseline_rows = baseline_cells[cell].get("repetitions", [])
        candidate_rows = candidate_cells[cell].get("repetitions", [])
        cells[cell] = {
            metric: _paired_change(baseline_rows, candidate_rows, metric)
            for metric in metrics
        }

    return {
        "schema_version": "1.0",
        "mode": baseline["mode"],
        "engine": baseline.get("engine"),
        "baseline_build_id": baseline.get("build_id"),
        "candidate_build_id": candidate.get("build_id"),
        "interpretation": (
            "paired same-process prompt-path effect sizes; positive throughput or "
            "batching change is higher, while positive latency change is slower; "
            "evaluate every concurrency and metric separately"
        ),
        "turnover": cells,
    }


def main() -> int:
    args = _parse_args()
    try:
        result = compare(_load(args.baseline), _load(args.candidate))
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            if args.output.exists():
                raise ComparisonError(f"output already exists: {args.output}")
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except (ComparisonError, OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
