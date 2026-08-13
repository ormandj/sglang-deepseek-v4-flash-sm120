#!/usr/bin/env python3
"""Compare matched baseline and candidate engine-gate summaries."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


class ComparisonError(RuntimeError):
    """Raised when two engine gates cannot be compared directly."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--allow-decode-cell-mismatch", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ComparisonError(f"{path} does not contain a JSON object")
    return document


def _positive(value: Any, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"{name} is not numeric") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise ComparisonError(f"{name} is not positive and finite")
    return numeric


def _paired_change(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    metric: str,
    legacy_metric: str | None = None,
) -> dict[str, Any]:
    baseline_by_id = {str(row["id"]): row for row in baseline_rows}
    candidate_by_id = {str(row["id"]): row for row in candidate_rows}
    if len(baseline_by_id) != len(baseline_rows):
        raise ComparisonError(f"baseline repetition IDs are not unique for {metric}")
    if len(candidate_by_id) != len(candidate_rows):
        raise ComparisonError(f"candidate repetition IDs are not unique for {metric}")
    if not baseline_by_id or not candidate_by_id:
        raise ComparisonError(f"no repetition pairs exist for {metric}")
    if set(baseline_by_id) != set(candidate_by_id):
        raise ComparisonError(f"repetition IDs differ for {metric}")
    pairs: list[dict[str, Any]] = []
    log_ratios: list[float] = []
    for repetition in sorted(baseline_by_id):
        baseline_row = baseline_by_id[repetition]
        candidate_row = candidate_by_id[repetition]
        baseline = _positive(
            baseline_row.get(metric, baseline_row.get(legacy_metric or "")),
            f"baseline {repetition} {metric}",
        )
        candidate = _positive(
            candidate_row.get(metric, candidate_row.get(legacy_metric or "")),
            f"candidate {repetition} {metric}",
        )
        change = candidate / baseline - 1
        log_ratios.append(math.log(candidate / baseline))
        pairs.append(
            {
                "id": repetition,
                "baseline": baseline,
                "candidate": candidate,
                "change_percent": 100 * change,
            }
        )
    return {
        "pairs": pairs,
        "geometric_mean_change_percent": 100 * math.expm1(statistics.fmean(log_ratios)),
        "median_pair_change_percent": statistics.median(
            row["change_percent"] for row in pairs
        ),
    }


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    allow_decode_cell_mismatch: bool = False,
) -> dict[str, Any]:
    if baseline.get("mode") != candidate.get("mode"):
        raise ComparisonError("gate modes differ")
    baseline_decode = baseline.get("decode", {})
    candidate_decode = candidate.get("decode", {})
    baseline_cells = set(baseline_decode)
    candidate_cells = set(candidate_decode)
    if baseline_cells != candidate_cells and not allow_decode_cell_mismatch:
        raise ComparisonError("decode cell sets differ")
    common_cells = baseline_cells & candidate_cells
    if not common_cells:
        raise ComparisonError("decode gates have no common cells")

    decode: dict[str, Any] = {}
    metrics = (
        ("forward_passes_per_second", None),
        ("synthetic_decode_tokens_per_second", "useful_tokens_per_second"),
        (
            "output_tokens_per_forward_per_request",
            "useful_tokens_per_forward_per_request",
        ),
    )
    for cell in sorted(common_cells):
        baseline_rows = baseline_decode[cell].get("repetitions", [])
        candidate_rows = candidate_decode[cell].get("repetitions", [])
        decode[cell] = {
            metric: _paired_change(
                baseline_rows,
                candidate_rows,
                metric,
                legacy_metric,
            )
            for metric, legacy_metric in metrics
        }

    baseline_prefill = baseline.get("prefill", {})
    candidate_prefill = candidate.get("prefill", {})
    if set(baseline_prefill) != set(candidate_prefill):
        raise ComparisonError("prefill cell sets differ")
    prefill: dict[str, Any] = {}
    for cell in sorted(baseline_prefill):
        base_rate = _positive(
            baseline_prefill[cell].get("prompt_tokens_per_second"),
            f"baseline {cell} prefill rate",
        )
        candidate_rate = _positive(
            candidate_prefill[cell].get("prompt_tokens_per_second"),
            f"candidate {cell} prefill rate",
        )
        prefill[cell] = {
            "prompt_tokens_per_second": {
                "baseline": base_rate,
                "candidate": candidate_rate,
                "change_percent": 100 * (candidate_rate / base_rate - 1),
            }
        }

    return {
        "schema_version": "1.2",
        "mode": baseline["mode"],
        "baseline_engine": baseline.get("engine"),
        "candidate_engine": candidate.get("engine"),
        "baseline_build_id": baseline.get("build_id"),
        "candidate_build_id": candidate.get("build_id"),
        "interpretation": (
            "paired same-process prompt-path effect sizes; no independent-process "
            "confidence claim"
        ),
        "decode": decode,
        "decode_capacity": {
            "common_cells": sorted(common_cells),
            "baseline_only_cells": sorted(baseline_cells - candidate_cells),
            "candidate_only_cells": sorted(candidate_cells - baseline_cells),
        },
        "prefill": prefill,
    }


def main() -> int:
    args = _parse_args()
    try:
        result = compare(
            _load(args.baseline),
            _load(args.candidate),
            allow_decode_cell_mismatch=args.allow_decode_cell_mismatch,
        )
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
