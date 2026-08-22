#!/usr/bin/env python3
"""Validate and summarize a complete turnover/refill gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class SummaryError(RuntimeError):
    """Raised when a turnover artifact set is incomplete or invalid."""


EXPECTED_SHAPES = {
    "release-screen": {"repetitions": 3, "concurrencies": (8,)},
    "screen": {"repetitions": 3, "concurrencies": (1, 2, 4, 8)},
    "qualification": {"repetitions": 5, "concurrencies": (1, 2, 4, 8)},
    "publication": {"repetitions": 5, "concurrencies": (1, 2, 4, 8)},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(EXPECTED_SHAPES), required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise SummaryError(f"{path} does not contain a JSON object")
    return document


def _positive(value: Any, source: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise SummaryError(f"{source} is not numeric") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise SummaryError(f"{source} is not positive and finite")
    return numeric


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    mean = statistics.fmean(values)
    sample_stddev = statistics.stdev(values) if len(values) > 1 else None
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": mean,
        "max": max(values),
        "sample_stddev": sample_stddev,
        "sample_cv": sample_stddev / mean if sample_stddev is not None else None,
    }


def summarize(root: Path, *, mode: str, build_id: str) -> dict[str, Any]:
    expected_repetitions = EXPECTED_SHAPES[mode]["repetitions"]
    expected_concurrencies = EXPECTED_SHAPES[mode]["concurrencies"]
    documents: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for path in sorted(root.glob("c*/r*/turnover-analysis.json")):
        try:
            concurrency = int(path.parents[1].name.removeprefix("c"))
        except ValueError as exc:
            raise SummaryError(f"invalid turnover directory: {path}") from exc
        documents[concurrency].append((path.parent.name, _load(path)))
    if set(documents) != set(expected_concurrencies):
        raise SummaryError(
            f"turnover cells are {sorted(documents)}; expected {list(expected_concurrencies)}"
        )

    cells: dict[str, Any] = {}
    for concurrency in expected_concurrencies:
        runs = documents[concurrency]
        if len(runs) != expected_repetitions:
            raise SummaryError(
                f"C{concurrency} has {len(runs)} repetitions; expected {expected_repetitions}"
            )
        throughput: list[float] = []
        ttft: list[float] = []
        itl: list[float] = []
        refill_batching: list[float] = []
        repetitions: list[dict[str, Any]] = []
        shape: dict[str, int] | None = None
        for repetition, document in runs:
            if document.get("validation", {}).get("valid") is not True:
                raise SummaryError(f"C{concurrency}/{repetition} is invalid")
            document_cell = document.get("cell", {})
            if document_cell.get("target_concurrency") != concurrency:
                raise SummaryError(f"C{concurrency}/{repetition} has the wrong shape")
            try:
                document_shape = {
                    "target_concurrency": concurrency,
                    "expected_requests": int(document_cell["expected_requests"]),
                    "target_isl": int(document_cell["target_isl"]),
                    "target_osl": int(document_cell["target_osl"]),
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise SummaryError(
                    f"C{concurrency}/{repetition} has incomplete shape metadata"
                ) from exc
            if min(document_shape.values()) < 1:
                raise SummaryError(
                    f"C{concurrency}/{repetition} has invalid shape metadata"
                )
            if shape is None:
                shape = document_shape
            elif document_shape != shape:
                raise SummaryError(
                    f"C{concurrency}/{repetition} does not match the other repetitions"
                )
            requests = document.get("requests", {})
            output_rate = _positive(
                requests.get("output_tokens_per_second"),
                f"C{concurrency}/{repetition} output rate",
            )
            median_ttft = _positive(
                requests.get("time_to_first_token_ms", {}).get("median"),
                f"C{concurrency}/{repetition} median TTFT",
            )
            median_itl = _positive(
                requests.get("inter_token_latency_ms", {}).get("median"),
                f"C{concurrency}/{repetition} median ITL",
            )
            effective_batch = _positive(
                document.get("refill_batching", {}).get("requests_per_prefill_pass"),
                f"C{concurrency}/{repetition} requests per prefill pass",
            )
            throughput.append(output_rate)
            ttft.append(median_ttft)
            itl.append(median_itl)
            refill_batching.append(effective_batch)
            repetitions.append(
                {
                    "id": repetition,
                    "output_tokens_per_second": output_rate,
                    "median_ttft_ms": median_ttft,
                    "median_itl_ms": median_itl,
                    "requests_per_prefill_pass": effective_batch,
                }
            )
        cells[f"c{concurrency}"] = {
            "shape": shape,
            "repetitions": repetitions,
            "output_tokens_per_second": _distribution(throughput),
            "median_ttft_ms": _distribution(ttft),
            "median_itl_ms": _distribution(itl),
            "requests_per_prefill_pass": _distribution(refill_batching),
        }

    return {
        "schema_version": "1.0",
        "engine": "sglang",
        "build_id": build_id,
        "mode": mode,
        "interpretation": (
            "same-process closed-loop request-turnover signal; compare each "
            "concurrency independently from steady-state decode and cold prefill"
        ),
        "turnover": cells,
    }


def main() -> int:
    args = _parse_args()
    try:
        result = summarize(args.root, mode=args.mode, build_id=args.build_id)
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            if args.output.exists():
                raise SummaryError(f"output already exists: {args.output}")
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except (SummaryError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
