#!/usr/bin/env python3
"""Build the fixed paired-analysis vector from one completed block."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


class ResultError(RuntimeError):
    """Raised when an expected cell is missing or invalid."""


DECODE_CELLS = (1, 2, 4, 8, 16, 32)
PREFILL_CELLS = (
    ("8k_c1", "8k-c1", 8192, 1),
    ("8k_c2", "8k-c2", 8192, 2),
    ("8k_c4", "8k-c4", 8192, 4),
    ("64k_c1", "64k-c1", 65536, 1),
    ("128k_c1", "128k-c1", 131072, 1),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ResultError(f"{path} does not contain a JSON object")
    return document


def _finite_positive(value: Any, source: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ResultError(f"{source} is not numeric") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise ResultError(f"{source} is not positive and finite")
    return numeric


def _require_valid(document: dict[str, Any], source: str) -> None:
    if document.get("validation", {}).get("valid") is not True:
        raise ResultError(f"{source} is not a valid analyzed cell")


def build(block: Path) -> dict[str, Any]:
    manifest = _load(block / "block-manifest.json")
    cells: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}

    for concurrency in DECODE_CELLS:
        relative = Path("decode") / f"c{concurrency}" / "decode-analysis.json"
        analysis = _load(block / relative)
        _require_valid(analysis, str(relative))
        if analysis.get("decode", {}).get("target_concurrency") != concurrency:
            raise ResultError(f"{relative} has the wrong concurrency")
        name = f"decode_c{concurrency}_tokens_per_second"
        cells[name] = {
            "direction": "higher",
            "mde_fraction": 0.01,
            "value": _finite_positive(
                analysis["decode"]["tokens_per_second_ols"], str(relative)
            ),
        }
        sources[name] = str(relative)

    for label, directory, target_isl, concurrency in PREFILL_CELLS:
        relative = Path("prefill") / directory / "prefill-analysis.json"
        analysis = _load(block / relative)
        _require_valid(analysis, str(relative))
        cell = analysis.get("cell", {})
        if (
            cell.get("target_isl") != target_isl
            or cell.get("target_concurrency") != concurrency
        ):
            raise ResultError(f"{relative} has the wrong workload shape")
        throughput_name = f"prefill_{label}_tokens_per_second"
        ttft_name = f"prefill_{label}_median_ttft_ms"
        cells[throughput_name] = {
            "direction": "higher",
            "mde_fraction": 0.03,
            "value": _finite_positive(
                analysis["requests"]["aggregate_prompt_tokens_per_second"],
                str(relative),
            ),
        }
        cells[ttft_name] = {
            "direction": "lower",
            "mde_fraction": 0.03,
            "value": _finite_positive(
                analysis["requests"]["time_to_first_token_ms"]["median"],
                str(relative),
            ),
        }
        sources[throughput_name] = str(relative)
        sources[ttft_name] = str(relative)

    return {
        "schema_version": "1.0",
        "campaign": manifest.get("campaign"),
        "phase": manifest.get("phase"),
        "pair": manifest.get("pair"),
        "role": manifest.get("role"),
        "created_at_utc": manifest.get("created_at_utc"),
        "process_instance_id": manifest.get("process_instance_id"),
        "order": manifest.get("order"),
        "order_index": manifest.get("order_index"),
        "aiperf_random_seed": manifest.get("aiperf_random_seed"),
        "sampling_seed": manifest.get("sampling_seed"),
        "cells": cells,
        "sources": sources,
    }


def main() -> int:
    args = _parse_args()
    try:
        if args.output.exists():
            raise ResultError(f"output already exists: {args.output}")
        result = build(args.block)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (ResultError, OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
