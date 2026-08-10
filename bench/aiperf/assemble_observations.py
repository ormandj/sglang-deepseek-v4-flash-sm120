#!/usr/bin/env python3
"""Assemble completed baseline/candidate blocks into paired observations."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class AssemblyError(RuntimeError):
    """Raised when block provenance or pair ordering is inconsistent."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--seed-panel", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "final"), required=True)
    parser.add_argument("--pair-count", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise AssemblyError(f"{path} does not contain a JSON object")
    return document


def assemble(
    campaign_root: Path,
    seed_panel: dict[str, Any],
    phase: str,
    pair_count: int | None,
) -> dict[str, Any]:
    try:
        rows = seed_panel[phase]
    except (KeyError, TypeError) as exc:
        raise AssemblyError(f"seed panel has no valid {phase} rows") from exc
    if phase == "pilot":
        if pair_count not in (None, 3):
            raise AssemblyError("the frozen pilot requires exactly three pairs")
        selected = rows
    else:
        if pair_count is None or not 5 <= pair_count <= len(rows):
            raise AssemblyError(
                f"final pair count must be between 5 and {len(rows)}"
            )
        selected = rows[:pair_count]

    assembled: dict[str, dict[str, Any]] = {}
    campaign: str | None = None
    process_instances: set[str] = set()
    for row in selected:
        try:
            pair = str(row["pair"])
            order = [str(value) for value in row["order"]]
            aiperf_seed = int(row["aiperf_random_seed"])
            sampling_seed = int(row["sampling_seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AssemblyError("seed panel row is malformed") from exc
        sides: dict[str, dict[str, Any]] = {}
        for role in ("baseline", "candidate"):
            path = campaign_root / phase / f"{pair}-{role}" / "block-results.json"
            side = _load(path)
            expected = {
                "phase": phase,
                "pair": pair,
                "role": role,
                "order": order,
                "order_index": order.index(role),
                "aiperf_random_seed": aiperf_seed,
                "sampling_seed": sampling_seed,
            }
            for key, value in expected.items():
                if side.get(key) != value:
                    raise AssemblyError(
                        f"{path} {key} is {side.get(key)!r}; expected {value!r}"
                    )
            process_instance = side.get("process_instance_id")
            if not isinstance(process_instance, str) or len(process_instance) != 64:
                raise AssemblyError(f"{path} has no valid process instance ID")
            if process_instance in process_instances:
                raise AssemblyError(
                    f"{path} reuses serving process instance {process_instance}"
                )
            process_instances.add(process_instance)
            if campaign is None:
                campaign = str(side.get("campaign"))
            elif side.get("campaign") != campaign:
                raise AssemblyError(f"{path} belongs to a different campaign")
            sides[role] = side

        try:
            timestamps = {
                side_role: datetime.fromisoformat(str(sides[side_role]["created_at_utc"]))
                for side_role in ("baseline", "candidate")
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise AssemblyError(f"pair {pair} has an invalid block timestamp") from exc
        if timestamps[order[0]] >= timestamps[order[1]]:
            raise AssemblyError(
                f"pair {pair} ran in timestamp order "
                f"{min(timestamps, key=timestamps.get)}, expected {order[0]} first"
            )

        baseline_cells = sides["baseline"].get("cells")
        candidate_cells = sides["candidate"].get("cells")
        if not isinstance(baseline_cells, dict) or not isinstance(
            candidate_cells, dict
        ):
            raise AssemblyError(f"pair {pair} has no cells object")
        if baseline_cells.keys() != candidate_cells.keys():
            raise AssemblyError(f"pair {pair} baseline/candidate cells differ")
        for name, baseline_cell in baseline_cells.items():
            candidate_cell = candidate_cells[name]
            definition = {
                "direction": baseline_cell.get("direction"),
                "mde_fraction": baseline_cell.get("mde_fraction"),
            }
            if any(candidate_cell.get(key) != value for key, value in definition.items()):
                raise AssemblyError(f"pair {pair} cell {name} definition differs")
            target = assembled.setdefault(name, {**definition, phase: []})
            if any(target.get(key) != value for key, value in definition.items()):
                raise AssemblyError(f"cell {name} definition changes between pairs")
            target[phase].append(
                {
                    "pair": pair,
                    "baseline": baseline_cell.get("value"),
                    "candidate": candidate_cell.get("value"),
                }
            )

    return {
        "schema_version": "1.0",
        "campaign": campaign,
        "phase": phase,
        "pairs": len(selected),
        "cells": assembled,
    }


def main() -> int:
    args = _parse_args()
    try:
        if args.output.exists():
            raise AssemblyError(f"output already exists: {args.output}")
        result = assemble(
            args.campaign_root,
            _load(args.seed_panel),
            args.phase,
            args.pair_count,
        )
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (AssemblyError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
