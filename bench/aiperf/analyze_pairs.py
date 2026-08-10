#!/usr/bin/env python3
"""Plan and analyze paired serving-build comparisons on the log-ratio scale."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from scipy.stats import nct, t


class AnalysisError(RuntimeError):
    """Raised when paired observations violate the frozen design."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "compare"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--observations", type=Path, required=True)
        subparser.add_argument("--seed-panel", type=Path, required=True)
        subparser.add_argument("--output", type=Path)
        if command == "compare":
            subparser.add_argument("--plan", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise AnalysisError(f"{path} does not contain a JSON object")
    return document


def _seed_pair_ids(seed_panel: dict[str, Any], phase: str) -> list[str]:
    try:
        rows = seed_panel[phase]
        pair_ids = [str(row["pair"]) for row in rows]
    except (KeyError, TypeError) as exc:
        raise AnalysisError(f"seed panel has no valid {phase} rows") from exc
    if len(pair_ids) != len(set(pair_ids)):
        raise AnalysisError(f"seed panel {phase} pair IDs are not unique")
    return pair_ids


def _cell_definition(cell_name: str, cell: dict[str, Any]) -> tuple[str, float]:
    try:
        direction = str(cell["direction"])
        mde_fraction = float(cell["mde_fraction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError(f"cell {cell_name} has an invalid definition") from exc
    if direction not in ("higher", "lower"):
        raise AnalysisError(f"cell {cell_name} direction must be higher or lower")
    if not 0 < mde_fraction < 1:
        raise AnalysisError(f"cell {cell_name} MDE must be between zero and one")
    return direction, mde_fraction


def _paired_log_ratios(
    cell_name: str,
    observations: Any,
    *,
    direction: str,
    expected_pair_ids: list[str],
) -> tuple[list[float], list[dict[str, float | str]]]:
    if not isinstance(observations, list):
        raise AnalysisError(f"cell {cell_name} observations are not a list")
    by_pair: dict[str, dict[str, Any]] = {}
    for observation in observations:
        try:
            pair_id = str(observation["pair"])
        except (KeyError, TypeError) as exc:
            raise AnalysisError(f"cell {cell_name} has an observation without a pair") from exc
        if pair_id in by_pair:
            raise AnalysisError(f"cell {cell_name} repeats pair {pair_id}")
        by_pair[pair_id] = observation
    actual_pair_ids = list(by_pair)
    if actual_pair_ids != expected_pair_ids:
        raise AnalysisError(
            f"cell {cell_name} pair order is {actual_pair_ids}; "
            f"expected {expected_pair_ids}"
        )

    values: list[float] = []
    normalized: list[dict[str, float | str]] = []
    for pair_id in expected_pair_ids:
        observation = by_pair[pair_id]
        try:
            baseline = float(observation["baseline"])
            candidate = float(observation["candidate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalysisError(
                f"cell {cell_name} pair {pair_id} has invalid values"
            ) from exc
        if not math.isfinite(baseline) or baseline <= 0:
            raise AnalysisError(
                f"cell {cell_name} pair {pair_id} baseline is not positive and finite"
            )
        if not math.isfinite(candidate) or candidate <= 0:
            raise AnalysisError(
                f"cell {cell_name} pair {pair_id} candidate is not positive and finite"
            )
        ratio = candidate / baseline if direction == "higher" else baseline / candidate
        log_ratio = math.log(ratio)
        values.append(log_ratio)
        normalized.append(
            {
                "pair": pair_id,
                "baseline": baseline,
                "candidate": candidate,
                "improvement_log_ratio": log_ratio,
                "improvement_percent": math.expm1(log_ratio) * 100,
            }
        )
    return values, normalized


def _two_sided_paired_t_power(n: int, standardized_effect: float) -> float:
    if n < 2:
        raise ValueError("paired t power requires at least two pairs")
    if standardized_effect < 0:
        raise ValueError("standardized effect cannot be negative")
    critical = float(t.ppf(0.975, n - 1))
    noncentrality = standardized_effect * math.sqrt(n)
    return float(
        nct.cdf(-critical, n - 1, noncentrality)
        + nct.sf(critical, n - 1, noncentrality)
    )


def _required_pairs(
    standard_deviation: float,
    mde_fraction: float,
    *,
    minimum: int = 5,
    search_limit: int = 1000,
) -> tuple[int, float, float]:
    if standard_deviation == 0:
        return minimum, 1.0, 1.0
    effect = math.log1p(mde_fraction) / standard_deviation
    powers = {
        n: _two_sided_paired_t_power(n, effect)
        for n in range(minimum, search_limit + 1)
    }
    required = next((n for n, power in powers.items() if power >= 0.90), None)
    if required is None:
        raise AnalysisError(
            f"90% power was not reached within {search_limit} paired blocks"
        )
    return required, powers[minimum], powers[10]


def plan(observations: dict[str, Any], seed_panel: dict[str, Any]) -> dict[str, Any]:
    try:
        cells = observations["cells"]
    except (KeyError, TypeError) as exc:
        raise AnalysisError("observations have no cells object") from exc
    if not isinstance(cells, dict) or not cells:
        raise AnalysisError("observations cells must be a nonempty object")
    pilot_pair_ids = _seed_pair_ids(seed_panel, "pilot")
    if len(pilot_pair_ids) != 3:
        raise AnalysisError("the frozen design requires exactly three pilot pairs")

    results: dict[str, Any] = {}
    required_counts: list[int] = []
    for cell_name, cell in cells.items():
        if not isinstance(cell, dict):
            raise AnalysisError(f"cell {cell_name} is not an object")
        direction, mde_fraction = _cell_definition(cell_name, cell)
        values, normalized = _paired_log_ratios(
            cell_name,
            cell.get("pilot"),
            direction=direction,
            expected_pair_ids=pilot_pair_ids,
        )
        standard_deviation = statistics.stdev(values)
        required, power_at_five, power_at_ten = _required_pairs(
            standard_deviation, mde_fraction
        )
        required_counts.append(required)
        results[cell_name] = {
            "direction": direction,
            "mde_fraction": mde_fraction,
            "pilot_pairs": normalized,
            "pilot_geometric_improvement_percent": math.expm1(
                statistics.fmean(values)
            )
            * 100,
            "pilot_log_ratio_sample_stddev": standard_deviation,
            "required_pairs_for_90_percent_power": required,
            "power_at_five_pairs": power_at_five,
            "power_at_ten_pairs": power_at_ten,
        }

    maximum_required = max(required_counts)
    planned_pairs = max(5, min(10, maximum_required))
    return {
        "schema_version": "1.0",
        "alpha_two_sided": 0.05,
        "target_power": 0.90,
        "minimum_final_pairs": 5,
        "planned_maximum_pairs": 10,
        "maximum_required_pairs": maximum_required,
        "planned_final_pairs": planned_pairs,
        "powered_within_planned_maximum": maximum_required <= 10,
        "cells": results,
    }


def _classification(
    mean: float, low: float, high: float, mde_log: float
) -> str:
    if low > 0 and mean >= mde_log:
        return "clear_improvement"
    if high < 0 and mean <= -mde_log:
        return "clear_regression"
    if low >= -mde_log and high <= mde_log:
        return "practically_equivalent"
    return "inconclusive"


def _estimate(values: list[float], mde_fraction: float) -> dict[str, Any]:
    if len(values) < 2:
        raise AnalysisError("a paired estimate requires at least two pairs")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    standard_error = standard_deviation / math.sqrt(len(values))
    half_width = float(t.ppf(0.975, len(values) - 1)) * standard_error
    low = mean - half_width
    high = mean + half_width
    mde_log = math.log1p(mde_fraction)
    return {
        "pairs": len(values),
        "geometric_improvement_percent": math.expm1(mean) * 100,
        "ci95_improvement_percent": [
            math.expm1(low) * 100,
            math.expm1(high) * 100,
        ],
        "mean_improvement_log_ratio": mean,
        "log_ratio_sample_stddev": standard_deviation,
        "classification": _classification(mean, low, high, mde_log),
    }


def compare(
    observations: dict[str, Any], seed_panel: dict[str, Any], plan_document: dict[str, Any]
) -> dict[str, Any]:
    if plan_document.get("powered_within_planned_maximum") is not True:
        raise AnalysisError("plan is not powered within its planned maximum")
    try:
        planned_pairs = int(plan_document["planned_final_pairs"])
        cells = observations["cells"]
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError("plan or observations are malformed") from exc
    final_pair_ids = _seed_pair_ids(seed_panel, "final")[:planned_pairs]

    results: dict[str, Any] = {}
    for cell_name, plan_cell in plan_document.get("cells", {}).items():
        if cell_name not in cells:
            raise AnalysisError(f"final observations omit planned cell {cell_name}")
        direction, mde_fraction = _cell_definition(cell_name, cells[cell_name])
        if direction != plan_cell.get("direction") or mde_fraction != float(
            plan_cell.get("mde_fraction")
        ):
            raise AnalysisError(f"cell {cell_name} changed after planning")
        values, normalized = _paired_log_ratios(
            cell_name,
            cells[cell_name].get("final"),
            direction=direction,
            expected_pair_ids=final_pair_ids,
        )
        primary = _estimate(values, mde_fraction)
        leave_one_out: list[dict[str, Any]] = []
        for index, pair_id in enumerate(final_pair_ids):
            estimate = _estimate(values[:index] + values[index + 1 :], mde_fraction)
            estimate["omitted_pair"] = pair_id
            leave_one_out.append(estimate)
        classifications = {item["classification"] for item in leave_one_out}
        signs = {
            0 if item["mean_improvement_log_ratio"] == 0 else math.copysign(1, item["mean_improvement_log_ratio"])
            for item in leave_one_out
        }
        sensitivity_stable = (
            classifications == {primary["classification"]}
            and signs
            == {
                0
                if primary["mean_improvement_log_ratio"] == 0
                else math.copysign(1, primary["mean_improvement_log_ratio"])
            }
        )
        results[cell_name] = {
            "direction": direction,
            "mde_fraction": mde_fraction,
            "pairs": normalized,
            "estimate": primary,
            "leave_one_pair_out": leave_one_out,
            "sensitivity_stable": sensitivity_stable,
            "reported_classification": (
                primary["classification"] if sensitivity_stable else "unstable"
            ),
        }
    extra_cells = set(cells) - set(plan_document.get("cells", {}))
    if extra_cells:
        raise AnalysisError(f"final observations add unplanned cells: {sorted(extra_cells)}")
    return {
        "schema_version": "1.0",
        "planned_final_pairs": planned_pairs,
        "cells": results,
    }


def main() -> int:
    args = _parse_args()
    try:
        observations = _load_json(args.observations)
        seed_panel = _load_json(args.seed_panel)
        if args.command == "plan":
            result = plan(observations, seed_panel)
        else:
            result = compare(observations, seed_panel, _load_json(args.plan))
    except (AnalysisError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    if args.command == "plan" and not result["powered_within_planned_maximum"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
