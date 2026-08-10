from __future__ import annotations

from analyze_pairs import compare, plan


def _seed_panel() -> dict:
    return {
        "pilot": [{"pair": f"p0{i}"} for i in range(1, 4)],
        "final": [{"pair": f"f{i:02d}"} for i in range(1, 11)],
    }


def _observations() -> dict:
    return {
        "cells": {
            "decode_c1": {
                "direction": "higher",
                "mde_fraction": 0.01,
                "pilot": [
                    {"pair": "p01", "baseline": 100.0, "candidate": 102.0},
                    {"pair": "p02", "baseline": 101.0, "candidate": 103.0},
                    {"pair": "p03", "baseline": 99.0, "candidate": 101.0},
                ],
            },
            "prefill_ttft": {
                "direction": "lower",
                "mde_fraction": 0.03,
                "pilot": [
                    {"pair": "p01", "baseline": 1000.0, "candidate": 970.0},
                    {"pair": "p02", "baseline": 1010.0, "candidate": 980.0},
                    {"pair": "p03", "baseline": 990.0, "candidate": 960.0},
                ],
            },
        }
    }


def test_plan_uses_paired_pilot_variance_and_minimum_five() -> None:
    result = plan(_observations(), _seed_panel())
    assert result["powered_within_planned_maximum"] is True
    assert result["planned_final_pairs"] == 5
    assert result["cells"]["decode_c1"]["required_pairs_for_90_percent_power"] == 5


def test_compare_reports_improvement_and_stable_sensitivity() -> None:
    observations = _observations()
    plan_document = plan(observations, _seed_panel())
    for cell in observations["cells"].values():
        cell["final"] = []
    for index in range(1, 6):
        pair = f"f{index:02d}"
        observations["cells"]["decode_c1"]["final"].append(
            {"pair": pair, "baseline": 100.0, "candidate": 103.0}
        )
        observations["cells"]["prefill_ttft"]["final"].append(
            {"pair": pair, "baseline": 1000.0, "candidate": 950.0}
        )
    result = compare(observations, _seed_panel(), plan_document)
    decode = result["cells"]["decode_c1"]
    assert decode["reported_classification"] == "clear_improvement"
    assert decode["sensitivity_stable"] is True
    assert abs(decode["estimate"]["geometric_improvement_percent"] - 3.0) < 1e-9


def test_leave_one_out_marks_path_dependent_result_unstable() -> None:
    observations = _observations()
    plan_document = plan(observations, _seed_panel())
    for cell in observations["cells"].values():
        cell["final"] = []
    candidate_values = [104.0, 104.0, 104.0, 104.0, 85.0]
    for index, candidate in enumerate(candidate_values, 1):
        pair = f"f{index:02d}"
        observations["cells"]["decode_c1"]["final"].append(
            {"pair": pair, "baseline": 100.0, "candidate": candidate}
        )
        observations["cells"]["prefill_ttft"]["final"].append(
            {"pair": pair, "baseline": 1000.0, "candidate": 950.0}
        )
    result = compare(observations, _seed_panel(), plan_document)
    assert result["cells"]["decode_c1"]["reported_classification"] == "unstable"
