from __future__ import annotations

import pytest
from compare_engine_gates import ComparisonError, compare


def _summary(build_id: str, scale: float = 1.0) -> dict:
    return {
        "mode": "quick",
        "build_id": build_id,
        "decode": {
            "c1": {
                "repetitions": [
                    {
                        "id": "r01",
                        "forward_passes_per_second": 50 * scale,
                        "synthetic_decode_tokens_per_second": 300 * scale,
                        "output_tokens_per_forward_per_request": 6 * scale,
                    },
                    {
                        "id": "r02",
                        "forward_passes_per_second": 60 * scale,
                        "synthetic_decode_tokens_per_second": 330 * scale,
                        "output_tokens_per_forward_per_request": 5.5 * scale,
                    },
                ]
            }
        },
        "prefill": {
            "8k-c1": {
                "prompt_tokens_per_second": 8000 * scale,
                "median_ttft_ms": 1000 / scale,
            }
        },
    }


def test_compare_reports_matched_prompt_effects() -> None:
    result = compare(_summary("base"), _summary("candidate", 1.1))
    forward = result["decode"]["c1"]["forward_passes_per_second"]
    assert forward["geometric_mean_change_percent"] == pytest.approx(10)
    assert forward["median_pair_change_percent"] == pytest.approx(10)
    assert result["prefill"]["8k-c1"]["prompt_tokens_per_second"][
        "change_percent"
    ] == pytest.approx(10)
    assert set(result["prefill"]["8k-c1"]) == {"prompt_tokens_per_second"}
    assert result["schema_version"] == "1.2"


def test_compare_accepts_legacy_useful_token_field_names() -> None:
    baseline = _summary("base")
    for row in baseline["decode"]["c1"]["repetitions"]:
        row["useful_tokens_per_second"] = row.pop(
            "synthetic_decode_tokens_per_second"
        )
        row["useful_tokens_per_forward_per_request"] = row.pop(
            "output_tokens_per_forward_per_request"
        )

    result = compare(baseline, _summary("candidate", 1.1))

    assert result["decode"]["c1"]["synthetic_decode_tokens_per_second"][
        "geometric_mean_change_percent"
    ] == pytest.approx(10)


def test_compare_rejects_different_modes() -> None:
    candidate = _summary("candidate")
    candidate["mode"] = "qualification"
    with pytest.raises(ComparisonError, match="modes differ"):
        compare(_summary("base"), candidate)


def test_compare_rejects_unmatched_repetitions() -> None:
    candidate = _summary("candidate")
    candidate["decode"]["c1"]["repetitions"].pop()
    with pytest.raises(ComparisonError, match="repetition IDs differ"):
        compare(_summary("base"), candidate)


def test_compare_rejects_duplicate_repetition_ids() -> None:
    candidate = _summary("candidate")
    candidate["decode"]["c1"]["repetitions"][1]["id"] = "r01"
    with pytest.raises(
        ComparisonError, match="candidate repetition IDs are not unique"
    ):
        compare(_summary("base"), candidate)


def test_compare_can_retain_explicit_capacity_mismatch() -> None:
    baseline = _summary("base")
    baseline["decode"]["c32"] = baseline["decode"]["c1"]
    result = compare(
        baseline,
        _summary("candidate"),
        allow_decode_cell_mismatch=True,
    )
    assert result["decode_capacity"] == {
        "common_cells": ["c1"],
        "baseline_only_cells": ["c32"],
        "candidate_only_cells": [],
    }
