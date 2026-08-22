from __future__ import annotations

import pytest
from compare_engine_gates import ComparisonError
from compare_turnover_gates import compare


def _summary(build_id: str, scale: float = 1.0) -> dict:
    repetitions = [
        {
            "id": "r01",
            "output_tokens_per_second": 100 * scale,
            "median_ttft_ms": 200 / scale,
            "median_itl_ms": 20 / scale,
            "requests_per_prefill_pass": 2 * scale,
        },
        {
            "id": "r02",
            "output_tokens_per_second": 110 * scale,
            "median_ttft_ms": 220 / scale,
            "median_itl_ms": 22 / scale,
            "requests_per_prefill_pass": 2.2 * scale,
        },
    ]
    return {
        "mode": "qualification",
        "engine": "sglang",
        "build_id": build_id,
        "turnover": {
            "c1": {
                "shape": {
                    "target_concurrency": 1,
                    "expected_requests": 16,
                    "target_isl": 256,
                    "target_osl": 256,
                },
                "repetitions": repetitions,
            }
        },
    }


def test_compare_reports_each_turnover_dimension_separately() -> None:
    result = compare(_summary("baseline"), _summary("candidate", 1.1))
    c1 = result["turnover"]["c1"]
    assert c1["output_tokens_per_second"][
        "geometric_mean_change_percent"
    ] == pytest.approx(10)
    assert c1["median_ttft_ms"]["geometric_mean_change_percent"] == pytest.approx(
        100 * (1 / 1.1 - 1)
    )
    assert c1["requests_per_prefill_pass"][
        "median_pair_change_percent"
    ] == pytest.approx(10)


def test_compare_rejects_different_modes() -> None:
    candidate = _summary("candidate")
    candidate["mode"] = "screen"
    with pytest.raises(ComparisonError, match="modes differ"):
        compare(_summary("baseline"), candidate)


def test_compare_rejects_different_cell_sets() -> None:
    candidate = _summary("candidate")
    candidate["turnover"]["c2"] = candidate["turnover"]["c1"]
    with pytest.raises(ComparisonError, match="cell sets differ"):
        compare(_summary("baseline"), candidate)


def test_compare_rejects_different_turnover_shapes() -> None:
    candidate = _summary("candidate")
    candidate["turnover"]["c1"]["shape"]["expected_requests"] = 32
    with pytest.raises(ComparisonError, match="shapes differ for c1"):
        compare(_summary("baseline"), candidate)
