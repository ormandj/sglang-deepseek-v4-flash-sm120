from __future__ import annotations

import json

import pytest
from build_block_results import DECODE_CELLS, PREFILL_CELLS, ResultError, build


def _write_json(path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _block(tmp_path):
    _write_json(
        tmp_path / "block-manifest.json",
        {
            "campaign": "test",
            "phase": "pilot",
            "pair": "p01",
            "role": "candidate",
            "created_at_utc": "2026-08-10T00:00:00+00:00",
            "process_instance_id": "a" * 64,
            "order": ["baseline", "candidate"],
            "order_index": 1,
            "aiperf_random_seed": 1,
            "sampling_seed": 2,
        },
    )
    for concurrency in DECODE_CELLS:
        _write_json(
            tmp_path / "decode" / f"c{concurrency}" / "decode-analysis.json",
            {
                "validation": {"valid": True},
                "decode": {
                    "target_concurrency": concurrency,
                    "tokens_per_second_ols": 100.0 * concurrency,
                },
            },
        )
    for _, directory, target_isl, concurrency in PREFILL_CELLS:
        _write_json(
            tmp_path / "prefill" / directory / "prefill-analysis.json",
            {
                "validation": {"valid": True},
                "cell": {
                    "target_isl": target_isl,
                    "target_concurrency": concurrency,
                },
                "requests": {
                    "aggregate_prompt_tokens_per_second": 8000.0,
                    "time_to_first_token_ms": {"median": 1000.0},
                },
            },
        )
    return tmp_path


def test_build_extracts_fixed_performance_vector(tmp_path) -> None:
    result = build(_block(tmp_path))

    assert len(result["cells"]) == 16
    assert result["cells"]["decode_c32_tokens_per_second"]["value"] == 3200.0
    assert result["cells"]["prefill_128k_c1_median_ttft_ms"] == {
        "direction": "lower",
        "mde_fraction": 0.03,
        "value": 1000.0,
    }
    assert result["process_instance_id"] == "a" * 64


def test_build_rejects_analyzer_invalidity(tmp_path) -> None:
    block = _block(tmp_path)
    path = block / "decode" / "c4" / "decode-analysis.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["validation"]["valid"] = False
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ResultError, match="not a valid"):
        build(block)
