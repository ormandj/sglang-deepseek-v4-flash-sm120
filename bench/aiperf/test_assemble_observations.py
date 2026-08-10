from __future__ import annotations

import json

import pytest
from assemble_observations import AssemblyError, assemble


def _panel() -> dict:
    return {
        "pilot": [
            {
                "pair": f"p0{index}",
                "aiperf_random_seed": index,
                "sampling_seed": index,
                "order": (
                    ["baseline", "candidate"]
                    if index % 2
                    else ["candidate", "baseline"]
                ),
            }
            for index in range(1, 4)
        ]
    }


def _write_side(root, row: dict, role: str, value: float) -> None:
    directory = root / "pilot" / f"{row['pair']}-{role}"
    directory.mkdir(parents=True)
    document = {
        "campaign": "test-campaign",
        "phase": "pilot",
        "pair": row["pair"],
        "role": role,
        "created_at_utc": (
            "2026-08-10T01:00:00+00:00"
            if row["order"].index(role) == 0
            else "2026-08-10T02:00:00+00:00"
        ),
        "process_instance_id": (
            f"{row['pair']}-{role}".encode().hex().ljust(64, "0")[:64]
        ),
        "order": row["order"],
        "order_index": row["order"].index(role),
        "aiperf_random_seed": row["aiperf_random_seed"],
        "sampling_seed": row["sampling_seed"],
        "cells": {
            "decode_c1_tokens_per_second": {
                "direction": "higher",
                "mde_fraction": 0.01,
                "value": value,
            }
        },
    }
    (directory / "block-results.json").write_text(
        json.dumps(document), encoding="utf-8"
    )


def test_assembly_preserves_committed_pair_order(tmp_path) -> None:
    panel = _panel()
    for index, row in enumerate(panel["pilot"], 1):
        _write_side(tmp_path, row, "baseline", 100 + index)
        _write_side(tmp_path, row, "candidate", 102 + index)

    result = assemble(tmp_path, panel, "pilot", None)

    observations = result["cells"]["decode_c1_tokens_per_second"]["pilot"]
    assert [item["pair"] for item in observations] == ["p01", "p02", "p03"]
    assert observations[0] == {
        "pair": "p01",
        "baseline": 101,
        "candidate": 103,
    }


def test_assembly_rejects_changed_seed(tmp_path) -> None:
    panel = _panel()
    for row in panel["pilot"]:
        _write_side(tmp_path, row, "baseline", 100)
        _write_side(tmp_path, row, "candidate", 101)
    path = tmp_path / "pilot" / "p02-candidate" / "block-results.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["sampling_seed"] = 999
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(AssemblyError, match="sampling_seed"):
        assemble(tmp_path, panel, "pilot", None)
