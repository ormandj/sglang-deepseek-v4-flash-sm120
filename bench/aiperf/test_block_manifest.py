from __future__ import annotations

import json

import pytest
from block_manifest import ManifestError, build_manifest


def _panel() -> dict:
    return {
        "pilot": [
            {
                "pair": "p01",
                "aiperf_random_seed": 10,
                "sampling_seed": 11,
                "order": ["baseline", "candidate"],
            }
        ]
    }


def test_manifest_binds_seed_order_provenance_and_hashes(tmp_path) -> None:
    panel_path = tmp_path / "panel.json"
    lock_path = tmp_path / "lock.json"
    config_path = tmp_path / "config.yaml"
    panel_path.write_text(json.dumps(_panel()), encoding="utf-8")
    lock_path.write_text(json.dumps({"commit": "abc"}), encoding="utf-8")
    config_path.write_text("schemaVersion: '2.0'\n", encoding="utf-8")
    environment = {
        "build_id": "rc3",
        "image_ref": "image@sha256:deadbeef",
        "gitops_revision": "123",
        "project_revision": "456",
        "aiperf_revision": "abc",
        "model_revision": "model-commit",
        "model_name": "model",
        "tokenizer_path": "/models/model",
    }

    result = build_manifest(
        panel=_panel(),
        lock={"commit": "abc"},
        phase="pilot",
        pair="p01",
        role="candidate",
        campaign="baseline-candidate",
        configs=[config_path],
        environment=environment,
        process_instance_id="a" * 64,
        panel_path=panel_path,
        lock_path=lock_path,
    )

    assert result["order_index"] == 1
    assert result["aiperf_random_seed"] == 10
    assert result["sampling_seed"] == 11
    assert result["provenance"] == environment
    assert len(result["contract"]["config_sha256"]["config.yaml"]) == 64


def test_manifest_rejects_unlocked_aiperf_revision(tmp_path) -> None:
    panel_path = tmp_path / "panel.json"
    lock_path = tmp_path / "lock.json"
    panel_path.write_text(json.dumps(_panel()), encoding="utf-8")
    lock_path.write_text(json.dumps({"commit": "abc"}), encoding="utf-8")
    environment = {
        "build_id": "rc3",
        "image_ref": "image",
        "gitops_revision": "123",
        "project_revision": "456",
        "aiperf_revision": "wrong",
        "model_revision": "model-commit",
        "model_name": "model",
        "tokenizer_path": "/models/model",
    }

    with pytest.raises(ManifestError, match="does not match"):
        build_manifest(
            panel=_panel(),
            lock={"commit": "abc"},
            phase="pilot",
            pair="p01",
            role="baseline",
            campaign="baseline-candidate",
            configs=[],
            environment=environment,
            process_instance_id="a" * 64,
            panel_path=panel_path,
            lock_path=lock_path,
        )
