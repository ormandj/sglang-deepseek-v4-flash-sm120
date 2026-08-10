#!/usr/bin/env python3
"""Validate a committed seed row and create one performance-block manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ManifestError(RuntimeError):
    """Raised when a block does not match the frozen campaign contract."""


REQUIRED_ENVIRONMENT = {
    "build_id": "BENCH_BUILD_ID",
    "image_ref": "BENCH_IMAGE_REF",
    "gitops_revision": "BENCH_GITOPS_REVISION",
    "project_revision": "BENCH_PROJECT_REVISION",
    "aiperf_revision": "AIPERF_REVISION",
    "model_revision": "BENCH_MODEL_REVISION",
    "model_name": "MODEL_NAME",
    "tokenizer_path": "TOKENIZER_PATH",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "final"), required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--role", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--field",
        choices=("aiperf_random_seed", "sampling_seed", "order_index"),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ManifestError(f"{path} does not contain a JSON object")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identifier(name: str, value: str) -> str:
    if not value or any(not (char.isalnum() or char in "._-") for char in value):
        raise ManifestError(
            f"{name} may contain only letters, digits, dot, underscore, and hyphen"
        )
    return value


def _process_instance_id() -> str:
    try:
        stat = Path("/proc/1/stat").read_text(encoding="utf-8")
        fields_after_command = stat[stat.rfind(")") + 2 :].split()
        start_ticks = fields_after_command[19]
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        )
        hostname = os.environ["HOSTNAME"]
    except (IndexError, KeyError, OSError) as exc:
        raise ManifestError("cannot identify the serving process instance") from exc
    identity = f"{hostname}\0{boot_id.strip()}\0{start_ticks}".encode()
    return hashlib.sha256(identity).hexdigest()


def build_manifest(
    *,
    panel: dict[str, Any],
    lock: dict[str, Any],
    phase: str,
    pair: str,
    role: str,
    campaign: str,
    configs: list[Path],
    environment: dict[str, str],
    process_instance_id: str,
    panel_path: Path,
    lock_path: Path,
) -> dict[str, Any]:
    _identifier("campaign", campaign)
    _identifier("pair", pair)
    try:
        matching = [row for row in panel[phase] if row["pair"] == pair]
    except (KeyError, TypeError) as exc:
        raise ManifestError(f"seed panel has no valid {phase} rows") from exc
    if len(matching) != 1:
        raise ManifestError(
            f"seed panel contains {len(matching)} rows for {phase} pair {pair}"
        )
    row = matching[0]
    try:
        order = [str(value) for value in row["order"]]
        aiperf_seed = int(row["aiperf_random_seed"])
        sampling_seed = int(row["sampling_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"seed row {pair} is malformed") from exc
    if order not in (["baseline", "candidate"], ["candidate", "baseline"]):
        raise ManifestError(f"seed row {pair} has invalid order {order}")
    if aiperf_seed < 0 or sampling_seed < 0:
        raise ManifestError(f"seed row {pair} contains a negative seed")

    try:
        locked_revision = str(lock["commit"])
    except (KeyError, TypeError) as exc:
        raise ManifestError("AIPerf lock has no commit") from exc
    if environment["aiperf_revision"] != locked_revision:
        raise ManifestError(
            "AIPerf revision does not match the lock: "
            f"{environment['aiperf_revision']} != {locked_revision}"
        )

    config_hashes: dict[str, str] = {}
    for path in configs:
        if not path.is_file():
            raise ManifestError(f"config does not exist: {path}")
        if path.name in config_hashes:
            raise ManifestError(f"config name is not unique: {path.name}")
        config_hashes[path.name] = _sha256(path)

    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "campaign": campaign,
        "phase": phase,
        "pair": pair,
        "role": role,
        "order": order,
        "order_index": order.index(role),
        "aiperf_random_seed": aiperf_seed,
        "sampling_seed": sampling_seed,
        "process_instance_id": process_instance_id,
        "provenance": environment,
        "contract": {
            "seed_panel_sha256": _sha256(panel_path),
            "aiperf_lock_sha256": _sha256(lock_path),
            "config_sha256": config_hashes,
            "aiperf_workers": 1,
            "record_processors": 1,
            "client_placement": "serving_pod_localhost",
            "sampling": {"temperature": 1.0, "top_p": 0.95},
            "decode": {
                "input_tokens": 256,
                "output_tokens_per_request": {
                    "1": 18432,
                    "2": 16384,
                    "4": 12288,
                    "8": 8192,
                    "16": 6144,
                    "32": 4096,
                },
                "settle_seconds": 15,
                "tail_seconds": 3,
                "minimum_plateau_seconds": 30,
            },
            "prefill": {
                "8k_c1": {"input_tokens": 8192, "concurrency": 1, "requests": 40},
                "8k_c2": {"input_tokens": 8192, "concurrency": 2, "requests": 40},
                "8k_c4": {"input_tokens": 8192, "concurrency": 4, "requests": 40},
                "64k_c1": {
                    "input_tokens": 65536,
                    "concurrency": 1,
                    "requests": 10,
                },
                "128k_c1": {
                    "input_tokens": 131072,
                    "concurrency": 1,
                    "requests": 5,
                },
            },
        },
    }


def main() -> int:
    args = _parse_args()
    try:
        environment: dict[str, str] = {}
        for key, variable in REQUIRED_ENVIRONMENT.items():
            value = os.environ.get(variable, "")
            if not value or value == "unknown":
                raise ManifestError(f"{variable} must be set to a known value")
            environment[key] = value
        manifest = build_manifest(
            panel=_load_json(args.panel),
            lock=_load_json(args.lock),
            phase=args.phase,
            pair=args.pair,
            role=args.role,
            campaign=args.campaign,
            configs=args.config,
            environment=environment,
            process_instance_id=_process_instance_id(),
            panel_path=args.panel,
            lock_path=args.lock,
        )
        if args.field:
            print(manifest[args.field])
        if args.output:
            if args.output.exists():
                raise ManifestError(f"manifest already exists: {args.output}")
            args.output.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if not args.field and not args.output:
            raise ManifestError("one of --field or --output is required")
    except (ManifestError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
