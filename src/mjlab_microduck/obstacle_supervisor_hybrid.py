"""Compose exact HC2 center behavior with an HC4-L lateral specialist."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from mjlab_microduck.obstacle_supervisor_bc import (
    HC2_STAGE,
    HC4L_STAGE,
    HC4LH_STAGE,
    HC4R2H_STAGE,
    HC4R2_STAGE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compose_lateral_gated_supervisor(
    center_checkpoint: Path,
    lateral_checkpoint: Path,
    output_path: Path,
    *,
    lateral_gate_m: float = 0.06,
) -> Path:
    """Write a fail-closed HC2/HC4-L composition checkpoint."""
    if lateral_gate_m <= 0.0:
        raise ValueError("lateral_gate_m must be positive")
    center_checkpoint = center_checkpoint.resolve(strict=True)
    lateral_checkpoint = lateral_checkpoint.resolve(strict=True)
    output_path = output_path.resolve()
    manifest_path = output_path.with_suffix(".json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(output_path)
    center = torch.load(center_checkpoint, map_location="cpu", weights_only=False)
    lateral = torch.load(lateral_checkpoint, map_location="cpu", weights_only=False)
    if center.get("stage") != HC2_STAGE or center.get("decision") != (
        "offline-imitation-pass"
    ):
        raise ValueError("center checkpoint is not accepted HC2 imitation")
    if lateral.get("stage") != HC4L_STAGE or lateral.get("decision") != (
        "offline-imitation-pass"
    ):
        raise ValueError("lateral checkpoint is not eligible HC4-L imitation")
    if center.get("source_locomotion_checkpoint_sha256") != lateral.get(
        "source_locomotion_checkpoint_sha256"
    ):
        raise ValueError(
            "center and lateral checkpoints use different locomotion actors"
        )
    if center.get("model_config") != lateral.get("model_config"):
        raise ValueError("center and lateral checkpoints use different model configs")
    if (
        center.get("physical_motion_authorized") is not False
        or lateral.get("physical_motion_authorized") is not False
    ):
        raise ValueError("source checkpoint has invalid physical-motion authority")

    checkpoint = {
        "schema_version": 1,
        "stage": HC4LH_STAGE,
        "decision": "composition-complete-pending-rollout",
        "rollout_acceptance_required": True,
        "lateral_gate_m": lateral_gate_m,
        "model_config": lateral["model_config"],
        "model_state_dict": lateral["model_state_dict"],
        "center_model_state_dict": center["model_state_dict"],
        "source_locomotion_checkpoint_sha256": lateral[
            "source_locomotion_checkpoint_sha256"
        ],
        "center_supervisor_checkpoint": str(center_checkpoint),
        "center_supervisor_checkpoint_sha256": _sha256(center_checkpoint),
        "lateral_supervisor_checkpoint": str(lateral_checkpoint),
        "lateral_supervisor_checkpoint_sha256": _sha256(lateral_checkpoint),
        "physical_motion_authorized": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    manifest = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"model_state_dict", "center_model_state_dict"}
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return output_path


def compose_range_speed_gated_supervisor(
    far_checkpoint: Path,
    near_checkpoint: Path,
    output_path: Path,
    *,
    near_range_gate_m: float = 0.95,
    max_near_nominal_speed_mps: float = 0.40,
) -> Path:
    """Write a fail-closed HC4-LH/HC4-R2 composition checkpoint."""
    if near_range_gate_m <= 0.0:
        raise ValueError("near_range_gate_m must be positive")
    if not 0.0 < max_near_nominal_speed_mps <= 0.8:
        raise ValueError("max_near_nominal_speed_mps must be in (0, 0.8]")
    far_checkpoint = far_checkpoint.resolve(strict=True)
    near_checkpoint = near_checkpoint.resolve(strict=True)
    output_path = output_path.resolve()
    manifest_path = output_path.with_suffix(".json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(output_path)
    far = torch.load(far_checkpoint, map_location="cpu", weights_only=False)
    near = torch.load(near_checkpoint, map_location="cpu", weights_only=False)
    if far.get("stage") != HC4LH_STAGE or far.get("decision") not in {
        "composition-complete-pending-rollout",
        "accepted-simulation",
    }:
        raise ValueError("far checkpoint is not an eligible HC4-LH composition")
    if near.get("stage") != HC4R2_STAGE or near.get("decision") != (
        "offline-imitation-pass"
    ):
        raise ValueError("near checkpoint is not eligible HC4-R2 imitation")
    if far.get("source_locomotion_checkpoint_sha256") != near.get(
        "source_locomotion_checkpoint_sha256"
    ):
        raise ValueError("far and near checkpoints use different locomotion actors")
    if far.get("model_config") != near.get("model_config"):
        raise ValueError("far and near checkpoints use different model configs")
    if (
        far.get("physical_motion_authorized") is not False
        or near.get("physical_motion_authorized") is not False
    ):
        raise ValueError("source checkpoint has invalid physical-motion authority")
    required_far_fields = {
        "model_state_dict",
        "center_model_state_dict",
        "lateral_gate_m",
    }
    missing = required_far_fields.difference(far)
    if missing:
        raise ValueError(
            f"far checkpoint is missing composition fields: {sorted(missing)}"
        )

    checkpoint = {
        "schema_version": 1,
        "stage": HC4R2H_STAGE,
        "decision": "composition-complete-pending-rollout",
        "rollout_acceptance_required": True,
        "near_range_gate_m": near_range_gate_m,
        "max_near_nominal_speed_mps": max_near_nominal_speed_mps,
        "lateral_gate_m": far["lateral_gate_m"],
        "model_config": far["model_config"],
        "model_state_dict": far["model_state_dict"],
        "center_model_state_dict": far["center_model_state_dict"],
        "near_model_state_dict": near["model_state_dict"],
        "source_locomotion_checkpoint_sha256": far[
            "source_locomotion_checkpoint_sha256"
        ],
        "far_supervisor_checkpoint": str(far_checkpoint),
        "far_supervisor_checkpoint_sha256": _sha256(far_checkpoint),
        "near_supervisor_checkpoint": str(near_checkpoint),
        "near_supervisor_checkpoint_sha256": _sha256(near_checkpoint),
        "invalid_geometry_behavior": "execution-layer-immediate-stop",
        "physical_motion_authorized": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    manifest = {
        key: value
        for key, value in checkpoint.items()
        if not key.endswith("model_state_dict")
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("lateral", "range-speed"), default="lateral"
    )
    parser.add_argument("--center-checkpoint", type=Path)
    parser.add_argument("--lateral-checkpoint", type=Path)
    parser.add_argument("--far-checkpoint", type=Path)
    parser.add_argument("--near-checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lateral-gate-m", type=float, default=0.06)
    parser.add_argument("--near-range-gate-m", type=float, default=0.95)
    parser.add_argument("--max-near-nominal-speed-mps", type=float, default=0.40)
    args = parser.parse_args()
    if args.mode == "lateral":
        if args.center_checkpoint is None or args.lateral_checkpoint is None:
            parser.error(
                "lateral mode requires --center-checkpoint and --lateral-checkpoint"
            )
        compose_lateral_gated_supervisor(
            args.center_checkpoint,
            args.lateral_checkpoint,
            args.output,
            lateral_gate_m=args.lateral_gate_m,
        )
    else:
        if args.far_checkpoint is None or args.near_checkpoint is None:
            parser.error(
                "range-speed mode requires --far-checkpoint and --near-checkpoint"
            )
        compose_range_speed_gated_supervisor(
            args.far_checkpoint,
            args.near_checkpoint,
            args.output,
            near_range_gate_m=args.near_range_gate_m,
            max_near_nominal_speed_mps=args.max_near_nominal_speed_mps,
        )


if __name__ == "__main__":
    main()
