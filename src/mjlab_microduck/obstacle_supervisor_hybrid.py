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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center-checkpoint", required=True, type=Path)
    parser.add_argument("--lateral-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lateral-gate-m", type=float, default=0.06)
    args = parser.parse_args()
    compose_lateral_gated_supervisor(
        args.center_checkpoint,
        args.lateral_checkpoint,
        args.output,
        lateral_gate_m=args.lateral_gate_m,
    )


if __name__ == "__main__":
    main()
