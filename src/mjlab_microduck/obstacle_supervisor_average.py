"""Build an HC3-F supervisor by averaging only HC3-E speed-head updates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

HC3E_STAGE = "HC3E-interaction-speed-PPO"
HC3F_STAGE = "HC3F-seed-averaged-speed-head"
HC3G_STAGE = "HC3G-seed-consensus-speed-head"
SPEED_HEAD_WEIGHT = "network.4.weight"
SPEED_HEAD_BIAS = "network.4.bias"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_exact(
    current: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> bool:
    return current.keys() == reference.keys() and all(
        torch.equal(current[name], reference[name]) for name in current
    )


def _validate_speed_only_state(payload: dict, anchor: dict[str, torch.Tensor]) -> None:
    model = payload["model_state_dict"]
    if model.keys() != anchor.keys():
        raise ValueError("HC3-E model and anchor state dictionaries differ")
    if SPEED_HEAD_WEIGHT not in model or SPEED_HEAD_BIAS not in model:
        raise ValueError("HC3-E checkpoint lacks the expected speed head")
    if model[SPEED_HEAD_WEIGHT].shape[0] != 2 or model[SPEED_HEAD_BIAS].shape != (2,):
        raise ValueError("HC3-E speed head has an unexpected shape")
    for name in model:
        if name not in {SPEED_HEAD_WEIGHT, SPEED_HEAD_BIAS} and not torch.equal(
            model[name], anchor[name]
        ):
            raise ValueError(f"HC3-E modified frozen parameter {name}")
    if not torch.equal(model[SPEED_HEAD_WEIGHT][1], anchor[SPEED_HEAD_WEIGHT][1]):
        raise ValueError("HC3-E modified the frozen yaw weight row")
    if not torch.equal(model[SPEED_HEAD_BIAS][1], anchor[SPEED_HEAD_BIAS][1]):
        raise ValueError("HC3-E modified the frozen yaw bias")


def _aggregate_interaction_speed_checkpoints(
    checkpoint_paths: tuple[Path, ...], output_path: Path, *, consensus_only: bool
) -> Path:
    """Aggregate three or more compatible HC3-E speed heads."""
    if len(checkpoint_paths) < 3:
        raise ValueError("HC3-F requires at least three training-seed checkpoints")
    checkpoint_paths = tuple(path.resolve(strict=True) for path in checkpoint_paths)
    output_path = output_path.resolve()
    manifest_path = output_path.with_suffix(".json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(output_path)
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in checkpoint_paths
    ]
    seeds = [payload.get("seed") for payload in payloads]
    if None in seeds or len(set(seeds)) != len(seeds):
        raise ValueError("HC3-F inputs must have unique training seeds")

    reference = payloads[0]
    if reference.get("completed_iterations") != 1:
        raise ValueError("HC3-F is defined only for corrected HC3-E iteration 1")
    invariant_fields = (
        "action_authority",
        "completed_iterations",
        "min_interaction_speed_mps",
        "model_config",
        "reward_config",
        "source_locomotion_checkpoint_sha256",
        "source_supervisor_checkpoint_sha256",
        "training_cells",
    )
    for payload in payloads:
        if payload.get("stage") != HC3E_STAGE:
            raise ValueError("HC3-F accepts only HC3-E checkpoints")
        if payload.get("decision") != "training-complete-pending-rollout":
            raise ValueError("HC3-E input is not pending deterministic rollout")
        if payload.get("action_authority") != "interaction-speed-only":
            raise ValueError("HC3-E input has incompatible action authority")
        if payload.get("physical_motion_authorized") is not False:
            raise ValueError("HC3-E input has invalid physical-motion authority")
        if any(
            payload.get(field) != reference.get(field) for field in invariant_fields
        ):
            raise ValueError("HC3-E inputs do not share aggregation invariants")

    ppo_config = copy.deepcopy(reference["ppo_config"])
    ppo_config["iterations"] = reference["completed_iterations"]
    ppo_update_config = {
        key: value for key, value in ppo_config.items() if key != "iterations"
    }
    if any(
        {
            key: value
            for key, value in payload["ppo_config"].items()
            if key != "iterations"
        }
        != ppo_update_config
        for payload in payloads
    ):
        raise ValueError("HC3-E inputs do not share PPO update invariants")

    anchor = reference["anchor_model_state_dict"]
    if any(
        not _state_dict_exact(payload["anchor_model_state_dict"], anchor)
        for payload in payloads[1:]
    ):
        raise ValueError("HC3-E inputs do not share one exact HC2 anchor")
    for payload in payloads:
        _validate_speed_only_state(payload, anchor)

    speed_updates = torch.stack(
        [
            torch.cat(
                (
                    payload["model_state_dict"][SPEED_HEAD_WEIGHT][0].flatten()
                    - anchor[SPEED_HEAD_WEIGHT][0].flatten(),
                    payload["model_state_dict"][SPEED_HEAD_BIAS][0].reshape(1)
                    - anchor[SPEED_HEAD_BIAS][0].reshape(1),
                )
            )
            for payload in payloads
        ]
    )
    mean_update = speed_updates.mean(dim=0)
    consensus_mask = torch.logical_or(
        torch.all(speed_updates > 0.0, dim=0),
        torch.all(speed_updates < 0.0, dim=0),
    )
    averaged_model = copy.deepcopy(anchor)
    if consensus_only:
        mean_update = torch.where(
            consensus_mask, mean_update, torch.zeros_like(mean_update)
        )
        averaged_model[SPEED_HEAD_WEIGHT][0] += mean_update[:-1].reshape_as(
            averaged_model[SPEED_HEAD_WEIGHT][0]
        )
        averaged_model[SPEED_HEAD_BIAS][0] += mean_update[-1]
    else:
        averaged_model[SPEED_HEAD_WEIGHT][0] = torch.stack(
            [payload["model_state_dict"][SPEED_HEAD_WEIGHT][0] for payload in payloads]
        ).mean(dim=0)
        averaged_model[SPEED_HEAD_BIAS][0] = torch.stack(
            [payload["model_state_dict"][SPEED_HEAD_BIAS][0] for payload in payloads]
        ).mean(dim=0)
    _validate_speed_only_state({"model_state_dict": averaged_model}, anchor)
    parameter_count = averaged_model[SPEED_HEAD_WEIGHT][0].numel() + 1

    checkpoint = {
        "schema_version": 1,
        "stage": HC3G_STAGE if consensus_only else HC3F_STAGE,
        "decision": "aggregation-complete-pending-rollout",
        "rollout_acceptance_required": True,
        "action_authority": "interaction-speed-only",
        "min_interaction_speed_mps": reference["min_interaction_speed_mps"],
        "model_state_dict": averaged_model,
        "anchor_model_state_dict": copy.deepcopy(anchor),
        "model_config": copy.deepcopy(reference["model_config"]),
        "ppo_config": ppo_config,
        "reward_config": copy.deepcopy(reference["reward_config"]),
        "training_cells": copy.deepcopy(reference["training_cells"]),
        "completed_iterations": reference["completed_iterations"],
        "source_supervisor_checkpoint": reference["source_supervisor_checkpoint"],
        "source_supervisor_checkpoint_sha256": reference[
            "source_supervisor_checkpoint_sha256"
        ],
        "source_locomotion_checkpoint": reference["source_locomotion_checkpoint"],
        "source_locomotion_checkpoint_sha256": reference[
            "source_locomotion_checkpoint_sha256"
        ],
        "aggregation": {
            "method": (
                "unanimous-sign-arithmetic-mean"
                if consensus_only
                else "arithmetic-mean"
            ),
            "fields": [f"{SPEED_HEAD_WEIGHT}[0]", f"{SPEED_HEAD_BIAS}[0]"],
            "parameter_count": parameter_count,
            "sources": [
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "training_seed": payload["seed"],
                    "completed_iterations": payload["completed_iterations"],
                    "configured_iterations": payload["ppo_config"]["iterations"],
                }
                for path, payload in zip(checkpoint_paths, payloads, strict=True)
            ],
        },
        "physical_motion_authorized": False,
    }
    if consensus_only:
        retained_count = int(consensus_mask.sum())
        checkpoint["aggregation"]["retained_parameter_count"] = retained_count
        checkpoint["aggregation"]["anchor_parameter_count"] = (
            parameter_count - retained_count
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    manifest = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"model_state_dict", "anchor_model_state_dict"}
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return output_path


def average_interaction_speed_checkpoints(
    checkpoint_paths: tuple[Path, ...], output_path: Path
) -> Path:
    """Average all compatible HC3-E speed-head parameters into HC3-F."""
    return _aggregate_interaction_speed_checkpoints(
        checkpoint_paths, output_path, consensus_only=False
    )


def consensus_interaction_speed_checkpoints(
    checkpoint_paths: tuple[Path, ...], output_path: Path
) -> Path:
    """Average only unanimous-sign HC3-E speed updates into HC3-G."""
    return _aggregate_interaction_speed_checkpoints(
        checkpoint_paths, output_path, consensus_only=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--method",
        choices=("mean", "sign-consensus"),
        default="mean",
    )
    args = parser.parse_args()
    aggregate = (
        consensus_interaction_speed_checkpoints
        if args.method == "sign-consensus"
        else average_interaction_speed_checkpoints
    )
    aggregate(tuple(args.checkpoints), args.output)


if __name__ == "__main__":
    main()
