import json
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from mjlab_microduck.obstacle_supervisor_average import (
    HC3E_STAGE,
    HC3F_STAGE,
    HC3G_STAGE,
    SPEED_HEAD_BIAS,
    SPEED_HEAD_WEIGHT,
    average_interaction_speed_checkpoints,
    consensus_interaction_speed_checkpoints,
)
from mjlab_microduck.obstacle_supervisor_bc import (
    ObstacleSupervisor,
    SupervisorBcCfg,
)


def _checkpoint_payload(
    anchor: dict[str, torch.Tensor], seed: int, delta: float, configured_iterations: int
) -> dict:
    model = deepcopy(anchor)
    model[SPEED_HEAD_WEIGHT][0] += delta
    model[SPEED_HEAD_BIAS][0] += delta
    return {
        "schema_version": 1,
        "stage": HC3E_STAGE,
        "decision": "training-complete-pending-rollout",
        "action_authority": "interaction-speed-only",
        "min_interaction_speed_mps": 0.30,
        "model_config": asdict(SupervisorBcCfg()),
        "ppo_config": {
            "iterations": configured_iterations,
            "learning_rate": 1.0e-5,
        },
        "reward_config": {"collision": -10.0},
        "completed_iterations": 1,
        "seed": seed,
        "training_cells": [[0.5, 1.15], [0.8, 1.40]],
        "source_supervisor_checkpoint": "/source/hc2.pt",
        "source_supervisor_checkpoint_sha256": "a" * 64,
        "source_locomotion_checkpoint": "/source/locomotion.pt",
        "source_locomotion_checkpoint_sha256": "b" * 64,
        "model_state_dict": model,
        "anchor_model_state_dict": deepcopy(anchor),
        "physical_motion_authorized": False,
    }


def _write_inputs(tmp_path):
    anchor = ObstacleSupervisor().state_dict()
    paths = []
    for seed, delta, configured_iterations in zip(
        (109, 113, 127), (1.0, 2.0, 3.0), (4, 1, 1), strict=True
    ):
        path = tmp_path / f"seed-{seed}.pt"
        torch.save(
            _checkpoint_payload(anchor, seed, delta, configured_iterations), path
        )
        paths.append(path)
    return anchor, tuple(paths)


def test_average_changes_only_speed_head_and_writes_manifest(tmp_path):
    anchor, inputs = _write_inputs(tmp_path)
    output = tmp_path / "averaged" / "supervisor.pt"

    result = average_interaction_speed_checkpoints(inputs, output)

    assert result == output
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["stage"] == HC3F_STAGE
    assert payload["decision"] == "aggregation-complete-pending-rollout"
    assert payload["aggregation"]["parameter_count"] == 65
    assert payload["ppo_config"]["iterations"] == 1
    assert [
        source["configured_iterations"] for source in payload["aggregation"]["sources"]
    ] == [4, 1, 1]
    assert [
        source["training_seed"] for source in payload["aggregation"]["sources"]
    ] == [
        109,
        113,
        127,
    ]
    averaged = payload["model_state_dict"]
    torch.testing.assert_close(
        averaged[SPEED_HEAD_WEIGHT][0], anchor[SPEED_HEAD_WEIGHT][0] + 2.0
    )
    torch.testing.assert_close(
        averaged[SPEED_HEAD_BIAS][0], anchor[SPEED_HEAD_BIAS][0] + 2.0
    )
    torch.testing.assert_close(
        averaged[SPEED_HEAD_WEIGHT][1], anchor[SPEED_HEAD_WEIGHT][1]
    )
    torch.testing.assert_close(averaged[SPEED_HEAD_BIAS][1], anchor[SPEED_HEAD_BIAS][1])
    for name in averaged:
        if name not in {SPEED_HEAD_WEIGHT, SPEED_HEAD_BIAS}:
            assert torch.equal(averaged[name], anchor[name])
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert "model_state_dict" not in manifest
    assert manifest["physical_motion_authorized"] is False


def test_consensus_keeps_only_unanimous_update_coordinates(tmp_path):
    anchor, inputs = _write_inputs(tmp_path)
    payload = torch.load(inputs[2], map_location="cpu", weights_only=False)
    payload["model_state_dict"][SPEED_HEAD_WEIGHT][0, 0] = (
        anchor[SPEED_HEAD_WEIGHT][0, 0] - 3.0
    )
    torch.save(payload, inputs[2])
    output = tmp_path / "consensus" / "supervisor.pt"

    consensus_interaction_speed_checkpoints(inputs, output)

    result = torch.load(output, map_location="cpu", weights_only=False)
    assert result["stage"] == HC3G_STAGE
    assert result["aggregation"]["method"] == "unanimous-sign-arithmetic-mean"
    assert result["aggregation"]["retained_parameter_count"] == 64
    assert result["aggregation"]["anchor_parameter_count"] == 1
    assert torch.equal(
        result["model_state_dict"][SPEED_HEAD_WEIGHT][0, 0],
        anchor[SPEED_HEAD_WEIGHT][0, 0],
    )
    torch.testing.assert_close(
        result["model_state_dict"][SPEED_HEAD_WEIGHT][0, 1],
        anchor[SPEED_HEAD_WEIGHT][0, 1] + 2.0,
    )


def test_average_requires_three_unique_seeds(tmp_path):
    _, inputs = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="at least three"):
        average_interaction_speed_checkpoints(inputs[:2], tmp_path / "few.pt")

    duplicate = torch.load(inputs[2], map_location="cpu", weights_only=False)
    duplicate["seed"] = 109
    torch.save(duplicate, inputs[2])
    with pytest.raises(ValueError, match="unique training seeds"):
        average_interaction_speed_checkpoints(inputs, tmp_path / "duplicate.pt")


@pytest.mark.parametrize("corruption", ["yaw", "frozen", "iteration", "optimizer"])
def test_average_rejects_incompatible_input(tmp_path, corruption):
    _, inputs = _write_inputs(tmp_path)
    payload = torch.load(inputs[1], map_location="cpu", weights_only=False)
    if corruption == "yaw":
        payload["model_state_dict"][SPEED_HEAD_BIAS][1] += 1.0
    elif corruption == "frozen":
        payload["model_state_dict"]["network.0.bias"][0] += 1.0
    elif corruption == "iteration":
        payload["completed_iterations"] = 2
    else:
        payload["ppo_config"]["learning_rate"] = 2.0e-5
    torch.save(payload, inputs[1])

    with pytest.raises(ValueError):
        average_interaction_speed_checkpoints(inputs, tmp_path / "invalid.pt")
