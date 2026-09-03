from dataclasses import asdict

import pytest
import torch

from mjlab_microduck.obstacle_supervisor_bc import (
    HC2_STAGE,
    HC4L_STAGE,
    HC4LH_STAGE,
    ObstacleSupervisor,
    SupervisorBcCfg,
)
from mjlab_microduck.obstacle_supervisor_hybrid import (
    compose_lateral_gated_supervisor,
)


def _write_checkpoint(path, stage, locomotion_hash="a" * 64):
    torch.save(
        {
            "stage": stage,
            "decision": "offline-imitation-pass",
            "source_locomotion_checkpoint_sha256": locomotion_hash,
            "model_config": asdict(SupervisorBcCfg()),
            "model_state_dict": ObstacleSupervisor().state_dict(),
            "physical_motion_authorized": False,
        },
        path,
    )


def test_compose_lateral_gated_checkpoint(tmp_path):
    center = tmp_path / "center.pt"
    lateral = tmp_path / "lateral.pt"
    output = tmp_path / "hybrid" / "supervisor.pt"
    _write_checkpoint(center, HC2_STAGE)
    _write_checkpoint(lateral, HC4L_STAGE)

    compose_lateral_gated_supervisor(center, lateral, output)

    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["stage"] == HC4LH_STAGE
    assert payload["decision"] == "composition-complete-pending-rollout"
    assert payload["lateral_gate_m"] == 0.06
    assert payload["physical_motion_authorized"] is False
    assert output.with_suffix(".json").exists()


def test_compose_rejects_mismatched_locomotion_actor(tmp_path):
    center = tmp_path / "center.pt"
    lateral = tmp_path / "lateral.pt"
    _write_checkpoint(center, HC2_STAGE)
    _write_checkpoint(lateral, HC4L_STAGE, locomotion_hash="b" * 64)

    with pytest.raises(ValueError, match="different locomotion actors"):
        compose_lateral_gated_supervisor(center, lateral, tmp_path / "hybrid.pt")
