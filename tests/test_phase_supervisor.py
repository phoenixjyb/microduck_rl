import hashlib
import json

import pytest
import torch

from mjlab_microduck import obstacle_supervisor_bc as bc
from mjlab_microduck.hierarchical_obstacle_rollout import load_learned_supervisor


def observations():
    x = torch.zeros(3, 17)
    x[:, 0] = 0.375
    x[:, 7] = 1
    x[:, 13:16] = torch.eye(3)
    return x


def test_phase_experts_select_commands_and_isolate_gradients():
    model = bc.PhaseSeparatedSupervisor()
    x = observations()
    for i, expert in enumerate(model.experts):
        for parameter in expert.parameters():
            parameter.data.zero_()
        expert.network[-1].bias.data[:] = torch.tensor([i - 1.0, 1.0 - i])
    expected = torch.stack([model.experts[i](x[i:i + 1])[0] for i in range(3)])
    torch.testing.assert_close(model(x), expected)
    model(x[1:2]).sum().backward()
    assert model.experts[1].network[-1].bias.grad.abs().sum() > 0
    for i in (0, 2):
        assert all(p.grad is None or not bool(p.grad.any()) for p in model.experts[i].parameters())


@pytest.mark.parametrize("phase", [[0, 0, 0], [1, 1, 0], [.5, .5, 0], [float("nan"), 0, 0]])
def test_malformed_phase_commands_stop(phase):
    x = observations()[:1]
    x[:, 13:16] = torch.tensor(phase)
    torch.testing.assert_close(bc.PhaseSeparatedSupervisor()(x), torch.zeros(1, 2))


def test_phase_training_requires_coverage_and_valid_observations():
    x = observations()
    assert bc.phase_partition_counts(x, torch.ones(3, dtype=torch.bool)) == {
        "approach": 1, "interaction": 1, "recovery": 1,
    }
    with pytest.raises(ValueError, match="every phase"):
        bc.phase_partition_counts(x, torch.tensor([True, True, False]))
    x[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        bc.phase_partition_counts(x, torch.ones(3, dtype=torch.bool))


def test_u3_preserves_u2_dataset_contract():
    with pytest.raises(ValueError, match="exact ordered"):
        bc.validate_bc_dataset_contract(bc.HC4U3_STAGE, [{}], ("wrong",))


def test_phase_fit_manifest_and_runtime_roundtrip(tmp_path, monkeypatch):
    actor = tmp_path / "actor.pt"
    actor.write_bytes(b"test-locomotion")
    actor_hash = hashlib.sha256(actor.read_bytes()).hexdigest()
    x = observations().repeat(8, 1)
    paths = []
    for i in range(4):
        p = tmp_path / f"shard-{i}.pt"
        torch.save({
            "stage": "HC1-successful-teacher-trajectories" if i == 0 else "HC4R2-student-state-teacher-corrections",
            "checkpoint_sha256": actor_hash,
            "student_supervisor_checkpoint_sha256": bc.HC4U2_STUDENT_SHA256,
            "observations": x,
            "commands": torch.tensor([.3, 0.0]).repeat(len(x), 1),
            "episode_keys": torch.arange(8).repeat_interleave(3),
        }, p)
        paths.append(p)
    monkeypatch.setattr(bc, "HC4U2_REQUIRED_DATASET_SHA256", tuple(bc._sha256(p) for p in paths))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = tmp_path / "supervisor.pt"
    cfg = bc.SupervisorBcCfg(speed_mae_gate_mps=1.0, yaw_mae_gate_rps=1.0)
    bc.train_supervisor(tuple(paths), output, epochs=2, batch_size=24, stage=bc.HC4U3_STAGE, cfg=cfg)
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert manifest["architecture"] == "three-independent-phase-experts-v1"
    assert all(v > 0 for partition in manifest["phase_samples"].values() for v in partition.values())
    assert set(manifest["phase_validation_metrics"]) == {"approach", "interaction", "recovery"}
    progress = output.with_suffix(".progress.pt")
    assert torch.load(progress, weights_only=False)["epoch"] == 1
    with pytest.raises(ValueError, match="not eligible"):
        load_learned_supervisor(progress, actor, "cpu")
    loaded = load_learned_supervisor(output, actor, "cpu")
    assert isinstance(loaded, bc.PhaseSeparatedSupervisor)
    payload = torch.load(output, weights_only=False)
    reference = bc.PhaseSeparatedSupervisor(cfg)
    reference.load_state_dict(payload["model_state_dict"])
    torch.testing.assert_close(loaded(x), reference(x))
    payload["architecture"] = "unknown"
    torch.save(payload, output)
    with pytest.raises(ValueError, match="invalid phase architecture"):
        load_learned_supervisor(output, actor, "cpu")


def test_legacy_supervisor_state_layout_is_unchanged():
    assert set(bc.ObstacleSupervisor().state_dict()) == {
        f"network.{layer}.{field}" for layer in (0, 2, 4) for field in ("weight", "bias")
    }
