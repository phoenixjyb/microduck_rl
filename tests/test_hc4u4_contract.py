import copy
import hashlib
import json

import pytest
import torch

from mjlab_microduck import obstacle_supervisor_bc as bc
from mjlab_microduck.hc4u1_gate import PROTOCOL
from mjlab_microduck.hc4u4_gate import STAGE, compare_hc4u4_prescreen
from mjlab_microduck.hierarchical_obstacle_rollout import (
    FIRST_TERMINAL_OUTCOME_PROTOCOL,
    load_learned_supervisor,
    recording_controller_stage,
    rollout_stage,
)
from test_hc4u1_gate import _reports


def payloads():
    parent = [{"stage": "HC1-successful-teacher-trajectories"} for _ in range(7)]
    parent += [{"stage": "HC4R2-student-state-teacher-corrections"} for _ in range(3)]
    parent += [{
        "stage": "HC4R2-student-state-teacher-corrections",
        "student_supervisor_checkpoint_sha256": bc.HC4U2_STUDENT_SHA256,
    } for _ in range(3)]
    return parent + [{
        "stage": "HC4R2-student-state-teacher-corrections",
        "student_supervisor_checkpoint_sha256": bc.HC4U4_STUDENT_SHA256,
        "collection_seeds": [seed],
        "collection_window": PROTOCOL,
        "terminal_outcome_protocol": FIRST_TERMINAL_OUTCOME_PROTOCOL,
    } for seed in (317, 331, 337)]


def test_exact_corpus_and_parent_are_preserved():
    assert len(bc.HC4U4_REQUIRED_DATASET_SHA256) == 16
    assert bc.HC4U4_REQUIRED_DATASET_SHA256[:-3] == bc.HC4U2_REQUIRED_DATASET_SHA256
    bc.validate_bc_dataset_contract(bc.HC4U4_STAGE, payloads(), bc.HC4U4_REQUIRED_DATASET_SHA256)
    for hashes in (bc.HC4U4_REQUIRED_DATASET_SHA256[::-1], ("smoke",) * 16):
        with pytest.raises(ValueError, match="exact ordered"):
            bc.validate_bc_dataset_contract(bc.HC4U4_STAGE, payloads(), hashes)


@pytest.mark.parametrize("index,key,value", [
    (12, "student_supervisor_checkpoint_sha256", "wrong-parent"),
    (13, "student_supervisor_checkpoint_sha256", "wrong-student"),
    (14, "collection_seeds", [347]),
    (15, "collection_window", "all-attempts"),
    (13, "terminal_outcome_protocol", "old"),
    (13, "stage", "HC1-successful-teacher-trajectories"),
])
def test_data_identity_cannot_change(index, key, value):
    data = copy.deepcopy(payloads())
    data[index][key] = value
    with pytest.raises(ValueError):
        bc.validate_bc_dataset_contract(bc.HC4U4_STAGE, data, bc.HC4U4_REQUIRED_DATASET_SHA256)


@pytest.mark.parametrize("changes", [
    {"epochs": 199}, {"batch_size": 512}, {"seed": 43},
    {"cfg": bc.SupervisorBcCfg(hidden_dims=(128, 128))},
    {"cfg": bc.SupervisorBcCfg(speed_mae_gate_mps=1.0)},
])
def test_fit_configuration_cannot_change(tmp_path, changes):
    with pytest.raises(ValueError, match="predeclared fit configuration"):
        bc.train_supervisor((), tmp_path / "not-created.pt", stage=bc.HC4U4_STAGE, **changes)
    assert not list(tmp_path.iterdir())


def test_runtime_roundtrip_and_video_stays_closed(tmp_path):
    actor = tmp_path / "actor.pt"
    actor.write_bytes(b"synthetic-actor")
    model = bc.PhaseSeparatedSupervisor()
    candidate = tmp_path / "candidate.pt"
    torch.save({
        "stage": bc.HC4U4_STAGE, "decision": "offline-imitation-pass",
        "source_locomotion_checkpoint_sha256": hashlib.sha256(actor.read_bytes()).hexdigest(),
        "model_config": bc.asdict(bc.SupervisorBcCfg()),
        "architecture": "three-independent-phase-experts-v1",
        "model_state_dict": model.state_dict(),
    }, candidate)
    loaded = load_learned_supervisor(candidate, actor, "cpu")
    x = torch.zeros(3, 17)
    x[:, 13:16] = torch.eye(3)
    torch.testing.assert_close(loaded(x), model(x))
    assert rollout_stage(bc.HC4U4_STAGE) == STAGE
    with pytest.raises(ValueError, match="unsupported"):
        recording_controller_stage(bc.HC4U4_STAGE)


def reports(tmp_path, seed=347, **kwargs):
    paths = _reports(tmp_path, candidate_stage=STAGE, candidate_sha256="a" * 64, seed=seed, **kwargs)
    for path in paths:
        data = json.loads(path.read_text())
        data["terminal_outcome_protocol"] = FIRST_TERMINAL_OUTCOME_PROTOCOL
        data["obstacle_sensor_model"] = {field: 0.0 for field in (
            "range_noise_m", "bearing_noise_rad", "width_noise_m", "height_noise_m",
            "closing_rate_noise_mps", "dropout_probability",
        )}
        for case in data["cases"]:
            case["terminal_outcome_protocol"] = FIRST_TERMINAL_OUTCOME_PROTOCOL
        path.write_text(json.dumps(data))
    return paths


@pytest.mark.parametrize("seed", [347, 349, 353])
def test_only_fresh_predeclared_seeds_and_exact_checkpoint(tmp_path, seed):
    paths = reports(tmp_path, seed)
    assert compare_hc4u4_prescreen(*paths, candidate_sha256="a" * 64, seed=seed)["decision"] == "continue_fresh_seeds"
    with pytest.raises(ValueError, match="supervisor checkpoint"):
        compare_hc4u4_prescreen(*paths, candidate_sha256="b" * 64, seed=seed)
    for old_seed in (293, 307, 311, 317, 331, 337):
        with pytest.raises(ValueError, match="predeclared"):
            compare_hc4u4_prescreen(*paths, candidate_sha256="a" * 64, seed=old_seed)


@pytest.mark.parametrize("changes", [
    {"clean_pass_events": 63, "collision_events": 1},
    {"clean_pass_events": 63, "attempt_timeout_events": 1},
    {"motor_torque_utilization_p99": .601},
    {"approach_route_speed_mps": 0.0},
    {"recovery_route_speed_mps": 0.0},
])
def test_unchanged_numerical_gate_rejects_regressions(tmp_path, changes):
    paths = reports(tmp_path, candidate_overrides={(.30, .90, -.08): changes})
    assert compare_hc4u4_prescreen(*paths, candidate_sha256="a" * 64, seed=347)["decision"] == "stop"


@pytest.mark.parametrize("changes", [
    {"clean_pass_events": 65},
    {"recovery_route_speed_mps": float("nan")},
    {"terminal_outcome_protocol": "old"},
])
def test_malformed_evidence_rejected(tmp_path, changes):
    paths = reports(tmp_path)
    data = json.loads(paths[0].read_text())
    data["cases"][0].update(changes)
    paths[0].write_text(json.dumps(data))
    with pytest.raises(ValueError):
        compare_hc4u4_prescreen(*paths, candidate_sha256="a" * 64, seed=347)
