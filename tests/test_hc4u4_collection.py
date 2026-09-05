import json
from dataclasses import asdict

import pytest
import torch

from mjlab_microduck import hc4u4_collection as gate
from mjlab_microduck import hierarchical_obstacle_rollout as rollout
from test_hc4u1_gate import _write_report


def shard(tmp_path):
    report_path = _write_report(
        tmp_path / "report.json", stage=gate.STUDENT_STAGE,
        supervisor_sha256=gate.STUDENT_SHA256, forward_positions=(.9,), seed=317,
    )
    keys = torch.cat([i * 1_000_000 + torch.arange(64) for i in range(6)]).repeat_interleave(55)
    x = torch.zeros(len(keys), 17)
    x[:, 14] = 1
    dataset_path = tmp_path / "dataset.pt"
    payload = {
        "stage": "HC4R2-student-state-teacher-corrections",
        "checkpoint_sha256": gate.ACTOR_SHA256,
        "student_supervisor_checkpoint_sha256": gate.STUDENT_SHA256,
        "collection_window": gate.PROTOCOL,
        "terminal_outcome_protocol": gate.FIRST_TERMINAL_OUTCOME_PROTOCOL,
        "collection_seeds": [317], "observation_dim": 17,
        "teacher_config": asdict(gate.ObstacleTeacherCfg()),
        "command_fields": ["forward_speed_mps", "yaw_rate_rps"],
        "outcome_codes": {"clean_pass": 1, "collision": 2, "timeout": 3},
        "observations": x, "commands": torch.tensor([.3, 0.]).repeat(len(x), 1),
        "student_commands": torch.tensor([.28, 0.]).repeat(len(x), 1),
        "episode_keys": keys, "sample_outcome_codes": torch.ones(len(x), dtype=torch.int8),
    }
    report = json.loads(report_path.read_text())
    report.update({
        "terminal_outcome_protocol": gate.FIRST_TERMINAL_OUTCOME_PROTOCOL,
        "attempt_timeout_s": 12., "teacher_correction_dataset": str(dataset_path),
        "teacher_correction_dataset_samples": len(x), "teacher_correction_dataset_episodes": 384,
        "obstacle_sensor_model": {k: 0. for k in (
            "range_noise_m", "bearing_noise_rad", "width_noise_m", "height_noise_m",
            "closing_rate_noise_mps", "dropout_probability",
        )},
        "totals": {"clean_pass_events": 384, "collision_events": 0, "attempt_timeout_events": 0},
    })
    for c in report["cases"]:
        c.update(steps=700, terminal_outcome_protocol=gate.FIRST_TERMINAL_OUTCOME_PROTOCOL)
    torch.save(payload, dataset_path)
    report_path.write_text(json.dumps(report))
    return report_path, dataset_path, report, payload


def test_correction_shard_admits_data_not_policy(tmp_path):
    r, d, _, _ = shard(tmp_path)
    result = gate.validate_correction_shard(r, d, seed=317)
    assert result["decision"] == "admit-correction-shard-not-policy"
    assert result["episodes"] == 384 and result["samples"] == 21120
    assert result["dataset_sha256"] == gate._sha256(d)
    assert result["physical_motion_authorized"] is False
    with pytest.raises(ValueError, match="predeclared"):
        gate.validate_correction_shard(r, d, seed=311)


@pytest.mark.parametrize("corruption", ["student", "nan", "partial", "code", "labels", "bounds", "counts", "order", "protocol", "fall", "torque", "sensor"])
def test_collection_gate_fails_closed(tmp_path, corruption):
    r, d, report, payload = shard(tmp_path)
    if corruption == "student": payload["student_supervisor_checkpoint_sha256"] = "x" * 64
    if corruption == "nan": payload["observations"][0, 0] = float("nan")
    if corruption == "partial": payload["episode_keys"][0] = 999
    if corruption == "code": payload["sample_outcome_codes"][0] = 2
    if corruption == "labels": payload["student_commands"] = payload["commands"].clone()
    if corruption == "bounds": payload["commands"][0, 0] = 2.
    if corruption == "counts": report["cases"][0]["clean_pass_events"] = 63
    if corruption == "order": report["cases"] = report["cases"][::-1]
    if corruption == "protocol": report.pop("terminal_outcome_protocol")
    if corruption == "fall": report["cases"][0]["fall_events"] = 1
    if corruption == "torque": report["cases"][0]["motor_torque_utilization_p99"] = .61
    if corruption == "sensor": report["obstacle_sensor_model"]["range_noise_m"] = .02
    torch.save(payload, d)
    r.write_text(json.dumps(report))
    with pytest.raises(ValueError): gate.validate_correction_shard(r, d, seed=317)


def test_first_attempt_corrections_reach_collection_but_success_only_does_not(tmp_path, monkeypatch):
    actor, student = tmp_path / "actor.pt", tmp_path / "student.pt"
    actor.touch(); student.touch()
    class ReachedCase(Exception): pass
    def case(*args, **kwargs):
        assert kwargs["first_attempt_only"] and kwargs["collect_teacher_corrections"]
        raise ReachedCase
    monkeypatch.setattr(rollout, "_run_case", case)
    args = dict(num_envs=4, steps=700, speeds=(.3,), forward_positions=(.9,),
                lateral_positions=(-.08,), seeds=(317,), first_attempt_only=True)
    with pytest.raises(ReachedCase):
        rollout.run_rollout(actor, tmp_path / "run", supervisor_checkpoint=student,
                            collect_teacher_corrections=True, **args)
    with pytest.raises(ValueError, match="student corrections only"):
        rollout.run_rollout(actor, tmp_path / "bad", collect_success_dataset=True, **args)
