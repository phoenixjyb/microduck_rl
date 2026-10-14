from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from mjlab_microduck import v8_head_goal_boundary_probe as probe
from mjlab_microduck import v8_head_goal_schedule as schedule


def _window(arm):
    rows = []
    for step in range(schedule.PROBE_STEPS):
        stage = schedule.stage_index(arm, step, probe=True)
        ranges = schedule.ranges_at(arm, step, probe=True)
        previous = (None if stage == 0 else schedule.ranges_at(
            arm, schedule.PROBE_SCHEDULES[arm][stage][0] - 1, probe=True))
        minima, maxima, means, outside = [], [], [], []
        for component, (lo, hi) in enumerate(ranges):
            value = 0.0
            escaped = 0
            if arm == "staged" and stage > 0:
                prior_lo, prior_hi = previous[component]
                value = prior_hi + min(.001, hi - prior_hi)
                escaped = 1
            minima.append(0.0)
            maxima.append(value)
            means.append(.01 if value else 0.0)
            outside.append(escaped)
        rows.append(dict(before_step=step, completed_step=step + 1,
            schedule_kind="boundary-probe", arm=arm, stage=stage,
            ranges=[list(pair) for pair in ranges],
            previous_ranges=None if previous is None else [list(pair) for pair in previous],
            worlds=schedule.PROBE_WORLDS, command_min=minima, command_max=maxima,
            command_abs_mean=means, command_nonzero_count=[64 if x else 0 for x in means],
            command_outside_previous_range_count=outside))
    stages = {str(i): sum(row["stage"] == i for row in rows)
              for i in range(len(schedule.PROBE_SCHEDULES[arm]))}
    ranges_by_stage = {str(row["stage"]): row["ranges"] for row in rows}
    return dict(arm=arm, probe=True, schedule_kind="boundary-probe",
        steps=len(rows), worlds=schedule.PROBE_WORLDS, first_before_step=0,
        last_completed_step=len(rows), stage_action_steps=stages,
        ranges_by_stage=ranges_by_stage,
        command_min=[min(row["command_min"][i] for row in rows) for i in range(4)],
        command_max=[max(row["command_max"][i] for row in rows) for i in range(4)],
        command_abs_mean=[sum(row["command_abs_mean"][i] for row in rows)/len(rows)
                          for i in range(4)],
        command_nonzero_count=[sum(row["command_nonzero_count"][i] for row in rows)
                               for i in range(4)],
        command_outside_previous_range_count=[sum(row["command_outside_previous_range_count"][i]
                                                  for row in rows) for i in range(4)],
        command_sample_count=len(rows)*64, step_records=rows)


def _artifact(tmp_path, arm="staged"):
    out = tmp_path / arm / "boundary-smoke"
    out.mkdir(parents=True)
    window = _window(arm)
    rows = []
    for i, telemetry in enumerate(window["step_records"]):
        rows.append(dict(**telemetry, reward_mean=0.0, reward_min=0.0, reward_max=0.0,
            action_abs_max=.3, fall_worlds=int(i == 0), reset_worlds=int(i == 0),
            nan_terminations=0, bam_force_abs_mean_nm=.01, bam_force_abs_max_nm=.03,
            bam_speed_abs_mean_rad_s=.1, bam_speed_abs_max_rad_s=.2, state_abs_max=1.0,
            yaw_bias_cost_mean=.1, yaw_bias_weighted_reward_mean=-.1,
            yaw_bias_signed_ema_mean=0.0, selected_motor_cost_mean=.1,
            selected_motor_weighted_reward_mean=0.0,
            head_reward_snapshot={"step": i+1,
                "head_pose_tracking_weighted_rewards": [0.2]*64,
                "head_pose_tracking_weighted_reward_mean": 0.2,
                "head_pose_tracking_weighted_reward_min": 0.2,
                "head_pose_tracking_weighted_reward_max": 0.2,
                "head_pose_bias_weighted_rewards": [-0.1]*64,
                "head_pose_bias_weighted_reward_mean": -0.1,
                "head_pose_bias_weighted_reward_min": -0.1,
                "head_pose_bias_weighted_reward_max": -0.1}))
    recipe = dict(protocol=probe.PROTOCOL, arm=arm,
        shared_training_recipe_sha256="1"*64,
        full_arm_training_recipe_sha256="2"*64,
        schedule=schedule.stages_for(arm, probe=True), seed=971, worlds=64, steps=320)
    recipe["probe_recipe_sha256"] = hashlib.sha256(json.dumps(
        recipe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    stats = dict(protocol=probe.PROTOCOL, source="a"*40, arm=arm,
        parent_sha256=probe.PARENT_SHA256, checkpoint_sha256=probe.PARENT_SHA256, worlds=64, steps=320, seed=971,
        reward_mean=0.0, falls=1, resets=1, nan_terminations=0, action_abs_max=.3,
        actor_state_sha256_before="d"*64, actor_state_sha256_after="d"*64,
        normalizer_count=0, critic_constructed=False, optimizer_constructed=False, ppo_learning=False,
        head_goal_telemetry=window, step_records=rows,
        policy_acceptance=False, physical_motion_authorized=False)
    stats_path = out / "stats.json"
    stats_path.write_text(json.dumps(stats, sort_keys=True, allow_nan=False))
    receipt = dict(protocol=probe.PROTOCOL, source="a"*40, run_id="test-run", arm=arm,
        parent_sha256=probe.PARENT_SHA256, checkpoint_sha256=probe.PARENT_SHA256, output=str(out), runtime={}, host={},
        seed=971, worlds=64, steps=320, steps_per_update=24, schedule_kind="boundary-probe",
        stage_boundaries={"staged": [0, 80, 160], "control": [0]},
        stats_file="stats.json", stats_sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        ordinary_smoke_receipt_sha256={"control": "e"*64, "staged": "f"*64}, recipe=recipe,
        actor_state_sha256_before="d"*64, actor_state_sha256_after="d"*64,
        normalizer_count=0, critic_constructed=False, optimizer_constructed=False, ppo_learning=False,
        totals={"falls": 1, "resets": 1, "nan_terminations": 0, "reward_mean": 0.0,
                "action_abs_max": .3}, status="boundary-smoke-complete-not-accepted",
        policy_acceptance=False, physical_motion_authorized=False)
    (out / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, allow_nan=False))
    stats_path.chmod(0o444)
    (out / "receipt.json").chmod(0o444)
    return out


def test_receipt_recomputes_full_staged_boundaries_and_exposure(tmp_path):
    out = _artifact(tmp_path)
    receipt = probe.validate_receipt(out, "staged", "a"*40)
    assert receipt["status"] == "boundary-smoke-complete-not-accepted"


def test_control_receipt_requires_no_fabricated_widening_exposure(tmp_path):
    out = _artifact(tmp_path, "control")
    assert probe.validate_receipt(out, "control", "a"*40)["arm"] == "control"


def test_receipt_rejects_missing_actual_per_component_exposure(tmp_path):
    out = _artifact(tmp_path)
    stats_path = out / "stats.json"
    stats_path.chmod(0o644)
    stats = json.loads(stats_path.read_text())
    stats["head_goal_telemetry"]["step_records"][80][
        "command_outside_previous_range_count"][2] = 0
    stats_path.write_text(json.dumps(stats, sort_keys=True, allow_nan=False))
    receipt_path = out / "receipt.json"
    receipt_path.chmod(0o644)
    receipt = json.loads(receipt_path.read_text())
    receipt["stats_sha256"] = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, allow_nan=False))
    stats_path.chmod(0o444)
    receipt_path.chmod(0o444)
    with pytest.raises(ValueError, match="previous-stage exposure count"):
        probe.validate_receipt(out, "staged", "a"*40)


def test_receipt_rejects_source_or_actor_mutation(tmp_path):
    out = _artifact(tmp_path)
    with pytest.raises(ValueError, match="protocol, source, arm"):
        probe.validate_receipt(out, "staged", "0"*40)
    receipt_path = out / "receipt.json"
    receipt_path.chmod(0o644)
    receipt = json.loads(receipt_path.read_text())
    receipt["actor_state_sha256_after"] = "0"*64
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, allow_nan=False))
    receipt_path.chmod(0o444)
    with pytest.raises(ValueError, match="frozen actor unchanged"):
        probe.validate_receipt(out, "staged", "a"*40)


@pytest.mark.parametrize("arm", schedule.ARMS)
def test_probe_config_uses_accelerated_schedule_without_mutating_training_schedule(arm):
    from mjlab_microduck.tasks import mdp

    original_training_stages = schedule.stages_for(arm)
    cfg = NS(commands={"head_pose": NS(ranges=None)}, curriculum={
        "head_pose_range": NS(func=mdp.pose_command_range_curriculum,
                               params={"range_stages": schedule.stages_for(arm)})})
    probe._install_probe_schedule(cfg, arm)
    assert cfg.commands["head_pose"].ranges == schedule.ranges_at(arm, 0, probe=True)
    assert cfg.curriculum["head_pose_range"].params["range_stages"] == schedule.stages_for(
        arm, probe=True)
    assert schedule.stages_for(arm) == original_training_stages


def test_probe_motor_stream_uses_existing_supported_capacity_bucket():
    from types import SimpleNamespace

    import mujoco
    import torch

    motor = SimpleNamespace(name="motor", target="motor", trntype=mujoco.mjtTrn.mjTRN_JOINT,
                            gear=(1., 0., 0., 0., 0., 0.))
    joint = SimpleNamespace(name="motor", type=mujoco.mjtJoint.mjJNT_HINGE)
    robot = SimpleNamespace(spec=SimpleNamespace(actuators=(motor,), joints=(joint,)),
                            joint_names=("motor",), actuator_names=("motor",))
    stream = probe._make_motor_stream(robot, torch.device("cpu"))
    assert stream.num_envs == schedule.PROBE_WORLDS
    assert stream.names == ("motor",) and stream.joint_ids == (0,)
