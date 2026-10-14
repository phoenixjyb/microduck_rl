"""Bounded frozen-actor GPU probe of the staged head-goal boundaries.

This is an admission diagnostic only. It runs the original authenticated v8
actor for 320 real simulator steps, with no PPO runner, optimizer, resampling,
action filtering, or synthetic schedule counter movement.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import stat
import time
from pathlib import Path
from typing import Mapping

import torch

from . import integrated_turn_foundation as base
from . import v8_head_goal_schedule as schedule

PROTOCOL = "v8-staged-head-goals-boundary-smoke-v1"
PARENT_SHA256 = "dddd37576fa7d2cd17bb4201826a7fa8aec0404b7b5af1339b79d492e4a9355a"
CAP_SECONDS = 180
HEALTH_DEADLINE_SECONDS = 165


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _tree_finite(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_tree_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_tree_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_tree_sha256(value) -> str:
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode() + b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                digest.update(str(key).encode() + b"\0")
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode() + b"\0")

    visit(value)
    return digest.hexdigest()


def _write_new(path: Path, value) -> None:
    """Create a durable, never-overwritten JSON artifact."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o444)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _host_health(output: Path, started: float, *, final: bool = False):
    _require(time.monotonic() - started < (CAP_SECONDS if final else HEALTH_DEADLINE_SECONDS),
             "boundary probe remains inside declared 180s/165s deadlines")
    host = base.shared_host()
    if output.exists():
        row = dict(elapsed_s=time.monotonic() - started, host=host)
        with (output / "health.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return host


def _consume_reward_snapshots(env, sample, experiment, expected_step):
    """Read the already-produced yaw and selected-motor snapshots once."""
    yaw = getattr(env, "_microduck_yaw_bias_snapshot", None)
    selected = getattr(env, "_microduck_selected_load_snapshot", None)
    _require(isinstance(yaw, dict) and isinstance(selected, dict),
             "actual post-step yaw and selected-motor metric snapshots")
    _require(env.common_step_counter == expected_step
             and selected.get("step") == env.common_step_counter,
             "same-step pre-reset yaw and selected-motor snapshots")
    yaw_state = getattr(env, "_microduck_body_yaw_bias_state", None)
    _require(isinstance(yaw_state, dict) and yaw_state.get("step") == env.common_step_counter,
             "existing cached body-yaw EMA state is current")
    signed = yaw.get("signed_ema")
    bias_cost = yaw.get("cost")
    bias_reward = yaw.get("weighted_reward")
    selected_cost = selected.get("cost")
    selected_reward = selected.get("weighted")
    _require(all(isinstance(value, torch.Tensor) and value.shape == (env.num_envs,)
                 for value in (signed, bias_cost, bias_reward, selected_cost, selected_reward))
             and _tree_finite((signed, bias_cost, bias_reward, selected_cost, selected_reward)),
             "finite actual per-world yaw and selected-motor reward snapshots")
    _require(bool((bias_cost >= 0).all()) and bool((bias_reward <= 0).all())
             and torch.equal(bias_cost, signed.abs())
             and torch.allclose(bias_reward, -bias_cost, atol=1e-6, rtol=1e-5)
             and bool((selected_cost >= 0).all()) and bool((selected_reward == 0).all())
             and selected.get("weight") == 0.0,
             "frozen parent yaw and selected-motor reward signs")

    # Reconcile selected motor load against the already-consumed BAM stream.
    from . import v8_motor_headroom as headroom
    selected_cost_check, selected_reward_check, selected_force = (
        headroom.consume_motor_snapshot(env, sample, 0.0))
    _require(torch.allclose(selected_cost.double(), selected_cost_check.double(), atol=1e-6, rtol=1e-5)
             and torch.equal(selected_reward, selected_reward_check)
             and tuple(selected_force.shape) == (env.num_envs, len(headroom.ACTUATORS)),
             "inactive selected-load snapshot reconciles with actual named BAM motor stream")
    env._microduck_yaw_bias_snapshot = None
    return dict(yaw_bias_cost_mean=float(bias_cost.double().mean()),
                yaw_bias_weighted_reward_mean=float(bias_reward.double().mean()),
                yaw_bias_signed_ema_mean=float(signed.double().mean()),
                selected_motor_cost_mean=float(selected_cost.double().mean()),
                selected_motor_weighted_reward_mean=float(selected_reward.double().mean()))


def _validate_prior_smokes(root: Path, source: str, experiment) -> dict:
    """Require both same-source ordinary 64x5 smoke receipts before the probe."""
    validator = getattr(experiment, "validate_training_receipt", None)
    _require(callable(validator), "lesson independently validates standard smoke receipts")
    receipts = {arm: validator(root, arm, "smoke", source) for arm in schedule.ARMS}
    _require(set(receipts) == set(schedule.ARMS), "both arm smoke receipts authenticated")
    return {arm: hashlib.sha256(json.dumps(receipts[arm], sort_keys=True,
             separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            for arm in schedule.ARMS}


def _prepare_actor(env, wrapped, agent, experiment, authenticated_parent):
    from .v8_actor_checkpoint import FrozenV8Checkpoint

    checkpoint = FrozenV8Checkpoint(experiment.PARENT, experiment.PARENT_SHA)
    source_state = authenticated_parent.get("actor_state_dict")
    actor_state = checkpoint._actor_state
    _require(isinstance(source_state, Mapping) and set(source_state) == set(actor_state)
             and all(torch.equal(source_state[key], actor_state[key]) for key in actor_state),
             "typed frozen actor exactly matches fully authenticated original parent payload")
    actor = checkpoint.construct_actor(wrapped.get_observations(), agent)
    actor.to(device=env.device).eval()
    state_hash = _tensor_tree_sha256(actor.state_dict())
    _require(int(actor.obs_normalizer.count.item()) == 0
             and actor.obs_normalizer.eps == 0.0 and actor.obs_normalizer.until == 0,
             "original frozen actor affine has normalizer count zero")
    return checkpoint, actor, state_hash


def _install_probe_schedule(cfg, arm: str) -> None:
    """Give reset/update callbacks the accelerated schedule for this probe only."""
    from .tasks import mdp

    _require(arm in schedule.ARMS and "head_pose" in cfg.commands
             and "head_pose_range" in cfg.curriculum,
             "declared head command and existing pose-range curriculum")
    term = cfg.curriculum["head_pose_range"]
    _require(term.func is mdp.pose_command_range_curriculum
             and isinstance(term.params, dict) and "range_stages" in term.params,
             "existing live pose-range curriculum callback remains installed")
    cfg.commands["head_pose"].ranges = schedule.ranges_at(arm, 0, probe=True)
    term.params["range_stages"] = schedule.stages_for(arm, probe=True)


def _recipe_binding(experiment, arm: str) -> dict:
    recipe = experiment.recipe_match("smoke")
    _require(recipe.get("matched_after_declared_mask") is True
             and recipe.get("full_recipe_hashes_distinct") is True,
             "authenticated matched staged-head-goal smoke recipe")
    binding = dict(protocol=PROTOCOL, arm=arm,
        shared_training_recipe_sha256=recipe["comparison_recipe_sha256"],
        full_arm_training_recipe_sha256=recipe[
            "control_full_recipe_sha256" if arm == "control" else "treatment_full_recipe_sha256"],
        schedule=schedule.stages_for(arm, probe=True), seed=schedule.PROBE_SEED,
        worlds=schedule.PROBE_WORLDS, steps=schedule.PROBE_STEPS)
    binding = json.loads(json.dumps(binding, sort_keys=True, allow_nan=False))
    binding["probe_recipe_sha256"] = hashlib.sha256(json.dumps(
        binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return binding


def _make_motor_stream(robot, device):
    """Create the standard stream with its supported capacity bucket (256)."""
    from .motor_step_stream import MotorStepCostCfg, MotorStepStream

    return MotorStepStream.from_robot(robot, schedule.PROBE_WORLDS, device=device,
                                      cost_cfg=MotorStepCostCfg(), max_envs=256)


def run(args, experiment):
    """Run the exact 64-world x 320-step frozen-actor boundary admission probe."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from .normalized_onnx_reference import validate_actor_contract
    from . import official_native_reference as pinned
    _require(args.arm in schedule.ARMS, "declared boundary probe arm")
    _require(experiment.PARENT_SHA == PARENT_SHA256,
             "exact original frozen parent SHA identity")
    runtime = experiment.verify_source(args, training=True)
    root = Path(experiment.ROOT) / "artifacts/training" / args.run_id
    output = root / args.arm / "boundary-smoke"
    prior_smokes = _validate_prior_smokes(root, args.source, experiment)
    _require(base.LOCK.is_file() and not base.LOCK.is_symlink(), "existing regular shared GPU lease")
    info_path = base.LOCK.lstat()
    _require(stat.S_ISREG(info_path.st_mode) and info_path.st_uid == os.getuid(),
             "owned regular shared GPU lease file")

    with base.LOCK.open("r+") as lease:
        info = os.fstat(lease.fileno())
        _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(),
                 "owned regular GPU lease descriptor")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host = base.shared_host()
        _require(base.sha256(pinned.V8.path) == pinned.V8.sha256
                 and base.sha256(pinned.V8.path.with_name("manifest.json")) == pinned.V8.manifest_sha256,
                 "exact frozen v8 runtime artifact and manifest")
        output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        try:
            cfg, agent = experiment.prepare_config("smoke", arm=args.arm)
            cfg.seed = agent.seed = schedule.PROBE_SEED
            cfg.scene.num_envs = cfg.scene.terrain.num_envs = schedule.PROBE_WORLDS
            _install_probe_schedule(cfg, args.arm)
            cfg.sim.nan_guard.enabled = True
            _require(agent.clip_actions is None, "frozen policy remains unfiltered and unclipped")
            validate_actor_contract(cfg)
            authenticated_parent = experiment._read_parent()
            recipe = _recipe_binding(experiment, args.arm)
            torch.set_num_threads(1)
            torch.manual_seed(schedule.PROBE_SEED)
            torch.cuda.set_per_process_memory_fraction(.20, device=0)
            configure_torch_backends()
            env = ManagerBasedRlEnv(cfg, device="cuda:0")
            try:
                _require(env.step_dt == .02 and env.num_envs == schedule.PROBE_WORLDS
                         and type(env.common_step_counter) is int and env.common_step_counter == 0,
                         "fresh 64-world 20ms environment and local counter zero")
                wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
                checkpoint, actor, actor_hash_before = _prepare_actor(
                    env, wrapped, agent, experiment, authenticated_parent)
                robot = env.scene["robot"]
                stream = _make_motor_stream(robot, env.device)
                env._microduck_motor_step_stream = stream
                phase = torch.zeros(schedule.PROBE_WORLDS, dtype=torch.long, device=env.device)
                telemetry = schedule.HeadGoalTelemetry(
                    args.arm, probe=True, expected_worlds=schedule.PROBE_WORLDS)
                records = []
                falls = resets = nan_terminations = 0
                action_abs_max = 0.0
                reward_sum = 0.0
                obs = wrapped.get_observations()
                _require(obs["actor"].shape == (schedule.PROBE_WORLDS, 61)
                         and _tree_finite(obs), "fresh raw 61D actor observation")
                with torch.inference_mode():
                    for index in range(schedule.PROBE_STEPS):
                        if index % 32 == 0:
                            if callable(getattr(args, "health", None)):
                                args.health()
                            _host_health(output, started)
                        pre_state = torch.cat((
                            robot.data.root_link_pos_w,
                            robot.data.root_link_quat_w,
                            robot.data.root_link_lin_vel_b,
                            robot.data.root_link_ang_vel_b,
                            robot.data.joint_pos,
                            robot.data.joint_vel,
                        ), dim=-1)
                        _require(_tree_finite(pre_state), "finite pre-action root and joint state")
                        # The schedule is applied to the live manager before raw actor checks.
                        experiment.validate_head_contract(env)
                        head_row = telemetry.before(env, obs["actor"])
                        _require(torch.equal(obs["actor"][:, schedule.HEAD_SLOT],
                                             env.command_manager.get_command(schedule.HEAD_TERM)),
                                 "raw actor head slots 51:55 checked before inference")
                        actions = actor(obs)
                        _require(tuple(actions.shape) == (schedule.PROBE_WORLDS, 14)
                                 and _tree_finite(actions), "finite original 14D actor actions")
                        action_abs_max = max(action_abs_max, float(actions.abs().max()))
                        stream.begin(index, phase)
                        obs, rewards, dones, extras = wrapped.step(actions)
                        _require(_tree_finite((obs, rewards, dones, extras)),
                                 "finite actual post-step observations, rewards, dones and extras")
                        _require(tuple(rewards.shape) == (schedule.PROBE_WORLDS,),
                                 "one finite reward per world")
                        sample = stream.consume(dones.bool())
                        _require(_tree_finite((sample.force_nm, sample.speed_rad_s)),
                                 "finite post-step BAM motor stream snapshot")
                        head_row = telemetry.consume(env, sample)
                        head_rewards = experiment.capture_head_reward_snapshot(
                            env, expected_step=index + 1)
                        _require(_tree_finite(head_rewards),
                                 "finite actual post-step head reward columns")
                        reward_metrics = _consume_reward_snapshots(
                            env, sample, experiment, expected_step=index + 1)
                        fell = env.termination_manager.get_term("fell_over")
                        nan_end = env.termination_manager.get_term("nan_state")
                        _require(tuple(fell.shape) == tuple(nan_end.shape) == (schedule.PROBE_WORLDS,),
                                 "actual fall and NaN termination masks")
                        falls += int(fell.sum())
                        resets += int(dones.bool().sum())
                        nan_terminations += int(nan_end.sum())
                        _require(nan_terminations == 0, "zero NaN-state terminations")
                        reward_sum += float(rewards.double().mean())
                        post_state = torch.cat((
                            robot.data.root_link_pos_w,
                            robot.data.root_link_quat_w,
                            robot.data.root_link_lin_vel_b,
                            robot.data.root_link_ang_vel_b,
                            robot.data.joint_pos,
                            robot.data.joint_vel,
                        ), dim=-1)
                        _require(_tree_finite(post_state), "finite post-step root and joint state")
                        records.append(dict(**head_row,
                            reward_mean=float(rewards.double().mean()),
                            reward_min=float(rewards.double().min()),
                            reward_max=float(rewards.double().max()),
                            action_abs_max=float(actions.abs().max()),
                            fall_worlds=int(fell.sum()), reset_worlds=int(dones.bool().sum()),
                            nan_terminations=int(nan_end.sum()),
                            bam_force_abs_mean_nm=float(sample.force_nm.double().abs().mean()),
                            bam_force_abs_max_nm=float(sample.force_nm.double().abs().max()),
                            bam_speed_abs_mean_rad_s=float(sample.speed_rad_s.double().abs().mean()),
                            bam_speed_abs_max_rad_s=float(sample.speed_rad_s.double().abs().max()),
                            state_abs_max=float(post_state.abs().max()),
                            head_reward_snapshot=head_rewards, **reward_metrics))
                        if index % 32 == 31:
                            _require(time.monotonic() - started < HEALTH_DEADLINE_SECONDS,
                                     "probe physics completes within 165s internal deadline")

                actor_hash_after = _tensor_tree_sha256(actor.state_dict())
                _require(actor_hash_after == actor_hash_before,
                         "original frozen actor state hash unchanged")
                _require(int(actor.obs_normalizer.count.item()) == 0,
                         "frozen actor normalizer count remains zero")
                telemetry_summary = telemetry.drain(expected_steps=schedule.PROBE_STEPS)
                schedule.validate_window(telemetry_summary, args.arm, True,
                    schedule.PROBE_WORLDS, 0, schedule.PROBE_STEPS)
                _require(records and len(records) == schedule.PROBE_STEPS
                         and all(_tree_finite(row) for row in records),
                         "complete finite 320-step bounded statistics")
                _require(sum(row["fall_worlds"] for row in records) == falls
                         and sum(row["reset_worlds"] for row in records) == resets
                         and falls >= 0 and resets >= 0,
                         "retained actual fall and reset counts")
                _require(time.monotonic() - started < CAP_SECONDS,
                         "boundary probe completes within 180s outer cap")
                _host_health(output, started, final=True)
                stats = dict(protocol=PROTOCOL, source=args.source, arm=args.arm,
                    parent_sha256=experiment.PARENT_SHA,
                    checkpoint_sha256=checkpoint.sha256,
                    worlds=schedule.PROBE_WORLDS, steps=schedule.PROBE_STEPS,
                    seed=schedule.PROBE_SEED, reward_mean=reward_sum/schedule.PROBE_STEPS,
                    falls=falls, resets=resets, nan_terminations=nan_terminations,
                    action_abs_max=action_abs_max, actor_state_sha256_before=actor_hash_before,
                    actor_state_sha256_after=actor_hash_after, normalizer_count=int(actor.obs_normalizer.count.item()),
                    critic_constructed=False, optimizer_constructed=False, ppo_learning=False,
                    head_goal_telemetry=telemetry_summary, step_records=records,
                    policy_acceptance=False, physical_motion_authorized=False)
                stats_path = output / "stats.json"
                _write_new(stats_path, stats)
                receipt = dict(protocol=PROTOCOL, source=args.source, run_id=args.run_id,
                    arm=args.arm, parent_sha256=experiment.PARENT_SHA,
                    checkpoint_sha256=checkpoint.sha256, output=str(output),
                    runtime=runtime, host=host, seed=schedule.PROBE_SEED,
                    worlds=schedule.PROBE_WORLDS, steps=schedule.PROBE_STEPS,
                    steps_per_update=schedule.STEPS_PER_UPDATE,
                    schedule_kind="boundary-probe",
                    stage_boundaries={"staged": [0, 80, 160], "control": [0]},
                    stats_file=stats_path.name, stats_sha256=_sha256(stats_path),
                    ordinary_smoke_receipt_sha256=prior_smokes,
                    recipe=recipe,
                    actor_state_sha256_before=actor_hash_before,
                    actor_state_sha256_after=actor_hash_after,
                    normalizer_count=int(actor.obs_normalizer.count.item()),
                    critic_constructed=False, optimizer_constructed=False, ppo_learning=False,
                    totals={"falls": falls, "resets": resets, "nan_terminations": nan_terminations,
                            "reward_mean": reward_sum/schedule.PROBE_STEPS,
                            "action_abs_max": action_abs_max},
                    status="boundary-smoke-complete-not-accepted",
                    policy_acceptance=False, physical_motion_authorized=False)
                _write_new(output / "receipt.json", receipt)
                validate_receipt(output, args.arm, args.source, experiment=experiment)
                return receipt
            finally:
                env.close()
        except Exception as exc:
            failure = dict(protocol=PROTOCOL, source=args.source, run_id=args.run_id,
                           arm=args.arm, type=type(exc).__name__, error=str(exc),
                           elapsed_s=time.monotonic()-started,
                           policy_acceptance=False, physical_motion_authorized=False)
            try:
                _write_new(output / "failure.json", failure)
            except FileExistsError:
                pass
            raise


def validate_receipt(root, arm, source, *, experiment=None):
    """Independently authenticate and recompute one durable boundary receipt."""
    _require(arm in schedule.ARMS, "declared boundary receipt arm")
    root = Path(root)
    output = root if root.name == "boundary-smoke" else root / arm / "boundary-smoke"
    receipt_path = output / "receipt.json"
    stats_path = output / "stats.json"
    _require(receipt_path.is_file() and not receipt_path.is_symlink()
             and stats_path.is_file() and not stats_path.is_symlink(),
             "regular durable boundary receipt and stats files")
    receipt = json.loads(receipt_path.read_text())
    stats = json.loads(stats_path.read_text())
    for path in (receipt_path, stats_path):
        info = path.lstat()
        _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                 and info.st_mode & 0o222 == 0,
                 "owned regular immutable boundary artifact")
    _require(receipt.get("protocol") == stats.get("protocol") == PROTOCOL
             and receipt.get("source") == stats.get("source") == source
             and receipt.get("arm") == stats.get("arm") == arm
             and receipt.get("status") == "boundary-smoke-complete-not-accepted",
             "boundary receipt protocol, source, arm and status")
    _require(type(source) is str and len(source) == 40
             and all(char in "0123456789abcdef" for char in source),
             "exact lowercase source commit identity")
    _require(receipt.get("stats_file") == "stats.json"
             and receipt.get("stats_sha256") == _sha256(stats_path),
             "boundary stats file hash matches immutable receipt")
    _require((receipt.get("seed"), receipt.get("worlds"), receipt.get("steps"),
              receipt.get("schedule_kind")) ==
             (schedule.PROBE_SEED, schedule.PROBE_WORLDS, schedule.PROBE_STEPS, "boundary-probe")
             and (stats.get("seed"), stats.get("worlds"), stats.get("steps")) ==
             (schedule.PROBE_SEED, schedule.PROBE_WORLDS, schedule.PROBE_STEPS),
             "exact named 971 / 64-world / 320-step boundary protocol")
    _require(receipt.get("checkpoint_sha256") == stats.get("checkpoint_sha256") == PARENT_SHA256
             and receipt.get("critic_constructed") is False
             and stats.get("critic_constructed") is False
             and receipt.get("optimizer_constructed") is False
             and receipt.get("ppo_learning") is False
             and stats.get("critic_constructed") is False
             and stats.get("optimizer_constructed") is False
             and stats.get("ppo_learning") is False
             and receipt.get("normalizer_count") == stats.get("normalizer_count") == 0
             and receipt.get("actor_state_sha256_before") == receipt.get("actor_state_sha256_after")
             == stats.get("actor_state_sha256_before") == stats.get("actor_state_sha256_after"),
             "frozen actor unchanged, count-zero and optimizer-free inference")
    _require(receipt.get("ordinary_smoke_receipt_sha256")
             and set(receipt["ordinary_smoke_receipt_sha256"]) == set(schedule.ARMS),
             "both same-source ordinary smoke receipts precede the probe")
    recipe = receipt.get("recipe")
    declared_probe_schedule = json.loads(json.dumps(schedule.stages_for(arm, probe=True)))
    _require(isinstance(recipe, dict) and recipe.get("protocol") == PROTOCOL
             and recipe.get("arm") == arm
             and recipe.get("schedule") == declared_probe_schedule
             and (recipe.get("seed"), recipe.get("worlds"), recipe.get("steps")) ==
                 (schedule.PROBE_SEED, schedule.PROBE_WORLDS, schedule.PROBE_STEPS),
             "recipe receipt binds exact accelerated probe schedule and budget")
    recipe_digest = recipe.get("probe_recipe_sha256")
    recipe_body = {key: value for key, value in recipe.items() if key != "probe_recipe_sha256"}
    _require(recipe_digest == hashlib.sha256(json.dumps(recipe_body, sort_keys=True,
             separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
             "probe recipe hash recomputes from declared arm and schedule")
    telemetry = stats.get("head_goal_telemetry")
    schedule.validate_window(telemetry, arm, True, schedule.PROBE_WORLDS, 0, schedule.PROBE_STEPS)
    rows = stats.get("step_records")
    _require(isinstance(rows, list) and len(rows) == schedule.PROBE_STEPS
             and _tree_finite(rows), "complete finite 320-step retained diagnostics")
    telemetry_rows = telemetry["step_records"]
    for index, row in enumerate(rows):
        tracking = row.get("head_reward_snapshot", {})
        tracking_values = tracking.get("head_pose_tracking_weighted_rewards")
        bias_values = tracking.get("head_pose_bias_weighted_rewards")
        _require(row.get("fall_worlds", -1) >= 0 and row.get("reset_worlds", -1) >= 0
                 and row.get("nan_terminations", -1) == 0
                 and row.get("before_step") == index and row.get("completed_step") == index + 1
                 and row.get("stage") == telemetry_rows[index].get("stage")
                 and row.get("ranges") == telemetry_rows[index].get("ranges")
                 and row.get("yaw_bias_cost_mean", -1) >= 0
                 and row.get("yaw_bias_weighted_reward_mean", 1) <= 0
                 and math.isclose(row.get("yaw_bias_weighted_reward_mean"),
                                  -row.get("yaw_bias_cost_mean"), abs_tol=1e-6, rel_tol=1e-5)
                 and row.get("selected_motor_cost_mean", -1) >= 0
                 and row.get("selected_motor_weighted_reward_mean") == 0.0
                 and isinstance(tracking_values, list) and len(tracking_values) == schedule.PROBE_WORLDS
                 and isinstance(bias_values, list) and len(bias_values) == schedule.PROBE_WORLDS
                 and all(type(value) in (int, float) and math.isfinite(value)
                         for value in tracking_values + bias_values)
                 and all(value >= 0 for value in tracking_values)
                 and all(value <= 0 for value in bias_values)
                 and tracking.get("step") == index + 1
                 and math.isclose(tracking.get("head_pose_tracking_weighted_reward_mean", math.nan),
                     math.fsum(tracking_values) / schedule.PROBE_WORLDS, abs_tol=1e-7, rel_tol=1e-6)
                 and tracking.get("head_pose_tracking_weighted_reward_min") == min(tracking_values)
                 and tracking.get("head_pose_tracking_weighted_reward_max") == max(tracking_values)
                 and math.isclose(tracking.get("head_pose_bias_weighted_reward_mean", math.nan),
                     math.fsum(bias_values) / schedule.PROBE_WORLDS, abs_tol=1e-7, rel_tol=1e-6)
                 and tracking.get("head_pose_bias_weighted_reward_min") == min(bias_values)
                 and tracking.get("head_pose_bias_weighted_reward_max") == max(bias_values),
                 "contiguous finite reward, fall/reset and actual command records")
    _require(stats.get("falls") == sum(row["fall_worlds"] for row in rows)
             and stats.get("resets") == sum(row["reset_worlds"] for row in rows)
             and stats.get("nan_terminations") == 0
             and stats.get("actor_state_sha256_before") == stats.get("actor_state_sha256_after"),
             "receipt totals reconcile with retained actual per-step records")
    _require(receipt.get("totals") == {
                 "falls": stats["falls"], "resets": stats["resets"],
                 "nan_terminations": stats["nan_terminations"],
                 "reward_mean": stats["reward_mean"],
                 "action_abs_max": stats["action_abs_max"]},
             "receipt summary totals equal bounded stats artifact")
    if experiment is not None:
        parent = experiment._read_parent()
        expected_parent_actor_hash = _tensor_tree_sha256(parent["actor_state_dict"])
        expected_smokes = _validate_prior_smokes(output.parent.parent, source, experiment)
        _require(experiment.PARENT_SHA == PARENT_SHA256
                 and receipt.get("parent_sha256") == experiment.PARENT_SHA
                 and stats.get("parent_sha256") == experiment.PARENT_SHA
                 and receipt.get("checkpoint_sha256") == experiment.PARENT_SHA
                 and stats.get("checkpoint_sha256") == experiment.PARENT_SHA
                 and stats.get("actor_state_sha256_before") == expected_parent_actor_hash
                 and recipe == _recipe_binding(experiment, arm)
                 and receipt.get("ordinary_smoke_receipt_sha256") == expected_smokes,
                 "exact authenticated parent actor, recipe, and preceding smoke identities")
    else:
        _require(receipt.get("parent_sha256") == stats.get("parent_sha256") == PARENT_SHA256,
                 "exact original frozen parent identity")
    return receipt


def main(argv=None, experiment=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", required=True, choices=schedule.ARMS)
    args = parser.parse_args(argv)
    if experiment is None:
        from . import v8_staged_head_goals as experiment
    return run(args, experiment)


__all__ = ["PROTOCOL", "PARENT_SHA256", "CAP_SECONDS", "HEALTH_DEADLINE_SECONDS", "run",
           "validate_receipt", "main"]
