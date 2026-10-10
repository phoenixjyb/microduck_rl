"""Bounded fresh-policy walking/arc foundation on the integrated plant.

Not an old-checkpoint transfer, simulator qualification, or physical acceptance.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import time

import torch

from mjlab_microduck.evaluation import fix_velocity_commands
from mjlab_microduck.exploratory_turn_pilot import LOCK, shared_host
from mjlab_microduck.motor_step_stream import MotorStepCostCfg, MotorStepStream, install_metric
from mjlab_microduck.tasks.run import XL330_M288_RATED_NO_LOAD_SPEED_RAD_S as RATED_SPEED

ROOT = Path(__file__).resolve().parents[2]
TASK = "Mjlab-Velocity-Flat-MicroDuck"
PROTOCOL = "integrated-moving-turn-foundation-v1"
MODES = {"smoke": (64, 5, 821), "benchmark": (256, 10, 823), "foundation": (256, 1500, 827)}
SEEDS = (839, 853, 857)
CASES = ((0., 0.), (.30, 0.), (.20, -.20), (.20, .20))
EVAL_CHECKPOINTS = (750, 1499)
FOUNDATION_CAP_SECONDS = 2700


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, data):
    with Path(path).open("x") as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())


def prepare_config(mode):
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    worlds, updates, seed = MODES[mode]
    cfg, agent = load_env_cfg(TASK), load_rl_cfg(TASK)
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (.20, .40)
    command.ranges.lin_vel_y = (0., 0.)
    command.ranges.ang_vel_z = (-.50, .50)
    command.resampling_time_range = (3., 5.)
    command.heading_command = False
    command.ranges.heading = None
    command.rel_heading_envs = command.rel_world_envs = 0.
    command.rel_forward_envs = .25
    command.rel_turn_in_place_envs = 0.
    command.init_velocity_prob = 0.
    # Preserve upstream standing, smoothing, posture and CoM curricula. No
    # accelerated schedule or new yaw reward; this is a fresh foundation.
    cfg.sim.nan_guard.enabled = True
    install_metric(cfg)
    agent.logger, agent.upload_model = "tensorboard", False
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    agent.max_iterations, agent.save_interval = updates, 250
    return cfg, agent


def validate_commands(command):
    require(command.ndim == 2 and command.shape[1] == 3 and finite_tree(command), "finite twist")
    idle = (command == 0).all(-1)
    moving = (command[:, 0] >= .20-1e-6) & (command[:, 0] <= .40+1e-6)
    require(bool(((idle | moving) & (command[:, 1] == 0) &
                  (command[:, 2].abs() <= .50+1e-6)).all()), "declared moving/idle commands")


def training(output, mode, health, *, recipe=None, initialize=None,
             before_action=None, after_step=None, after_update=None):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls

    cfg, agent = prepare_config(mode) if recipe is None else recipe
    worlds, updates, seed = cfg.scene.num_envs, agent.max_iterations, agent.seed
    torch.manual_seed(seed)
    started = time.monotonic()
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = load_runner_cls(TASK)(wrapped, asdict(agent), str(output), "cuda:0")
        if initialize is not None:
            initialize(runner, env)
        stream = MotorStepStream.from_robot(env.scene["robot"], worlds, device=env.device,
                                            cost_cfg=MotorStepCostCfg())
        env._microduck_motor_step_stream = stream
        phase = torch.zeros(worlds, dtype=torch.long, device=env.device)
        counters = dict(steps=0, updates=0, idle=0, straight=0, negative_yaw=0, positive_yaw=0)
        motor_rows, reward_sum, falls, nan_ends = [], 0., 0, 0
        pending = None
        act, step, update, save = runner.alg.act, wrapped.step, runner.alg.update, runner.save

        def checked_act(obs):
            nonlocal pending
            if before_action is not None:
                before_action(env)
            require(pending is None and obs["actor"].shape == (worlds, 61) and finite_tree(obs),
                    "finite actor/critic input and actor order")
            command = env.command_manager.get_command("twist")
            validate_commands(command)
            require(torch.equal(obs["actor"][:, 48:51], command), "raw actor command equals manager")
            pending = command.clone()
            actions = act(obs)
            require(actions.shape == (worlds, 14) and finite_tree(actions), "finite 14 motor actions")
            return actions

        def checked_step(actions):
            nonlocal pending, reward_sum, falls, nan_ends
            command = env.command_manager.get_command("twist")
            require(pending is not None and torch.equal(command, pending), "actor command consumed by physics")
            idle = (command == 0).all(-1)
            counters["idle"] += int(idle.sum())
            counters["straight"] += int(((command[:, 2].abs() < .01) & ~idle).sum())
            counters["negative_yaw"] += int((command[:, 2] < -.01).sum())
            counters["positive_yaw"] += int((command[:, 2] > .01).sum())
            stream.begin(counters["steps"], phase)
            obs, rewards, dones, extras = step(actions)
            require(finite_tree((obs, rewards)), "finite rollout")
            sample = stream.consume(dones.bool())
            if after_step is not None:
                after_step(env, sample, extras)
            motor_rows.append(torch.stack((sample.force_nm, sample.speed_rad_s), -1))
            falls += int(env.termination_manager.get_term("fell_over").sum())
            nan_ends += int(env.termination_manager.get_term("nan_state").sum())
            require(nan_ends == 0, "NaN termination (sanitized rewards are not raw-term proof)")
            reward_sum += float(rewards.mean())
            counters["steps"] += 1
            pending = None
            return obs, rewards, dones, extras

        def checked_update():
            nonlocal reward_sum, falls, nan_ends
            losses = update()
            require(finite_tree(losses), "finite optimizer losses")
            counters["updates"] += 1
            if counters["updates"] % 25 == 0 or counters["updates"] == updates:
                require(finite_tree(runner.alg.save()), "finite models and Adam every25 updates/final")
            motors = torch.stack(motor_rows).double()
            force, speed = motors[..., 0], motors[..., 1]
            utilization = force.abs() / .60
            row = dict(update=counters["updates"], steps=counters["steps"],
                       elapsed_s=time.monotonic()-started, mean_reward=reward_sum/24,
                       falls=falls, fall_fraction=falls/(worlds*24), nan_terminations=nan_ends,
                       torque_p99=float(torch.quantile(utilization.flatten(), .99)),
                       soft_limit_fraction=float((utilization > .70).double().mean()),
                       rated_speed_exceed_fraction=float((speed.abs() > RATED_SPEED).double().mean()),
                       mean_abs_mechanical_power_w=float((force*speed).abs().sum(-1).mean()),
                       thermal_load_proxy=float(utilization.square().mean()), losses=losses)
            if after_update is not None:
                row.update(after_update(env))
            require(row["torque_p99"] <= 1.5 and row["rated_speed_exceed_fraction"] <= .10,
                    "gross fresh-policy motor abort guard")
            # Fresh random-policy falls are expected. Admission uses strict
            # first-episode held-out gates, not an early training fall guard.
            with (output / "updates.jsonl").open("a") as log:
                log.write(json.dumps(row, allow_nan=False)+"\n")
                log.flush()
            if counters["updates"] % 25 == 0 or counters["updates"] == updates:
                health()
                print("FOUNDATION_PROGRESS", json.dumps(row, allow_nan=False), flush=True)
            motor_rows.clear()
            reward_sum, falls, nan_ends = 0., 0, 0
            return losses

        def durable_save(path, infos=None):
            temporary = str(path)+".partial"
            save(temporary, infos)
            with open(temporary, "rb") as payload:
                os.fsync(payload.fileno())
            os.replace(temporary, path)
            directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)

        runner.alg.act, wrapped.step, runner.alg.update, runner.save = checked_act, checked_step, checked_update, durable_save
        runner.save(str(output / "initial.pt"))
        runner.learn(num_learning_iterations=updates, init_at_random_ep_len=False)
        require(counters["steps"] == updates*24 and counters["updates"] == updates, "complete PPO budget")
        require(all(counters[k] > 0 for k in ("idle", "straight", "negative_yaw", "positive_yaw")), "actual command coverage")
        checkpoint = output / f"model_{updates-1}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        require(finite_tree(payload) and payload["optimizer_state_dict"]["state"], "durable finite model/Adam")
        return dict(status="training-complete-not-accepted", mode=mode, **counters,
                    worlds=worlds, seed=seed, elapsed_s=time.monotonic()-started,
                    transitions=worlds*counters["steps"], checkpoint=str(checkpoint),
                    checkpoint_sha256=sha256(checkpoint), motor_stream=stream.provenance())
    finally:
        env.close()


def response_diagnostics(velocities, motors, positions, limits, names, yaw, *, startup=60):
    """Descriptive signed response; never changes the admission predicate."""
    require(velocities.ndim == motors.ndim-1 == positions.ndim == 3,
            "diagnostic trajectory ranks")
    steps, worlds, joints = positions.shape
    require(velocities.shape == (steps, worlds, 2) and motors.shape == (steps, worlds, joints, 2)
            and limits.shape == (worlds, joints, 2) and len(names) == len(set(names)) == joints
            and finite_tree((velocities, motors, positions, limits)), "finite diagnostic identity/layout")
    require(bool((limits[..., 1] > limits[..., 0]).all()), "finite ordered hard joint bounds")
    settled = velocities[startup:]
    force = motors[..., 0]
    distance = torch.minimum(positions-limits[..., 0], limits[..., 1]-positions)
    return dict(protocol="signed-body-response-and-joint-load-v1", joint_columns=list(names),
        steps=steps, startup_steps=startup, settled_steps=len(settled),
        mean_body_vx_per_world=settled[..., 0].mean(0).tolist() if len(settled) else None,
        mean_body_yaw_per_world=settled[..., 1].mean(0).tolist() if len(settled) else None,
        mean_abs_body_yaw_per_world=settled[..., 1].abs().mean(0).tolist() if len(settled) else None,
        yaw_correct_sign_fraction_per_world=(settled[..., 1]*yaw > 0).double().mean(0).tolist()
            if len(settled) and yaw != 0 else None,
        abs_force_p99_nm_by_joint=torch.quantile(force.abs().flatten(0, 1), .99, dim=0).tolist(),
        mean_signed_force_nm_by_joint=force.mean((0, 1)).tolist(),
        soft_limit_fraction_by_joint=(force.abs()/.60 > .70).double().mean((0, 1)).tolist(),
        hard_stop_margin_rad=.05,
        hard_stop_proximity_fraction_by_joint=(distance <= .05).double().mean((0, 1)).tolist(),
        hard_range_violation_fraction_by_joint=(distance < 0).double().mean((0, 1)).tolist(),
        min_distance_to_hard_stop_rad_by_joint=distance.amin((0, 1)).tolist(),
        velocity_and_position_timing="pre-action body-frame state; startup excluded only for response",
        force_timing="post-decimation pre-reset control-step stream; all steps included",
        positions_not_synchronous_with_force=True, policy_acceptance=False, physical_motion_authorized=False)


def joint_target_diagnostics(positions, targets, constraints, limits, names):
    """Previous applied targets and generalized constraint loads; no clipping.

    All samples are at the next pre-action boundary. Constraint force is the
    last derived physics value, not a simultaneous hard-stop force sensor.
    """
    require(positions.ndim == 3 and positions.shape == targets.shape == constraints.shape
            and limits.shape == (*positions.shape[1:], 2)
            and len(names) == len(set(names)) == positions.shape[-1]
            and finite_tree((positions, targets, constraints, limits))
            and bool((limits[..., 1] > limits[..., 0]).all()), "finite target diagnostic layout")
    def stats(values):
        return dict(mean=values.mean((0, 1)).tolist(), minimum=values.amin((0, 1)).tolist(),
                    maximum=values.amax((0, 1)).tolist())
    distance = torch.minimum(targets-limits[..., 0], limits[..., 1]-targets)
    return dict(protocol="previous-applied-joint-target-v1", joint_columns=list(names),
        position_rad=stats(positions), previous_applied_target_rad=stats(targets),
        target_outside_configured_range_fraction=(distance < 0).double().mean((0, 1)).tolist(),
        generalized_constraint_force_nm=stats(constraints),
        constraint_abs_p99_nm=torch.quantile(constraints.abs().flatten(0, 1), .99, dim=0).tolist(),
        configured_limits_rad=limits.tolist(),
        timing="pre-action state and previous applied target; constraint force has solver integration lag",
        constraint_includes_contacts_and_limits=True, hard_stop_force_isolated=False,
        policy_acceptance=False, physical_motion_authorized=False)


def evaluate_case(checkpoint, seed, speed, yaw, health, *, diagnostics=False, target_diagnostics=False):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    require(not target_diagnostics or diagnostics, "target capture requires position diagnostics")
    worlds, steps, startup = 8, 240, 60
    torch.manual_seed(seed)
    cfg, agent = load_env_cfg(TASK, play=True), load_rl_cfg(TASK)
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    cfg.curriculum.clear()
    cfg.events.pop("push_robot", None)
    cfg.sim.nan_guard.enabled = True
    fix_velocity_commands(cfg, speed, yaw)
    install_metric(cfg)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        require(env.step_dt == .02, "20ms evaluation control step")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = load_runner_cls(TASK)(wrapped, asdict(agent), device="cuda:0")
        runner.load(str(checkpoint), strict=True, map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
        obs, robot = wrapped.get_observations(), env.scene["robot"]
        stream = MotorStepStream.from_robot(robot, worlds, device=env.device, cost_cfg=MotorStepCostCfg())
        env._microduck_motor_step_stream = stream
        phase = torch.zeros(worlds, dtype=torch.long, device=env.device)
        velocities, motor_rows, terminals = [], [], []
        positions, targets, constraints = [], [], []
        limits = robot.data.joint_pos_limits[:, stream.joint_ids].cpu().clone() if diagnostics else None
        with torch.inference_mode():
            for step in range(steps):
                if step % 120 == 0:
                    health()
                command = env.command_manager.get_command("twist")
                require(torch.allclose(command, command.new_tensor([speed, 0., yaw]).expand_as(command),
                                       atol=1e-6, rtol=0), "fixed evaluation command")
                require(obs["actor"].shape == (worlds, 61) and finite_tree(obs), "finite evaluation obs")
                require(torch.equal(obs["actor"][:, 48:51], command), "evaluation actor command")
                velocity = torch.stack((robot.data.root_link_lin_vel_b[:, 0], robot.data.root_link_ang_vel_b[:, 2]), -1)
                require(finite_tree(velocity), "finite measured body velocity")
                velocities.append(velocity.cpu().clone())
                if diagnostics:
                    positions.append(robot.data.joint_pos[:, stream.joint_ids].cpu().clone())
                if target_diagnostics:
                    targets.append(robot.data.joint_pos_target[:, stream.joint_ids].cpu().clone())
                    dofs = robot.data.indexing.joint_v_adr[list(stream.joint_ids)]
                    constraints.append(robot.data.data.qfrc_constraint[:, dofs].cpu().clone())
                actions = policy(obs)
                require(actions.shape == (worlds, 14) and finite_tree(actions), "finite evaluation actions")
                stream.begin(step, phase)
                obs, rewards, dones, _ = wrapped.step(actions)
                require(finite_tree((obs, rewards)), "finite evaluation output")
                sample = stream.consume(dones.bool())
                motor_rows.append(torch.stack((sample.force_nm, sample.speed_rad_s), -1).cpu())
                if bool(dones.any()):
                    terminals = dones.nonzero().flatten().tolist()
                    break
        v, m = torch.stack(velocities).double(), torch.stack(motor_rows).double()
        f, s = m[..., 0], m[..., 1]
        u = f.abs()/.60
        settled = v[startup:]
        result = dict(seed=seed, speed=speed, yaw=yaw, worlds=worlds, steps=len(v),
                    complete=len(v) == steps and not terminals, terminal_worlds=terminals,
                    checkpoint_sha256=sha256(checkpoint), torque_p99=float(torch.quantile(u.flatten(), .99)),
                    rated_speed_exceed_fraction=float((s.abs() > RATED_SPEED).double().mean()),
                    soft_limit_fraction=float((u > .70).double().mean()),
                    mean_abs_mechanical_power_w=float((f*s).abs().sum(-1).mean()),
                    thermal_load_proxy=float(u.square().mean()),
                    speed_mae=(settled[..., 0]-speed).abs().mean(0).tolist() if len(settled) else None,
                    yaw_mae=(settled[..., 1]-yaw).abs().mean(0).tolist() if len(settled) else None,
                    motor_stream=stream.provenance())
        if diagnostics:
            result["response_diagnostics"] = response_diagnostics(v, m, torch.stack(positions).double(),
                limits.double(), stream.names, yaw, startup=startup)
        if target_diagnostics:
            result["joint_target_diagnostics"] = joint_target_diagnostics(torch.stack(positions).double(),
                torch.stack(targets).double(), torch.stack(constraints).double(), limits.double(), stream.names)
        return result
    finally:
        env.close()


def decision(rows):
    require(len(rows) == len(SEEDS)*len(CASES), "complete fixed final matrix")
    failures = []
    for row, (seed, case) in zip(rows, ((s, c) for s in SEEDS for c in CASES), strict=True):
        require((row["seed"], row["speed"], row["yaw"]) == (seed, *case), "ordered held-out cases")
        require(finite_tree(row), "finite evaluation report")
        if not row["complete"]:
            failures.append([seed, *case, "first-episode-terminal"])
            continue
        require(row["worlds"] == 8 and row["steps"] == 240 and not row["terminal_worlds"], "first episode coverage")
        require(len(row["speed_mae"]) == len(row["yaw_mae"]) == 8 and
                all(v >= 0 for v in row["speed_mae"]+row["yaw_mae"]), "eight-world error coverage")
        speed_limit = .03 if case[0] == 0 else .10
        if max(row["speed_mae"]) > speed_limit or max(row["yaw_mae"]) > .10:
            failures.append([seed, *case, "speed-or-yaw-tracking"])
        if row["torque_p99"] > .60 or row["rated_speed_exceed_fraction"] != 0:
            failures.append([seed, *case, "motor-envelope"])
    return dict(decision="foundation-ready-for-next-experiment" if not failures else "foundation-not-ready",
                failures=failures, policy_acceptance=False, simulator_qualified=False,
                physical_motion_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=(*MODES, "evaluate"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    require(re.fullmatch(r"[a-z0-9-]{1,64}", args.run_id), "safe run id")
    def git(*argv):
        return subprocess.check_output(("git", "-C", str(ROOT), *argv), text=True).strip()
    require(git("rev-parse", "HEAD") == args.source and not git("status", "--porcelain"), "exact clean source")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "literal isolated device selection")
    require(LOCK.is_file() and not LOCK.is_symlink(), "existing nonsymlink GPU lock")
    root = ROOT / "artifacts/training" / args.run_id
    output = root / args.mode
    with LOCK.open("r+") as lease:
        info = os.fstat(lease.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(), "owned regular GPU lock")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host = shared_host()
        if args.mode in ("benchmark", "foundation"):
            smoke = json.loads((root / "smoke/result.json").read_text())
            require(smoke["source"] == args.source and smoke["status"] == "training-complete-not-accepted"
                    and smoke["updates"] == 5 and smoke["worlds"] == 64, "same-source64 smoke")
        if args.mode == "foundation":
            bench = json.loads((root / "benchmark/result.json").read_text())
            require(bench["source"] == args.source and bench["updates"] == 10 and bench["worlds"] == 256
                    and bench["status"] == "training-complete-not-accepted", "same-source256 benchmark")
            require(bench["elapsed_s"]*150*1.5 < FOUNDATION_CAP_SECONDS-60,
                    "measured1500 update budget fits declared cap")
        output.mkdir(parents=True, exist_ok=False)
        runtime = {name: metadata.version(name) for name in
                   ("torch", "warp-lang", "mujoco", "mujoco-warp", "mjlab", "better-actuator-models")}
        write_new(output / "launch.json", dict(protocol=PROTOCOL, source=args.source,
                  mode=args.mode, runtime=runtime, host=host, old_policy_parent=None,
                  policy_acceptance=False, physical_motion_authorized=False))
        def health():
            sample = shared_host()
            with (output / "health.jsonl").open("a") as log:
                log.write(json.dumps(dict(utc=time.time(), **sample), allow_nan=False)+"\n")
                log.flush(); os.fsync(log.fileno())
            return sample
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20, device=0)
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        try:
            if args.mode == "evaluate":
                trained = json.loads((root / "foundation/result.json").read_text())
                require(trained["source"] == args.source and trained["updates"] == 1500 and
                        trained["status"] == "training-complete-not-accepted", "exact complete foundation")
                require(sha256(root / "foundation/model_1499.pt") == trained["checkpoint_sha256"], "final bytes")
                all_rows = {}
                for iteration in EVAL_CHECKPOINTS:
                    checkpoint = root / f"foundation/model_{iteration}.pt"
                    require(checkpoint.is_file(), "predeclared checkpoint exists")
                    rows = []
                    for seed in SEEDS:
                        for index, (speed, yaw) in enumerate(CASES):
                            row = evaluate_case(checkpoint, seed, speed, yaw, health)
                            write_new(output / f"k{iteration}-s{seed}-c{index}.json", row)
                            rows.append(row)
                    all_rows[str(iteration)] = rows
                result = {**decision(all_rows["1499"]), "cases": all_rows,
                          "selection": "final1499 only;750 descriptive, no checkpoint fishing"}
            else:
                result = training(output, args.mode, health)
            write_new(output / "result.json", dict(protocol=PROTOCOL, source=args.source, **result))
            health()
        except Exception as exc:
            write_new(output / "failure.json", dict(type=type(exc).__name__, error=str(exc), source=args.source))
            raise


if __name__ == "__main__":
    main()
