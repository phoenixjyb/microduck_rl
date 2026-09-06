"""Bounded no-obstacle speed response; diagnoses but never admits training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import time

import torch

from mjlab_microduck.evaluation import fix_velocity_commands
from mjlab_microduck.first_attempt_smoke import ACTOR_SHA256, require, sha256
from mjlab_microduck.motor_audit_smoke import DEPENDENCIES, JOINTS, VERSIONS
from mjlab_microduck.motor_step_stream import MotorStepStream, MotorStepCostCfg, install_metric
from mjlab_microduck.recovery_ab import DEADLINE, verify_source, write_new
from mjlab_microduck.recovery_measurement import RecoveryMeasurement
from mjlab_microduck.rollout_repeatability import ROOT, ACTOR, check_host
from mjlab_microduck.tasks.run import XL330_M288_RATED_NO_LOAD_SPEED_RAD_S

PROTOCOL = "frozen-straight-speed-s383-v1"
TASK = "Mjlab-Run-MotorAware-Flat-MicroDuck"
OUTPUT = ROOT / "artifacts/evaluations" / PROTOCOL
SEED, NUM_ENVS, STEPS, SETTLE = 383, 8, 400, 100
SPEED, STEP_DT = .3, .02
SERVICE_SECONDS, CLOSEOUT_SECONDS = 240, 300


def prepare_config():
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    cfg, agent = load_env_cfg(TASK, play=True), load_rl_cfg(TASK)
    cfg.seed = agent.seed = SEED
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = NUM_ENVS
    fix_velocity_commands(cfg, SPEED, 0.)
    cfg.curriculum.clear()
    cfg.events.pop("push_robot", None)
    agent.logger, agent.upload_model = "tensorboard", False
    require("obstacle" not in cfg.scene.entities, "no obstacle in straight control")
    install_metric(cfg)
    return cfg, agent


def velocity_rows(world_velocity, body_velocity, quaternion, route_dir):
    """Keep full body-frame x and planar route projection distinct."""
    n = world_velocity.shape[0]
    require(world_velocity.shape == body_velocity.shape == (n, 3)
            and quaternion.shape == (n, 4) and route_dir.shape == (n, 2), "velocity frame shapes")
    require(all(bool(torch.isfinite(v).all()) for v in (world_velocity, body_velocity, quaternion, route_dir)),
            "finite frame measurements")
    require(bool(torch.allclose(torch.linalg.vector_norm(route_dir, dim=1), torch.ones_like(route_dir[:, 0]),
                                atol=1e-5, rtol=0)), "unit route direction")
    require(bool(torch.allclose(torch.linalg.vector_norm(quaternion, dim=1), torch.ones_like(quaternion[:, 0]),
                                atol=1e-5, rtol=0)), "unit quaternion")
    qw, qx, qy, qz = quaternion.unbind(-1)
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy.square() + qz.square()))
    heading = yaw - torch.atan2(route_dir[:, 1], route_dir[:, 0])
    heading = torch.atan2(torch.sin(heading), torch.cos(heading))
    cross_dir = torch.stack((-route_dir[:, 1], route_dir[:, 0]), -1)
    return torch.stack((body_velocity[:, 0], (world_velocity[:, :2] * route_dir).sum(-1),
                        (world_velocity[:, :2] * cross_dir).sum(-1), heading), -1).detach().clone()


def summarize(velocities, commands, legacy_force, legacy_speed, pre_force, pre_speed, terminal_steps,
              measurement):
    """Consume retained bounded tensors; no simulator or policy action."""
    count = velocities.shape[0]
    require(1 <= count <= STEPS and velocities.shape == (count, NUM_ENVS, 4)
            and commands.shape == (count, NUM_ENVS, 3), "bounded sample shapes")
    require(all(v.shape == (count, NUM_ENVS, 14) for v in
                (legacy_force, legacy_speed, pre_force, pre_speed)), "named motor coverage")
    require(all(bool(torch.isfinite(v).all()) for v in
                (velocities, commands, legacy_force, legacy_speed, pre_force, pre_speed)), "finite sample values")
    require(type(terminal_steps) is list and all(type(i) is int and 0 <= i < count for i in terminal_steps),
            "terminal step accounting")
    result = dict(protocol=PROTOCOL, decision="diagnostic-only", physical_motion_authorized=False,
                  policy_acceptance=False, training_admitted=False, reopens_recovery_ab=False,
                  sample_steps=count, startup_steps=min(SETTLE, count), settled_steps=max(0, count-SETTLE),
                  terminal_steps=terminal_steps, speed_mps=SPEED, seed=SEED, num_envs=NUM_ENVS,
                  step_dt_s=STEP_DT, warmup_failures_ignored=False,
                  velocity_sampling="pre-control-step; body x and initial-heading world projection",
                  stable_route_response=measurement, groups={})
    for name, start in (("all", 0), ("settled", SETTLE)):
        if count <= start: continue
        v = velocities[start:].double()
        force = legacy_force[start:].float().abs() / .6
        speed = legacy_speed[start:].float().abs() / XL330_M288_RATED_NO_LOAD_SPEED_RAD_S
        pre = pre_force[start:].double().abs() / .6
        pre_velocity = pre_speed[start:].double().abs() / XL330_M288_RATED_NO_LOAD_SPEED_RAD_S
        result["groups"][name] = dict(
            steps=count-start, body_forward_per_env_mean=v[:, :, 0].mean(0).tolist(),
            route_forward_per_env_mean=v[:, :, 1].mean(0).tolist(),
            body_forward_mean=float(v[:, :, 0].mean()), route_forward_mean=float(v[:, :, 1].mean()),
            route_speed_p05=float(torch.quantile(v[:, :, 1].flatten(), .05)),
            route_speed_p95=float(torch.quantile(v[:, :, 1].flatten(), .95)),
            cross_route_abs_mean=float(v[:, :, 2].abs().mean()), heading_abs_max=float(v[:, :, 3].abs().max()),
            legacy_torque_p99=float(torch.quantile(force.flatten(), .99)),
            legacy_rated_speed_exceed_fraction=float((speed > 1).float().mean()),
            pre_reset_torque_p99=float(torch.quantile(pre.flatten(), .99)),
            pre_reset_rated_speed_exceed_fraction=float((pre_velocity > 1).double().mean()),
            pre_reset_squared_utilization_mean=float(pre.square().mean()),
            pre_reset_joint_p99={joint: float(torch.quantile(pre[:, :, i].flatten(), .99))
                                 for i, joint in enumerate(JOINTS)})
    expected = torch.tensor([SPEED, 0., 0.], device=commands.device, dtype=commands.dtype)
    command_ok = bool(torch.allclose(commands, expected.expand_as(commands), atol=1e-6, rtol=0))
    failures = []
    if not command_ok: failures.append("command-not-fixed")
    if terminal_steps: failures.append("terminal-including-startup")
    if count != STEPS: failures.append("incomplete-control")
    for group, values in result["groups"].items():
        if values["legacy_torque_p99"] > .60: failures.append(f"{group}-legacy-torque")
        if values["legacy_rated_speed_exceed_fraction"] != 0: failures.append(f"{group}-legacy-rated-speed")
        if values["pre_reset_rated_speed_exceed_fraction"] != 0: failures.append(f"{group}-pre-reset-rated-speed")
    result["safety_failures"] = failures
    if failures:
        result["classification"] = "safety-or-coverage-stop"
    else:
        settled = result["groups"]["settled"]
        body_ok = all(abs(x - SPEED) <= .03 for x in settled["body_forward_per_env_mean"])
        route_ok = all(abs(x - SPEED) <= .03 for x in settled["route_forward_per_env_mean"])
        stable = measurement["counts"]["recovered-in-window"] == NUM_ENVS
        result.update(body_mean_in_band_all_envs=body_ok, route_mean_in_band_all_envs=route_ok,
                      stable_route_window_all_envs=stable)
        result["classification"] = ("straight-body-mean-outside-band" if not body_ok else
            "body-route-response-diverge" if not route_ok else
            "mean-tracking-but-instantaneous-window-missed" if not stable else "straight-response-within-both-criteria")
    return result


def run_control(*, device="cuda:0"):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls

    cfg, agent = prepare_config()
    torch.manual_seed(SEED)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
        require(env.step_dt == STEP_DT, "exact control timestep")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = load_runner_cls(TASK)(wrapped, asdict(agent), device=device)
        runner.load(str(ACTOR), map_location=device)
        policy = runner.get_inference_policy(device=device)
        observations = wrapped.get_observations()
        require(observations["actor"].shape == (NUM_ENVS, 61), "frozen 61D actor observation")
        robot = env.scene["robot"]
        stream = MotorStepStream.from_robot(robot, NUM_ENVS, device=device, cost_cfg=MotorStepCostCfg())
        require(stream.names == JOINTS, "exact named unit-gear motor layout")
        env._microduck_motor_step_stream = stream
        q = robot.data.root_link_quat_w
        yaw = torch.atan2(2 * (q[:, 0]*q[:, 3] + q[:, 1]*q[:, 2]), 1 - 2*(q[:, 2].square()+q[:, 3].square()))
        route = torch.stack((yaw.cos(), yaw.sin()), -1).clone()
        observer = RecoveryMeasurement(NUM_ENVS, SPEED, STEP_DT)
        data = {k: [] for k in ("velocities", "commands", "legacy_force", "legacy_speed", "pre_force", "pre_speed")}
        terminals = []
        with torch.inference_mode():
            for step in range(STEPS):
                velocities = velocity_rows(robot.data.root_link_lin_vel_w, robot.data.root_link_lin_vel_b,
                                           robot.data.root_link_quat_w, route)
                phase = torch.full((NUM_ENVS,), 0 if step < SETTLE else 2, device=device, dtype=torch.long)
                observer.begin(step, phase, velocities[:, 1])
                command = env.command_manager.get_command("twist").detach().clone()
                require(bool(torch.allclose(command, torch.tensor([SPEED, 0., 0.], device=device).expand_as(command),
                                             atol=1e-6, rtol=0)), "fixed command mutated")
                require(all(bool(torch.isfinite(x).all()) for x in observations.values()), "finite actor inputs")
                actions = policy(observations)
                require(bool(torch.isfinite(actions).all()), "finite actions")
                stream.begin(step, phase)
                observations, rewards, dones, _ = wrapped.step(actions)
                sample = stream.consume(dones.bool())
                observer.finish(dones.bool())
                require(bool(torch.isfinite(rewards).all()) and all(bool(torch.isfinite(x).all())
                        for x in observations.values()), "finite step outputs")
                for key, value in dict(velocities=velocities, commands=command,
                        legacy_force=robot.data.actuator_force, legacy_speed=robot.data.joint_vel[:, stream.joint_ids],
                        pre_force=sample.force_nm, pre_speed=sample.speed_rad_s).items():
                    data[key].append(value.detach().cpu().clone())
                if bool(dones.any()):
                    terminals.append(step)
                    break  # stop all environments; no reset episode may enter the control
        report = summarize(**{k: torch.stack(v) for k, v in data.items()}, terminal_steps=terminals,
                           measurement=observer.report())
        report.update(task=TASK, checkpoint_sha256=sha256(ACTOR), motor_stream=stream.provenance(),
                      actor_observation_shape=list(observations["actor"].shape))
        return report
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    verify_source(args.source)
    require(dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=SERVICE_SECONDS+CLOSEOUT_SECONDS)
            < DEADLINE, "control and closeout must fit before deadline")
    require(sha256(ACTOR) == ACTOR_SHA256, "frozen actor hash")
    require({k: importlib.metadata.version(k) for k in VERSIONS} == VERSIONS, "frozen runtime")
    import mjlab
    dependencies = {k: sha256(Path(mjlab.__file__).parent / k) for k in DEPENDENCIES}
    require(dependencies == DEPENDENCIES, "audited dependency hashes")
    host = check_host()
    OUTPUT.mkdir(exist_ok=False)
    write_new(OUTPUT / "launch.json", dict(protocol=PROTOCOL, source=args.source,
        actor_sha256=ACTOR_SHA256, versions=VERSIONS, dependencies=dependencies, preflight=host,
        seed=SEED, num_envs=NUM_ENVS, steps=STEPS, settle_steps=SETTLE, speed=SPEED,
        command_yaw_rate=0., task=TASK, service_seconds=SERVICE_SECONDS, started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        numerical_environment={k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONHASHSEED",
            "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "CUDA_LAUNCH_BLOCKING")}))
    start = time.monotonic()
    try:
        report = run_control()
        report["source"] = args.source
        write_new(OUTPUT / "response.json", report)
        decision = dict(protocol=PROTOCOL, classification=report["classification"],
                        report_sha256=sha256(OUTPUT / "response.json"), safety_failures=report["safety_failures"],
                        policy_acceptance=False, training_admitted=False, physical_motion_authorized=False,
                        reopens_recovery_ab=False)
    except Exception as error:
        decision = dict(protocol=PROTOCOL, classification="runtime-failure-stop", error=f"{type(error).__name__}: {error}",
                        policy_acceptance=False, training_admitted=False, physical_motion_authorized=False,
                        reopens_recovery_ab=False)
    write_new(OUTPUT / "decision.json", decision)
    write_new(OUTPUT / "runtime.json", dict(wall_seconds=time.monotonic()-start))
    print(json.dumps(decision, sort_keys=True), flush=True)
    raise SystemExit(2 if decision["classification"] in ("runtime-failure-stop", "safety-or-coverage-stop") else 0)


if __name__ == "__main__": main()
