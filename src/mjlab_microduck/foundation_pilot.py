"""Predeclared F1 command-focused fine-tune; simulation pilot, never promotion."""

from dataclasses import asdict
import argparse
import datetime as dt
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import torch

from mjlab_microduck.evaluation import fix_velocity_commands
from mjlab_microduck.first_attempt_smoke import ACTOR_SHA256, require, sha256
from mjlab_microduck.motor_audit_smoke import DEPENDENCIES, VERSIONS
from mjlab_microduck.motor_step_stream import MotorStepStream, MotorStepCostCfg, install_metric
from mjlab_microduck.recovery_ab import verify_source, write_new
from mjlab_microduck.rollout_repeatability import ROOT, ACTOR, check_host, read
from mjlab_microduck.tasks.motor_aware import _freeze_other_step_curricula
from mjlab_microduck.tasks.run import XL330_M288_RATED_NO_LOAD_SPEED_RAD_S as RATED_SPEED

TASK = "Mjlab-Run-MotorAware-Flat-MicroDuck"
PROTOCOL = "f1-fixed030-v1"
DEADLINE = dt.datetime(2026, 9, 7, 1, 4, tzinfo=dt.timezone.utc)
TRAIN_STOP = DEADLINE - dt.timedelta(minutes=10)
MODES = {"smoke": (387, 10, 240), "pilot": (389, 500, 1200)}
NUM_ENVS = 256
PARENT_STEP = 192000
START_ITERATION = 7999


def prepare_config(mode):
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    seed, iterations, _ = MODES[mode]
    cfg, agent = load_env_cfg(TASK), load_rl_cfg(TASK)
    _freeze_other_step_curricula(cfg, frozenset())
    for key in ("standing_envs", "velocity_command_ranges", "head_pose_range", "body_pose_range"):
        del cfg.curriculum[key]
    fix_velocity_commands(cfg, .30, 0.)
    # Materialize the already-completed parent reward stages before first reset.
    cfg.rewards["motor_torque_load"].weight = -2.
    cfg.rewards["action_rate_l2"].weight = -1.
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = NUM_ENVS
    agent.max_iterations, agent.save_interval = iterations, 50
    agent.logger, agent.upload_model = "tensorboard", False
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    install_metric(cfg)
    return cfg, agent


def restore_parent(runner, path):
    """Resume ALL learned state, advance update label once, retain saved env time."""
    runner.load(str(path), strict=True, map_location=runner.device)
    require(runner.current_learning_iteration == 7998, "parent update identity")
    require(runner.env.unwrapped.common_step_counter == PARENT_STEP, "parent curriculum time")
    rates = {g["lr"] for g in runner.alg.optimizer.param_groups}
    require(len(rates) == 1 and all(math.isfinite(x) and x > 0 for x in rates), "restored optimizer rate")
    # PPO's Python learning_rate is not serialized; synchronize it with restored Adam.
    runner.alg.learning_rate = next(iter(rates))
    runner.current_learning_iteration = START_ITERATION


def training_failures(row):
    """Gross stochastic-rollout stop guards, NOT deterministic acceptance gates."""
    failures = []
    if not all(type(v) in (int, float) and math.isfinite(v) for v in row.values()):
        return ["nonfinite-training-metric"]
    if row["fall_fraction"] > .50: failures.append("training-fall-burst")
    if row["pre_reset_torque_p99"] > .85: failures.append("training-torque-burst")
    if row["rated_speed_exceed_fraction"] > .01: failures.append("training-rated-speed-burst")
    return failures


def finite_gradient(gradient):
    require(bool(torch.isfinite(gradient).all()), "finite gradient before optimizer step")
    return gradient


def runtime_identity():
    import mjlab
    versions = {k: importlib.metadata.version(k) for k in VERSIONS}
    dependencies = {k: sha256(Path(mjlab.__file__).parent / k) for k in DEPENDENCIES}
    require(versions == VERSIONS and dependencies == DEPENDENCIES, "frozen runtime")
    return dict(versions=versions, dependencies=dependencies)


def train(mode, output):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    class DurableRunner(MjlabOnPolicyRunner):
        def save(self, path, infos=None):
            target = Path(path)
            temporary = target.with_suffix(".pt.pending")
            require(not temporary.exists(), "no checkpoint overwrite in progress")
            super().save(str(temporary), infos)
            with temporary.open("rb") as handle: os.fsync(handle.fileno())
            os.replace(temporary, target)
            fd = os.open(target.parent, os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)

    cfg, agent = prepare_config(mode)
    configure_torch_backends()
    torch.manual_seed(agent.seed)
    dump_yaml(output / "params/env.yaml", asdict(cfg))
    dump_yaml(output / "params/agent.yaml", asdict(agent))
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        # The wrapper performs the first reset. Apply the parent's completed
        # domain curricula there, rather than one episode later after load().
        env.common_step_counter = PARENT_STEP
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = DurableRunner(wrapped, asdict(agent), str(output), "cuda:0")
        restore_parent(runner, ACTOR)
        for model in (runner.alg.actor, runner.alg.critic):
            for parameter in model.parameters(): parameter.register_hook(finite_gradient)
        # Observe unsanitized manager output before the repository's existing NaN patch.
        # Finite runs are identical; a masked nonfinite value must not look healthy.
        from mjlab_microduck.tasks import mdp
        def checked_rewards(dt):
            result = mdp._orig_reward_compute(env.reward_manager, dt)
            require(bool(torch.isfinite(result).all()) and all(bool(torch.isfinite(x).all())
                    for x in env.reward_manager._episode_sums.values()), "finite raw rewards")
            return result
        env.reward_manager.compute = checked_rewards
        def checked_returns(obs):
            mdp._orig_compute_returns(runner.alg, obs)
            require(bool(torch.isfinite(runner.alg.storage.advantages).all())
                    and bool(torch.isfinite(runner.alg.storage.returns).all()), "finite raw returns")
        runner.alg.compute_returns = checked_returns
        require(wrapped.get_observations()["actor"].shape == (NUM_ENVS, 61), "61D actor retained")
        runner.save(str(output / "initial.pt"), dict(protocol=PROTOCOL, before_updates=True))
        stream = MotorStepStream.from_robot(env.scene["robot"], NUM_ENVS, device=env.device,
                                          cost_cfg=MotorStepCostCfg())
        env._microduck_motor_step_stream = stream
        phase = torch.zeros(NUM_ENVS, dtype=torch.long, device=env.device)
        force_rows, speed_rows, falls = [], [], []
        original_step, original_update = wrapped.step, runner.alg.update
        updates = 0
        started = time.monotonic()

        def guarded_step(actions):
            require(dt.datetime.now(dt.timezone.utc) < TRAIN_STOP, "training closeout deadline")
            require(time.monotonic()-started < MODES[mode][2]-30, "bounded training runtime")
            require(bool(torch.isfinite(actions).all()), "finite actions before step")
            command = env.command_manager.get_command("twist")
            require(bool(torch.allclose(command, command.new_tensor([.3, 0., 0.]).expand_as(command),
                                        atol=1e-6, rtol=0)), "fixed training command")
            stream.begin(stream.next_step, phase)
            result = original_step(actions)
            obs, rewards, dones, _ = result
            sample = stream.consume(dones.bool())
            require(bool(torch.isfinite(rewards).all()) and all(bool(torch.isfinite(x).all())
                    for x in obs.values()), "finite returned training tensors")
            force_rows.append(sample.force_nm)
            speed_rows.append(sample.speed_rad_s)
            falls.append(env.termination_manager.terminated.detach().clone())
            return result

        def guarded_update():
            nonlocal updates
            f, v = torch.stack(force_rows).double(), torch.stack(speed_rows).double()
            u = f.abs()/.6
            row = dict(update=updates+1, common_step=env.common_step_counter,
                fall_fraction=float(torch.stack(falls).any(0).double().mean()),
                pre_reset_torque_p99=float(torch.quantile(u.flatten(), .99)),
                rated_speed_exceed_fraction=float((v.abs() > RATED_SPEED).double().mean()),
                soft_limit_fraction=float((u > .7).double().mean()),
                squared_load_proxy=float(u.square().mean()),
                mechanical_abs_power_w=float((f*v).abs().mean()),
                motor_cost=float((u.square()+4*(u-.7).clamp_min(0).square()).mean()),
                elapsed_s=time.monotonic()-started)
            if updates % 25 == 0:
                temperature = int(read("nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"))
                require(temperature < 80, "GPU temperature guard")
                for service in ("recomo-ai-mission-vllm.service", "recomo-ai-mission-subject-model-worker.service"):
                    require(read("systemctl", "show", service, "-p", "ActiveState", "--value") == "inactive",
                            "protected service changed state")
                row["gpu_temperature_c"] = temperature
            failures = training_failures(row)
            with (output / "rollout-metrics.jsonl").open("a") as handle:
                handle.write(json.dumps({**row, "failures": failures}, allow_nan=False)+"\n")
                handle.flush(); os.fsync(handle.fileno())
            require(not failures, ",".join(failures))
            losses = original_update()
            require(all(math.isfinite(float(v)) for v in losses.values()), "finite optimizer losses")
            require(all(bool(torch.isfinite(p).all()) for m in (runner.alg.actor, runner.alg.critic)
                        for p in m.parameters()), "finite learned parameters")
            updates += 1
            force_rows.clear(); speed_rows.clear(); falls.clear()
            return losses

        wrapped.step, runner.alg.update = guarded_step, guarded_update
        runner.learn(agent.max_iterations, init_at_random_ep_len=True)
        final = output / f"model_{START_ITERATION+agent.max_iterations-1}.pt"
        require(final.is_file() and updates == agent.max_iterations, "complete bounded update count")
        return dict(status="training-complete-not-accepted", updates=updates,
            final_checkpoint=str(final), final_sha256=sha256(final),
            common_step=env.common_step_counter, parent_step=PARENT_STEP,
            wall_seconds=time.monotonic()-started, motor_stream=stream.provenance())
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    verify_source(args.source)
    require(dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=MODES[args.mode][2]) < TRAIN_STOP,
            "budget fits training cutoff")
    require(sha256(ACTOR) == ACTOR_SHA256, "parent checkpoint hash")
    runtime, host = runtime_identity(), check_host()
    output = ROOT / "artifacts/training" / f"{PROTOCOL}-{args.mode}"
    if args.mode == "pilot":
        smoke = json.loads((output.parent / f"{PROTOCOL}-smoke/result.json").read_text())
        require(smoke["status"] == "training-complete-not-accepted" and smoke["updates"] == 10
                and smoke["source"] == args.source, "same-source completed smoke")
        require(smoke["wall_seconds"] * 50 * 1.5 < 1200, "conservative measured pilot budget")
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "launch.json", dict(protocol=PROTOCOL, source=args.source, mode=args.mode,
        parent_sha256=ACTOR_SHA256, runtime=runtime, host=host, seed=MODES[args.mode][0],
        iterations=MODES[args.mode][1], num_envs=NUM_ENVS, deadline=DEADLINE.isoformat()))
    try:
        result = train(args.mode, output)
    except Exception as exc:
        write_new(output / "failure.json", dict(error_type=type(exc).__name__, error=str(exc),
                  source=args.source, policy_acceptance=False))
        raise
    write_new(output / "result.json", {**result, "source": args.source, "policy_acceptance": False})


if __name__ == "__main__":
    main()
