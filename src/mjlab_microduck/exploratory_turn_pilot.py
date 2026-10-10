"""Small live PPO experiment, explicitly separate from simulator admission."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import subprocess

import torch

from mjlab_microduck import foundation_pilot as base
from mjlab_microduck.evaluation import fix_velocity_commands
from mjlab_microduck.first_attempt_smoke import ACTOR_SHA256, canonical, require, sha256
from mjlab_microduck.motor_step_stream import MotorStepStream, MotorStepCostCfg, install_metric
from mjlab_microduck.tasks.run import XL330_M288_RATED_NO_LOAD_SPEED_RAD_S as RATED_SPEED

PROTOCOL = "exploratory-gentle-turn-v1"
GPU = "GPU-f21e0304-3b55-b6eb-4993-946e7ee1f6dd"
MODES = {"smoke": (701, 10, 360), "pilot": (709, 128, 900)}
CASES = ((.30, 0.), (.20, -.20), (.20, .20))
SERVICES = ("recomo-ai-mission-vllm.service", "recomo-ai-mission-subject-model-worker.service")
LOCK = Path("/home/converge/.local/state/microduck-gpu0.lock")


def read(*args):
    return subprocess.check_output(args, text=True, timeout=10).strip()


def shared_host():
    """Observe shared occupancy; never stop the retained DINO worker."""
    row = read("nvidia-smi", "--query-gpu=uuid,memory.used,memory.free,temperature.gpu",
               "--format=csv,noheader,nounits").split(",")
    require(len(row) == 4 and row[0].strip() == GPU, "exact native GPU")
    used, free, temperature = map(lambda v: int(v.strip()), row[1:])
    require(used <= 6144 and free >= 8192 and temperature < 75, "shared GPU budget/temperature")
    processes = []
    for line in read("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                     "--format=csv,noheader,nounits").splitlines():
        pid, name, memory = (x.strip() for x in line.split(",", 2))
        require(int(pid) == os.getpid() or name ==
                "/home/converge/Tonghao/VLM/grounding_dino_cpp_dev/build_worker/grounding_dino_cpp_worker",
                "unexpected competing GPU process")
        processes.append(dict(pid=int(pid), name=name, memory_mib=memory))
    states = {}
    for namespace in ((), ("--user",)):
        for service in SERVICES:
            key = ("user:" if namespace else "system:") + service
            states[key] = read("systemctl", *namespace, "show", service, "-p", "ActiveState", "--value")
            require(states[key] == "inactive", "protected service active: " + key)
    return dict(memory_used_mib=used, memory_free_mib=free, temperature_c=temperature,
                processes=processes, protected_services=states)


def prepare_config(mode):
    cfg, agent = base.prepare_config(mode)
    seed, updates, _ = MODES[mode]
    twist = cfg.commands["twist"]
    twist.resampling_time_range = (3., 5.)
    twist.ranges.lin_vel_x = (.15, .30)
    twist.ranges.ang_vel_z = (-.20, .20)
    cfg.seed = agent.seed = seed
    agent.max_iterations, agent.save_interval = updates, 10
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    cfg.sim.nan_guard.enabled = True
    return cfg, agent


def validate_commands(command):
    require(command.shape == (256, 3) and bool(torch.isfinite(command).all()), "finite 256-world commands")
    require(bool(((command[:, 0] >= .15-1e-6) & (command[:, 0] <= .30+1e-6)).all())
            and bool((command[:, 1] == 0).all())
            and bool((command[:, 2].abs() <= .20+1e-6).all()), "gentle command bounds")


class CommandObserver:
    def __init__(self, host_check=shared_host):
        self.host_check, self.steps, self.actors = host_check, 0, 0
        self.previous = None
        self.pending = None
        self.positive = self.negative = 0
        self.changed_world_commands = self.turn_world_steps = 0

    def before_actor(self, env, observations):
        require(observations["actor"].shape == (256, 61)
                and all(bool(torch.isfinite(v).all()) for v in observations.values()), "unchanged finite actor input")
        require(self.pending is None, "ordered actor call")
        command = env.command_manager.get_command("twist")
        validate_commands(command)
        require(torch.equal(observations["actor"][:, 48:51], command), "fresh raw actor command")
        self.pending = command.detach().clone()
        self.actors += 1

    def after_actor(self, env, observations, actions):
        require(actions.shape == (256, 14) and bool(torch.isfinite(actions).all()), "finite motor actions")
        require(torch.equal(observations["actor"][:, 48:51], self.pending), "stored PPO command")

    def before_step(self, env, command):
        validate_commands(command)
        require(self.pending is not None and torch.equal(command, self.pending), "consumed command at physics entry")
        if self.previous is not None:
            self.changed_world_commands += int((command != self.previous).any(-1).sum())
        self.previous = command.detach().clone()
        self.turn_world_steps += int((command[:, 2].abs() > .01).sum())
        self.positive += int((command[:, 2] > 0).sum())
        self.negative += int((command[:, 2] < 0).sum())
        if self.steps % 120 == 0:
            self.host_check()

    def after_step(self, env, dones):
        validate_commands(env.command_manager.get_command("twist"))
        require(dones.shape == (256,) and dones.dtype == torch.bool, "completed 256-world step")
        self.pending = None
        self.steps += 1

    def finish(self, expected):
        require(self.steps == self.actors == expected and self.turn_world_steps > 0
                and self.changed_world_commands > 0 and self.positive > 0 and self.negative > 0
                and self.pending is None, "live command coverage")
        return dict(steps=self.steps, actor_calls=self.actors,
                    changed_world_commands=self.changed_world_commands,
                    turn_world_steps=self.turn_world_steps, positive_yaw_samples=self.positive,
                    negative_yaw_samples=self.negative, actor_command_equal=True, commands_modified=False)


def evaluation_case(checkpoint, speed, yaw, health=shared_host):
    """First-episode body-frame response; no initial-heading projection on turns."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    seed, worlds, steps, startup = 719, 8, 240, 60
    torch.manual_seed(seed)
    cfg, agent = load_env_cfg(base.TASK, play=True), load_rl_cfg(base.TASK)
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    cfg.curriculum.clear()
    cfg.events.pop("push_robot", None)
    cfg.sim.nan_guard.enabled = True
    fix_velocity_commands(cfg, speed, yaw)
    install_metric(cfg)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        require(env.step_dt == .02, "20ms control timestep")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = load_runner_cls(base.TASK)(wrapped, asdict(agent), device="cuda:0")
        runner.load(str(checkpoint), strict=True, map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
        obs, robot = wrapped.get_observations(), env.scene["robot"]
        stream = MotorStepStream.from_robot(robot, worlds, device=env.device, cost_cfg=MotorStepCostCfg())
        env._microduck_motor_step_stream = stream
        velocities, forces, speeds, terminals = [], [], [], []
        phase = torch.zeros(worlds, dtype=torch.long, device=env.device)
        with torch.inference_mode():
            for step in range(steps):
                if step % 120 == 0: health()
                require(obs["actor"].shape == (worlds, 61)
                        and all(bool(torch.isfinite(v).all()) for v in obs.values()), "finite evaluation input")
                command = env.command_manager.get_command("twist")
                require(bool(torch.allclose(command, command.new_tensor([speed, 0., yaw]).expand_as(command),
                                           atol=1e-6, rtol=0)), "fixed held-out command")
                v = torch.stack((robot.data.root_link_lin_vel_b[:, 0], robot.data.root_link_ang_vel_b[:, 2]), -1)
                require(bool(torch.isfinite(v).all()), "finite measured velocity")
                velocities.append(v.detach().cpu().clone())
                actions = policy(obs)
                require(bool(torch.isfinite(actions).all()), "finite evaluation action")
                stream.begin(step, phase)
                obs, rewards, dones, _ = wrapped.step(actions)
                require(bool(torch.isfinite(rewards).all())
                        and all(bool(torch.isfinite(v).all()) for v in obs.values()), "finite evaluation output")
                sample = stream.consume(dones.bool())
                forces.append(sample.force_nm.cpu()); speeds.append(sample.speed_rad_s.cpu())
                if bool(dones.any()):
                    terminals = dones.nonzero().flatten().tolist()
                    break  # never count reset episodes as success
        v, f, s = map(lambda rows: torch.stack(rows).double(), (velocities, forces, speeds))
        require(bool(torch.isfinite(f).all()) and bool(torch.isfinite(s).all()), "finite pre-reset motors")
        u = f.abs() / .6
        settled = v[startup:]
        return dict(seed=seed, speed_mps=speed, yaw_rad_s=yaw, worlds=worlds, steps=len(v),
            complete=len(v) == steps and not terminals, terminal_worlds=terminals,
            checkpoint_sha256=sha256(checkpoint), torque_p99=float(torch.quantile(u.flatten(), .99)),
            rated_speed_exceed_fraction=float((s.abs() > RATED_SPEED).double().mean()),
            soft_limit_fraction=float((u > .7).double().mean()),
            mechanical_abs_power_w=float((f*s).abs().mean()),
            settled_speed_mae_per_world=(settled[:, :, 0]-speed).abs().mean(0).tolist() if len(settled) else None,
            settled_yaw_mae_per_world=(settled[:, :, 1]-yaw).abs().mean(0).tolist() if len(settled) else None,
            motor_stream=stream.provenance())
    finally:
        env.close()


def compare(parent, candidate):
    """Research signal only. No simulator, obstacle, hop or physical promotion."""
    canonical([parent, candidate])
    require(len(parent) == len(candidate) == len(CASES), "full predeclared matrix")
    failures = []
    for i, (p, c) in enumerate(zip(parent, candidate, strict=True)):
        require((p["seed"], p["speed_mps"], p["yaw_rad_s"]) ==
                (c["seed"], c["speed_mps"], c["yaw_rad_s"]) == (719, *CASES[i]), "paired evaluation")
        for row in (p, c):
            require(type(row["complete"]) is bool and row["worlds"] == 8
                    and 1 <= row["steps"] <= 240, "evaluation coverage")
            if row["complete"]:
                require(row["steps"] == 240 and not row["terminal_worlds"], "complete first episode")
                for key in ("settled_speed_mae_per_world", "settled_yaw_mae_per_world"):
                    require(len(row[key]) == 8 and all(type(x) in (int, float) and x >= 0 for x in row[key]),
                            "eight finite world errors")
            if not row["complete"] or row["torque_p99"] > .60 or row["rated_speed_exceed_fraction"] != 0:
                failures.append("incomplete-or-motor-gate")
        if p["complete"] and c["complete"]:
            for key, slack in (("settled_speed_mae_per_world", .01), ("settled_yaw_mae_per_world", .03)):
                if any(a > b+slack for a,b in zip(c[key], p[key], strict=True)):
                    failures.append(key+"-regression")
    improved = (not failures and any(sum(c["settled_speed_mae_per_world"]) <
                sum(p["settled_speed_mae_per_world"])-.04 for p,c in zip(parent,candidate,strict=True)))
    return dict(decision="exploratory-improvement" if improved else "no-demonstrated-improvement",
                failures=failures, policy_acceptance=False, physical_motion_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=(*MODES, "evaluate"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    require(re.fullmatch(r"[a-z0-9-]{1,64}", args.run_id) is not None, "safe unique run id")
    base.verify_source(args.source)
    require(sha256(base.ACTOR) == ACTOR_SHA256, "retained parent bytes")
    runtime = base.runtime_identity()
    require(LOCK.is_file() and not LOCK.is_symlink(), "existing cooperative GPU lock")
    with LOCK.open("r+") as lease:
        require(stat.S_ISREG(os.fstat(lease.fileno()).st_mode)
                and os.fstat(lease.fileno()).st_uid == os.getuid(), "owned regular GPU lock")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host = shared_host()
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20, device=0)
        root = base.ROOT / "artifacts/training" / args.run_id
        output = root / args.mode
        if args.mode == "pilot":
            smoke = json.loads((root / "smoke/result.json").read_text())
            require(smoke["source"] == args.source and smoke["updates"] == 10
                    and smoke["status"] == "training-complete-not-accepted", "same-source real optimizer smoke")
            require(smoke["wall_seconds"] * 12.8 * 1.5 < 870, "measured pilot fits cap")
        output.mkdir(parents=True, exist_ok=False)
        base.write_new(output / "launch.json", dict(protocol=PROTOCOL, source=args.source,
            runtime=runtime, host=host, parent_sha256=ACTOR_SHA256, mode=args.mode,
            policy_acceptance=False, physical_motion_authorized=False))
        def health():
            snapshot = shared_host()
            with (output / "gpu-health.jsonl").open("a") as log:
                log.write(json.dumps({"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                      **snapshot}, allow_nan=False)+"\n")
                log.flush(); os.fsync(log.fileno())
            return snapshot
        try:
            if args.mode == "evaluate":
                training = json.loads((root / "pilot/result.json").read_text())
                candidate = root / "pilot/model_8126.pt"
                require(training["source"] == args.source and training["updates"] == 128
                        and training["status"] == "training-complete-not-accepted"
                        and sha256(candidate) == training["final_sha256"], "durable exact candidate")
                reports = {}
                for arm, checkpoint in (("parent", base.ACTOR), ("candidate", candidate)):
                    reports[arm] = []
                    for i, (speed, yaw) in enumerate(CASES):
                        row = evaluation_case(checkpoint, speed, yaw, health=health)
                        base.write_new(output / f"{arm}-{i}.json", row)
                        reports[arm].append(row)
                result = compare(reports["parent"], reports["candidate"])
            else:
                _, _, seconds = MODES[args.mode]
                result = base.train(args.mode, output, config_factory=prepare_config, protocol=PROTOCOL,
                    stop_at=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=seconds),
                    max_seconds=seconds, command_validator=CommandObserver(host_check=health))
            base.write_new(output / "result.json", {**result, "source": args.source, "policy_acceptance": False})
        except Exception as exc:
            base.write_new(output / "failure.json", dict(error_type=type(exc).__name__, error=str(exc),
                           source=args.source, policy_acceptance=False))
            raise


if __name__ == "__main__":
    main()
