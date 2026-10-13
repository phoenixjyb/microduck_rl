"""Five-update import integration smoke; no foundation run or promotion."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata as metadata
import json
import os
import re
import stat
import subprocess
import time
from dataclasses import asdict

import torch

from . import integrated_turn_foundation as base
from . import official_native_reference as pinned
from .normalized_onnx_reference import NormalizedOnnxReference, validate_actor_contract
from .v8_actor_seed import FrozenV8Actor, import_v8_actor

PROTOCOL = "v8-actor-only-native-integration-smoke-v1"
CAP_SECONDS = 300
INITIAL_STD = .3
SEED = 947


def prepare_config():
    cfg, agent = base.prepare_config("smoke")
    cfg.seed = agent.seed = SEED
    agent.actor.class_name = "mjlab_microduck.v8_actor_seed:FrozenV8Actor"
    agent.actor.distribution_cfg = {
        "class_name": "rsl_rl.modules:GaussianDistribution",
        "init_std": INITIAL_STD, "std_type": "scalar"}
    agent.experiment_name = PROTOCOL
    validate_actor_contract(cfg)
    base.require(cfg.scene.num_envs == 64 and agent.max_iterations == 5
                 and agent.clip_actions is None, "unfiltered 64-world five-update smoke")
    return cfg, agent


def initialize(runner, env, output):
    """Import only a fresh actor; prove critic/optimizer/clocks untouched."""
    from .integrated_fresh_foundation import normalizer_count

    alg = runner.alg
    base.require(type(alg.actor) is FrozenV8Actor and not alg.optimizer.state
                 and type(alg.optimizer) is torch.optim.Adam
                 and runner.current_learning_iteration == env.common_step_counter == 0,
                 "fresh actor/critic/Adam and zero target clocks required")
    base.require(normalizer_count(alg.actor) == normalizer_count(alg.critic) == 0,
                 "empty target normalizer counters")
    ownership = tuple(id(p) for group in alg.optimizer.param_groups for p in group["params"])
    actor_ids = tuple(id(p) for p in alg.actor.parameters())
    critic_before = {k: v.detach().clone() for k, v in alg.critic.state_dict().items()}
    std_before = {k: v.detach().clone() for k, v in alg.actor.distribution.state_dict().items()}
    raw = runner.env.get_observations()["actor"].detach().cpu().clone()
    reference = NormalizedOnnxReference(pinned.V8.path, pinned.V8.sha256)
    robot = env.scene["robot"]
    ids, names = robot.find_joints("^(?!passive_).*", preserve_order=True)
    reference.validate_robot(robot, tuple(ids), tuple(names))
    alg.actor.cpu()
    try:
        report = import_v8_actor(alg.actor, pinned.V8.path, raw)
    finally:
        alg.actor.to(env.device)
    base.require(tuple(id(p) for p in alg.actor.parameters()) == actor_ids
                 and tuple(id(p) for group in alg.optimizer.param_groups for p in group["params"]) == ownership
                 and not alg.optimizer.state, "unchanged optimizer ownership and empty Adam")
    base.require(all(torch.equal(v, critic_before[k]) for k, v in alg.critic.state_dict().items())
                 and all(torch.equal(v, std_before[k]) for k, v in alg.actor.distribution.state_dict().items()),
                 "critic and fresh exploration not imported or modified")
    base.require(runner.current_learning_iteration == env.common_step_counter == 0,
                 "no inherited runner or curriculum clocks")
    report.update(initial_std=INITIAL_STD, initial_learning_rate=alg.learning_rate,
                  critic_normalizer_count=0, adam_state_entries=0,
                  target_iteration=0, target_common_step_counter=0,
                  actor_class="mjlab_microduck.v8_actor_seed:FrozenV8Actor")
    base.write_new(output / "actor-import.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    base.require(re.fullmatch(r"[0-9a-f]{40}", args.source) is not None
                 and re.fullmatch(r"[a-z0-9-]{1,64}", args.run_id) is not None,
                 "safe exact source and unique run ID required")
    def git(*argv):
        return subprocess.check_output(("git", "-C", str(base.ROOT), *argv), text=True).strip()
    base.require(git("rev-parse", "HEAD") == args.source and not git("status", "--porcelain"),
                 "clean exact committed native source")
    base.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "literal isolated CUDA0 required")
    base.require(base.sha256(pinned.V8.path) == pinned.V8.sha256
                 and base.sha256(pinned.V8.path.with_name("manifest.json")) == pinned.V8.manifest_sha256,
                 "exact retained v8 policy and manifest")
    runtime = {p: metadata.version(p) for p in (*pinned.RUNTIME, "rsl-rl-lib", "onnx", "onnxruntime")}
    base.require(all(runtime[k].split("+")[0] == v for k, v in pinned.RUNTIME.items())
                 and runtime["rsl-rl-lib"] == "5.0.1", "unchanged native runtime pins")
    cfg, agent = prepare_config()
    base.require(base.LOCK.is_file() and not base.LOCK.is_symlink(), "existing regular GPU lease")
    with base.LOCK.open("r+") as lease:
        info = os.fstat(lease.fileno())
        base.require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(), "owned GPU lease")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host = base.shared_host()
        output = base.ROOT / "artifacts/training" / args.run_id
        output.mkdir(parents=True, exist_ok=False)
        base.write_new(output / "launch.json", dict(source=args.source, protocol=PROTOCOL,
            seed=SEED, worlds=64, updates=5, steps_per_update=24, cap_seconds=CAP_SECONDS,
            host=host, runtime=runtime, artifact_sha256=pinned.V8.sha256,
            manifest_sha256=pinned.V8.manifest_sha256, actor_cfg=asdict(agent.actor),
            initial_std=INITIAL_STD, actor_normalization="frozen-exported-affine-once",
            ppo_resume=False, foundation_training=False, policy_acceptance=False,
            physical_motion_authorized=False))
        started = time.monotonic()
        def health():
            base.require(time.monotonic() - started < CAP_SECONDS - 15, "bounded smoke deadline")
            sample = base.shared_host()
            with (output / "health.jsonl").open("a") as log:
                log.write(json.dumps(dict(utc=time.time(), **sample), allow_nan=False) + "\n")
                log.flush(); os.fsync(log.fileno())
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20, device=0)
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        try:
            initialization = {}
            def seed_runner(runner, env):
                initialization.update(initialize(runner, env, output))
            result = base.training(output, "smoke", health, recipe=(cfg, agent), initialize=seed_runner)
            health()
            # Check serialization and numerical validity; no held-out claim.
            checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
            actor_state = checkpoint["actor_state_dict"]
            base.require(base.finite_tree(actor_state)
                         and actor_state["obs_normalizer.count"].item() == 0
                         and bytes(actor_state["v8_seed_digest"].tolist()).hex() == pinned.V8.sha256,
                         "durable finite imported actor with frozen empty updater")
            # Velocity runner.save uses its normal baked-normalizer export and
            # task metadata path. Its warning-only export failures are fatal to
            # this smoke receipt instead of being mistaken for success.
            from .integrated_motor_refinement import onnx_check
            result["onnx"] = onnx_check(output / f"{output.name}.onnx")
            result["initial_checkpoint_sha256"] = base.sha256(output / "initial.pt")
            base.write_new(output / "result.json", dict(source=args.source,
                protocol=PROTOCOL, **result, actor_import=initialization,
                total_elapsed_s=time.monotonic()-started, policy_acceptance=False,
                foundation_training=False, physical_motion_authorized=False))
        except Exception as error:
            base.write_new(output / "failure.json", dict(source=args.source,
                type=type(error).__name__, error=str(error)))
            raise


if __name__ == "__main__":
    main()
