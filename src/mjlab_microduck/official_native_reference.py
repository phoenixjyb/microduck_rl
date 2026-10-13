"""Fresh native-plant reference diagnostic, not C1 replay or PPO training."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time

import torch

from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import integrated_joint_limit_refinement as limits
from mjlab_microduck.normalized_onnx_reference import NormalizedOnnxReference

PROTOCOL = "official-velstand-native-reference-v1"
POLICY_SHA = "1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45"
MANIFEST_SHA = "622048c2c23ea58942023f66fd16b189a875fd169e88d85beb16ebbe63b20c94"
POLICY = base.ROOT / "artifacts/community/intake-20261001/official-velstand/policy.onnx"
MANIFEST = POLICY.with_name("manifest.json")
CAP_SECONDS = 600
RUNTIME = {"torch": "2.9.1", "warp-lang": "1.12.0", "mujoco": "3.10.0",
           "mujoco-warp": "3.8.1", "mjlab": "1.3.0", "better-actuator-models": "1.0.1"}


@dataclass(frozen=True)
class ReferenceSpec:
    protocol: str
    path: Path
    sha256: str
    manifest_sha256: str
    hub_revision: str


ARCHIVE = ReferenceSpec(PROTOCOL, POLICY, POLICY_SHA, MANIFEST_SHA,
                        "1b56c396825c052a4e26e95cf2b8d8298af9e9b4")
V8 = ReferenceSpec("official-velstand-v8-native-reference-v1",
    base.ROOT / "artifacts/community/intake-20261013/official-velstand-v8/policy.onnx",
    "2fc7f8ee6acd7d0d3f66a99927b5d5a56589aeed02ec714de329b252d5ccf492",
    "c566390738dd90a54c1e2de829aaab0e904c636949e2c2c6158ee7bbb889b818",
    "665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448")
REFERENCES = {"archive": ARCHIVE, "v8": V8}


def validate_spec(spec):
    base.require(isinstance(spec, ReferenceSpec) and spec in (ARCHIVE, V8),
                 "only explicitly predeclared reference artifacts")


def evaluate_matrix(policy, health, output, *, evaluator=None, spec=ARCHIVE):
    """Exactly 12 ordered cases; no command remap, checkpoint search or retry."""
    validate_spec(spec)
    evaluator = base.evaluate_case if evaluator is None else evaluator
    rows = []
    for seed in base.SEEDS:
        for index, (speed, yaw) in enumerate(base.CASES):
            health()
            row = evaluator(spec.path, seed, speed, yaw, health, diagnostics=True,
                            yaw_trace=True, gait_trace=True, onnx_reference=policy)
            base.require(row["checkpoint_sha256"] == spec.sha256
                         and (row["seed"], row["speed"], row["yaw"]) == (seed, speed, yaw),
                         "bound reference artifact and ordered case")
            base.write_new(output / f"s{seed}-c{index}.json", row)
            rows.append(row)
    return dict(protocol=spec.protocol, artifact_sha256=spec.sha256, cases=rows,
                numerical_gate=limits.decision(rows), policy_acceptance=False,
                simulator_qualified=False, physical_motion_authorized=False,
                ppo_checkpoint=False, command_remap=False,
                interpretation="reference diagnostic on current plain native plant; not teacher admission")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reference", choices=tuple(REFERENCES), default="archive")
    args = parser.parse_args()
    spec = REFERENCES[args.reference]
    base.require(re.fullmatch(r"[a-f0-9]{40}", args.source) is not None
                 and re.fullmatch(r"[a-z0-9-]{1,64}", args.run_id) is not None, "exact safe source/run")
    def git(*argv):
        return subprocess.check_output(("git", "-C", str(base.ROOT), *argv), text=True).strip()
    base.require(git("rev-parse", "HEAD") == args.source and not git("status", "--porcelain"),
                 "clean exact committed native source")
    base.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "literal isolated GPU selection")
    base.require(base.sha256(spec.path) == spec.sha256
                 and base.sha256(spec.path.with_name("manifest.json")) == spec.manifest_sha256,
                 "retained official policy and manifest bytes")
    runtime = {name: metadata.version(name) for name in (*RUNTIME, "onnx", "onnxruntime")}
    base.require(all(runtime[k].split("+")[0] == v for k, v in RUNTIME.items()), "unchanged pinned native runtime")
    policy = NormalizedOnnxReference(spec.path, spec.sha256)
    base.require(base.LOCK.is_file() and not base.LOCK.is_symlink(), "existing nonsymlink lease")
    with base.LOCK.open("r+") as lease:
        info = os.fstat(lease.fileno())
        base.require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(), "owned regular GPU lease")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host = base.shared_host()
        output = base.ROOT / "artifacts/evaluations" / args.run_id
        output.mkdir(parents=True, exist_ok=False)
        base.write_new(output / "launch.json", dict(protocol=spec.protocol, source=args.source,
            artifact_sha256=spec.sha256, manifest_sha256=spec.manifest_sha256,
            reference=args.reference, hub_revision=spec.hub_revision, runtime=runtime, host=host,
            inference_backend=policy.provenance(), seeds=list(base.SEEDS), cases=list(base.CASES),
            cap_seconds=CAP_SECONDS, training=False, ppo_checkpoint=False,
            policy_acceptance=False, physical_motion_authorized=False))
        started = time.monotonic()
        def health():
            base.require(time.monotonic() - started < CAP_SECONDS - 15, "bounded reference deadline")
            sample = base.shared_host()
            with (output / "health.jsonl").open("a") as log:
                log.write(json.dumps(dict(utc=time.time(), **sample), allow_nan=False) + "\n")
                log.flush(); os.fsync(log.fileno())
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20, device=0)
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        try:
            result = evaluate_matrix(policy, health, output, spec=spec)
            health()
            base.write_new(output / "result.json", dict(source=args.source,
                elapsed_s=time.monotonic()-started, **result))
        except Exception as error:
            base.write_new(output / "failure.json", dict(source=args.source,
                type=type(error).__name__, error=str(error)))
            raise


if __name__ == "__main__":
    main()
