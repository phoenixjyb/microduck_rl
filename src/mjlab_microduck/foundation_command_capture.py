"""Bounded borrowed-session capture core; no environment creation or launcher.

The caller owns environment lifetime, immutable evidence persistence, checkpoint
loading, exclusive GPU acquisition and an external hard process timeout.
"""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time

import torch

from mjlab_microduck import foundation_command_map as mapping
from mjlab_microduck.command_delivery import PROTOCOL as DELIVERY, prepare_actor_command_input
from mjlab_microduck.evaluation import fix_velocity_commands
from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.speed_response_control import (
    prepare_config as original_config, route_position_rows, velocity_rows,
)

PROTOCOL = "foundation-command-map-capture-core-v1"


def prepare_config(cell):
    """Distinct map configuration, with historical defaults left untouched."""
    cell.__post_init__()
    cfg, agent = original_config(seed=cell.seed)
    fix_velocity_commands(cfg, cell.speed_mps, 0.)
    return cfg, agent


def bind_files(cell, checkpoint, runtime_files):
    """Check bytes against explicit pins; not proof of loading or execution.

runtime_files maps labels to (path, expected_sha256). The caller must bind that
pin set to a reviewed source manifest; this function cannot establish completeness.
"""
    cell.__post_init__()
    require(type(runtime_files) is dict and bool(runtime_files), "nonempty explicit runtime pins")
    records = {}
    entries = {"checkpoint": (checkpoint, mapping.CHECKPOINTS[cell.policy])}
    require("checkpoint" not in runtime_files, "reserved checkpoint label")
    entries.update(runtime_files)
    for label, (path, expected) in entries.items():
        require(type(label) is str and bool(label), "named artifact")
        require(type(expected) is str and len(expected) == 64
                and all(c in '0123456789abcdef' for c in expected), "exact SHA256 pin")
        path = Path(path).absolute()
        require(not any(p.is_symlink() for p in (path, *path.parents)), "no symlink artifact path")
        require(path.is_file(), "regular artifact")
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        require(actual == expected, "artifact hash mismatch: "+label)
        records[label] = dict(path=str(path), sha256=actual, bytes=len(raw))
    return dict(protocol=PROTOCOL,cell=cell.identity(),files=records, declared_file_bytes_verified=True,
                checkpoint_loaded=False, runtime_pin_coverage_verified=False,
                simulation_executed=False, policy_acceptance=False)


def actor_digest(actor):
    """Hash tensor parameters/buffers, including normalizer buffers, on CPU."""
    require(isinstance(actor, torch.nn.Module) and not any(m.training for m in actor.modules()),
            "actor and all submodules must be in evaluation mode")
    digest = hashlib.sha256()
    state = actor.state_dict()
    require(bool(state), "nonempty actor state")
    for name, tensor in sorted(state.items()):
        require(isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided,
                "tensor-only actor state")
        value = tensor.detach().cpu().contiguous()
        require(bool(torch.isfinite(value).all()), "finite actor state")
        metadata = canonical(dict(name=name, shape=list(value.shape), dtype=str(value.dtype))).encode()
        digest.update(len(metadata).to_bytes(8, 'big')); digest.update(metadata)
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, 'big')); digest.update(raw)
    return digest.hexdigest()


class CaptureFailure(RuntimeError):
    """Caller must retain partial raw frames as failure evidence, never rescore as success."""

    def __init__(self, cause, frames, inference_steps, simulation_steps):
        super().__init__(str(cause))
        self.frames = frames
        self.inference_steps = inference_steps
        self.simulation_steps = simulation_steps


@dataclass(frozen=True)
class Capture:
    trace: mapping.Trace
    cached_commands: torch.Tensor
    evidence: dict


def capture(cell, wrapped, actor, stream, *, budget_seconds, clock=time.monotonic, on_frame=None):
    """Use existing mjlab-style objects; cooperative timeout cannot interrupt a hung step.

No checkpoint is loaded or environment constructed here. No reset/close is
performed on borrowed objects. Factory/launcher must close them in finally.
"""
    cell.__post_init__()
    require(type(budget_seconds) in (float, int) and math.isfinite(budget_seconds)
            and 0 < budget_seconds <= 120, "bounded 120-second cell budget")
    start = clock(); last_time = start
    require(math.isfinite(start), "finite monotonic clock")
    frames, inference_steps, simulation_steps = [], 0, 0

    def guard():
        nonlocal last_time
        now = clock()
        require(math.isfinite(now) and now >= last_time, "monotonic clock regression")
        require(now-start < budget_seconds, "cell budget expired")
        last_time = now

    def snapshot(values):
        return {key: value.detach().cpu().clone() for key, value in values.items()}

    try:
        guard()
        initial_actor = actor_digest(actor)
        env = wrapped.unwrapped
        require(env.num_envs == mapping.NUM_ENVS and env.step_dt == mapping.STEP_DT,
                "exact eight-environment 50Hz session")
        require(tuple(stream.names) == tuple(mapping.JOINTS) and stream.next_step == 0,
                "fresh named pre-reset stream")
        require(getattr(env, '_microduck_motor_step_stream', None) is stream,
                "stream must be installed on this environment")
        robot = env.scene['robot']
        observations = wrapped.get_observations()
        q = robot.data.root_link_quat_w
        yaw = torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]), 1-2*(q[:,2].square()+q[:,3].square()))
        route = torch.stack((yaw.cos(),yaw.sin()),-1).detach().clone()
        origin = robot.data.root_link_pos_w.detach().clone()
        previous = torch.zeros_like(yaw)
        event_ranges = {name: list(env.event_manager.get_term_cfg(name).params['ranges'])
                        for name in ('randomize_com','randomize_head_com')}
        require(event_ranges == dict(randomize_com=[-.003,.003], randomize_head_com=[-.003,.003]),
                "declared evaluation CoM ranges")
        for step in range(mapping.STEPS):
            guard()
            velocity = velocity_rows(robot.data.root_link_lin_vel_w, robot.data.root_link_lin_vel_b,
                                     robot.data.root_link_quat_w, route)
            position = route_position_rows(robot.data.root_link_pos_w, origin, route)
            heading = torch.atan2(velocity[:,3].sin(),velocity[:,3].cos())
            previous = previous + ((-heading).clamp(-.35,.35)-previous).clamp(-.02,.02)
            target = torch.stack((torch.full_like(previous,cell.speed_mps),torch.zeros_like(previous),previous),-1)
            command = env.command_manager.get_command('twist')
            command.copy_(target)
            prepared = prepare_actor_command_input(observations,command,env.observation_manager,
                                                   protocol=DELIVERY,step=step)
            require(torch.equal(prepared.issued,prepared.consumed), "same-step actor command")
            frame = snapshot(dict(velocity=velocity,position=position,issued=prepared.issued,
                                  consumed=prepared.consumed,cached=prepared.cached))
            frames.append(frame)
            require(all(bool(torch.isfinite(x).all()) for x in prepared.observations.values()),
                    "finite full policy observations")
            guard()
            with torch.inference_mode():
                actions = actor(prepared.observations)
            inference_steps += 1
            require(actions.shape == (8,14) and bool(torch.isfinite(actions).all()), "finite14D actions")
            require(torch.equal(prepared.observations['actor'][:,48:51],prepared.consumed),
                    "raw command unchanged across inference")
            require(torch.equal(command,prepared.issued), "command unchanged before physics")
            guard()
            phase = torch.full((8,),0 if step < 100 else 2,device=command.device)
            stream.begin(step,phase)
            with torch.inference_mode():
                observations,rewards,dones,_ = wrapped.step(actions)
            simulation_steps += 1
            require(dones.dtype in (torch.bool,torch.int32,torch.int64) and dones.shape == (8,)
                    and bool(((dones == 0) | (dones == 1)).all()), "exact terminal flags")
            dones = dones.bool()  # The installed RSL wrapper returns integer0/1.
            sample = stream.consume(dones)
            frame.update(snapshot(dict(legacy_force=robot.data.actuator_force,
                legacy_speed=robot.data.joint_vel[:,stream.joint_ids],pre_force=sample.force_nm,
                pre_speed=sample.speed_rad_s,dones=dones)))
            if on_frame is not None:
                on_frame(step,snapshot(frame))  # Sink cannot mutate retained tensors.
            require(bool(torch.isfinite(rewards).all())
                    and all(bool(torch.isfinite(x).all()) for x in observations.values()),
                    "finite post-step rewards and observations")
            guard()
            if bool(dones.any()):
                break
        require(actor_digest(actor) == initial_actor, "actor or normalizer state mutated")
        guard()
        trace = mapping.Trace(**{key: torch.stack([row[key] for row in frames])
                                 for key in mapping.Trace.__dataclass_fields__ if key != 'joint_names'},
                              joint_names=tuple(stream.names))
        mapping.score(cell,trace)  # Reject malformed/overflowed data before successful return.
        guard()
        return Capture(trace,torch.stack([row['cached'] for row in frames]),dict(
            protocol=PROTOCOL,cell=cell.identity(),inference_calls_completed=inference_steps,
            step_calls_completed=simulation_steps,actor_state_sha256=initial_actor,
            actor_state_unchanged=True,live_event_ranges=event_ranges,delivery=DELIVERY,
            checkpoint_loaded_verified=False,runtime_verified=False,physics_execution_verified=False,
            policy_acceptance=False,training_admitted=False,physical_motion_authorized=False))
    except Exception as exc:
        raise CaptureFailure(exc,frames,inference_steps,simulation_steps) from exc
