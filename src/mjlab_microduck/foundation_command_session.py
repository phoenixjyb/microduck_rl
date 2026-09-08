"""Native session and durable cell evidence, not a campaign/service launcher.

This module exposes no CLI. GPU deployment requires a separate reviewed launch
manifest, clean source, exclusive-host guard and hard process timeout.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import math
import os
from pathlib import Path
import time

import torch
import yaml

from mjlab_microduck import foundation_command_capture as core
from mjlab_microduck import foundation_command_map as mapping
from mjlab_microduck import foundation_reset_evidence as reset_evidence
from mjlab_microduck.checkpoint_inference_audit import audit_actor_state
from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.foundation_pilot import runtime_identity
from mjlab_microduck.motor_step_stream import MotorStepStream, MotorStepCostCfg

PROTOCOL = "foundation-command-map-native-cell-v1"
ITERATIONS = dict(original=7998,narrow=8498)
COMMON_STEPS = dict(original=192000,narrow=204000)


def _plain_path(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path,*path.parents)), "no symlink evidence path")
    return path


def load_actor(cell, checkpoint, actor_cfg):
    """Hash the exact bytes deserialized by weights-only loading, strictly restore CPU actor."""
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    cell.__post_init__()
    path = _plain_path(checkpoint)
    require(path.is_file(), "regular checkpoint")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == mapping.CHECKPOINTS[cell.policy], "pinned checkpoint bytes")
    payload = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    require(type(payload) is dict and type(payload.get('iter')) is int
            and payload['iter'] == ITERATIONS[cell.policy], "fixed checkpoint iteration")
    common_step = payload['infos']['env_state']['common_step_counter']
    require(type(common_step) is int and common_step == COMMON_STEPS[cell.policy], "fixed saved environment time")
    state = payload['actor_state_dict']
    audit = audit_actor_state(state,actor_cfg)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = MLPModel(TensorDict({'actor':torch.zeros(8,61)},batch_size=[8]),
                         {'actor':['actor']},'actor',14,hidden_dims=actor_cfg.hidden_dims,
                         activation=actor_cfg.activation,obs_normalization=True,
                         distribution_cfg=deepcopy(actor_cfg.distribution_cfg))
        model.load_state_dict(state,strict=True)
        model.eval()
        require(set(model.state_dict()) == set(state) and all(
            v.dtype == state[k].dtype and torch.equal(v,state[k])
            for k,v in model.state_dict().items()), "exact actor/normalizer tensor restoration")
    return model,dict(checkpoint_sha256=digest,checkpoint_bytes=len(raw),saved_iteration=payload['iter'],
                      common_step=common_step,actor_state_sha256=core.actor_digest(model),
                      strict_actor_restore=True,optimizer_restored=False,critic_restored=False,
                      actor_audit=audit,policy_acceptance=False)


def _native_types():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    return ManagerBasedRlEnv,RslRlVecEnvWrapper


@contextmanager
def native_session(cell, checkpoint, *, device):
    """Close a successfully created env even if wrapper/reset/stream setup fails.

No training runner is constructed. The context owns one explicit native session.
Construction failures inside the native constructor itself need process teardown.
"""
    require(device in ('cpu','cuda:0'), "explicit supported simulation device")
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == ('' if device == 'cpu' else '0'),
            "explicit isolated device environment")
    before = runtime_identity()  # Existing version and selected dependency pins.
    cfg,agent = core.prepare_config(cell)
    config_yaml = {name:yaml.dump(asdict(value),sort_keys=False)
                   for name,value in (('env',cfg),('agent',agent))}
    actor,loading = load_actor(cell,checkpoint,agent.actor)
    env_type,wrapper_type = _native_types()
    torch.manual_seed(cell.seed)
    env = env_type(cfg=cfg,device=device)
    try:
        wrapped = wrapper_type(env,clip_actions=agent.clip_actions)
        reset_common_step = env.common_step_counter
        # Match the retained wrapper-then-load order without loading optimizer.
        env.common_step_counter = loading['common_step']
        actor = actor.to(device)
        require(core.actor_digest(actor) == loading['actor_state_sha256'], "unchanged state after device transfer")
        stream = MotorStepStream.from_robot(env.scene['robot'],8,device=device,cost_cfg=MotorStepCostCfg())
        env._microduck_motor_step_stream = stream
        runtime = dict(selected_pins=before,complete_runtime_equivalence_verified=False)
        runtime['reset_evidence'] = reset_evidence.snapshot(env,cell,reset_common_step=reset_common_step)
        reset_evidence.validate(runtime['reset_evidence'],cell,device=device,common_step=loading['common_step'])
        yield dict(wrapped=wrapped,actor=actor,stream=stream,loading=loading,runtime=runtime,
                   config_yaml=config_yaml,native_backend='mjlab',device=device)
        require(runtime_identity() == before, "selected runtime changed during capture")
        require(hashlib.sha256(_plain_path(checkpoint).read_bytes()).hexdigest() == loading['checkpoint_sha256'],
                "checkpoint changed during capture")
    finally:
        env.close()


def _fsync_dir(path):
    fd = os.open(path,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


class Evidence:
    """Exclusive output directory, durable completed-frame journal, manifest last."""

    def __init__(self,path):
        self.path = _plain_path(path)
        require(self.path.parent.is_dir(), "existing explicit output parent")
        self.path.mkdir()  # Existing directories, including closed cells, refuse reuse.
        _fsync_dir(self.path.parent)
        self.next_frame = 0
        self.journal = (self.path/'frames.jsonl').open('x')
        try:
            self.journal.flush(); os.fsync(self.journal.fileno()); _fsync_dir(self.path)
        except Exception:
            self.journal.close()
            raise

    def write(self,name,raw):
        require(type(name) is str and name not in ('','.', '..','frames.jsonl','manifest.json')
                and '/' not in name and '\\' not in name, "exclusive evidence basename")
        with (self.path/name).open('xb') as output:
            output.write(raw); output.flush(); os.fsync(output.fileno())
        _fsync_dir(self.path)

    def json(self,name,value):
        self.write(name,(canonical(value)+'\n').encode())

    def frame(self,step,values):
        require(type(step) is int and step == self.next_frame and step < mapping.STEPS,
                "ordered bounded frame journal")
        # JSON disallows NaN. If corrupt samples cannot be journaled, the caller
        # retains them in the failure tensor payload instead, never as a pass.
        row = dict(step=step,tensors={k:dict(dtype=str(v.dtype),shape=list(v.shape),value=v.tolist())
                                     for k,v in values.items()})
        self.journal.write(canonical(row)+'\n')
        self.journal.flush(); os.fsync(self.journal.fileno())
        self.next_frame += 1

    def close(self):
        if not self.journal.closed:
            try:
                self.journal.flush(); os.fsync(self.journal.fileno())
            finally:
                self.journal.close()

    def seal(self,status, *, guard=None):
        require(status in ('captured-diagnostic-only','runtime-failure-stop'), "explicit closure outcome")
        self.close()
        files = {}
        for path in sorted(self.path.iterdir()):
            require(path.is_file() and not path.is_symlink() and path.name != 'manifest.json',
                    "regular new evidence files")
            raw = path.read_bytes()
            files[path.name] = dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
        if guard is not None:
            guard()
        with (self.path/'manifest.json').open('x') as output:
            output.write(canonical(dict(protocol=PROTOCOL,status=status,files=files,
                                        policy_acceptance=False,physical_motion_authorized=False))+'\n')
            output.flush(); os.fsync(output.fileno())
        _fsync_dir(self.path)


def _tensor_bytes(value):
    buffer = io.BytesIO(); torch.save(value,buffer); return buffer.getvalue()


def retain_cell(cell, checkpoint, output, *, device, session_factory=native_session,
                budget_seconds=120, clock=time.monotonic):
    """Retain one explicit cell. Caller must guard host/source and enforce hard timeout.

session_factory is a test seam; a replacement never gains verified-native status.
This is not an 18-cell launcher, a retry API or an authorization mechanism.
"""
    require(type(budget_seconds) in (int,float) and 0 < budget_seconds <= 120, "bounded cell budget")
    cell.__post_init__()
    evidence = Evidence(output)
    started = last_time = None

    def checked_clock():
        nonlocal started, last_time
        now = clock()
        require(math.isfinite(now), "finite cell clock")
        if started is None:
            started = last_time = now
        require(now >= last_time, "cell clock regressed")
        require(now-started < budget_seconds, "cell budget including construction, capture and retention")
        last_time = now
        return now

    try:
        checked_clock()
        evidence.json('launch.json',dict(protocol=PROTOCOL,cell=cell.identity(),device=device,
                                       budget_seconds=budget_seconds,policy_acceptance=False))
        with session_factory(cell,checkpoint,device=device) as session:
            require(set(session['config_yaml']) == {'env','agent'}, "both configuration snapshots required")
            for name,value in session['config_yaml'].items():
                require(name in ('env','agent'), "known configuration snapshot")
                evidence.write(name+'.yaml',value.encode())
            evidence.json('loading.json',session['loading'])
            evidence.json('runtime.json',session['runtime'])
            remaining = budget_seconds-(checked_clock()-started)
            require(0 < remaining <= budget_seconds, "construction consumed cell budget or clock regressed")
            result = core.capture(cell,session['wrapped'],session['actor'],session['stream'],
                                  budget_seconds=remaining,clock=checked_clock,on_frame=evidence.frame)
            require(result.evidence['actor_state_sha256'] == session['loading']['actor_state_sha256'],
                    "captured actor matches loaded tensor digest")
        # Context exit/cleanup and post-capture checks must succeed before success.
        checked_clock()
        score = mapping.score(cell,result.trace)
        evidence.write('trace.pt',_tensor_bytes({**asdict(result.trace),'cached_commands':result.cached_commands}))
        evidence.json('capture.json',result.evidence)
        evidence.json('score.json',score)
        checked_clock()
        evidence.json('decision.json',dict(status='captured-diagnostic-only',classification=score['classification'],
                       native_factory_used=session_factory is native_session,policy_acceptance=False,
                       training_admitted=False,physical_motion_authorized=False))
        evidence.seal('captured-diagnostic-only',guard=checked_clock)
        return score
    except Exception as exc:
        try:
            if isinstance(exc,core.CaptureFailure):
                evidence.write('partial-failure.pt',_tensor_bytes(exc.frames))
            evidence.json('failure.json',dict(status='runtime-failure-stop',error_type=type(exc).__name__,
                           error=str(exc),journaled_complete_steps=evidence.next_frame,policy_acceptance=False))
            evidence.seal('runtime-failure-stop')
        except Exception as retention_error:
            # A failed filesystem cannot promise durable failure evidence. Keep
            # the causal exception and surface this separately, never return a score.
            exc.add_note(f"Failure-evidence retention also failed: {retention_error}")
        raise
    finally:
        evidence.close()
