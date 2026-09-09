"""Caller-owned first-attempt collection; no simulator allocation or GPU launcher."""
from copy import deepcopy
import math
import time

import torch

from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_control_evidence as control
from mjlab_microduck import stance_evaluation_bundle as bundle
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck.first_attempt_smoke import canonical
from mjlab_microduck.first_attempt_smoke import require


@torch.no_grad()
def collect(env, actor, binding, *, deadline_monotonic, policy_tick_limit=250, clock=time.monotonic):
    """Use only inside a caller's seeded, leased, independently timed service.

    CPU inference is intentional and matches bundle replay. Environment remains
    caller-owned; never reset it, replace its arrays or suppress first failures.
    A deadline/prefix limit retains partial evidence, not a completed attempt.
    The monotonic check cannot preempt a hung kernel: an outer hard timeout is
    mandatory for live use. Source, runtime and seed provenance remain external.
    """
    trace.validate_binding(binding)
    require(type(policy_tick_limit) is int and 1 <= policy_tick_limit <= 250, 'bounded evaluation ticks')
    require(type(deadline_monotonic) in (float, int) and math.isfinite(deadline_monotonic)
            and clock() < deadline_monotonic, 'evaluation wall budget before stepping')
    require(env.n == binding['worlds'] and str(env.device) == binding['capture_device']
            and env.live.all(), 'matching fresh evaluation environment')
    require(all(p.device.type == 'cpu' and not p.requires_grad for p in actor.parameters()), 'restored frozen CPU evaluation actor')
    recorder = trace.FirstAttemptTrace(binding, env.snapshot())
    controls = dict(protocol=control.PROTOCOL, binding=deepcopy(binding), ticks=[])
    reason = 'policy-tick-limit'
    for _ in range(policy_tick_limit):
        if not env.live.any(): reason = 'all-first-attempts-complete'; break
        if clock() >= deadline_monotonic: reason = 'wall-budget-exhausted'; break
        obs = env.observations()['actor']
        action_cpu = checkpoint.infer(actor, obs.detach().cpu())
        action = action_cpu.to(env.device)
        result = env.step(action, capture_control=True)
        recorder.append(result, obs, action)
        controls['ticks'].append(trace.owned(result['control_evidence']))
    require(controls['ticks'], 'no complete policy tick to retain')
    if not env.live.any(): reason = 'all-first-attempts-complete'
    return dict(payload=recorder.payload(), control_evidence=controls,
        collection=dict(stop_reason=reason, policy_ticks=len(controls['ticks']),
            actor_device='cpu', physics_device=str(env.device), seed_initialization_validated=False,
            independent_gpu_supervision_validated=False, checkpoint_admitted=False))


def evaluate_owned_case(directory, env, *, binding, checkpoint_raw, checkpoint_identity,
                        runtime_raw, launch_raw, deadline_monotonic, policy_tick_limit=250):
    """Restore/check before stepping; retain and independently replay afterward.

    Does not create an environment or grant a GPU lease. Partial prefixes may be
    retained but the campaign refuses them. Retains no new learned policy.
    """
    directory = bundle.files.native._plain_path(directory)
    if directory.exists(): raise FileExistsError('evaluation output already exists')
    actor, _, compiled = bundle.checked_inputs(binding, checkpoint_raw, checkpoint_identity, runtime_raw, launch_raw)
    require(canonical(plant.describe(env.native)) == canonical(compiled), 'owned environment compiled plant')
    plant.check_trace(dict(binding=binding, initial=trace.owned(env.snapshot()), ticks=[]), compiled)
    result = collect(env, actor, binding, deadline_monotonic=deadline_monotonic, policy_tick_limit=policy_tick_limit)
    digest, score = bundle.write_bundle(directory, result['payload'], control_evidence=result['control_evidence'],
        binding=binding, checkpoint_raw=checkpoint_raw, checkpoint_identity=checkpoint_identity,
        runtime_raw=runtime_raw, launch_raw=launch_raw)
    checked = bundle.verify_bundle(directory, digest, binding=binding, checkpoint_identity=checkpoint_identity)
    require(checked == score, 'immediate evaluation bundle replay')
    return dict(manifest_sha256=digest, score=score, collection=result['collection'])
