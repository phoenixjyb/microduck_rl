"""The lean-lesson throughput probe: timing only, never a capability.

These tests pin the properties the predeclaration
``docs/experiments/2026-09-16-stance-lean-lesson-throughput-probe.md`` depends on:
that expired authority is never a window, that the cap comes from the measured
worst case rather than a mean or an estimate, and that the optimizer component is
measured on the CPU and is never physics evidence.
"""
import hashlib
import time

import pytest
import torch

from mjlab_microduck import stance_checkpoint as cp
from mjlab_microduck import stance_eager_learning as eager
from mjlab_microduck import stance_lean_lesson as lean
from mjlab_microduck import stance_lean_throughput as probe

SOURCE, LAUNCH, RUNTIME = 'a'*40, 'b'*64, 'c'*64


def synthetic_parent(monkeypatch):
    """An eager final export standing in for the pinned artifact."""
    learner = eager.EagerLearner()
    meta = eager.identity(cp.LEAN_PARENT_SOURCE, LAUNCH, RUNTIME, learner,
                          cp.LEAN_PARENT_ITERATION)
    raw = cp.encode(learner.actor, learner.critic, meta)
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(cp, 'LEAN_PARENT_SHA256', digest)
    return cp.load_lean_parent(raw, digest, meta)


def test_declared_probe_scope_matches_the_predeclaration():
    assert probe.PROTOCOL == 'football-b1n-lean-lesson-throughput-v1'
    assert probe.MODULE == 'mjlab_microduck.stance_lean_throughput'
    assert (probe.WORLDS, probe.TICKS) == (64, 24)
    assert probe.WORLDS == lean.WORLDS and probe.TICKS == 24
    assert (probe.WARMUP_UPDATES, probe.MEASURED_UPDATES) == (2, 8)
    assert probe.TARGET_UPDATES == lean.UPDATES == 256
    assert probe.SAFETY == 1.25
    assert probe.WATCHDOG_MARGIN_SECONDS == 60
    assert probe.CLOSEOUT_SECONDS == 600
    assert probe.MAX_WINDOW_SECONDS == 3600
    assert (probe.PROBE_CHILD_SECONDS, probe.PROBE_SERVICE_SECONDS) == (600, 660)
    assert probe.PROBE_CHILD_SECONDS < probe.PROBE_SERVICE_SECONDS
    assert probe.PARENT_SECONDS_PER_UPDATE == 735.963/128


def test_expired_authority_is_never_a_window():
    now = int(time.time())
    probe.check_window(now+1800, launching=True)
    probe.check_window(now+3600)
    # 660 s service + 600 s closeout + 60 s margin = 1320 s is the declared floor.
    probe.check_window(now+1321, launching=True)
    for offset in (-1, 0, 60, 600, 1320):
        with pytest.raises(ValueError, match='fresh 22-to-60-minute window|in the future'):
            probe.check_window(now+offset, launching=True)
    for offset in (-3600, 3601, 7200):
        with pytest.raises(ValueError, match='in the future|inside 60 minutes'):
            probe.check_window(now+offset, launching=True)
    with pytest.raises(ValueError, match='explicit integer deadline'):
        probe.check_window(float(now+1800), launching=True)
    with pytest.raises(ValueError, match='explicit integer deadline'):
        probe.check_window(True, launching=True)
    # The stale CUDA-probe cutoff is expired and must not be reusable as a window.
    assert probe.host.CUTOFF < time.time()
    with pytest.raises(ValueError, match='in the future'):
        probe.check_window(int(probe.host.CUTOFF), launching=True)
    with pytest.raises(ValueError, match='in the future'):
        probe.check_window(int(probe.host.CUTOFF)-1, launching=True)


def test_summarize_is_nearest_rank_and_refuses_degenerate_input():
    ordered = [float(v) for v in range(1, 11)]
    assert probe.quantile(ordered, .95) == 10.0
    assert probe.quantile(ordered, .5) == 5.0
    assert probe.quantile(ordered, .1) == 1.0
    summary = probe.summarize([1.0, 2.0, 3.0, 4.0])
    assert summary == dict(count=4, mean=2.5, median=2.5, p95=4.0, max=4.0,
                           series=[1.0, 2.0, 3.0, 4.0])
    for bad in ([], [1.0], [1.0, float('nan')], [1.0, float('inf')], [1.0, 0.0],
                [1.0, -2.0], [1, 2.0], 'nope'):
        with pytest.raises(ValueError):
            probe.summarize(bad)
    with pytest.raises(ValueError):
        probe.quantile([], .95)


def test_caps_use_the_worst_case_and_keep_the_watchdog_inside():
    collection = probe.summarize([5.0]*8)
    optimizer = probe.summarize([0.12]*8)
    derived = probe.caps(collection, optimizer, 165.0)
    # ceil(256 * 5.12) + ceil(165.0) = 1311 + 165
    assert derived['per_update_worst_seconds'] == pytest.approx(5.12)
    assert derived['predicted_child_seconds'] == 1476
    assert derived['service_seconds'] == 1845
    assert derived['child_seconds'] == 1785
    assert derived['child_seconds'] < derived['service_seconds']
    assert derived['service_seconds']-derived['child_seconds'] == probe.WATCHDOG_MARGIN_SECONDS
    assert derived['closeout_seconds'] == 600
    assert derived['parent_estimate_seconds'] == 1472
    # The cap must follow the maximum, not the mean.
    spiky = probe.summarize([1.0, 1.0, 1.0, 40.0])
    assert probe.caps(spiky, optimizer, 0.0)['per_update_worst_seconds'] > spiky['mean']+optimizer['max']
    with pytest.raises(ValueError, match='measured setup seconds'):
        probe.caps(collection, optimizer, -1.0)
    with pytest.raises(ValueError, match='summarized components'):
        probe.caps({'max': 1.0}, optimizer, 1.0)


def test_plan_declares_no_optimizer_and_no_admission():
    deadline = int(time.time())+1800
    value = probe.plan(SOURCE, {'machine': 'x'}, deadline)
    assert value['protocol'] == probe.PROTOCOL and value['source'] == SOURCE
    assert value['deadline_unix'] == deadline
    assert (value['worlds'], value['ticks']) == (64, 24)
    assert (value['warmup_updates'], value['measured_updates']) == (2, 8)
    assert value['target_updates'] == 256 and value['safety_factor'] == 1.25
    assert value['parent_checkpoint_sha256'] == cp.LEAN_PARENT_SHA256
    assert value['parent_iteration'] == 127 and value['parent_file'] == 'model_127.pt'
    assert value['optimizer_steps'] == 0
    for key in ('checkpoint_admitted', 'training_admitted', 'learned_stance',
                'physical_motion_authorized'):
        assert value[key] is False
    with pytest.raises(ValueError, match='explicit integer deadline'):
        probe.plan(SOURCE, {}, 0)


def test_optimizer_component_is_cpu_only_and_not_physics_evidence(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    parent = synthetic_parent(monkeypatch)
    result = probe.measure_optimizer(parent)
    assert result['device'] == 'cpu'
    assert result['stand_in'] is True
    assert result['physics_evidence'] is False
    assert result['optimizer_steps'] == probe.MEASURED_UPDATES
    assert result['samples'] == probe.TICKS*probe.WORLDS
    summary = result['summarize']
    assert summary['count'] == probe.MEASURED_UPDATES
    assert len(summary['series']) == probe.MEASURED_UPDATES
    assert summary['max'] >= summary['mean'] >= 0
    assert torch.cuda.is_initialized() is False


def test_optimizer_component_refuses_to_run_with_cuda_visible(monkeypatch):
    parent = synthetic_parent(monkeypatch)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError, match='CPU-only optimizer measurement'):
        probe.measure_optimizer(parent)


def test_decide_admits_nothing_and_composes_both_components(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    parent = synthetic_parent(monkeypatch)
    optimizer = probe.measure_optimizer(parent)
    collection = dict(summarize=probe.summarize([5.63]*8), setup_seconds=165.0,
                      optimizer_steps=0, physics_device='cuda:0')
    decision = probe.decide(collection, optimizer)
    assert decision['protocol'] == probe.PROTOCOL
    assert decision['decision'] == 'throughput-measured-not-a-capability'
    assert decision['collection'] == collection and decision['optimizer'] == optimizer
    for key in ('checkpoint_admitted', 'training_admitted', 'learned_stance',
                'physical_motion_authorized'):
        assert decision[key] is False
    assert decision['caps']['child_seconds'] < decision['caps']['service_seconds']
    assert decision['optimizer']['stand_in'] is True
