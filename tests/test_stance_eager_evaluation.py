"""CPU evaluation integration; fixtures never establish learned stance."""
from copy import deepcopy
from hashlib import sha256
import os
import subprocess
import sys
import time

import pytest
import torch

from mjlab_microduck import stance_eager_evaluation as evaluation
from mjlab_microduck import stance_checkpoint as cp
from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_evaluation_bundle as bundle
from mjlab_microduck import stance_evaluation_worker as worker
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime


@pytest.fixture(scope='module')
def fixture_inputs():
    learner = evaluation.training.EagerLearner(); entries = []; raw = {}
    for iteration in (-1, 127):
        meta = evaluation.training.identity(evaluation.TRAINING_SOURCE, 'b'*64, 'c'*64, learner, iteration)
        # Both are fresh synthetic models; no fake claim that this is trained.
        data = cp.encode(learner.actor, learner.critic, meta); raw[iteration] = data
        entries.append(dict(file='initial.pt' if iteration == -1 else 'model_127.pt',
                            sha256=sha256(data).hexdigest(), identity=meta))
    retained = dict(source=evaluation.TRAINING_SOURCE, report_sha256=evaluation.TRAINING_REPORT, checkpoints=entries)
    return evaluation.plan('d'*40, {}, 'e'*64, retained, 123456), raw


def test_exact_separate_matrix_no_optimizer_or_best_checkpoint(fixture_inputs):
    launch, _ = fixture_inputs
    assert [(c['binding']['checkpoint_iteration'], c['binding']['evaluation_seed']) for c in launch['cases']] == [
        (iteration, seed) for iteration in (-1, 127) for seed in (541, 547, 557)]
    assert launch['attempts_required'] == 768 and launch['worlds_per_case'] == 128
    assert launch['optimizer_steps'] == 0 and launch['forward_graph'] is False
    assert launch['child_timeout_seconds'] == 900 and launch['service_timeout_seconds'] == 960
    assert not launch['checkpoint_admitted'] and not launch['physical_motion_authorized']


@pytest.mark.parametrize('iteration', [-1, 127])
def test_strict_frozen_diagnostic_restore_not_pilot(fixture_inputs, iteration):
    launch, raws = fixture_inputs
    entry = launch['retained_training']['checkpoints'][iteration == 127]
    actor, receipt = cp.load_eager_diagnostic(raws[iteration], entry['sha256'], entry['identity'])
    assert not actor.training and all(not p.requires_grad and p.device.type == 'cpu' for p in actor.parameters())
    assert not receipt['checkpoint_admitted'] and not receipt['optimizer_restored']
    with pytest.raises(ValueError, match='only declared pilot'):
        cp.load_evaluation(raws[iteration], entry['sha256'], entry['identity'])
    wrong = deepcopy(entry['identity']); wrong['iteration'] = 3
    with pytest.raises(ValueError, match='initializer/final'): cp.load_eager_diagnostic(raws[iteration], entry['sha256'], wrong)
    wrong['purpose'] = 'pilot'; wrong['iteration'] = 128
    with pytest.raises(ValueError, match='initializer/final'): cp.load_eager_diagnostic(raws[iteration], entry['sha256'], wrong)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('iteration,protocol,valid', [(-1, trace.PROTOCOL, False),
    (127, trace.PROTOCOL, False), (128, trace.EAGER_PROTOCOL, False), (511, trace.EAGER_PROTOCOL, False),
    (-1, trace.EAGER_PROTOCOL, True), (127, trace.EAGER_PROTOCOL, True)])
def test_no_iteration_relabeling(fixture_inputs, iteration, protocol, valid):
    binding = deepcopy(fixture_inputs[0]['cases'][0]['binding'])
    binding.update(checkpoint_iteration=iteration, protocol=protocol)
    if valid: trace.validate_binding(binding)
    else:
        with pytest.raises(ValueError): trace.validate_binding(binding)


def test_real_cpu_eager_bundle_replays_same_physics_and_controls(tmp_path, fixture_inputs):
    launch, raws = fixture_inputs; case = launch['cases'][0]; entry = case['checkpoint']
    env = WarpStanceRuntime(2, device='cpu')
    runtime = plant.runtime_bytes('d'*40, env.native)
    binding = {**case['binding'], 'worlds': 2, 'capture_device': 'cpu', 'runtime_sha256': sha256(runtime).hexdigest()}
    del binding['launch_sha256']
    launch_raw = bundle.launch_bytes(binding, entry['identity']); binding['launch_sha256'] = sha256(launch_raw).hexdigest()
    result = worker.evaluate_owned_case(tmp_path/'case', env, binding=binding,
        checkpoint_raw=raws[-1], checkpoint_identity=entry['identity'], runtime_raw=runtime,
        launch_raw=launch_raw, deadline_monotonic=time.monotonic()+60, policy_tick_limit=1)
    score = bundle.verify_bundle(tmp_path/'case', result['manifest_sha256'], binding=binding, checkpoint_identity=entry['identity'])
    assert score == result['score'] and score['protocol'] == trace.EAGER_PROTOCOL
    assert score['complete_attempts'] == 0 and score['strict_checkpoint_checked'] and score['motor_commit_masks_checked']
    assert not score['checkpoint_admitted'] and not torch.cuda.is_initialized()


def synthetic_scores(launch, passes=122):
    scores = {}
    for case in launch['cases']:
        attempts = [dict(world_id=i, complete_first_attempt=True, candidate_pass=i < passes,
            last_physics_step=2500 if i < passes else 400, hard_failure=i >= passes,
            gates={'full_duration': i < passes}) for i in range(128)]
        score = dict(protocol=trace.EAGER_PROTOCOL, binding=case['binding'], attempts=attempts,
            complete_attempts=128, numerical_passes=passes, terminal_contact_records_checked=128)
        score.update({k: True for k in ('trajectory_continuity_validated', 'strict_checkpoint_checked',
            'deterministic_actor_replay_checked', 'compiled_plant_checked', 'nominal_reset_checked',
            'terminal_contact_summary_checked', 'kinematic_observations_checked', 'action_slew_checked',
            'delayed_motor_targets_checked', 'motor_commit_masks_checked', 'voltage_history_checked')})
        score.update({k: False for k in ('checkpoint_admitted', 'learned_stance_accepted',
                                      'physical_motion_authorized', 'provenance_validated')})
        scores[case['name']] = score
    return scores


@pytest.mark.parametrize('passes,accepted', [(121, False), (122, True), (128, True), (0, False)])
def test_original_per_seed_threshold_and_no_capability_admission(fixture_inputs, passes, accepted):
    launch, _ = fixture_inputs; result = evaluation.summarize(launch, synthetic_scores(launch, passes))
    assert result['final_numerical_gate_passed'] is accepted and result['complete_attempts'] == 768
    assert not result['learned_stance_accepted'] and not result['football_balance_accepted']


@pytest.mark.parametrize('damage', ['missing_case', 'partial', 'false_count', 'admitted', 'changed_plan'])
def test_missing_evidence_cannot_become_completion(fixture_inputs, damage):
    launch = deepcopy(fixture_inputs[0]); scores = synthetic_scores(launch)
    one = scores['final-seed-557']
    if damage == 'missing_case': scores.pop('initial-seed-541')
    elif damage == 'partial': one['attempts'][0]['complete_first_attempt'] = False
    elif damage == 'false_count': one['numerical_passes'] = 123
    elif damage == 'admitted': one['checkpoint_admitted'] = True
    else: launch['worlds_per_case'] = 64
    with pytest.raises(ValueError): evaluation.summarize(launch, scores)


def test_training_hash_before_tensor_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation.training, 'output_path', lambda source: tmp_path)
    evaluation.files.write_json(tmp_path/'report.json', {})
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('hash before load'))
    with pytest.raises(ValueError, match='training report hash'): evaluation.training_inputs()


@pytest.mark.parametrize('key', [None, 'MainPID', 'ActiveState', 'RuntimeMaxUSec', 'KillMode'])
def test_independent_service_gate(monkeypatch, key):
    fields = dict(MainPID=str(os.getpid()), ActiveState='active', RuntimeMaxUSec='16min', KillMode='control-group')
    if key: fields[key] = 'wrong'
    monkeypatch.setattr(evaluation.host, 'read', lambda *a: '\n'.join(k+'='+v for k, v in fields.items()))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    if key:
        with pytest.raises(ValueError): evaluation.check_service('a'*40)
    else: evaluation.check_service('a'*40)


def test_launch_hash_before_training_or_runtime_check(tmp_path, monkeypatch):
    evaluation.files.write_json(tmp_path/'launch.json', {})
    monkeypatch.setattr(evaluation, 'output_path', lambda source: tmp_path)
    monkeypatch.setattr(evaluation, 'training_inputs', lambda: pytest.fail('hash first'))
    with pytest.raises(ValueError, match='evaluation launch hash'): evaluation.inputs_check('a'*40, 'b'*64)


def test_cli_is_evaluation_only():
    result = subprocess.run([sys.executable, '-m', evaluation.MODULE, '--help'],
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0 and '{prepare,supervise,child}' in result.stdout
    for option in ('--resume', '--graph', '--iterations', '--train'): assert option not in result.stdout
