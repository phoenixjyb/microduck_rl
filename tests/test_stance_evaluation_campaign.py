"""Synthetic matrix planning/dispatch decisions, never full held-out evidence."""
from copy import deepcopy
from hashlib import sha256
import pytest
import torch

from mjlab_microduck import stance_checkpoint as cp
from mjlab_microduck import stance_evaluation_campaign as campaign
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_attempt_trace as trace


@pytest.fixture(scope='module')
def inputs():
    actor, critic = cp.fresh_models(521)
    entries = []; raw = {}
    for iteration in trace.CHECKPOINTS:
        # Deliberately synthetic metadata: no training iterations occurred.
        identity = dict(protocol=cp.PROTOCOL, source='a'*40, runtime_sha256='b'*64,
            training_launch_sha256='c'*64, purpose='pilot', training_seed=521, worlds=512,
            iteration=iteration, initial_state_sha256=cp.state_hash(cp.states_of(actor, critic)),
            architecture=deepcopy(cp.ARCHITECTURE))
        raw[iteration] = cp.encode(actor, critic, identity)
        entries.append(dict(sha256=sha256(raw[iteration]).hexdigest(), identity=identity))
    runtime = plant.runtime_bytes('d'*40, plant.build_entity().compile())
    plan = campaign.make_plan('d'*40, sha256(runtime).hexdigest(), entries)
    return plan, runtime, raw


def fake_scores(plan, *, final_passes=122, early_passes=0):
    result = {}
    for case in plan['cases']:
        n = final_passes if case['binding']['checkpoint_iteration'] == 511 else early_passes
        score = dict(binding=deepcopy(case['binding']), attempts=[dict(world_id=i,
            complete_first_attempt=True, candidate_pass=i < n) for i in range(128)],
            complete_attempts=128, numerical_passes=n, terminal_contact_records_checked=128)
        score.update({k: False for k in ('provenance_validated', 'checkpoint_admitted', 'learned_stance_accepted', 'physical_motion_authorized')})
        score.update({k: True for k in ('trajectory_continuity_validated', 'strict_checkpoint_checked',
            'deterministic_actor_replay_checked', 'compiled_plant_checked', 'nominal_reset_checked',
            'kinematic_observations_checked', 'terminal_contact_summary_checked', 'action_slew_checked',
            'delayed_motor_targets_checked', 'motor_commit_masks_checked', 'voltage_history_checked')})
        result[case['name']] = score
    return result


def test_all_checkpoint_bytes_preflight_without_cuda(inputs):
    plan, runtime, raw = inputs
    assert campaign.preflight(plan, runtime, raw) == dict(input_bytes_checked=True, matrix_cases=12, gpu_launch_authorized=False)
    assert len(plan['cases']) == 12 and plan['attempts_required'] == 1536
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('damage', ['missing_case', 'reorder', 'seed', 'threshold', 'worlds', 'device', 'flag', 'extra'])
def test_predeclared_matrix_cannot_drift(inputs, damage):
    plan = deepcopy(inputs[0])
    if damage == 'missing_case': plan['cases'].pop()
    elif damage == 'reorder': plan['checkpoints'].reverse()
    elif damage == 'seed': plan['cases'][0]['binding']['evaluation_seed'] = 999
    elif damage == 'threshold': plan['final_required_passes_per_seed'] = 121
    elif damage == 'worlds': plan['worlds_per_case'] = 127
    elif damage == 'device': plan['actor_device'] = 'cuda:0'
    elif damage == 'flag': plan['gpu_launch_authorized'] = True
    else: plan['extra'] = 1
    with pytest.raises(ValueError): campaign.validate_plan(plan)


def test_checkpoints_must_come_from_one_declared_training_run(inputs):
    entries = deepcopy(inputs[0]['checkpoints']); entries[1]['identity']['training_launch_sha256'] = 'e'*64
    with pytest.raises(ValueError, match='matched pilot'): campaign.make_plan('d'*40, inputs[0]['runtime_sha256'], entries)


@pytest.mark.parametrize('damage', ['runtime', 'checkpoint', 'missing_checkpoint'])
def test_input_bytes_cannot_be_replaced(inputs, damage):
    plan, runtime, raw = deepcopy(inputs)
    if damage == 'runtime': runtime += b'x'
    elif damage == 'checkpoint': raw[128] += b'x'
    else: raw.pop(511)
    with pytest.raises(ValueError): campaign.preflight(plan, runtime, raw)


def test_122_of_128_each_final_seed_not_best_early_checkpoint(inputs):
    plan = inputs[0]
    passed = campaign._summarize(plan, fake_scores(plan))
    assert passed['final_numerical_gate_passed'] and passed['total_complete_attempts'] == 1536
    assert not passed['checkpoint_admitted'] and not passed['football_balance_accepted']
    assert not campaign._summarize(plan, fake_scores(plan, final_passes=121, early_passes=128))['final_numerical_gate_passed']
    scores = fake_scores(plan, final_passes=128)
    key = plan['cases'][-1]['name']; scores[key]['numerical_passes'] = 121
    for i in range(121, 128): scores[key]['attempts'][i]['candidate_pass'] = False
    assert not campaign._summarize(plan, scores)['final_numerical_gate_passed']


@pytest.mark.parametrize('damage', ['missing', 'duplicate_world', 'prefix', 'false_count', 'admission', 'unchecked', 'seed', 'contacts'])
def test_incomplete_or_mislabeled_scores_cannot_enter_matrix(inputs, damage):
    plan = inputs[0]; scores = fake_scores(plan); key = plan['cases'][0]['name']; score = scores[key]
    if damage == 'missing': scores.pop(key)
    elif damage == 'duplicate_world': score['attempts'][1]['world_id'] = 0
    elif damage == 'prefix': score['attempts'][0]['complete_first_attempt'] = False
    elif damage == 'false_count': score['numerical_passes'] = 128
    elif damage == 'admission': score['checkpoint_admitted'] = True
    elif damage == 'unchecked': score['delayed_motor_targets_checked'] = False
    elif damage == 'seed': score['binding']['evaluation_seed'] = 547
    else: score['terminal_contact_records_checked'] = 127
    with pytest.raises(ValueError): campaign._summarize(plan, scores)


def fixture_directories(tmp_path, plan):
    root = tmp_path/'matrix'; root.mkdir()
    for case in plan['cases']: (root/case['name']).mkdir()
    return root


def test_index_reverifies_every_case_and_refuses_rehashed_decision(tmp_path, inputs, monkeypatch):
    plan = inputs[0]; root = fixture_directories(tmp_path, plan); scores = fake_scores(plan); calls = []
    hashes = {case['name']: sha256(case['name'].encode()).hexdigest() for case in plan['cases']}
    def verified(path, digest, *, binding, checkpoint_identity):
        calls.append(path.name)
        case = next(c for c in plan['cases'] if c['name'] == path.name)
        assert digest == hashes[path.name] and binding == case['binding'] and checkpoint_identity == case['checkpoint_identity']
        return deepcopy(scores[path.name])
    # Synthetic dispatch fixture only; actual bundle verification has its own
    # real short-CPU tests. Empty directories below are not physical evidence.
    monkeypatch.setattr(campaign.bundle, 'verify_bundle', verified)
    digest, summary = campaign.retain_index(root, plan, hashes)
    assert calls == [c['name'] for c in plan['cases']]
    assert campaign.verify_index(root, digest, plan) == summary and len(calls) == 24
    raw = campaign.files.file_bytes(root/'matrix.json'); index = campaign.files.parse(raw)
    index['summary']['checkpoint_admitted'] = True
    changed = campaign.plan_bytes(index); (root/'matrix.json').write_bytes(changed)
    with pytest.raises(ValueError, match='decision replay'): campaign.verify_index(root, sha256(changed).hexdigest(), plan)


def test_failure_never_publishes_index(tmp_path, inputs, monkeypatch):
    plan = inputs[0]; root = fixture_directories(tmp_path, plan)
    def invalid(*_a, **_kw): raise ValueError('synthetic corrupt bundle')
    monkeypatch.setattr(campaign.bundle, 'verify_bundle', invalid)
    with pytest.raises(ValueError, match='corrupt bundle'):
        campaign.retain_index(root, plan, {c['name']: 'e'*64 for c in plan['cases']})
    assert not (root/'matrix.json').exists()
