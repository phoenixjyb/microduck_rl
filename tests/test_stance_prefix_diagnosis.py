"""Synthetic evidence tests; no GPU or model-quality claims."""
from copy import deepcopy
from hashlib import sha256
import io
import sys

import pytest
import torch

from mjlab_microduck import stance_prefix_diagnosis as diagnosis


def test_bits_numbers_discrete_and_structure_are_separate():
    zero = diagnosis.compare_tree({'x': torch.tensor([0.])}, {'x': torch.tensor([-0.])})
    assert not zero['bitwise_equal'] and zero['discrete_equal'] and zero['structure_equal']
    assert zero['fields']['x']['max_abs_difference'] == 0
    numeric = diagnosis.compare_tree({'x': torch.tensor([0., 1.])}, {'x': torch.tensor([0., 1.25])})
    assert numeric['fields']['x']['max_abs_difference'] == .25
    assert numeric['fields']['x']['first']['first_index'] == [1]
    discrete = diagnosis.compare_tree({'x': torch.tensor([True])}, {'x': torch.tensor([False])})
    assert not discrete['discrete_equal'] and discrete['structure_equal']
    changed = diagnosis.compare_tree({'x': torch.zeros(2, 1)}, {'x': torch.zeros(1, 2)})
    assert not changed['structure_equal']  # Never broadcast mismatched fields.
    assert diagnosis.compare_tree(torch.zeros(0), torch.zeros(0))['bitwise_equal']
    assert not diagnosis.compare_tree(0., -0.)['bitwise_equal']


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), torch.tensor([float('nan')]),
                               torch.tensor([1j]), object(), {1: 2}])
def test_invalid_leaves_fail_even_when_both_inputs_match(bad):
    with pytest.raises(ValueError): diagnosis.compare_tree(bad, bad)


def test_no_contact_reordering_or_terminal_slot_omission():
    a = {'contacts': [{'geom': 1, 'force': .5}, {'geom': 2, 'force': .6}]}
    r = diagnosis.compare_tree(a, {'contacts': list(reversed(a['contacts']))})
    assert not r['discrete_equal'] and 'contacts/*/force' in r['fields']
    assert not diagnosis.compare_tree([None, {'x': 1}], [None])['structure_equal']
    assert not diagnosis.compare_tree({'x': 0}, {})['structure_equal']
    assert not diagnosis.compare_tree(None, {'x': 1})['structure_equal']


def prefix(n):
    qpos = torch.zeros(n, 21); qpos[:, 2] = .12; qpos[:, 3] = 1
    qvel = torch.zeros(n, 20)
    state = dict(tilt=torch.zeros(n), root_velocity=qvel[:, :3].clone(), height=qpos[:, 2].clone(),
        support=torch.ones(n, 2), torque=torch.zeros(n, 14), joint_velocity=torch.zeros(n, 14),
        hard_limit=torch.zeros(n, dtype=torch.bool), forbidden_contact=torch.zeros(n, dtype=torch.bool),
        warning=torch.zeros(n, dtype=torch.bool))
    observation = dict(actor=torch.zeros(n, 44), critic=torch.zeros(n, 50))
    frame = dict(physics_steps=torch.zeros(n, dtype=torch.long), qpos=qpos, qvel=qvel,
        soft_limit_mask=torch.zeros(n, 14, dtype=torch.bool), state=state, observation=observation)
    result = dict(reward=torch.zeros(n), terminated=torch.zeros(n, dtype=torch.bool),
        timed_out=torch.zeros(n, dtype=torch.bool), episode_steps=torch.zeros(n, dtype=torch.long),
        executed_steps=torch.zeros(n, dtype=torch.long), live=torch.ones(n, dtype=torch.bool),
        term_sums={}, observation=observation, boundaries=[frame], terminal_records=[None]*n,
        optimizer_launched=False)
    control = {k: torch.zeros(n, width) for k, width in
        (('correction', 10), ('target', 14), ('previous', 14), ('voltage', 1), ('kp', 1),
         ('friction', 20), ('damping', 20), ('ctrl', 14))}
    control['queue'] = torch.zeros(n, 3, 14)
    return dict(result=result, control=control, integration=dict(qpos=qpos, qvel=qvel,
        time=torch.zeros(n), qacc_warmstart=torch.zeros(n, 20)))


def test_first_boundary_uses_time_order_not_alphabetical_result_fields():
    a = [prefix(2) for _ in range(4)]; b = deepcopy(a)
    b[2]['result']['boundaries'][0]['qvel'][1, 5] = .125
    result = diagnosis.compare_prefixes(a, b)
    assert result['initial_boundary']['bitwise_equal']
    first = result['first_divergent_boundary']
    assert (first['tick'], first['boundary']) == (2, 0)
    assert first['comparison']['fields']['qvel']['first']['first_index'] == [1, 5]
    assert not result['solver_phase_attribution_established']


def test_missing_boundary_is_not_an_exact_prefix():
    a = [prefix(2) for _ in range(4)]; b = deepcopy(a)
    b[1]['result']['boundaries'].append(deepcopy(b[1]['result']['boundaries'][0]))
    result = diagnosis.compare_prefixes(a, b)
    assert result['first_divergent_boundary']['tick'] == 1
    assert result['first_divergent_boundary']['boundary'] == 1
    assert not result['ticks'][1]['complete_tree']['structure_equal']


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path/'input'; root.mkdir(); cases = []
    for i, (n, graph) in enumerate(diagnosis.probe.CASES):
        data = prefix(n)
        case = dict(worlds=n, graph=graph, graph_bound=graph, device='cuda:0', warp_device='cuda:0',
            synthetic_terminal=True, optimizer_steps=0, training_admitted=False, physical_motion_authorized=False,
            ticks=[dict(tick=t, state_sha256=diagnosis.probe.tree_hash(data), reset_sha256='b'*64,
                        executed_steps=[0]*n) for t in range(24)],
            timing=[dict(step_s=.1, reset_s=0) for _ in range(24)], diagnostic_prefix=[])
        for tick in range(4):
            raw = io.BytesIO(); torch.save(data, raw); name = f'case-{i}-tick-{tick}.pt'
            (root/name).write_bytes(raw.getvalue())
            case['diagnostic_prefix'].append(dict(file=name, sha256=sha256(raw.getvalue()).hexdigest()))
        diagnosis.files.write_json(root/f'case-{i}.json', case); cases.append(case)
    decision = diagnosis.probe.decide(cases)
    diagnosis.files.write_json(root/'decision.json', decision)
    diagnosis.files.write_json(root/'launch.json', dict(protocol=diagnosis.probe.PROTOCOL, source='a'*40))
    (root/'child.log').write_text('Synthetic CPU fixture only\n')
    hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    diagnosis.files.write_json(root/'report.json', dict(protocol=diagnosis.probe.PROTOCOL,
        files=hashes, launch_sha256=hashes['launch.json'], result=decision, decision=decision['decision'],
        training_admitted=False, child=dict(returncode=0)))
    return root, sha256((root/'report.json').read_bytes()).hexdigest()


def test_complete_bundle_is_recomputed_deterministically_without_admission(bundle):
    root, digest = bundle
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    a = diagnosis.diagnose(root, digest); b = diagnosis.diagnose(root, digest)
    assert diagnosis.canonical(a) == diagnosis.canonical(b)
    assert a['decision'] == 'diagnostic-only'
    assert not a['training_admitted'] and not a['graph_equivalence_established']
    assert all(p['first_divergent_boundary'] is None for p in a['pairs'])
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('damage', ['report', 'raw', 'case', 'extra', 'missing', 'symlink'])
def test_damaged_input_fails_before_deserialization(bundle, damage, monkeypatch):
    root, digest = bundle
    if damage == 'report': digest = '0'*64
    if damage in ('raw', 'case'):
        path = root/('case-0-tick-0.pt' if damage == 'raw' else 'case-0.json')
        path.write_bytes(path.read_bytes()+b' ')
    if damage == 'extra': (root/'extra').touch()
    if damage == 'missing': (root/'case-0-tick-0.pt').unlink()
    if damage == 'symlink':
        raw = root/'case-0-tick-0.pt'; target = root.parent/'outside.pt'
        raw.rename(target); raw.symlink_to(target)
    monkeypatch.setattr(diagnosis.torch, 'load', lambda *a, **k: pytest.fail('unverified deserialization'))
    with pytest.raises(ValueError): diagnosis.diagnose(root, digest)


def test_value_hash_and_receipt_are_checked_before_pairing(bundle):
    root, _ = bundle
    case = diagnosis.files.parse((root/'case-0.json').read_bytes())
    case['ticks'][0]['state_sha256'] = '0'*64
    with pytest.raises(ValueError, match='value hash'): diagnosis.load_prefix(root, case, 0, 0)
    case = diagnosis.files.parse((root/'case-0.json').read_bytes())
    case['ticks'][0]['executed_steps'][0] = 9
    with pytest.raises(ValueError, match='executed-step'): diagnosis.load_prefix(root, case, 0, 0)


def test_cli_refuses_input_mutation_or_existing_output(bundle, monkeypatch):
    root, digest = bundle
    for target in (root/'derived.json', root/'report.json', root):
        monkeypatch.setattr(sys, 'argv', ['diagnose', '--input', str(root), '--report-sha256', digest,
                                        '--output', str(target)])
        with pytest.raises(ValueError, match='fresh derived'): diagnosis.main()
