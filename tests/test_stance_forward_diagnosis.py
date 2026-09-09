"""Synthetic CPU fixtures are not graph or trained-policy acceptance evidence."""
from copy import deepcopy
from hashlib import sha256
import sys

import pytest
import torch

from mjlab_microduck import stance_forward_diagnosis as diagnosis
from mjlab_microduck import stance_forward_probe as probe
from test_stance_forward_probe import cpu_batch, synthetic_files


@pytest.fixture
def completed(cpu_batch, monkeypatch):
    root, result = cpu_batch
    monkeypatch.setattr(probe, 'WORLDS', (2,))
    synthetic_files(root, result)
    launch = dict(protocol=probe.PROTOCOL, source='a'*40, worlds=[2], graph_order=list(probe.ORDER),
        seed=523, motor_preparations_per_batch=1, integration_steps=0, optimizer_steps=0,
        training_admitted=False, physical_motion_authorized=False)
    (root/'launch.json').write_text(probe.canonical(launch)+'\n')
    checked = probe.verify(root)
    manifest = {p.name: sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    report = dict(protocol=probe.PROTOCOL, child={'returncode': 0}, training_admitted=False,
        launch_sha256=manifest['launch.json'], files=manifest, result=checked, decision=checked['decision'])
    probe.host.supervisor.write_json(root/'report.json', report)
    return root, sha256((root/'report.json').read_bytes()).hexdigest()


def samples(root):
    return [torch.load(root/f'output-2-{i}.pt', map_location='cpu', weights_only=True) for i in range(8)]


def test_completed_verifier_and_diagnosis_leave_all_inputs_unchanged(completed, monkeypatch):
    root, digest = completed
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    monkeypatch.setattr(probe, 'WarpStanceRuntime', lambda *a, **k: pytest.fail('resimulation'))
    original = probe.verify(root, report_sha256=digest)
    a = diagnosis.diagnose(root, digest); b = diagnosis.diagnose(root, digest)
    assert probe.canonical(a) == probe.canonical(b)
    assert a['source_decision'] == original and a['decision'] == 'diagnostic-only'
    assert not a['training_admitted'] and not a['graph_equivalence_established']
    assert not a['physical_motion_authorized'] and len(a['batches'][0]['pairs']) == 7
    assert [p['kind'] for p in a['batches'][0]['pairs']] == ['cross', 'cross', 'eager', 'cross', 'eager', 'eager', 'cross']
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('name', ['report.json', 'input-2.pt', 'output-2-7.pt', 'child.log'])
def test_any_manifest_tampering_fails_before_tensor_loading(completed, monkeypatch, name):
    root, digest = completed; path = root/name
    path.write_bytes(path.read_bytes()+b' ')
    monkeypatch.setattr(probe.torch, 'load', lambda *a, **k: pytest.fail('unverified tensor loading'))
    with pytest.raises(ValueError, match='hash'): diagnosis.diagnose(root, digest)


@pytest.mark.parametrize('damage', ['missing', 'extra', 'symlink'])
def test_inventory_and_symlink_fail_closed(completed, damage):
    root, digest = completed
    if damage == 'missing': (root/'output-2-4.pt').unlink()
    elif damage == 'extra': (root/'unexpected').write_text('unexpected')
    else:
        target = root.parent/'saved.pt'; (root/'output-2-4.pt').rename(target)
        (root/'output-2-4.pt').symlink_to(target)
    with pytest.raises(ValueError): diagnosis.diagnose(root, digest)


@pytest.mark.parametrize('damage', ['decision', 'returncode', 'protocol'])
def test_rehashed_report_still_cannot_change_original_semantics(completed, damage):
    root, _ = completed; path = root/'report.json'
    r = diagnosis.files.parse(path.read_bytes())
    if damage == 'decision': r['result']['training_admitted'] = True
    elif damage == 'returncode': r['child']['returncode'] = 1
    else: r['protocol'] = 'unsupported'
    path.write_text(probe.canonical(r)+'\n')
    with pytest.raises(ValueError): diagnosis.diagnose(root, sha256(path.read_bytes()).hexdigest())


def test_permutation_is_explanatory_only_and_preserves_raw_decision(cpu_batch):
    root, _ = cpu_batch; values = samples(root)
    for row in values[1]['constraints']:
        for key in row: row[key] = row[key].flip(0)
    before = probe.throughput.tree_hash(values)
    original = probe.comparison(values)
    pair = diagnosis.compare_samples(values, 2)['pairs'][0]
    assert not pair['raw_rows']['bitwise_equal'] and pair['dof_aligned_rows']['bitwise_equal']
    assert original['decision'] == 'candidate-difference-requires-diagnosis'
    assert probe.comparison(values) == original and probe.throughput.tree_hash(values) == before


def test_signed_zero_force_stays_a_bit_difference_after_alignment(cpu_batch):
    root, _ = cpu_batch; values = samples(root)
    values[0]['constraints'][0]['force'][0] = 0.
    values[1]['constraints'][0]['force'][0] = -0.
    pair = diagnosis.compare_samples(values, 2)['pairs'][0]['dof_aligned_rows']
    assert not pair['bitwise_equal']
    assert pair['fields']['force']['max_abs_difference'] == 0


@pytest.mark.parametrize('damage', ['duplicate', 'foreign_id', 'type', 'nan', 'rows'])
def test_ambiguous_or_nonfinite_row_domains_rejected(cpu_batch, damage):
    root, _ = cpu_batch; value = samples(root)[0]; row = value['constraints'][0]
    if damage == 'duplicate': row['id'][0] = row['id'][1]
    elif damage == 'foreign_id': row['id'][0] = 0
    elif damage == 'type': row['type'][0] = 0
    elif damage == 'nan': row['force'][0] = float('nan')
    else:
        value['solver']['nefc'][0] = 15
        for key in row: row[key] = torch.cat((row[key], row[key][:1]))
    with pytest.raises(ValueError): diagnosis.row_tables(value, 2)


def test_coordinate_units_and_provenance_are_not_mixed():
    a = {k: torch.zeros(2, 20) for k in probe.DYNAMICS}; b = deepcopy(a)
    b['qacc'][1, 0] = .25; b['qfrc_bias'][1, 6] = .5
    r = diagnosis.coordinate_differences(a, b)
    assert r['qacc'][0] == dict(dof=0, max_abs=.25, world=1, unit='m/s^2')
    assert r['qacc'][3]['unit'] == 'rad/s^2'
    assert r['qfrc_bias'][0]['unit'] == 'N'
    assert r['qfrc_bias'][6] == dict(dof=6, max_abs=.5, world=1, unit='N*m')
    b['qacc'][0, 0] = float('inf')
    with pytest.raises(ValueError, match='finite'): diagnosis.coordinate_differences(a, b)


@pytest.mark.parametrize('kind', ['inside', 'existing'])
def test_cli_refuses_output_inside_source_or_overwrite(tmp_path, monkeypatch, kind):
    source = tmp_path/'input'; source.mkdir()
    target = source/'new.json' if kind == 'inside' else tmp_path/'existing.json'
    if kind == 'existing': target.write_text('preserve')
    monkeypatch.setattr(sys, 'argv', ['diagnose', '--input', str(source), '--report-sha256', 'a'*64,
                                    '--output', str(target)])
    monkeypatch.setattr(diagnosis, 'diagnose', lambda *a: pytest.fail('unsafe output analyzed'))
    with pytest.raises(ValueError, match='fresh derived'): diagnosis.main()
    if kind == 'existing': assert target.read_text() == 'preserve'
