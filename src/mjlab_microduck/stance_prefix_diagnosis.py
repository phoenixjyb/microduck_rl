"""CPU-only, hash-bound diagnosis of retained throughput prefixes; no admission."""

import argparse
from hashlib import sha256
import io
import math
from pathlib import Path

import torch

from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck import foundation_command_campaign as files
from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_throughput_probe as probe

PROTOCOL = 'football-b1n-throughput-prefix-diagnosis-v1'
PAIRS = ((0, 1), (0, 2), (3, 4), (3, 5))
PREFIX_TICKS = 4
EXPECTED_FILES = {'launch.json', 'decision.json', 'child.log'} | {
    f'case-{i}.json' for i in range(6)
} | {f'case-{i}-tick-{tick}.pt' for i in range(6) for tick in range(PREFIX_TICKS)}


def compare_tree(left, right):
    """Keep bit, numerical, discrete and structural differences distinct.

    No reordering, rounding or tolerance. Wildcards aggregate repeated leaves
    for a compact report; first/max locations retain the original ordered path.
    Shape/type changes are structural mismatches, never broadcast comparisons.
    """
    result = dict(bitwise_equal=True, discrete_equal=True, structure_equal=True,
                  first_difference=None, fields={})

    def mark(path, kind, count, **details):
        result['bitwise_equal'] = False
        entry = dict(path=path, kind=kind, **details)
        if result['first_difference'] is None: result['first_difference'] = entry
        key = '/'.join('*' if type(k) is int else k for k in path)
        field = result['fields'].setdefault(key, dict(kind=kind, occurrences=0,
            differing_values=0, first=entry, max_abs_difference=None, maximum=None))
        field['occurrences'] += 1; field['differing_values'] += count
        if kind == 'float':
            if field['max_abs_difference'] is None or details['max_abs_difference'] > field['max_abs_difference']:
                field['max_abs_difference'] = details['max_abs_difference']
                field['maximum'] = entry
        elif kind == 'discrete': result['discrete_equal'] = False
        else:
            result['structure_equal'] = False
            result['discrete_equal'] = False

    def finite(value):
        if isinstance(value, torch.Tensor):
            require(value.device.type == 'cpu' and value.layout == torch.strided
                    and value.dtype in (torch.float32, torch.float64, torch.int32, torch.int64, torch.bool),
                    'owned CPU tensor dtype/layout')
            if value.is_floating_point(): require(torch.isfinite(value).all(), 'nonfinite diagnostic tensor')
        elif type(value) is dict:
            require(all(type(k) is str for k in value), 'string tree keys')
            for v in value.values(): finite(v)
        elif type(value) in (list, tuple):
            for v in value: finite(v)
        else:
            require(type(value) in (float, int, bool, str, type(None)), 'supported diagnostic leaf')
            if type(value) is float: require(math.isfinite(value), 'nonfinite diagnostic scalar')

    def visit(a, b, path):
        if type(a) is not type(b):
            mark(path, 'structure', 1, left_type=type(a).__name__, right_type=type(b).__name__); return
        if isinstance(a, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                mark(path, 'structure', 1, left_layout=[str(a.dtype), list(a.shape)],
                     right_layout=[str(b.dtype), list(b.shape)]); return
            aa, bb = a.contiguous(), b.contiguous()
            # Compare each element's bytes, including the sign of zero.
            bits = (aa.reshape(-1).view(torch.uint8).reshape(-1, aa.element_size()) !=
                    bb.reshape(-1).view(torch.uint8).reshape(-1, bb.element_size())).any(1)
            if not bits.any(): return
            flat = int(bits.nonzero()[0, 0])
            index = list(torch.unravel_index(torch.tensor(flat), a.shape)) if a.ndim else []
            details = dict(first_index=[int(i) for i in index],
                           left=aa.reshape(-1)[flat].item(), right=bb.reshape(-1)[flat].item())
            if a.is_floating_point():
                diff = (aa.double()-bb.double()).abs()
                maximum = int(diff.reshape(-1).argmax())
                details.update(max_abs_difference=float(diff.max()),
                    max_index=[int(i) for i in torch.unravel_index(torch.tensor(maximum), a.shape)] if a.ndim else [])
            mark(path, 'float' if a.is_floating_point() else 'discrete', int(bits.sum()), **details)
        elif type(a) is dict:
            if a.keys() != b.keys():
                mark(path, 'structure', 1, left_keys=sorted(a), right_keys=sorted(b))
            for key in sorted(a.keys() & b.keys()): visit(a[key], b[key], path+[key])
        elif type(a) in (tuple, list):
            if len(a) != len(b):
                mark(path, 'structure', 1, left_length=len(a), right_length=len(b))
            for i, (v, w) in enumerate(zip(a, b)): visit(v, w, path+[i])
        elif type(a) is float:
            if a.hex() != b.hex():
                mark(path, 'float', 1, left=a, right=b, max_abs_difference=abs(a-b))
        elif a != b: mark(path, 'discrete', 1, left=a, right=b)

    finite(left); finite(right); visit(left, right, [])
    canonical(result)
    return result


def validate_prefix(data, worlds):
    require(set(data) == {'result', 'control', 'integration'}, 'exact captured prefix')
    result = data['result']
    require(set(result) == {'reward', 'terminated', 'timed_out', 'episode_steps', 'executed_steps',
        'live', 'term_sums', 'observation', 'boundaries', 'terminal_records', 'optimizer_launched'}
        and result['optimizer_launched'] is False, 'exact non-optimizer result')
    for key in ('terminated', 'timed_out', 'live'):
        trace.tensor(result[key], (worlds,), torch.bool, key)
    for key in ('episode_steps', 'executed_steps'):
        trace.tensor(result[key], (worlds,), torch.int64, key)
    trace.tensor(result['reward'], (worlds,), torch.float32, 'reward')
    require(type(result['boundaries']) is list and 1 <= len(result['boundaries']) <= 11,
            'bounded physical boundary prefix')
    for frame in result['boundaries']: trace.validate_frame(frame, worlds)
    require(type(result['terminal_records']) is list and len(result['terminal_records']) == worlds,
            'all terminal slots retained')
    require(set(data['integration']) == {'qpos', 'qvel', 'time', 'qacc_warmstart'}, 'integration fields')
    for key, width in (('qpos', 21), ('qvel', 20), ('qacc_warmstart', 20), ('time', None)):
        trace.tensor(data['integration'][key], (worlds, width) if width else (worlds,), torch.float32, key)
    require(set(data['control']) == {'correction', 'target', 'queue', 'previous', 'voltage',
        'kp', 'friction', 'damping', 'ctrl'}, 'control fields')
    # Check all remaining leaves too, including terminal contact lists.
    compare_tree(data, data)


def load_prefix(root, case, index, tick):
    name = f'case-{index}-tick-{tick}.pt'
    descriptor = case['diagnostic_prefix'][tick]
    raw = files.file_bytes(root/name, limit=16*1024*1024)
    require(descriptor == dict(file=name, sha256=sha256(raw).hexdigest()), 'raw prefix descriptor hash')
    data = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    require(probe.tree_hash(data) == case['ticks'][tick]['state_sha256'], 'captured value hash')
    validate_prefix(data, case['worlds'])
    require(data['result']['executed_steps'].tolist() == case['ticks'][tick]['executed_steps'],
            'retained executed-step receipt')
    return data


def compare_prefixes(left, right):
    require(len(left) == len(right) == PREFIX_TICKS, 'four complete diagnostic ticks')
    initial = compare_tree(left[0]['result']['boundaries'][0], right[0]['result']['boundaries'][0])
    ticks = []; first_boundary = None
    for tick, (a, b) in enumerate(zip(left, right)):
        frames = []
        for index, (fa, fb) in enumerate(zip(a['result']['boundaries'], b['result']['boundaries'])):
            compared = compare_tree(fa, fb)
            if first_boundary is None and not compared['bitwise_equal']:
                first_boundary = dict(tick=tick, boundary=index, comparison=compared)
            frames.append(compared)
        counts = [len(v['result']['boundaries']) for v in (a, b)]
        if first_boundary is None and counts[0] != counts[1]:
            first_boundary = dict(tick=tick, boundary=min(counts),
                comparison=compare_tree({'boundary_count': counts[0]}, {'boundary_count': counts[1]}))
        ticks.append(dict(tick=tick, complete_tree=compare_tree(a, b), boundaries=frames))
    return dict(initial_boundary=initial, first_divergent_boundary=first_boundary, ticks=ticks,
                full_forward_inputs_retained=False, solver_phase_attribution_established=False)


def diagnose(root, report_sha256):
    require(not torch.cuda.is_initialized(), 'CPU-only prefix diagnosis')
    root = files.native._plain_path(Path(root)); files.hex_id(report_sha256, 64)
    raw = files.file_bytes(root/'report.json', limit=2*1024*1024)
    require(sha256(raw).hexdigest() == report_sha256, 'independent source report hash')
    report = files.parse(raw)
    require(set(report['files']) == EXPECTED_FILES and {p.name for p in root.iterdir()} == EXPECTED_FILES | {'report.json'},
            'exact retained input inventory')
    records = {}
    for name in sorted(EXPECTED_FILES):
        raw = files.file_bytes(root/name, limit=16*1024*1024)
        require(sha256(raw).hexdigest() == report['files'][name],
                'retained input file hash: '+name)
        if name.endswith('.json'): records[name] = files.parse(raw)
    cases = [records[f'case-{i}.json'] for i in range(6)]
    decision = probe.decide(cases)
    require(report['protocol'] == probe.PROTOCOL and report['training_admitted'] is False
            and report['result'] == decision and report['decision'] == decision['decision']
            and records['decision.json'] == decision, 'original recomputed decision')
    require(report['child']['returncode'] == 0, 'completed original probe')
    launch = records['launch.json']
    require(report['launch_sha256'] == report['files']['launch.json'] and launch['protocol'] == probe.PROTOCOL,
            'launch identity')
    files.hex_id(launch['source'], 40)
    require(all(len(c['diagnostic_prefix']) == PREFIX_TICKS for c in cases), 'all raw prefixes retained')
    for i, case in enumerate(cases):
        for tick, saved in enumerate(case['diagnostic_prefix']):
            name = f'case-{i}-tick-{tick}.pt'
            require(saved == dict(file=name, sha256=report['files'][name]), 'report-bound prefix descriptor')
    pairs = []
    for i, j in PAIRS:
        a = [load_prefix(root, cases[i], i, t) for t in range(PREFIX_TICKS)]
        b = [load_prefix(root, cases[j], j, t) for t in range(PREFIX_TICKS)]
        pairs.append(dict(cases=[i, j], worlds=cases[i]['worlds'], **compare_prefixes(a, b)))
    require(not torch.cuda.is_initialized(), 'diagnosis must not initialize CUDA')
    return dict(protocol=PROTOCOL, source_report_sha256=report_sha256, source_commit=launch['source'],
        source_decision=decision, input_files=report['files'], pairs=pairs,
        scope='four retained ticks only; no resimulation; no numerical acceptance',
        decision='diagnostic-only', training_admitted=False, graph_equivalence_established=False,
        physical_motion_authorized=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--report-sha256', required=True)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    source = files.native._plain_path(args.input)
    target = files.native._plain_path(args.output)
    require(not target.exists() and source != target and source not in target.parents,
            'fresh derived output outside immutable input')
    result = diagnose(source, args.report_sha256)
    files.write_json(target, result)
    print(canonical(dict(output=str(target), sha256=sha256(files.file_bytes(target)).hexdigest(),
                         decision=result['decision'], graph_equivalence_established=False)))


if __name__ == '__main__': main()
