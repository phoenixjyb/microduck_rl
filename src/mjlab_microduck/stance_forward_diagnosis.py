"""CPU-only secondary friction-row explanation; never rescoring or admission."""
import argparse
from hashlib import sha256
import io
from pathlib import Path

import torch

from mjlab_microduck import foundation_command_campaign as files
from mjlab_microduck import stance_forward_probe as probe
from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.stance_prefix_diagnosis import compare_tree

PROTOCOL = 'football-b1n-forward-row-diagnosis-v1'


def row_tables(value, n):
    probe.validate_output(value, n)
    require((value['solver']['nefc'] == 14).all(), 'friction-only row domain')
    raw = {key: torch.stack([r[key] for r in value['constraints']]) for key in probe.CONSTRAINTS}
    require((raw['type'] == 1).all(), 'only DOF friction rows can be aligned')
    ids, order = raw['id'].sort(dim=1)
    require(torch.equal(ids, torch.arange(6, 20, dtype=torch.int32).expand(n, -1)),
            'unique complete motor DOF identities 6 through 19')
    # Advanced indexing creates independent copies; the original ordered tables
    # and original simulator decisions are never modified or canonicalized.
    worlds = torch.arange(n).unsqueeze(1)
    aligned = {key: value[worlds, order].clone() for key, value in raw.items()}
    return raw, aligned


def coordinate_differences(left, right):
    """Per-coordinate maxima; never attach one unit to a mixed free-joint vector."""
    result = {}
    for key in probe.DYNAMICS:
        a, b = left[key], right[key]
        require(a.shape == b.shape and a.ndim == 2 and a.shape[1] == 20
                and a.dtype == b.dtype == torch.float32 and a.device.type == b.device.type == 'cpu'
                and torch.isfinite(a).all() and torch.isfinite(b).all(), 'finite owned generalized vector')
        delta = (a.double()-b.double()).abs()
        maxima, worlds = delta.max(dim=0)
        acceleration = key in ('qacc', 'qacc_smooth')
        result[key] = [dict(dof=i, max_abs=float(maxima[i]), world=int(worlds[i]),
            unit=('m/s^2' if i < 3 else 'rad/s^2') if acceleration else ('N' if i < 3 else 'N*m'))
            for i in range(20)]
    return result


def compare_samples(samples, n):
    require(len(samples) == 8, 'eight retained forward samples')
    tables = [row_tables(value, n) for value in samples]
    pairs = []
    for i in range(1, 8):
        pairs.append(dict(calls=[0, i], kind='cross' if probe.ORDER[i] else 'eager',
            raw_rows=compare_tree(tables[0][0], tables[i][0]),
            dof_aligned_rows=compare_tree(tables[0][1], tables[i][1]),
            dynamics=coordinate_differences(samples[0]['dynamics'], samples[i]['dynamics'])))
    return dict(worlds=n, reference_call=0, aligned_row_ids=list(range(6, 20)), pairs=pairs,
                comparison_scope='calls 1 through 7 versus call 0; not independent trials')


def diagnose(root, report_sha256):
    require(not torch.cuda.is_initialized(), 'CPU-only forward diagnosis')
    root = files.native._plain_path(Path(root))
    original = probe.verify(root, report_sha256=report_sha256)

    def read(name, expected, *, empty=False):
        raw = files.file_bytes(root/name, limit=probe.SNAPSHOT_LIMIT, allow_empty=empty)
        require(sha256(raw).hexdigest() == expected, 'diagnosis input file hash: '+name)
        return raw

    report = files.parse(read('report.json', report_sha256))
    launch = files.parse(read('launch.json', report['files']['launch.json']))
    batches = []
    for n in probe.WORLDS:
        samples = []
        for i in range(8):
            name = f'output-{n}-{i}.pt'
            samples.append(torch.load(io.BytesIO(read(name, report['files'][name])),
                                      map_location='cpu', weights_only=True))
        batches.append(compare_samples(samples, n))
    # End-to-end immutable-input check, including files not used for alignment.
    for name, expected in report['files'].items(): read(name, expected, empty=name == 'child.log')
    read('report.json', report_sha256)
    require(not torch.cuda.is_initialized(), 'diagnosis must not initialize CUDA')
    return dict(protocol=PROTOCOL, source_report_sha256=report_sha256,
        source_commit=launch['source'], input_files=report['files'], source_decision=original,
        batches=batches, decision='diagnostic-only', graph_equivalence_established=False,
        training_admitted=False, physical_motion_authorized=False,
        scope='prepared friction-only forward snapshots; no integration, tolerance or trajectory acceptance')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--report-sha256', required=True)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    source = files.native._plain_path(args.input); target = files.native._plain_path(args.output)
    require(not target.exists() and source != target and source not in target.parents,
            'fresh derived output outside immutable input')
    value = diagnose(source, args.report_sha256)
    files.write_json(target, value)
    print(canonical(dict(output=str(target), sha256=sha256(files.file_bytes(target)).hexdigest(),
                         decision=value['decision'])))


if __name__ == '__main__': main()
