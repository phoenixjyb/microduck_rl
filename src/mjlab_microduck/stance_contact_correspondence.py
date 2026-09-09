"""CPU structural contact-candidate audit; not physical matching or admission."""
from collections import defaultdict

import torch

from mjlab_microduck.first_attempt_smoke import require
from mjlab_microduck.stance_contact_evidence import validate_contacts
from mjlab_microduck.stance_prefix_diagnosis import compare_tree
from mjlab_microduck.stance_throughput_probe import tree_hash

PROTOCOL = 'football-b1n-contact-candidate-audit-v1'
FIELDS = {'worldid', 'geom', 'dim', 'dist', 'pos', 'frame', 'friction', 'efc_address', 'force'}


def groups(table, nworld):
    require(type(nworld) is int and 0 < nworld <= 65536, 'bounded positive world count')
    require(type(table) is dict and set(table) == FIELDS, 'complete contact table')
    require(all(isinstance(v, torch.Tensor) and v.device.type == 'cpu' and v.layout == torch.strided
                for v in table.values()), 'owned CPU contact tensors')
    require(table['worldid'].ndim == 1 and len(table['worldid']) <= 1048576, 'bounded contact row count')
    validate_contacts(table, nworld, torch.device('cpu'))
    require(torch.isin(table['dim'], torch.tensor([1, 3, 4, 6])).all(), 'supported contact dimension')
    require((table['efc_address'] >= -1).all(), 'valid excluded-contact address sentinel')
    require((table['geom'][:, 0] != table['geom'][:, 1]).all(), 'distinct contact geometries')
    compare_tree(table, table)  # Reject unsupported dtypes as well as nonfinite leaves.
    result = defaultdict(list)
    for row in range(len(table['worldid'])):
        # Do not sort the geom pair: reversing it changes frame/force semantics.
        # Allocation addresses are not identities; only active/excluded status
        # is keyed. Multiple points for one geom pair remain explicitly ambiguous.
        key = (int(table['worldid'][row]), int(table['geom'][row, 0]),
               int(table['geom'][row, 1]), int(table['dim'][row]),
               bool(table['efc_address'][row, 0] >= 0))
        result[key].append(row)
    return result


def audit_contacts(left, right, nworld):
    """Explain unique structural candidates without guessing spatial tolerances.

    Callers loading files must independently verify their manifests before tensor
    deserialization. This in-memory helper hashes the actual ordered values; it
    does not authenticate a source file, replay state or physical contact identity.
    """
    a = groups(left, nworld); b = groups(right, nworld)
    identities = dict(left=tree_hash(left), right=tree_hash(right))
    candidates = []; issues = []
    for key in sorted(a.keys() | b.keys()):
        ar, br = a.get(key, []), b.get(key, [])
        entry = dict(key=list(key), left_rows=list(ar), right_rows=list(br))
        if len(ar) == len(br) == 1:
            candidates.append(entry)
        else:
            reasons = []
            for side, rows in (('left', ar), ('right', br)):
                if not rows: reasons.append('missing-'+side)
                elif len(rows) > 1: reasons.append('ambiguous-'+side)
            issues.append(dict(**entry, reasons=reasons))
    compared = None
    if issues:
        status = 'unresolved-correspondence'
    elif not candidates:
        status = 'no-contact-evidence'
    else:
        status = 'unique-key-candidates'
        ai = torch.tensor([p['left_rows'][0] for p in candidates], dtype=torch.long)
        bi = torch.tensor([p['right_rows'][0] for p in candidates], dtype=torch.long)
        # Compare copies and retain raw contact frames/forces without transforming
        # them. Even an exact aligned match is not numerical/physical acceptance.
        compared = compare_tree({k: v[ai].clone() for k, v in left.items()},
                                {k: v[bi].clone() for k, v in right.items()})
    require(identities == dict(left=tree_hash(left), right=tree_hash(right)), 'unchanged ordered contact inputs')
    return dict(protocol=PROTOCOL, decision='diagnostic-only', status=status,
        key_fields=['world', 'geom0', 'geom1', 'dimension', 'included'],
        input_value_sha256=identities, candidates=candidates, issues=issues, comparison=compared,
        world_contact_counts=dict(left=torch.bincount(left['worldid'].long(), minlength=nworld).tolist(),
                                  right=torch.bincount(right['worldid'].long(), minlength=nworld).tolist()),
        physical_contact_identity_established=False, graph_equivalence_established=False,
        training_admitted=False, physical_motion_authorized=False)
