"""Same-input finite-predicate checks during short, unchanged integration cases.

The audited eager solver is not trajectory-repeatable. Do not compare two
separate solves or introduce a new tolerance. Shadow both finite predicates on
the identical current solved tensors, and check exact bit preservation instead.
This temporary wrapper is confined to each diagnostic-owned runtime object.
"""

from hashlib import sha256

import torch

from mjlab_microduck import stance_cuda_probe as host
from mjlab_microduck import stance_solved_field_check as checker
from mjlab_microduck import stance_finite_check_probe as probe
from mjlab_microduck.first_attempt_smoke import canonical, require


PROTOCOL = 'football-b1n-packed-same-input-integration-v1'


def cases(device):
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime

    records = []
    faults = []
    def factory(nworld, *, device):
        env = WarpStanceRuntime(nworld, device=device, solved_field_check='packed')
        require(env.forward_graph is None, 'unchanged eager integration')
        record = dict(worlds=nworld, checked_forwards=0)
        records.append(record)
        selected = env._check_solved_fields
        def observed():
            values = {name: env._view(name) for name in checker.FIELDS}
            before = {name: value.contiguous().view(torch.uint8).clone() for name, value in values.items()}
            checker.legacy_check(values)
            selected()  # Actual selected runtime method, including fresh views/mapping.
            require(all(torch.equal(before[name], value.contiguous().view(torch.uint8))
                        for name, value in values.items()), 'same solved-input bits preserved')
            require(env.forward_graph is None and env.solved_field_check == 'packed',
                    'unchanged selected eager runtime')
            record['checked_forwards'] += 1
        env._check_solved_fields = observed
        if not faults:
            values = {name: env._view(name) for name in checker.FIELDS}
            before = {name: value.contiguous().view(torch.uint8).clone() for name, value in values.items()}
            faults.extend(probe.errors(checker, values))  # Copies only, never simulator views.
            require(all(torch.equal(before[name], value.contiguous().view(torch.uint8))
                        for name, value in values.items()), 'fault fixtures leave solved views unchanged')
        env.reset(torch.ones(nworld, dtype=torch.bool, device=env.device))
        return env

    payload = host.cases(device, runtime_factory=factory)
    receipt = dict(protocol=PROTOCOL, device=device, solved_field_check='packed',
        checker_sha256=host.digest(checker.__file__), fields=list(checker.FIELDS),
        runtimes=records, faults=faults, faults_sha256=sha256(canonical(faults).encode()).hexdigest(),
        same_input_predicates_agree=True, input_bits_unchanged=True,
        separate_trajectory_equivalence_claimed=False, forward_graph=False,
        policy_inferences=0, optimizer_updates=0, timing_qualified=False,
        learned_stance=False, football_balance=False, physical_motion_authorized=False)
    validate(receipt, device)
    return payload, receipt


def validate(receipt, device):
    require(receipt['protocol'] == PROTOCOL and receipt['device'] == device
            and receipt['solved_field_check'] == 'packed', 'bound same-input predicate integration')
    require(receipt['checker_sha256'] == host.digest(checker.__file__)
            and receipt['fields'] == list(checker.FIELDS), 'exact reviewed checker and fields')
    require(canonical(receipt['runtimes']) == canonical([dict(worlds=2, checked_forwards=41),
                                                       dict(worlds=2, checked_forwards=42)]),
            'all normal, terminal-isolation and reset forwards checked')
    expected = probe.errors(checker, {name: torch.zeros(2, 1) for name in checker.FIELDS})
    require(canonical(receipt['faults']) == canonical(expected)
            and receipt['faults_sha256'] == sha256(canonical(expected).encode()).hexdigest(),
            'all identical copy-only first-field faults')
    require(receipt['same_input_predicates_agree'] is True and receipt['input_bits_unchanged'] is True
            and all(receipt[key] is False for key in ('separate_trajectory_equivalence_claimed',
                'forward_graph', 'timing_qualified', 'learned_stance', 'football_balance',
                'physical_motion_authorized'))
            and all(type(receipt[key]) is int and receipt[key] == 0
                    for key in ('policy_inferences', 'optimizer_updates')),
            'non-admitting finite predicate evidence')
