"""Finite per-row progressive pulse declaration, separate from frozen D1.

Declaration/math only: a matching force array does not prove physical delivery,
first-attempt evidence, learner integration, job admission or skill acceptance.
"""
from copy import deepcopy
from hashlib import sha256
import re

from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_lesson_plan as lesson
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = 'football-b1d-progressive-row-schedule-v1'
MAX_ROWS = 64
SPLITS = ('training', 'held-out')


def declaration(source, stage, split, cell_ids):
    require(type(source) is str and re.fullmatch('[0-9a-f]{40}', source) is not None,
            'exact schedule source revision')
    require(type(stage) is str and stage in lesson.STAGES and type(split) is str and split in SPLITS,
            'explicit finite schedule stage and split')
    require(type(cell_ids) in (list, tuple) and 1 <= len(cell_ids) <= MAX_ROWS
            and all(type(c) is str for c in cell_ids), 'one to 64 explicit schedule rows')
    catalog = {c['id']: c for c in lesson.cells(stage, held_out=split == 'held-out')}
    require(all(c in catalog for c in cell_ids), 'only declared split cells may bind rows')
    # Repeated rows are intentional for future vectorized attempts. This does
    # not claim distinct randomized worlds or mutate an episode's cell later.
    return dict(protocol=PROTOCOL, source=source, stage=stage, split=split,
        worlds=len(cell_ids), cell_ids=list(cell_ids),
        row_cells=[deepcopy(catalog[c]) for c in cell_ids],
        parent_checkpoint_sha256=baseline.CHECKPOINT_SHA256,
        actor_dim=44, critic_dim=50, action_dim=10, dt=.002, total_steps=2500,
        auto_reset=False, first_attempt_only=True, observation_expanded=False,
        raw_perception=False, execution_admitted=False, **baseline.FALSE_FLAGS)


def checked(value):
    require(type(value) is dict and all(k in value for k in ('source', 'stage', 'split', 'cell_ids')),
            'complete per-row schedule declaration')
    expected = declaration(value['source'], value['stage'], value['split'], value['cell_ids'])
    require(canonical(value) == canonical(expected), 'unchanged exact typed schedule declaration')
    return deepcopy(expected)


def binding_sha256(value):
    return sha256(canonical(checked(value)).encode()).hexdigest()


def expected_wrenches(value, steps, accepted, nbody, trunk_body_id):
    """Full zero-initialized matrices for explicit accepted per-row clocks."""
    value = checked(value); n = value['worlds']
    require(type(steps) is list and len(steps) == n
            and all(type(s) is int and 0 <= s <= value['total_steps'] for s in steps),
            'exact per-row bounded integer schedule clocks')
    require(type(accepted) is list and len(accepted) == n
            and all(type(a) is bool for a in accepted), 'exact per-row accepted schedule mask')
    require(type(nbody) is int and 1 < nbody <= 64 and type(trunk_body_id) is int
            and 0 < trunk_body_id < nbody, 'actual ordered schedule body binding')
    xfrc, qfrc = [], []
    for cell, step, live in zip(value['row_cells'], steps, accepted):
        x = [[0.]*6 for _ in range(nbody)]; q = [0.]*20
        if live and cell['onset_step'] <= step < cell['onset_step']+cell['duration_steps']:
            x[trunk_body_id][:3] = list(cell['force_world_newtons'])
        xfrc.append(x); qfrc.append(q)
    return xfrc, qfrc


def window_mask(value, steps, accepted):
    """Typed phase-capture window, including explicitly zero-force controls."""
    value = checked(value)
    # Reuse the complete typed clock/mask checks, not any installed force state.
    expected_wrenches(value, steps, accepted, 2, 1)
    return [live and c['onset_step'] <= step < c['onset_step']+c['duration_steps']
            for c, step, live in zip(value['row_cells'], steps, accepted)]


def validate_wrenches(xfrc, qfrc, value, steps, accepted, nbody, trunk_body_id):
    value = checked(value)
    expected_x, expected_q = expected_wrenches(value, steps, accepted, nbody, trunk_body_id)
    baseline.force._numeric_tree(xfrc, (value['worlds'], nbody, 6), 'scheduled xfrc_applied')
    baseline.force._numeric_tree(qfrc, (value['worlds'], 20), 'scheduled qfrc_applied')
    require(xfrc == expected_x and qfrc == expected_q, 'exact complete scheduled wrench arrays')
    return True
