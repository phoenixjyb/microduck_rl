"""Source-only progressive recovery lesson proposal, not execution admission.

Separate from frozen D1: no existing checkpoint, pulse, loader, runtime or trace
allowlist is widened. These finite force tables require their own actual runtime,
parent installation, timing qualification and independent evaluator before use.
"""
import re

from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = 'football-b1d-progressive-recovery-lesson-plan-v1'
TRAINING_SEEDS = (653, 659)
HELD_OUT_SEEDS = (671, 677, 683)
COMMON_CHECKPOINTS = (64, 128, 192, 255)
STAGES = ('gentle', 'timing', 'dose')
# Literal float32 components, not unrecorded random normalization. Diagonals
# belong only to held-out evaluation, never the cardinal training distribution.
CARDINALS = (('+x', (1., 0., 0.)), ('-x', (-1., 0., 0.)),
             ('+y', (0., 1., 0.)), ('-y', (0., -1., 0.)))
DIAGONALS = (('++', (1.4142135381698608, 1.4142135381698608, 0.)),
             ('+-', (1.4142135381698608, -1.4142135381698608, 0.)),
             ('-+', (-1.4142135381698608, 1.4142135381698608, 0.)),
             ('--', (-1.4142135381698608, -1.4142135381698608, 0.)))


def _cell(name, force, onset, duration):
    return dict(id=name, force_world_newtons=list(force), torque_world_nm=[0., 0., 0.],
        onset_step=onset, duration_steps=duration, total_steps=2500,
        application='trunk-inertial-com', dt=.002, first_attempt_only=True, auto_reset=False)


def cells(stage, *, held_out=False):
    require(stage in STAGES and type(held_out) is bool, 'explicit planned recovery stage/split')
    result = [_cell('zero-wrench', (0., 0., 0.), 500, 10)]
    times = (500,) if stage == 'gentle' else (250, 500, 750)
    doses = ((2., 10),) if stage != 'dose' else ((2., 10), (2., 20), (4., 10))
    for onset in times:
        for magnitude, duration in doses:
            for direction, vector in CARDINALS:
                result.append(_cell(f'{direction}-{int(magnitude)}n-{duration}steps-t{onset}',
                    tuple(magnitude*v for v in vector), onset, duration))
    if held_out:
        for onset in (375, 625):
            for direction, vector in DIAGONALS:
                result.append(_cell(f'diagonal-{direction}-2n-10steps-t{onset}', vector, onset, 10))
    return result


def expected_wrench(cell, step, accepted, nbody, trunk_body_id):
    """Pure proposal math, never permission to install a force in a runtime."""
    allowed = cells('dose', held_out=True)
    require(type(cell) is dict and any(canonical(cell) == canonical(c) for c in allowed),
            'exact finite typed declared lesson cell')
    require(type(step) is int and 0 <= step <= 2500 and type(accepted) is bool,
            'bounded planned pulse clock/mask')
    require(type(nbody) is int and 1 < nbody <= 64 and type(trunk_body_id) is int
            and 0 < trunk_body_id < nbody, 'caller must bind actual ordered trunk body')
    xfrc = [[0.]*6 for _ in range(nbody)]; qfrc = [0.]*20
    if accepted and cell['onset_step'] <= step < cell['onset_step']+cell['duration_steps']:
        xfrc[trunk_body_id][:3] = list(cell['force_world_newtons'])
    return xfrc, qfrc


def plan(source):
    require(type(source) is str and re.fullmatch('[0-9a-f]{40}', source) is not None, 'exact proposed source')
    return dict(protocol=PROTOCOL, source=source, status='source-plan-only',
        parent=dict(source=baseline.TRAINING_SOURCE, training_seed=baseline.TRAINING_SEED,
            iteration=baseline.ITERATION, sha256=baseline.CHECKPOINT_SHA256,
            actor_dim=44, critic_dim=50, action_dim=10, restore='weights-only-fresh-optimizer'),
        training_seeds=list(TRAINING_SEEDS), held_out_seeds=list(HELD_OUT_SEEDS),
        worlds_proposed=64, updates_proposed_per_stage=256, common_checkpoints=list(COMMON_CHECKPOINTS),
        stages=[dict(name=stage, training_cells=cells(stage), held_out_cells=cells(stage, held_out=True),
            zero_attempt_fraction=.2, pulse_attempt_fraction=.8,
            pulse_sampling='uniform-explicit-cells-excluding-zero',
            held_out_worlds_per_cell=128, held_out_cycles=1,
            consecutive_passing_evaluations_required=2) for stage in STAGES],
        numerical_gates='unchanged-full-D1-stance-motor-first-attempt-gates',
        runtime_required='new-explicit-per-row-schedule-adapter-never-patch-nominal-step',
        loader_required='new-exact-D1-parent-actor-and-critic-loader-never-alias-model127',
        evaluator_required='new-student-checkpoint-and-pulse-trace-namespace',
        training_reset='only-after-transition-and-first-terminal-evidence-retained',
        observation_expanded=False, raw_perception=False, thermal_model_applied=False,
        execution_admitted=False, gpu_job_predeclared=False, learner_available=False,
        calibrated_ball_model=False, **baseline.FALSE_FLAGS)


def promotion(rows, *, stage):
    """Fail-closed numerical proposal screen; no executable skill admission.

    Require both training seeds and every held-out seed/cell at two adjacent
    common checkpoints. Missing, duplicated or partial cells cannot
    borrow a pass from another seed or checkpoint. This pure schema still needs
    an authenticated raw-evidence caller; JSON assertions alone prove nothing.
    """
    expected = {(evaluation, training_seed, seed, c['id']) for evaluation in (0, 1)
        for training_seed in TRAINING_SEEDS for seed in HELD_OUT_SEEDS for c in cells(stage, held_out=True)}
    require(type(rows) is list and len(rows) == len(expected), 'complete proposed held-out matrix')
    require(all(type(r) is dict and set(r) == {'evaluation', 'training_seed', 'seed', 'cell', 'checkpoint',
        'first_attempts', 'attempts_passed', 'complete_force_checks', 'raw_evidence_verified'} for r in rows),
        'exact proposed cell receipt fields')
    require(all(type(r['evaluation']) is int and type(r['training_seed']) is int
            and type(r['seed']) is int and type(r['cell']) is str for r in rows), 'typed evaluation and seed identities')
    keys = [(r['evaluation'], r['training_seed'], r['seed'], r['cell']) for r in rows]
    require(len(set(keys)) == len(keys) and set(keys) == expected, 'unique complete held-out cells')
    checkpoints = [next(r['checkpoint'] for r in rows if r['evaluation'] == evaluation) for evaluation in (0, 1)]
    pairs = list(zip(COMMON_CHECKPOINTS, COMMON_CHECKPOINTS[1:]))
    require(all(type(c) is int for c in checkpoints) and tuple(checkpoints) in pairs
            and all(type(r['checkpoint']) is int and r['checkpoint'] == checkpoints[r['evaluation']] for r in rows),
            'two adjacent common declared checkpoints before stage progression')
    require(all(type(r['first_attempts']) is int and r['first_attempts'] == 128
            and type(r['attempts_passed']) is int and 0 <= r['attempts_passed'] <= 128
            and type(r['complete_force_checks']) is bool and type(r['raw_evidence_verified']) is bool
            for r in rows), 'typed complete first-attempt and force receipts')
    passed = all(r['attempts_passed'] == 128 and r['complete_force_checks'] is True
                 and r['raw_evidence_verified'] is True for r in rows)
    return dict(protocol=PROTOCOL, stage=stage, common_checkpoints=checkpoints,
        numerical_proposal_passed=passed, rows_checked=len(rows),
        raw_provenance_enforced_by_this_pure_screen=False,
        execution_admitted=False, **baseline.FALSE_FLAGS)
