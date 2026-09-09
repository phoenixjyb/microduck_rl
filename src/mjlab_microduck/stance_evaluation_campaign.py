"""Exact held-out matrix and retained-bundle replay; no GPU allocation or launch."""
from copy import deepcopy
from hashlib import sha256

from mjlab_microduck import foundation_command_campaign as files
from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_evaluation_bundle as bundle
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = 'football-b1n-heldout-matrix-v1'
INDEX_PROTOCOL = 'football-b1n-heldout-index-v1'
WORLDS, REQUIRED_PASSES = 128, 122


def plan_bytes(plan):
    return (canonical(plan)+'\n').encode()


def make_plan(source, runtime_sha256, checkpoints):
    """Metadata-only declaration; preflight separately checks all actual bytes."""
    files.hex_id(source, 40); files.hex_id(runtime_sha256, 64)
    require(type(checkpoints) is list and len(checkpoints) == 4, 'four declared checkpoints')
    common = None; cases = []
    for iteration, entry in zip(trace.CHECKPOINTS, checkpoints):
        require(set(entry) == {'sha256', 'identity'}, 'exact checkpoint entry')
        files.hex_id(entry['sha256'], 64)
        identity = entry['identity']; checkpoint.validate_identity(identity, evaluation=True)
        require(identity['iteration'] == iteration, 'ordered common checkpoints')
        training = {k: v for k, v in identity.items() if k != 'iteration'}
        if common is None: common = training
        require(training == common, 'one matched pilot training run')
        for seed in trace.SEEDS:
            binding = dict(protocol=trace.PROTOCOL, source=source, runtime_sha256=runtime_sha256,
                checkpoint_sha256=entry['sha256'], checkpoint_iteration=iteration,
                evaluation_seed=seed, worlds=WORLDS, capture_device='cuda:0')
            launch = bundle.launch_bytes(binding, identity)
            binding['launch_sha256'] = sha256(launch).hexdigest()
            cases.append(dict(name=f'iteration-{iteration:04d}-seed-{seed}', binding=binding,
                              checkpoint_identity=deepcopy(identity)))
    return dict(protocol=PROTOCOL, source=source, runtime_sha256=runtime_sha256,
        checkpoints=deepcopy(checkpoints), cases=cases, actor_device='cpu', physics_device='cuda:0',
        policy_ticks=250, physics_steps=2500, worlds_per_case=WORLDS,
        cases_required=12, attempts_required=1536, final_attempts_required=384,
        final_required_passes_per_seed=REQUIRED_PASSES, final_iteration=511,
        reset_policy='one-nominal-first-attempt-no-auto-reset',
        seed_interpretation='nominal-deterministic-repeatability-not-randomized-generalization',
        gpu_launch_authorized=False, checkpoint_admitted=False, physical_motion_authorized=False)


def validate_plan(plan):
    expected = make_plan(plan['source'], plan['runtime_sha256'], plan['checkpoints'])
    require(canonical(plan) == canonical(expected), 'exact predeclared stance matrix')


def preflight(plan, runtime_raw, checkpoint_bytes):
    """CPU-only exact-byte/plant/model restore checks, before any case allocation."""
    validate_plan(plan)
    require(type(runtime_raw) is bytes and sha256(runtime_raw).hexdigest() == plan['runtime_sha256'], 'matrix runtime byte identity')
    plant.checked_runtime(files.parse(runtime_raw), plan['source'])
    require(set(checkpoint_bytes) == set(trace.CHECKPOINTS), 'all checkpoint bytes before evaluation')
    for iteration, entry in zip(trace.CHECKPOINTS, plan['checkpoints']):
        checkpoint.load_evaluation(checkpoint_bytes[iteration], entry['sha256'], entry['identity'])
    return dict(input_bytes_checked=True, matrix_cases=12, gpu_launch_authorized=False)


def _summarize(plan, scores):
    """Internal: scores must come from full bundle verification, not saved claims."""
    require(set(scores) == {c['name'] for c in plan['cases']}, 'complete unique matrix coverage')
    rows = []; final = []
    for case in plan['cases']:
        score = scores[case['name']]
        require(score['binding'] == case['binding'], 'case score binding')
        attempts = score['attempts']
        require(type(attempts) is list and len(attempts) == WORLDS
                and [a['world_id'] for a in attempts] == list(range(WORLDS)), 'exact ordered attempt inventory')
        require(all(a['complete_first_attempt'] is True for a in attempts)
                and type(score['complete_attempts']) is int and score['complete_attempts'] == WORLDS,
                'every attempt must be complete, not a truncated prefix')
        require(all(type(a['candidate_pass']) is bool for a in attempts), 'boolean attempt decisions')
        passes = sum(a['candidate_pass'] for a in attempts)
        require(type(score['numerical_passes']) is int and score['numerical_passes'] == passes, 'recomputed attempt counts')
        for flag in ('provenance_validated', 'checkpoint_admitted', 'learned_stance_accepted', 'physical_motion_authorized'):
            require(score[flag] is False, 'no implicit provenance or skill admission')
        for flag in ('trajectory_continuity_validated', 'strict_checkpoint_checked', 'deterministic_actor_replay_checked',
            'compiled_plant_checked', 'nominal_reset_checked', 'kinematic_observations_checked',
            'terminal_contact_summary_checked', 'action_slew_checked', 'delayed_motor_targets_checked',
            'motor_commit_masks_checked', 'voltage_history_checked'):
            require(score[flag] is True, 'required bundle replay check: '+flag)
        require(score['terminal_contact_records_checked'] == WORLDS, 'all first terminal contacts checked')
        row = dict(case=case['name'], iteration=case['binding']['checkpoint_iteration'],
                   seed=case['binding']['evaluation_seed'], complete_attempts=WORLDS, numerical_passes=passes,
                   meets_95_percent=passes >= REQUIRED_PASSES)
        rows.append(row)
        if row['iteration'] == 511: final.append(row['meets_95_percent'])
    passed = len(final) == 3 and all(final)
    return dict(protocol=PROTOCOL, plan_sha256=sha256(plan_bytes(plan)).hexdigest(), rows=rows,
        total_complete_attempts=1536, final_complete_attempts=384,
        final_numerical_gate_passed=passed, decision='nominal-candidate-numerical-only' if passed else 'rejected',
        provenance_validated=False, checkpoint_admitted=False, learned_stance_accepted=False,
        football_balance_accepted=False, physical_motion_authorized=False)


def _verified_scores(directory, plan, hashes):
    require(set(hashes) == {case['name'] for case in plan['cases']}, 'exact matrix bundle hashes')
    scores = {}
    for case in plan['cases']:  # Sequential, bounded CPU replay; never parallel GPU work.
        digest = hashes[case['name']]; files.hex_id(digest, 64)
        scores[case['name']] = bundle.verify_bundle(directory/case['name'], digest,
            binding=case['binding'], checkpoint_identity=case['checkpoint_identity'])
    return scores


def retain_index(directory, plan, independently_retained_bundle_hashes):
    """Verify every bundle before exclusively publishing the matrix index last."""
    validate_plan(plan); directory = files.native._plain_path(directory)
    require({p.name for p in directory.iterdir()} == {c['name'] for c in plan['cases']}, 'exact matrix directory inventory')
    summary = _summarize(plan, _verified_scores(directory, plan, independently_retained_bundle_hashes))
    index = dict(protocol=INDEX_PROTOCOL, plan_sha256=sha256(plan_bytes(plan)).hexdigest(),
        bundle_hashes=deepcopy(independently_retained_bundle_hashes), summary=summary)
    files.write_json(directory/'matrix.json', index)
    return sha256(files.file_bytes(directory/'matrix.json')).hexdigest(), summary


def verify_index(directory, expected_index_sha256, plan):
    validate_plan(plan); directory = files.native._plain_path(directory)
    raw = files.file_bytes(directory/'matrix.json', limit=1024*1024)
    require(sha256(raw).hexdigest() == expected_index_sha256, 'independent matrix hash')
    index = files.parse(raw)
    require(set(index) == {'protocol', 'plan_sha256', 'bundle_hashes', 'summary'}
            and index['protocol'] == INDEX_PROTOCOL and index['plan_sha256'] == sha256(plan_bytes(plan)).hexdigest(), 'matrix plan binding')
    require({p.name for p in directory.iterdir()} == {c['name'] for c in plan['cases']} | {'matrix.json'}, 'exact retained matrix inventory')
    summary = _summarize(plan, _verified_scores(directory, plan, index['bundle_hashes']))
    require(canonical(index['summary']) == canonical(summary), 'matrix decision replay mismatch')
    return summary
