from copy import deepcopy

import pytest

from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_lean_evaluation as ev

from test_stance_lean_evaluation import evaluate_plan, pinned_probe, synthetic_scores


def test_original_wrapper_matches_pure_case_helper(monkeypatch, tmp_path):
    pinned_probe(monkeypatch, tmp_path)
    launch = evaluate_plan()
    scores = synthetic_scores(launch)

    assert ev.summarize(launch, scores) == ev.summarize_cases(
        launch['cases'], scores, ev.LESSON)


def test_helper_preserves_any_checkpoint_as_the_distinct_winner(monkeypatch, tmp_path):
    pinned_probe(monkeypatch, tmp_path)
    launch = evaluate_plan()
    cases = deepcopy(launch['cases'])
    scores = synthetic_scores(launch)
    for case in cases:
        case['name'] = f"full-{case['binding']['checkpoint_iteration']}-{case['binding']['evaluation_seed']}"
        case['binding']['protocol'] = trace.LEAN_REPLICATION_PROTOCOL
    remapped = {}
    for original, case in zip(launch['cases'], cases):
        score = deepcopy(scores[original['name']])
        score['binding'] = deepcopy(case['binding'])
        score['protocol'] = trace.LEAN_REPLICATION_PROTOCOL
        remapped[case['name']] = score
        if case['binding']['checkpoint_iteration'] != 128:
            for attempt in score['attempts']:
                attempt['candidate_pass'] = False
            score['numerical_passes'] = 0

    summary = ev.summarize_cases(cases, remapped, ev.REPLICATION)
    assert summary['protocol'] == ev.REPLICATION['protocol']
    assert summary['passing_checkpoints'] == [128]
    assert summary[ev.REPLICATION['gate_key']] is True
    assert summary['complete_attempts'] == 1536
    assert summary['checkpoint_admitted'] is False
    assert summary['learned_stance_accepted'] is False
    assert summary['football_balance_accepted'] is False
    assert summary['physical_motion_authorized'] is False


@pytest.mark.parametrize('damage', [
    'missing_pair', 'reordered_pair', 'duplicate_pair', 'duplicate_name',
    'missing_score', 'prefix',
])
def test_helper_refuses_incomplete_or_misordered_case_evidence(monkeypatch, tmp_path, damage):
    pinned_probe(monkeypatch, tmp_path)
    launch = evaluate_plan()
    cases = deepcopy(launch['cases'])
    scores = synthetic_scores(launch)
    if damage == 'missing_pair':
        cases.pop()
    elif damage == 'reordered_pair':
        cases[0], cases[1] = cases[1], cases[0]
    elif damage == 'duplicate_pair':
        cases[1]['binding']['checkpoint_iteration'] = cases[0]['binding']['checkpoint_iteration']
        cases[1]['binding']['evaluation_seed'] = cases[0]['binding']['evaluation_seed']
    elif damage == 'duplicate_name':
        cases[1]['name'] = cases[0]['name']
    elif damage == 'missing_score':
        scores.pop(cases[-1]['name'])
    elif damage == 'prefix':
        scores[cases[0]['name']]['attempts'].pop()

    with pytest.raises(ValueError):
        ev.summarize_cases(cases, scores, ev.LESSON)
