"""Cross-seed verdict for the fresh-seed replication; records, never admits.

Declared by ``docs/experiments/2026-09-17-stance-lean-replication.md``. That
document's decision rule is about the **set** of three replication seeds, not
about any one of them:

- ``lean-replication-passed``          all three declared seeds pass
- ``lean-replication-seed-dependent``  some but not all pass
- ``lean-replication-rejected``        none pass
- ``lean-replication-incomplete``      a seed did not produce a verifiable run

The last one is why this is a module rather than a one-line ``all()``. A seed
that failed to *run* is not a seed that failed its *gate*, and quietly dropping
it from the denominator would let two seeds be reported as three. So an
unverifiable seed short-circuits to ``incomplete`` and no other verdict is
recorded, exactly as the predeclaration requires.

A per-seed verdict is deliberately not the experiment's verdict. One seed passing
says nothing about whether the recipe replicates, which is the entire question.
This module will not emit ``passed`` from fewer than all three.

Each seed's verdict is **re-derived from retained bytes** through
``evaluation.verify``, which replays every bundle from its manifest, rather than
read out of that run's own summary. Nothing here admits a checkpoint, accepts a
learned stance, authorizes motion, or relaxes the 0.0873 rad gate.
"""

from copy import deepcopy
from hashlib import sha256

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_lean_evaluation as evaluation
from mjlab_microduck.first_attempt_smoke import canonical, require

MODULE = 'mjlab_microduck.stance_lean_replication_campaign'
PROTOCOL = 'football-b1n-lean-replication-campaign-v1'
SEEDS = checkpoint.LEAN_REPLICATION_SEEDS
PASSED, REJECTED = evaluation.REPLICATION['passed'], evaluation.REPLICATION['rejected']
DECISIONS = ('lean-replication-passed', 'lean-replication-seed-dependent',
             'lean-replication-rejected', 'lean-replication-incomplete')
MODE = 'evaluate'
host, files = evaluation.host, evaluation.files


def evaluation_root(source, seed):
    """One replication seed's retained comparison directory."""
    return evaluation.output_path(source, MODE, evaluation.REPLICATION, seed)


def campaign_root(source):
    files.hex_id(source, 40)
    return host.ROOT/'artifacts/evaluations'/('stance-lean-replication-campaign-'+source[:12])


def seed_verdict(source, seed, training_source=None):
    """One seed's verdict, re-derived from its retained comparison bytes.

    Reads the report only to find the comparison hash and to check that the
    recorded decision is one this campaign may combine; the verdict itself comes
    from ``evaluation.verify``, which replays all twelve bundles.
    """
    root = evaluation_root(source, seed)
    report_raw = files.file_bytes(root/'report.json')
    report = files.parse(report_raw)
    require(report['protocol'] == evaluation.REPLICATION['protocol'],
            'a replication comparison report')
    require(report['decision'] in (PASSED, REJECTED),
            'a completed per-seed replication verdict')
    require(report['tilt_gate_relaxed'] is False, 'a per-seed verdict with the gate unrelaxed')
    launch_raw = files.file_bytes(root/'launch.json')
    launch = files.parse(launch_raw)
    retained = launch['retained_training']
    actual_training_source = retained['source']
    if training_source is not None:
        require(actual_training_source == training_source,
                'each replication seed judged the declared training source')
    summary = evaluation.verify(root, launch, report['comparison_sha256'], evaluation.REPLICATION)
    require(summary['decision'] == report['decision'], 're-derived per-seed verdict')
    return dict(summary, training_source=actual_training_source,
        training_report_sha256=retained['report_sha256'],
        report_sha256=sha256(report_raw).hexdigest(),
        launch_sha256=sha256(launch_raw).hexdigest())


def decide(verdicts):
    """The predeclared rule, applied to three verified per-seed verdicts.

    ``verdicts`` maps each declared seed to its summary, or to an error record
    when that seed could not be verified. A missing or broken seed is
    ``incomplete`` and outranks every other outcome: two passing seeds plus one
    that never ran is not two-thirds of a pass, and the third seed is not
    dropped from the denominator to make it look like one.
    """
    require(set(verdicts) == set(SEEDS), 'all three declared replication seeds')
    for seed in SEEDS:
        verdict = verdicts[seed]
        if 'error' in verdict:
            return 'lean-replication-incomplete'
        require(verdict['decision'] in (PASSED, REJECTED),
                'per-seed verdicts this campaign may combine')
    passing = [seed for seed in SEEDS if verdicts[seed]['decision'] == PASSED]
    if len(passing) == len(SEEDS): return 'lean-replication-passed'
    if passing: return 'lean-replication-seed-dependent'
    return 'lean-replication-rejected'


def campaign(source, training_source, *, write=True):
    """Judge all three declared seeds and record the declared verdict.

    ``training_source`` names the commit that produced the three replication
    runs. It is authenticated here rather than taken on trust: each seed's own
    evaluation already pins its training archive by hash, and this only checks
    that the three agree on which run they judged.
    """
    files.hex_id(source, 40)
    require(type(training_source) is str, 'the replication training source to judge')
    files.hex_id(training_source, 40)
    verdicts = {}
    for seed in SEEDS:
        try:
            summary = seed_verdict(source, seed, training_source)
            require(summary['protocol'] == evaluation.REPLICATION['protocol'],
                    'a replication comparison summary')
            require(summary['training_source'] == training_source,
                    'each replication seed judged the declared training source')
            verdicts[seed] = summary
        except Exception as exc:
            # Recorded, never swallowed: an unrunnable seed must not be able to
            # leave the remaining seeds looking like a full set.
            verdicts[seed] = dict(error=type(exc).__name__, message=str(exc),
                                  error_notes=[str(note) for note in getattr(exc, '__notes__', [])])
    decision = decide(verdicts)
    require(decision in DECISIONS, 'a declared replication verdict')
    result = dict(protocol=PROTOCOL, source=source, training_source=training_source,
        seeds=list(SEEDS), mode=MODE, decision=decision,
        passing_seeds=[seed for seed in SEEDS
                       if verdicts[seed].get('decision') == PASSED],
        per_seed={str(seed): (verdicts[seed] if 'error' in verdicts[seed] else dict(
            decision=verdicts[seed]['decision'],
            training_source=verdicts[seed]['training_source'],
            training_report_sha256=verdicts[seed]['training_report_sha256'],
            report_sha256=verdicts[seed]['report_sha256'],
            launch_sha256=verdicts[seed]['launch_sha256'],
            passing_checkpoints=verdicts[seed]['passing_checkpoints'],
            complete_attempts=verdicts[seed]['complete_attempts'],
            tilt_gate_rad=verdicts[seed]['tilt_gate_rad'],
            tilt_gate_relaxed=verdicts[seed]['tilt_gate_relaxed']))
            for seed in SEEDS},
        tilt_gate_rad=evaluation.TILT_GATE_RAD, tilt_gate_relaxed=False,
        # A pass admits only a nominal stance candidate. Small-disturbance
        # recovery still precedes B1 acceptance, so every flag stays false.
        checkpoint_admitted=False, learned_stance_accepted=False,
        football_balance_accepted=False, physical_motion_authorized=False)
    if write:
        root = campaign_root(source)
        root.mkdir(exist_ok=False)
        files.write_json(root/'campaign.json', result)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--training-source', required=True)
    args = parser.parse_args()
    print(canonical(campaign(args.source, args.training_source)))


if __name__ == '__main__': main()
