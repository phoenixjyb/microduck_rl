"""CPU-only contract tests for the bounded WSL collection profile."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from mjlab_microduck import stance_wsl_collection_profile as profile


SOURCE = 'a' * 40
TOOL_SHA = 'b' * 64
LAUNCH_SHA = 'c' * 64


def collection(offset=0.0):
    updates = []
    for index in range(10):
        ticks = [0.4 + offset + index / 1000] * 24
        updates.append(dict(update=index, warmup=index < 2,
                            seconds=sum(ticks) + 0.2,
                            tick_seconds=ticks, tick_max_seconds=max(ticks)))
    measured = [row['seconds'] for row in updates[2:]]
    return dict(updates=updates, setup_seconds=0.5 + offset,
                summarize=profile.throughput.summarize(measured),
                optimizer_steps=0, physics_device='cuda:0',
                ticks_per_update=24, worlds=64)


def cgroup(before=100, after=140, cpu_max='200000 100000'):
    base = dict(usage_usec=before, nr_periods=10, nr_throttled=2,
                throttled_usec=5, extra_counter=7)
    end = {key: value + (after - before) for key, value in base.items()}
    return (dict(path='/user.slice/test.scope', cpu_max=cpu_max, counters=base),
            dict(path='/user.slice/test.scope', cpu_max=cpu_max, counters=end))


def batch(name, offset=0.0):
    before, after = cgroup()
    return dict(name=name, instrumented=name == 'python-profile',
                collection=collection(offset), cpu_before=before,
                cpu_after=after, cpu_delta=profile.cpu_delta(before, after))


def valid_batches():
    return [batch('baseline-before'), batch('python-profile', 0.1),
            batch('baseline-after', 0.02)]


def test_plan_binds_separate_tool_runtime_and_never_admits_training(monkeypatch):
    monkeypatch.setattr(profile.execution, 'PROFILE',
                        profile.execution.select(profile.execution.WSL))
    monkeypatch.setattr(profile, 'tool_identity',
                        lambda source, digest: dict(source=source, sha256=digest))
    monkeypatch.setattr(profile.host, 'identity', lambda source: dict(source=source))
    monkeypatch.setattr(profile.q, 'validate_timing_basis', lambda source: dict(source=source))
    result = profile.plan(SOURCE, TOOL_SHA, 123456)
    assert result['tool'] == dict(source=SOURCE, sha256=TOOL_SHA)
    assert result['inputs'] == dict(source=profile.RUNTIME_SOURCE)
    assert (result['worlds'], result['ticks_per_update'], result['warmup_updates'],
            result['measured_updates']) == (64, 24, 2, 8)
    assert result['batches'] == list(profile.BATCHES)
    assert (result['child_seconds'], result['service_seconds'],
            result['closeout_seconds']) == (900, 960, 600)
    assert (result['memory_max_bytes'], result['cpu_quota_per_sec_usec'],
            result['nice']) == (6 * 1024**3, 2_000_000, 10)
    assert result['instrumented_times_excluded_from_qualification'] is True
    assert all(result[key] is False for key in profile.NO_ADMISSION)


def test_non_wsl_profile_refuses_before_host_or_tool_checks(monkeypatch):
    monkeypatch.setattr(profile.execution, 'PROFILE', profile.execution.select(profile.execution.DEFAULT))
    with pytest.raises(ValueError, match='fixed authorized WSL profile'):
        profile.plan(SOURCE, TOOL_SHA, 123456)


def test_tool_bytes_match_independently_pinned_commit_and_hash(monkeypatch):
    raw = Path(profile.__file__).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    calls = []
    def git(command, **kwargs):
        calls.append(command)
        return raw
    monkeypatch.setattr(profile.subprocess, 'check_output', git)
    value = profile.tool_identity(SOURCE, digest)
    assert calls == [['git','show',SOURCE+':'+profile.TOOL_PATH]]
    assert value == dict(source=SOURCE,file=profile.TOOL_PATH,sha256=digest,
                         runtime_source=profile.RUNTIME_SOURCE)
    with pytest.raises(ValueError, match='independently pinned tool bytes'):
        profile.tool_identity(SOURCE, 'e'*64)
    monkeypatch.setattr(profile.subprocess, 'check_output', lambda *args, **kwargs: raw+b'changed')
    with pytest.raises(ValueError, match='independently pinned tool bytes'):
        profile.tool_identity(SOURCE, digest)


def test_unbounded_service_refuses_before_gpu_work(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(profile, 'checked', lambda *args: dict(deadline_unix=123456))
    monkeypatch.setattr(profile, 'output_path', lambda *args: tmp_path)
    monkeypatch.setattr(profile.throughput, 'check_window', lambda *args, **kwargs: None)
    monkeypatch.setattr(profile.host, 'read', lambda *args: 'infinity')
    with pytest.raises(ValueError, match='independently bounded service'):
        profile.supervise(SOURCE, TOOL_SHA, LAUNCH_SHA)
    assert list(tmp_path.iterdir()) == []


def test_summarize_uses_both_baselines_and_reports_profile_separately():
    batches = valid_batches()
    summary = profile.summarize_batches(batches)
    expected = batches[0]['collection']['summarize']['series'] + batches[2]['collection']['summarize']['series']
    assert summary['baseline'] == profile.throughput.summarize(expected)
    assert summary['profiled'] == batches[1]['collection']['summarize']
    assert summary['decision'] == 'profile-complete-not-training-qualification'
    assert summary['python_profile_scope'] == 'inclusive-wall-time-not-GPU-kernel-attribution'
    assert all(summary[key] is False for key in profile.NO_ADMISSION)


@pytest.mark.parametrize('damage', [
    'missing-batch', 'reordered', 'instrumentation', 'nonfinite',
    'inconsistent-summary', 'negative-counter', 'changed-quota',
])
def test_summarize_rejects_incomplete_or_inconsistent_evidence(damage):
    batches = deepcopy(valid_batches())
    if damage == 'missing-batch':
        batches.pop()
    elif damage == 'reordered':
        batches[0], batches[1] = batches[1], batches[0]
    elif damage == 'instrumentation':
        batches[0]['instrumented'] = True
    elif damage == 'nonfinite':
        batches[1]['collection']['updates'][3]['tick_seconds'][0] = float('nan')
    elif damage == 'inconsistent-summary':
        batches[2]['collection']['summarize']['max'] += 1
    elif damage == 'negative-counter':
        batches[0]['cpu_after']['counters']['usage_usec'] -= 1
    else:
        batches[0]['cpu_after']['cpu_max'] = '300000 100000'
    with pytest.raises(ValueError):
        profile.summarize_batches(batches)


def test_cpu_delta_rederives_all_service_counters_and_requires_unchanged_quota():
    before, after = cgroup()
    derived = profile.cpu_delta(before, after)
    assert derived == dict(cpu_max='200000 100000',
                           counters={key: 40 for key in before['counters']},
                           scope='whole-service-not-child-only')
    changed = deepcopy(after)
    changed['cpu_max'] = '100000 100000'
    with pytest.raises(ValueError, match='unchanged service cgroup'):
        profile.cpu_delta(before, changed)
    regressed = deepcopy(after)
    regressed['counters']['nr_throttled'] = before['counters']['nr_throttled'] - 1
    with pytest.raises(ValueError, match='monotone cgroup counters'):
        profile.cpu_delta(before, regressed)
    wrong_quota = deepcopy(before)
    wrong_quota['cpu_max'] = '300000 100000'
    wrong_quota_after = deepcopy(after)
    wrong_quota_after['cpu_max'] = '300000 100000'
    with pytest.raises(ValueError, match='declared 200 percent CPU quota'):
        profile.cpu_delta(wrong_quota, wrong_quota_after)


def test_profile_rows_are_sorted_and_validate_rows_reject_bad_records():
    import cProfile

    def leaf():
        return sum(range(5))

    profiler = cProfile.Profile()
    profiler.runcall(leaf)
    rows = profile.profile_rows(profiler)
    assert rows == sorted(rows, key=lambda row: (-row['cumulative_seconds'], row['file'],
                                                  row['line'], row['function']))
    profile.validate_rows(rows)
    invalid = deepcopy(rows)
    invalid[0]['primitive_calls'] = -1
    with pytest.raises(ValueError):
        profile.validate_rows(invalid)
    with pytest.raises(ValueError):
        profile.validate_rows([])


@pytest.mark.parametrize('damage', ['fields', 'calls', 'time', 'duplicate'])
def test_validate_rows_rejects_malformed_profile_records(damage):
    row = dict(file='module.py', line=1, function='f', primitive_calls=1,
               calls=1, self_seconds=0.1, cumulative_seconds=0.2)
    rows = [deepcopy(row)]
    if damage == 'fields':
        rows[0]['extra'] = True
    elif damage == 'calls':
        rows[0]['primitive_calls'] = 2
    elif damage == 'time':
        rows[0]['self_seconds'] = float('inf')
    else:
        rows.append(deepcopy(row))
    with pytest.raises(ValueError):
        profile.validate_rows(rows)


def test_verify_evidence_replays_hashes_contract_and_summary(tmp_path, monkeypatch):
    batches = valid_batches()
    summary = profile.summarize_batches(batches)
    tool_sha = hashlib.sha256(b'runner').hexdigest()
    launch = dict(protocol=profile.PROTOCOL,
        tool=dict(sha256=tool_sha, runtime_source=profile.RUNTIME_SOURCE, file=profile.TOOL_PATH),
        inputs=dict(source=profile.RUNTIME_SOURCE),
        **{key: False for key in profile.NO_ADMISSION})
    launch.update(batches=list(profile.BATCHES), worlds=64, ticks_per_update=24,
                  seed=523, warmup_updates=2, measured_updates=8,
                  child_seconds=900, service_seconds=960, closeout_seconds=600,
                  memory_max_bytes=6*1024**3, cpu_quota_per_sec_usec=2_000_000,
                  nice=10, optimizer_steps=0, forward_graph=False,
                  instrumented_times_excluded_from_qualification=True,
                  runtime_sha256=hashlib.sha256(b'runtime').hexdigest())
    values = {
        'launch.json': launch,
        'integration.json': {'fixture': True},
        'baseline-before.json': batches[0],
        'python-profile.json': batches[1],
        'baseline-after.json': batches[2],
        'summary.json': summary,
        'python-functions.json': [dict(file='f.py', line=1, function='f',
            primitive_calls=1, calls=1, self_seconds=0.1, cumulative_seconds=0.2)],
    }
    names = {'runner.py', 'parent.pt', 'runtime.json', 'launch.json', 'integration.json',
             'child.log', 'baseline-before.json', 'python-profile.json', 'baseline-after.json',
             'python-functions.json', 'summary.json'}
    payloads = {'runner.py': b'runner', 'parent.pt': b'parent', 'runtime.json': b'runtime',
                'child.log': b''}
    payloads.update({name: json.dumps(value, sort_keys=True).encode()
                     for name, value in values.items() if name != 'launch.json'})
    payloads['launch.json'] = json.dumps(launch, sort_keys=True).encode()
    launch_sha = hashlib.sha256(payloads['launch.json']).hexdigest()
    payloads['report.json'] = b'report'
    for name, raw in payloads.items():
        (tmp_path / name).write_bytes(raw)
    hashes = {name: hashlib.sha256(payloads[name]).hexdigest() for name in names}
    report = dict(protocol=profile.PROTOCOL,
        decision='profile-complete-not-training-qualification', launch_sha256=launch_sha,
        files=hashes, child=dict(returncode=0))
    report.update({key: False for key in profile.NO_ADMISSION})
    (tmp_path / 'report.json').write_text(json.dumps(report))
    monkeypatch.setattr(profile.files, 'file_bytes', lambda path, **kwargs: Path(path).read_bytes())
    monkeypatch.setattr(profile.files, 'parse', lambda raw: json.loads(raw))
    monkeypatch.setattr(profile.host, 'digest', lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest())
    monkeypatch.setattr(profile.host, 'validate_payload', lambda payload, sha: None)
    monkeypatch.setattr(profile.host, 'check_log', lambda path: None)
    monkeypatch.setattr(profile.throughput.checkpoint, 'LEAN_PARENT_SHA256',
                        hashlib.sha256(b'parent').hexdigest())
    result = profile.verify_evidence(tmp_path, launch_sha, tool_sha)
    assert result['summary'] == summary
    assert result['live_host_rechecked'] is False

    (tmp_path / 'unexpected.txt').write_text('extra')
    with pytest.raises(ValueError, match='exact profile evidence inventory'):
        profile.verify_evidence(tmp_path, launch_sha, tool_sha)
    (tmp_path / 'unexpected.txt').unlink()

    original_log = (tmp_path / 'child.log').read_bytes()
    (tmp_path / 'child.log').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='all profile hashes'):
        profile.verify_evidence(tmp_path, launch_sha, tool_sha)
    (tmp_path / 'child.log').write_bytes(original_log)

    def reseal_launch(change):
        current = json.loads((tmp_path / 'launch.json').read_bytes())
        change(current)
        raw = json.dumps(current, sort_keys=True).encode()
        (tmp_path / 'launch.json').write_bytes(raw)
        new_sha = hashlib.sha256(raw).hexdigest()
        current_report = json.loads((tmp_path / 'report.json').read_bytes())
        current_report['launch_sha256'] = new_sha
        current_report['files']['launch.json'] = new_sha
        (tmp_path / 'report.json').write_text(json.dumps(current_report))
        return new_sha

    bad_contract_sha = reseal_launch(lambda value: value.update(cpu_quota_per_sec_usec=1_900_000))
    with pytest.raises(ValueError, match='fixed non-learning profiling protocol'):
        profile.verify_evidence(tmp_path, bad_contract_sha, tool_sha)
    bad_source_sha = reseal_launch(
        lambda value: value['inputs'].update(source='e' * 40))
    with pytest.raises(ValueError, match='honest separate tool/runtime binding'):
        profile.verify_evidence(tmp_path, bad_source_sha, tool_sha)
