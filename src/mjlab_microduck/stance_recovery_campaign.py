"""One measured, sequential frozen five-case baseline; never a learner."""
import argparse
import gc
from hashlib import sha256
import math
import os
import time

import torch

from mjlab_microduck import stance_recovery_probe as probe
from mjlab_microduck import stance_recovery_contract as contract
from mjlab_microduck import stance_recovery_matrix as matrix
from mjlab_microduck.first_attempt_smoke import require

host, files, profile, evidence = probe.host, probe.files, probe.profile, probe.evidence
MODULE = 'mjlab_microduck.stance_recovery_campaign'
TIMING_SOURCE = 'a44c1d70803012acf791de995a3d384c51162955'
TIMING_REPORT_SHA = '36a4473f2dded6be55e1402b63efd4377f2b0b53127fd1af5ea4d45e664d7e3a'
CHILD_SECONDS, SERVICE_SECONDS, CASE_SECONDS = 1352, 1560, 250
PARENT_SECONDS, LAUNCH_RESERVE = 208, 1800


def check_window(*, launching=False, now=None):
    remaining = contract.CUTOFF-(time.time() if now is None else now)
    require(remaining > (LAUNCH_RESERVE if launching else 0),
            'complete frozen matrix and closeout before fixed 08:00 Shanghai cutoff')


def output_path(source):
    files.hex_id(source, 40)
    return host.ROOT/'artifacts/evaluations'/('stance-wsl-d1-frozen-five-case-'+source[:12])


def service_name(source):
    output_path(source)
    return 'microduck-wsl-d1-frozen-five-case-'+source[:12]+'.service'


def measured_budget(report):
    require(report['source'] == TIMING_SOURCE
            and report['decision'] == 'frozen-recovery-timing-probe-replayed'
            and report['full_length_probe_completed'] is True
            and report['child']['returncode'] == 0
            and all(report[k] is False for k in contract.FALSE_FLAGS), 'successful non-admitting R1 timing')
    capture, score = report['capture'], report['score']
    require(capture['source'] == TIMING_SOURCE and capture['optimizer_steps'] == 0
            and capture['backend'] == dict(torch_device='cuda:0', warp_is_cuda=True)
            and capture['collection']['policy_ticks'] == contract.POLICY_TICKS
            and score['case'] == contract.PROBE_CASE
            and score['numerical_diagnostic']['gates']['full_duration'] is True
            and score['pulse']['checked_physics_steps'] == contract.TOTAL_STEPS
            and score['pulse']['complete_pulse_delivery'] is True
            and all(capture[k] is score[k] is False for k in contract.FALSE_FLAGS),
            'full-length authenticated timing case')
    timings = [capture['construction_seconds'], capture['collection']['elapsed_seconds'],
        capture['serialization_seconds'], capture['elapsed_seconds'],
        report['child']['elapsed_s'], report['elapsed_seconds'], report['supervisor_replay_seconds']]
    require(all(type(t) is float and math.isfinite(t) and t > 0 for t in timings), 'finite measured phases')
    repeating = sum(timings[:3]); entry = timings[3]-repeating; wrapper = timings[4]-timings[3]
    parent = timings[5]-timings[4]+4*timings[6]
    predicted = entry+wrapper+len(contract.CASE_NAMES)*repeating
    require(entry > 0 and wrapper > 0 and parent > 0
            and math.ceil(predicted*1.25) == 1075 and math.ceil(repeating*1.5) == CASE_SECONDS
            and math.ceil((parent+60)*1.5) == 202
            and files.LEAN_EVALUATION_CHILD_SECONDS == CHILD_SECONDS
            and 900 < math.ceil(predicted*1.25) <= CHILD_SECONDS
            and SERVICE_SECONDS == CHILD_SECONDS+PARENT_SECONDS
            and LAUNCH_RESERVE == SERVICE_SECONDS+contract.CLOSEOUT_SECONDS+contract.MARGIN_SECONDS,
            'unchanged measured matrix watchdog and closeout declaration')
    return dict(repeating_case_seconds=repeating, entry_seconds=entry, wrapper_seconds=wrapper,
        predicted_child_seconds=predicted, child_multiplier=1.25, child_rounded_seconds=1075,
        predicted_parent_seconds=parent, parent_comparison_reserve_seconds=60,
        parent_multiplier=1.5, parent_rounded_seconds=202, child_seconds=CHILD_SECONDS,
        service_seconds=SERVICE_SECONDS, case_seconds=CASE_SECONDS, memory_bytes=2*1024**3,
        cpu_quota_percent=200, nice=10, kill_mode='control-group', launch_reserve_seconds=LAUNCH_RESERVE)


def timing_prerequisite():
    require(not torch.cuda.is_initialized(), 'fresh CPU timing qualification before CUDA initialization')
    root = probe.output_path(TIMING_SOURCE)
    raw = files.file_bytes(root/'report.json')
    require(sha256(raw).hexdigest() == TIMING_REPORT_SHA, 'exact retained successful R1 report bytes')
    report = files.parse(raw); budget = measured_budget(report)
    for name, digest in report['files'].items():
        require(type(name) is str and '/' not in name and name not in ('.', '..')
                and host.digest(root/name) == digest, 'retained timing artifact '+str(name))
    record = report['capture']
    raw = files.file_bytes(root/'capture.pt', limit=contract.CAPTURE_LIMIT)
    require(record['capture_sha256'] == sha256(raw).hexdigest()
            and record['capture_bytes'] == len(raw), 'complete R1 raw bytes before replay')
    cp = files.file_bytes(root/'checkpoint.pt', limit=probe.evaluation.checkpoint.LIMIT)
    launch = files.parse(files.file_bytes(root/'launch.json'))
    require(evidence.verify(raw, record['capture_sha256'], cp, launch['declaration']) == report['score'],
            'fresh complete R1 actor/control/recorded-state timing rescore')
    return dict(source=TIMING_SOURCE, report_sha256=TIMING_REPORT_SHA, files=report['files'], budget=budget)


def launch_plan(source, prefix_sha):
    prefix = probe.checked(source, prefix_sha)
    gc.collect()  # Closed CPU qualification fixtures, never an active runtime.
    return dict(protocol=matrix.PROTOCOL, source=source, prefix_launch_sha256=prefix_sha,
        prefix_output=str(probe.output_path(source)), prefix_launch=prefix,
        cases=list(contract.CASE_NAMES), timing=timing_prerequisite(), **contract.FALSE_FLAGS)


def prepare(source):
    check_window(launching=True)
    prepared = probe.prepare(source)
    gc.collect()  # Release the completed CPU prefix before whole R1 loading.
    launch = launch_plan(source, prepared['launch_sha256'])
    root = files.native._plain_path(output_path(source)); root.mkdir(exist_ok=False)
    files.write_json(root/'launch.json', launch)
    return dict(output=str(root), service=service_name(source), launch_sha256=host.digest(root/'launch.json'))


def checked(source, launch_sha):
    files.hex_id(launch_sha, 64); root = output_path(source)
    raw = files.file_bytes(root/'launch.json')
    require(sha256(raw).hexdigest() == launch_sha, 'exact fresh matrix launch bytes')
    launch = files.parse(raw)
    require(launch == launch_plan(source, launch['prefix_launch_sha256']),
            'unchanged exact matrix archive/source/plant/profile/prefix/timing')
    return launch


def case_paths(root, index):
    require(type(index) is int and 0 <= index < len(contract.CASE_NAMES), 'ordered bounded matrix index')
    return root/f'case-{index}.pt', root/f'case-{index}.json'


def child(source, launch_sha, fd, started):
    require(type(started) is float and 0 < time.monotonic()-started < CHILD_SECONDS, 'same-clock bounded matrix entry')
    check_window(launching=True); probe.smoke.inherited_lease(fd)
    launch = checked(source, launch_sha)
    require(time.monotonic()-started < 180, 'matrix CPU entry leaves capture reserve')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.is_available(), 'sole inherited-lease matrix CUDA child')
    host.wait_idle(); root = output_path(source)
    prefix = launch['prefix_launch']; identity = prefix['selected_checkpoint']['identity']
    cp = files.file_bytes(probe.output_path(source)/'checkpoint.pt', limit=probe.evaluation.checkpoint.LIMIT)
    from mjlab_microduck.stance_recovery_runtime import RecoveryRuntime
    for index, case in enumerate(contract.CASE_NAMES):
        check_window()
        case_start = time.monotonic()
        require(started+CHILD_SECONDS-case_start >= CASE_SECONDS, 'reserve complete next fresh matrix case')
        probe.seed_reset(); actor = probe.restored_actor(cp, identity)
        built = time.monotonic(); env = RecoveryRuntime([case], device='cuda:0')
        construction = time.monotonic()-built
        try:
            value = evidence.collect(env, actor, prefix['declaration'], identity,
                deadline_monotonic=min(case_start+CASE_SECONDS-60, started+CHILD_SECONDS-120))
        except Exception as exc:
            # The existing collector cannot return its local partial trace on
            # an exception. Preserve an honest failure receipt, not a fabricated
            # normal capture or successful prefix. Prior cases stay immutable.
            files.write_json(root/f'case-{index}-failure.json', dict(protocol=matrix.PROTOCOL,
                source=source, case=case, index=index, launch_sha256=launch_sha,
                stage='collection', error_type=type(exc).__name__, error=str(exc),
                current_case_trace_retained=False, optimizer_steps=0, **contract.FALSE_FLAGS))
            raise
        require(time.monotonic()-case_start < CASE_SECONDS-60, 'fixed matrix case serialization reserve')
        serial = time.monotonic(); raw = evidence.encode(value)
        tensor_path, record_path = case_paths(root, index)
        probe.write_capture(tensor_path, raw)
        record = dict(protocol=matrix.PROTOCOL, source=source, case=case, index=index,
            launch_sha256=launch_sha, capture_sha256=sha256(raw).hexdigest(), capture_bytes=len(raw),
            backend=value['backend'], construction_seconds=construction, collection=value['collection'],
            serialization_seconds=time.monotonic()-serial, case_elapsed_seconds=time.monotonic()-case_start,
            optimizer_steps=0, **contract.FALSE_FLAGS)
        files.write_json(record_path, record)
        require(record['case_elapsed_seconds'] < CASE_SECONDS, 'complete matrix case within fixed cap')
        complete = value['collection']['policy_ticks'] == contract.POLICY_TICKS
        del env, actor, value, raw
        gc.collect()  # Only after the case is closed and its evidence durable.
        require(complete, 'partial or early-failed matrix case retained; no reset or retry')


def replay_cases(source, launch, launch_sha):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'independent CPU-only matrix scorer')
    root = output_path(source); receipts = []
    cp = files.file_bytes(probe.output_path(source)/'checkpoint.pt', limit=probe.evaluation.checkpoint.LIMIT)
    for index, case in enumerate(contract.CASE_NAMES):
        path, metadata_path = case_paths(root, index)
        record = files.parse(files.file_bytes(metadata_path)); raw = files.file_bytes(path, limit=contract.CAPTURE_LIMIT)
        require(record['protocol'] == matrix.PROTOCOL and record['source'] == source
                and record['case'] == case and record['index'] == index and record['launch_sha256'] == launch_sha
                and record['capture_sha256'] == sha256(raw).hexdigest() and record['capture_bytes'] == len(raw)
                and record['backend'] == dict(torch_device='cuda:0', warp_is_cuda=True)
                and record['optimizer_steps'] == 0 and all(record[k] is False for k in contract.FALSE_FLAGS),
                'authenticated actual ordered CUDA case before tensor loading')
        phases = [record[k] for k in ('construction_seconds', 'serialization_seconds', 'case_elapsed_seconds')]
        require(all(type(t) is float and math.isfinite(t) and 0 < t < CASE_SECONDS for t in phases)
                and record['serialization_seconds'] < 60
                and phases[0]+record['collection']['elapsed_seconds']+phases[1] <= phases[2],
                'nonoverlapping bounded case timing')
        receipt = matrix.checked_case(raw, record['capture_sha256'], cp, launch['prefix_launch']['declaration'])
        require(receipt['score']['collection'] == record['collection'], 'retained matrix timing matches full payload')
        files.write_json(root/f'case-{index}-replay.json', receipt)
        receipts.append(receipt); del raw; gc.collect()
    comparison = matrix.compare(receipts)
    files.write_json(root/'comparison.json', comparison)
    return comparison


def supervise(source, launch_sha):
    started = time.monotonic(); root = output_path(source)
    report = dict(protocol=matrix.PROTOCOL, source=source, launch_sha256=launch_sha,
        decision='frozen-five-case-diagnostic-failed', **contract.FALSE_FLAGS)
    try:
        props = {k: host.read('systemctl', '--user', 'show', service_name(source), '-p', k, '--value')
            for k in ('MainPID', 'RuntimeMaxUSec', 'KillMode', 'ActiveState', 'MemoryMax', 'CPUQuotaPerSecUSec', 'Nice')}
        require(props == dict(MainPID=str(os.getpid()), RuntimeMaxUSec='26min', KillMode='control-group',
            ActiveState='active', MemoryMax=str(2*1024**3), CPUQuotaPerSecUSec='2s', Nice='10'), 'independently capped matrix service')
        report['service_properties'] = props
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CUDA-hidden matrix parent')
        check_window(launching=True); launch = checked(source, launch_sha)
        require({p.name for p in root.iterdir()} == {'launch.json'}, 'one fresh matrix attempt')
        with files.gpu_lease() as fd:
            report['idle_before'] = host.wait_idle(); child_started = time.monotonic()
            command = [str(host.ROOT/'.venv/bin/python'), '-m', MODULE, 'child', '--source', source,
                '--launch-sha256', launch_sha, '--lock-fd', str(fd), '--started-monotonic', repr(child_started)]
            env = files.child_environment(); env.update(profile.settings())
            def guard():
                check_window(); host.check_log(root/'child.log')
                require(host.read('git', 'rev-parse', 'HEAD') == source
                        and not host.read('git', 'status', '--porcelain'), 'frozen clean live matrix source')
                require(probe.d0.filmbrain_state() == launch['prefix_launch']['preserved_filmbrain'], 'preserved FilmBrain')
            require(files.LEAN_EVALUATION_CHILD_SECONDS == CHILD_SECONDS, 'unchanged existing 1352-second watchdog')
            report['child'] = files.supervised_lean_evaluation(command, root/'child.log', cwd=host.ROOT,
                env=env, lock_fd=fd, guard=guard)
            host.check_log(root/'child.log')
            require(checked(source, launch_sha) == launch, 'unchanged post-child matrix qualification')
            replay_start = time.monotonic()
            comparison = replay_cases(source, launch, launch_sha)
            report['supervisor_replay_seconds'] = time.monotonic()-replay_start
            report['comparison'] = comparison; report['idle_after'] = host.wait_idle()
        elapsed = time.monotonic()-started
        require(elapsed < SERVICE_SECONDS, 'complete matrix service within fixed cap')
        report.update(decision=comparison['decision'], elapsed_seconds=elapsed,
            files={p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()})
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), error_notes=getattr(exc, '__notes__', []))
        try:
            report['files'] = {p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        except Exception as inventory_exc:
            report['inventory_error'] = str(inventory_exc)
        raise
    finally:
        files.write_json(root/'report.json', report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'supervise', 'child'))
    parser.add_argument('--source', required=True); parser.add_argument('--launch-sha256')
    parser.add_argument('--lock-fd', type=int); parser.add_argument('--started-monotonic', type=float)
    a = parser.parse_args(argv)
    if a.mode == 'prepare': print(prepare(a.source))
    elif a.mode == 'supervise': supervise(a.source, a.launch_sha256)
    else: child(a.source, a.launch_sha256, a.lock_fd, a.started_monotonic)


if __name__ == '__main__':
    main()
