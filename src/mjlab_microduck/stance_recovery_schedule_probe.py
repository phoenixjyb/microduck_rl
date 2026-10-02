"""One bounded genuine CPU zero/diagonal diagnostic, no GPU or training."""
import argparse
import gc
from hashlib import sha256
import io
import os
import random
import time

import numpy as np
import torch

from mjlab_microduck import stance_recovery_probe as retained
from mjlab_microduck import stance_recovery_parent as parent
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck import stance_recovery_schedule_trace as evidence
from mjlab_microduck.first_attempt_smoke import require

host, files, baseline, profile = retained.host, retained.files, retained.contract, retained.profile
MODULE = 'mjlab_microduck.stance_recovery_schedule_probe'
MATRIX_SOURCE = 'e9b9d498363714deda1214026724aa41382170a9'
MATRIX_CLOSEOUT_SHA256 = 'a44d87779cca18b3e4df899095126d85d9958472469b2fe23c7c99ad73e38f15'
MATRIX_REPORT_SHA256 = 'f9b659da15f53f276f0dedbe61e9afcaf98b8c9f3602ca5171a5ffb8ee798a84'
SERVICE_SECONDS, CASE_COLLECTION_SECONDS, LAUNCH_RESERVE = 240, 60, 480
CELL_IDS = ('zero-wrench', 'diagonal-++-2n-10steps-t375')


def check_window(*, launching=False, now=None):
    require(baseline.CUTOFF-(time.time() if now is None else now) > (LAUNCH_RESERVE if launching else 0),
            'CPU diagnostic and closeout before fixed 08:00 Shanghai cutoff')


def output_path(source):
    files.hex_id(source, 40)
    return host.ROOT/'artifacts/evaluations'/('stance-wsl-cpu-scheduled-diagonal-'+source[:12])


def service_name(source):
    output_path(source)
    return 'microduck-cpu-scheduled-diagonal-'+source[:12]+'.service'


def prerequisite():
    root = host.ROOT/'artifacts/evaluations/stance-wsl-d1-frozen-five-case-e9b9d4983637'
    raw = files.file_bytes(root/'independent-closeout.json')
    require(sha256(raw).hexdigest() == MATRIX_CLOSEOUT_SHA256, 'exact independent five-case closeout')
    closeout = files.parse(raw); report_raw = files.file_bytes(root/'report.json')
    require(sha256(report_raw).hexdigest() == MATRIX_REPORT_SHA256, 'exact successful five-case report')
    report = files.parse(report_raw)
    require(closeout['source'] == report['source'] == MATRIX_SOURCE
            and closeout['whole_cpu_rescore_identical'] is True and closeout['cases_checked'] == 5
            and closeout['decision'] == report['decision'] == 'frozen-five-case-baseline-passed'
            and all(closeout[k] is report[k] is False for k in baseline.FALSE_FLAGS), 'non-admitting closed prerequisite')
    require(set(closeout['files_rehashed']) == set(report['files']) | {'report.json'}, 'complete closed matrix inventory')
    require({p.name for p in root.iterdir()} == set(closeout['files_rehashed']) | {'independent-closeout.json'},
            'no unlisted closed matrix artifacts')
    for name, info in closeout['files_rehashed'].items():
        require(type(name) is str and '/' not in name and name not in ('.', '..')
                and (root/name).stat().st_size == info['bytes']
                and host.digest(root/name) == info['sha256'], 'unchanged matrix artifact '+str(name))
    prefix = retained.output_path(MATRIX_SOURCE)
    require({p.name for p in prefix.iterdir()} == set(closeout['prefix_files_rehashed']),
            'no unlisted selected CPU-prefix artifacts')
    for name, info in closeout['prefix_files_rehashed'].items():
        require(type(name) is str and '/' not in name and name not in ('.', '..')
                and (prefix/name).stat().st_size == info['bytes']
                and host.digest(prefix/name) == info['sha256'], 'unchanged selected CPU-prefix artifact '+str(name))
    cp = files.file_bytes(prefix/'checkpoint.pt', limit=retained.evaluation.checkpoint.LIMIT)
    require(sha256(cp).hexdigest() == baseline.CHECKPOINT_SHA256, 'selected immutable real parent bytes')
    return dict(source=MATRIX_SOURCE, closeout_sha256=MATRIX_CLOSEOUT_SHA256,
        report_sha256=MATRIX_REPORT_SHA256, checkpoint_sha256=baseline.CHECKPOINT_SHA256), cp


def seed_cpu():
    random.seed(evidence.EVALUATION_SEED); np.random.seed(evidence.EVALUATION_SEED)
    torch.random.default_generator.manual_seed(evidence.EVALUATION_SEED)


def protected_state():
    return {name: host.read(*command) for name, command in host.execution.service_commands(files.SERVICES)}


def run(source):
    started = time.monotonic(); check_window(launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only scheduled probe entry')
    props = {k: host.read('systemctl', '--user', 'show', service_name(source), '-p', k, '--value')
        for k in ('MainPID', 'ActiveState', 'RuntimeMaxUSec', 'MemoryMax', 'CPUQuotaPerSecUSec', 'Nice', 'KillMode')}
    require(props == dict(MainPID=str(os.getpid()), ActiveState='active', RuntimeMaxUSec='4min',
        MemoryMax=str(2*1024**3), CPUQuotaPerSecUSec='2s', Nice='10', KillMode='control-group'),
        'independently capped genuine CPU scheduled service')
    source_identity = host.identity(source); actual_profile = profile.checked_receipt()
    preserved = retained.d0.filmbrain_state()
    # Protected services are observed only; this CPU workflow never starts,
    # stops or shares ownership of a GPU process.
    services_before = protected_state()
    require(all(v == 'inactive' for v in services_before.values()), 'protected services remain inactive')
    previous, cp = prerequisite()
    before_rng = torch.random.get_rng_state().clone()
    loaded = parent.load_parent(cp)
    require(torch.equal(before_rng, torch.random.get_rng_state()), 'parent preparation preserves owned CPU RNG')
    frozen, _ = retained.evaluation.checkpoint.load_lean_replication_evaluation(cp, baseline.CHECKPOINT_SHA256, parent.expected_identity())
    obs = torch.arange(88, dtype=torch.float32).reshape(2, 44)/100
    require(torch.equal(retained.evaluation.checkpoint.infer(loaded['actor'], obs),
        retained.evaluation.checkpoint.infer(frozen, obs)), 'exact real-parent deterministic actor shadow')
    parent_receipt = loaded['receipt']; del loaded, frozen; gc.collect()
    from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
    declarations = [schedule.declaration(source, 'gentle', 'held-out', [cell]) for cell in CELL_IDS]
    root = files.native._plain_path(output_path(source)); root.mkdir(exist_ok=False)
    report = dict(protocol=evidence.PROTOCOL, source=source, decision='cpu-scheduled-diagonal-diagnostic-failed',
        service_properties=props, **baseline.FALSE_FLAGS)
    try:
        seed_cpu(); env = ScheduledRecoveryRuntime(declarations[0], device='cpu')
        compiled = env.binding
        launch = dict(protocol=evidence.PROTOCOL, source=source, source_identity=source_identity,
            prerequisite=previous, parent_preparation=parent_receipt, cpu_math_profile=actual_profile,
            preserved_filmbrain=preserved, services_before=services_before, compiled_plant=compiled,
            declarations=declarations, evaluation_seed=evidence.EVALUATION_SEED,
            service_seconds=SERVICE_SECONDS, case_collection_seconds=CASE_COLLECTION_SECONDS,
            launch_reserve_seconds=LAUNCH_RESERVE, synthetic_observations_for_shadow_only=True,
            actual_first_attempts_have_no_counter_support_hooks=True, optimizer_steps=0, **baseline.FALSE_FLAGS)
        files.write_json(root/'launch.json', launch); retained.write_capture(root/'checkpoint.pt', cp)
        scores = []; prefixes = []
        for index, declaration in enumerate(declarations):
            check_window()
            require(time.monotonic()-started < SERVICE_SECONDS-90, 'reserve complete next CPU case and retention')
            require(host.identity(source) == source_identity and retained.d0.filmbrain_state() == preserved
                    and protected_state() == services_before, 'unchanged CPU source and unrelated/protected services')
            if index:
                seed_cpu(); env = ScheduledRecoveryRuntime(declaration, device='cpu')
            require(env.binding == compiled, 'identical fresh compiled scheduled plant')
            case_started = time.monotonic()
            try:
                value = evidence.collect(env, cp, deadline_monotonic=case_started+CASE_COLLECTION_SECONDS)
            except Exception as exc:
                files.write_json(root/f'case-{index}-failure.json', dict(stage='collection', source=source,
                    index=index, cell=declaration['cell_ids'][0],
                    error_type=type(exc).__name__, error=str(exc), current_case_trace_retained=False, **baseline.FALSE_FLAGS))
                raise
            collection_wall_seconds = time.monotonic()-case_started
            raw = evidence.encode(value); capture_sha = sha256(raw).hexdigest()
            retained.write_capture(root/f'case-{index}.pt', raw)
            files.write_json(root/f'case-{index}.json', dict(protocol=evidence.PROTOCOL, source=source,
                declaration=declaration, capture_bytes=len(raw), capture_sha256=capture_sha,
                collection=value['collection'], collection_wall_seconds=collection_wall_seconds,
                case_elapsed_seconds=time.monotonic()-case_started, **baseline.FALSE_FLAGS))
            del value, env; gc.collect()
            score = evidence.verify(raw, capture_sha, cp, declaration, compiled)
            reread = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            prefix = evidence.prefix_hash(reread, 375) if score['pulse']['checked_physics_steps'] > 375 else None
            del reread, raw; gc.collect()
            files.write_json(root/f'case-{index}-replay.json', score); scores.append(score); prefixes.append(prefix)
            require(0 < collection_wall_seconds < CASE_COLLECTION_SECONDS
                    and 0 < score['collection']['elapsed_seconds'] < CASE_COLLECTION_SECONDS,
                    'over-budget CPU collection retained and scored; stop without retry')
            require(score['numerical_diagnostic']['gates']['full_duration'] is True
                    and score['numerical_diagnostic']['complete_first_attempt'] is True
                    and score['collection']['policy_ticks'] == 250
                    and score['pulse']['checked_physics_steps'] == 2500
                    and score['pulse']['complete_pulse_delivery'] is True
                    and score['pulse']['complete_phase_checks'] is True,
                    'partial CPU first attempt retained; stop without retry')
        require(prefixes[0] is not None and prefixes[0] == prefixes[1], 'exact mid-tick pre-push matched control/physical prefixes')
        require(host.identity(source) == source_identity and retained.d0.filmbrain_state() == preserved
                and protected_state() == services_before and not torch.cuda.is_initialized()
                and time.monotonic()-started < SERVICE_SECONDS, 'unchanged bounded CPU closeout')
        passed = all(s['numerical_diagnostic']['candidate_pass'] is True for s in scores)
        report.update(decision='cpu-scheduled-diagonal-probe-passed' if passed else 'cpu-scheduled-diagonal-probe-rejected',
            scores=scores, prefixes=prefixes, prefixes_identical=True, genuine_cpu_first_attempts=2,
            parent_preparation=parent_receipt, cuda_initialized=False, optimizer_steps=0,
            filmbrain_unchanged=True, protected_services_inactive=True)
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        try:
            report['files'] = {p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        except Exception as inventory_error:
            report['inventory_error'] = str(inventory_error)
        files.write_json(root/'report.json', report)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--source', required=True)
    run(p.parse_args(argv).source)


if __name__ == '__main__':
    main()
