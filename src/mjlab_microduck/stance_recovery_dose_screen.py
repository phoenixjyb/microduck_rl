"""Three fixed CPU dose cases for the unchanged parent, never training."""
import argparse
import gc
from hashlib import sha256
import io
import os
import time

import torch

from mjlab_microduck import stance_recovery_schedule_probe as base
from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = 'football-b1d-cpu-dose-screen-v1'
CELL_IDS = ('zero-wrench', '+x-2n-20steps-t250', '+x-4n-10steps-t250')
PREFIX_STEP = 250
SERVICE_SECONDS, COLLECTION_SECONDS, LAUNCH_RESERVE = 180, 60, 360
CASE_CLOSEOUT_RESERVE = 20
CLOSED_SOURCE = '1ae0a2956e10626679cabcac3a18c5c40ec383c3'
CLOSED_SHA256 = '168738a55372d46734b360ce4ebe4a99c0660ebeb7f5ebe6d61c785644ccaac5'


def output_path(source):
    base.files.hex_id(source, 40)
    return base.host.ROOT/'artifacts/evaluations'/('stance-wsl-cpu-dose-screen-'+source[:12])


def service_name(source):
    output_path(source)
    return 'microduck-cpu-dose-screen-'+source[:12]+'.service'


def check_window(*, launching=False, now=None):
    require(base.baseline.CUTOFF-(time.time() if now is None else now) > (LAUNCH_RESERVE if launching else 0),
            'CPU dose screen and closeout before fixed 08:00 Shanghai cutoff')


def prerequisite():
    root = base.output_path(CLOSED_SOURCE)
    raw = base.files.file_bytes(root/'independent-closeout.json')
    require(sha256(raw).hexdigest() == CLOSED_SHA256, 'exact completed scheduled CPU closeout')
    value = base.files.parse(raw)
    require(value['source'] == CLOSED_SOURCE and value['whole_cpu_rescore_identical'] is True
            and value['cases_checked'] == 2 and value['decision'] == 'cpu-scheduled-diagonal-probe-passed'
            and all(value[k] is False for k in base.baseline.FALSE_FLAGS), 'non-admitting completed CPU prerequisite')
    require({p.name for p in root.iterdir()} == set(value['files_rehashed']) | {'independent-closeout.json'},
            'exact completed scheduled CPU inventory')
    for name, info in value['files_rehashed'].items():
        require(type(name) is str and '/' not in name and name not in ('.', '..')
                and (root/name).stat().st_size == info['bytes'] and base.host.digest(root/name) == info['sha256'],
                'unchanged completed scheduled CPU artifact '+str(name))
    _, cp = base.prerequisite()
    return dict(source=CLOSED_SOURCE, closeout_sha256=CLOSED_SHA256), cp


def run(source):
    started = time.monotonic(); check_window(launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only dose screen')
    props = {k: base.host.read('systemctl', '--user', 'show', service_name(source), '-p', k, '--value')
        for k in ('MainPID', 'ActiveState', 'RuntimeMaxUSec', 'MemoryMax', 'CPUQuotaPerSecUSec', 'Nice', 'KillMode')}
    require(props == dict(MainPID=str(os.getpid()), ActiveState='active', RuntimeMaxUSec='3min',
        MemoryMax=str(2*1024**3), CPUQuotaPerSecUSec='2s', Nice='10', KillMode='control-group'), 'exact capped dose-screen service')
    identity = base.host.identity(source); cpu = base.profile.checked_receipt()
    preserved = base.retained.d0.filmbrain_state(); services = base.protected_state()
    require(all(v == 'inactive' for v in services.values()), 'protected services inactive')
    previous, cp = prerequisite()
    declarations = [base.schedule.declaration(source, 'dose', 'held-out', [c]) for c in CELL_IDS]
    root = base.files.native._plain_path(output_path(source)); root.mkdir(exist_ok=False)
    report = dict(protocol=PROTOCOL, source=source, decision='cpu-dose-screen-failed', optimizer_steps=0,
                  **base.baseline.FALSE_FLAGS)
    try:
        from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
        base.seed_cpu(); env = ScheduledRecoveryRuntime(declarations[0], device='cpu'); compiled = env.binding
        launch = dict(protocol=PROTOCOL, source=source, source_identity=identity, cpu_math_profile=cpu,
            prerequisite=previous, declarations=declarations, compiled_plant=compiled,
            preserved_filmbrain=preserved, services_before=services, evaluation_seed=671,
            service_seconds=SERVICE_SECONDS, collection_seconds=COLLECTION_SECONDS, prefix_step=PREFIX_STEP,
            service_properties=props, optimizer_steps=0, **base.baseline.FALSE_FLAGS)
        base.files.write_json(root/'launch.json', launch); base.retained.write_capture(root/'checkpoint.pt', cp)
        scores=[]; prefixes=[]
        for index, declaration in enumerate(declarations):
            check_window()
            require(time.monotonic()-started < SERVICE_SECONDS-COLLECTION_SECONDS-CASE_CLOSEOUT_RESERVE,
                    'reserve next full CPU collection and retention')
            require(base.host.identity(source) == identity and base.retained.d0.filmbrain_state() == preserved
                    and base.protected_state() == services, 'unchanged dose-screen source and services')
            if index:
                base.seed_cpu(); env = ScheduledRecoveryRuntime(declaration, device='cpu')
            require(env.binding == compiled, 'identical fresh dose-screen compiled plant')
            case_started = time.monotonic()
            try:
                value = base.evidence.collect(env, cp, deadline_monotonic=case_started+COLLECTION_SECONDS)
            except Exception as exc:
                base.files.write_json(root/f'case-{index}-failure.json', dict(index=index,
                    error_type=type(exc).__name__, error=str(exc), current_case_trace_retained=False))
                raise
            wall = time.monotonic()-case_started
            raw = base.evidence.encode(value); digest = sha256(raw).hexdigest()
            base.retained.write_capture(root/f'case-{index}.pt', raw)
            base.files.write_json(root/f'case-{index}.json', dict(source=source, declaration=declaration,
                capture_sha256=digest, capture_bytes=len(raw), collection_wall_seconds=wall,
                collection=value['collection'], **base.baseline.FALSE_FLAGS))
            del env, value; gc.collect()
            score = base.evidence.verify(raw, digest, cp, declaration, compiled)
            value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            prefix = base.evidence.prefix_hash(value, PREFIX_STEP) if score['pulse']['checked_physics_steps'] > PREFIX_STEP else None
            del raw, value; gc.collect()
            base.files.write_json(root/f'case-{index}-replay.json', score); scores.append(score); prefixes.append(prefix)
            require(0 < wall < COLLECTION_SECONDS and 0 < score['collection']['elapsed_seconds'] < COLLECTION_SECONDS,
                    'over-budget dose trace retained and scored; no retry')
            require(score['numerical_diagnostic']['complete_first_attempt'], 'incomplete dose attempt retained; no retry')
        require(scores[0]['numerical_diagnostic']['candidate_pass'], 'failed zero control cannot establish a dose deficit')
        matched = prefixes[0] is not None and all(p == prefixes[0] for p in prefixes)
        delivered = all(s['pulse']['complete_pulse_delivery'] and s['pulse']['complete_phase_checks'] for s in scores)
        failed = [s['cell'] for s in scores if not s['numerical_diagnostic']['candidate_pass']]
        decision = ('cpu-dose-screen-inconclusive' if not matched or not delivered else
                    'frozen-dose-screen-no-deficit' if not failed else 'frozen-dose-screen-measured-deficit')
        require(base.host.identity(source) == identity and base.retained.d0.filmbrain_state() == preserved
                and base.protected_state() == services and not torch.cuda.is_initialized()
                and time.monotonic()-started < SERVICE_SECONDS, 'unchanged bounded CPU dose closeout')
        report.update(decision=decision, scores=scores, prefixes=prefixes, prefixes_identical=matched,
            complete_force_delivery=delivered, failed_cells=failed, genuine_cpu_first_attempts=3,
            filmbrain_unchanged=True, protected_services_inactive=True, cuda_initialized=False)
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        try:
            report['files'] = {p.name: base.host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        except Exception as error:
            report['inventory_error'] = str(error)
        base.files.write_json(root/'report.json', report)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--source', required=True)
    run(p.parse_args(argv).source)


if __name__ == '__main__':
    main()
