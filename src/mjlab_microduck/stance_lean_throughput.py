"""Lean-lesson throughput probe: size the watchdog from measurement.

Declared by ``docs/experiments/2026-09-16-stance-lean-lesson-throughput-probe.md``
under protocol ``football-b1n-lean-lesson-throughput-v1``.

This is a **timing probe, not a training run**. It exports no checkpoint, admits
nothing, and establishes no capability. It exists because the lean-lesson
predeclaration forbids sizing the watchdog and service cap from an estimate, and
because that estimate is biased low: the parent's 5.7497 s/update average includes
early updates whose episodes terminated at ~1.16 s, whereas a weight-initialized
policy runs near-full episodes from update 0.

One update costs two things on two devices, so both are measured separately on the
same host and reported separately:

* **Component A, collection** — CUDA0, 64 worlds, 24 ticks, real eager physics,
  driven by the *pinned frozen parent* policy. No optimizer is constructed, so
  this keeps the "never an optimizer" shape of the earlier source-bound probes.
  Using the parent policy rather than random actions is what makes episode lengths
  match the real run, which is precisely what the parent's average got wrong.
  Measured in the CUDA child.
* **Component B, optimizer** — the same PPO update on 64x24 samples, timed on the
  host CPU with CUDA hidden. Measured in the CPU-only supervisor, because the CUDA
  child cannot satisfy a CPU-only guard. Two development-host runs put this at
  0.1164 s mean / 0.1758 s max and 0.4565 s mean / 0.9801 s max respectively
  (2.024% and 7.939% of the parent's per-update total). The spread is real, which
  is why the cap uses the maximum plus a declared safety factor, never a mean.

A third development-host run changed the warm-up: its series began
``2.0236 0.4876 0.6353 ...`` — the first *measured* update still carried a 2.02 s
first-touch transient against a ~0.29 s steady state, so one discarded warm-up
update was not enough. Two are discarded, or the watchdog would be sized to a
one-off transient.
"""

import argparse
from copy import deepcopy
from hashlib import sha256
import io
import math
import os
import random
import statistics as st
import time

import numpy as np
import torch

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_cuda_probe as host
from mjlab_microduck import stance_lean_lesson as lean
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_training_smoke as smoke
from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.stance_ppo import STEPS

MODULE = 'mjlab_microduck.stance_lean_throughput'
PROTOCOL = 'football-b1n-lean-lesson-throughput-v1'
WORLDS, TICKS = lean.WORLDS, STEPS
WARMUP_UPDATES, MEASURED_UPDATES = 2, 8
TARGET_UPDATES = lean.UPDATES
SAFETY = 1.25
WATCHDOG_MARGIN_SECONDS = 60
CLOSEOUT_SECONDS = 600
MAX_WINDOW_SECONDS = 3600
# The probe's own caps. Declared generously and fixed: the probe is short, and
# these are not the numbers it exists to produce. 900/960 reuses the proven
# eager-learning pair and the `supervised_stance_smoke` wrapper, whose 900 s bound
# is fixed. `supervised_process` cannot be used here: it enforces
# `timeout <= CELL_SECONDS` (120), which is far too short for a nine-update probe.
PROBE_CHILD_SECONDS, PROBE_SERVICE_SECONDS = 900, 960
# Recorded from the parent eager run, used only to show the estimate being replaced.
PARENT_SECONDS_PER_UPDATE = 735.963/128


def parent_path():
    """The frozen parent export on this host; read-only input, never written."""
    return (host.ROOT/'artifacts/evaluations'
            /('stance-eager-learning-'+checkpoint.LEAN_PARENT_SOURCE[:12])
            /checkpoint.LEAN_PARENT_FILE)


def quantile(ordered, q):
    """Nearest-rank, declared: no interpolation is invented between samples."""
    require(len(ordered) > 0, 'nonempty sample for a quantile')
    return ordered[max(1, math.ceil(q*len(ordered)))-1]


def summarize(series):
    require(type(series) is list and len(series) >= 2, 'at least two measured samples')
    require(all(type(v) is float and math.isfinite(v) and v > 0 for v in series),
            'finite positive timing series')
    ordered = sorted(series)
    return dict(count=len(series), mean=st.mean(series), median=st.median(ordered),
                p95=quantile(ordered, .95), max=ordered[-1], series=list(series))


def caps(collection, optimizer, setup_seconds):
    """Map the measured worst case onto the lean-lesson child and service caps."""
    keys = {'count', 'mean', 'median', 'p95', 'max', 'series'}
    require(set(collection) == keys and set(optimizer) == keys, 'summarized components')
    require(type(setup_seconds) is float and math.isfinite(setup_seconds) and setup_seconds >= 0,
            'measured setup seconds')
    per_update = collection['max']+optimizer['max']
    predicted = math.ceil(TARGET_UPDATES*per_update)+math.ceil(setup_seconds)
    service = math.ceil(SAFETY*predicted)
    child = service-WATCHDOG_MARGIN_SECONDS
    require(child > 0 and child < service, 'child watchdog strictly inside the service cap')
    return dict(target_updates=TARGET_UPDATES, per_update_worst_seconds=per_update,
        collection_worst_seconds=collection['max'], optimizer_worst_seconds=optimizer['max'],
        setup_seconds=setup_seconds, predicted_child_seconds=predicted, safety_factor=SAFETY,
        service_seconds=service, child_seconds=child, watchdog_margin_seconds=WATCHDOG_MARGIN_SECONDS,
        closeout_seconds=CLOSEOUT_SECONDS,
        parent_estimate_seconds=math.ceil(PARENT_SECONDS_PER_UPDATE*TARGET_UPDATES))


def check_window(deadline, *, launching=False):
    """A fresh absolute window. Expired authority is never a window."""
    require(type(deadline) is int, 'explicit integer deadline')
    now = time.time()
    require(math.isfinite(now), 'finite clock')
    remaining = deadline-now
    require(remaining > 0, 'a window in the future; expired authority is not a window')
    require(remaining <= MAX_WINDOW_SECONDS, 'one bounded job inside 60 minutes')
    if launching:
        require(remaining > PROBE_SERVICE_SECONDS+CLOSEOUT_SECONDS+WATCHDOG_MARGIN_SECONDS,
                'measured probe needs a fresh 27-to-60-minute window')


class RolloutStandIn:
    """Timing harness for component B. Not a physics plant, and never claimed as one.

    The optimizer's cost is fixed by the rollout tensor shapes, not by the plant,
    so a stand-in fills the same 64x24 buffer without touching CUDA. It is used
    only to time ``LeanStanceLearner.update`` and is never evidence about physics.
    """

    n = WORLDS
    device = torch.device('cpu')

    def __init__(self):
        self.live = torch.ones(self.n, dtype=torch.bool)
        self.tick = 0
        self.obs = torch.zeros(self.n, 50)

    def observations(self):
        return dict(actor=self.obs[:, :44].clone(), critic=self.obs.clone())

    def step(self, action):
        self.tick += 1
        self.obs[:, 0] = self.tick/10000
        self.obs[:, 1] = float(action.abs().mean())
        terminated = torch.zeros(self.n, dtype=torch.bool)
        timed_out = torch.zeros(self.n, dtype=torch.bool)
        self.live = ~(terminated | timed_out)
        self.records = [None]*self.n
        return dict(observation=self.observations(), reward=torch.full((self.n,), .03),
            terminated=terminated, timed_out=timed_out, live=self.live.clone(),
            executed_steps=torch.full((self.n,), 10, dtype=torch.long),
            terminal_records=deepcopy(self.records), boundaries=[{'synthetic': True}])

    def reset(self, rows):
        self.live[rows] = True
        return deepcopy(self.records)


def measure_collection(env, actor):
    """Real CUDA0 collection under the frozen parent policy; no optimizer."""
    require(isinstance(actor, torch.nn.Module) and not actor.training, 'frozen parent policy')
    bridge = smoke.PhysicsBridge(env)
    started = time.monotonic()
    bridge.reset(torch.ones(WORLDS, dtype=torch.bool))
    setup_seconds = time.monotonic()-started
    updates = []
    for update in range(WARMUP_UPDATES+MEASURED_UPDATES):
        tick_seconds = []
        update_started = time.monotonic()
        for _ in range(TICKS):
            obs = bridge.observations()
            action = checkpoint.infer(actor, obs['actor'])
            tick_started = time.monotonic()
            bridge.step(action)
            tick_seconds.append(time.monotonic()-tick_started)
            if not bridge.live.all():
                bridge.reset(~bridge.live)
        updates.append(dict(update=update, warmup=update < WARMUP_UPDATES,
            seconds=time.monotonic()-update_started, tick_seconds=tick_seconds,
            tick_max_seconds=max(tick_seconds)))
    measured = [u['seconds'] for u in updates if not u['warmup']]
    require(len(measured) == MEASURED_UPDATES, 'declared measured updates')
    return dict(updates=updates, setup_seconds=setup_seconds, summarize=summarize(measured),
        optimizer_steps=0, physics_device=str(env.device), ticks_per_update=TICKS, worlds=WORLDS)


def measure_optimizer(parent):
    """Component B in the CPU-only supervisor. CUDA must be hidden and uninitialized."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only optimizer measurement')
    learner = lean.LeanStanceLearner(parent)
    env = RolloutStandIn()
    seconds = []
    for update in range(WARMUP_UPDATES+MEASURED_UPDATES):
        for _ in range(TICKS):
            learner.collect_one(env)
        started = time.monotonic()
        learner.update()
        elapsed = time.monotonic()-started
        if update >= WARMUP_UPDATES:
            seconds.append(elapsed)
    require(len(seconds) == MEASURED_UPDATES, 'declared measured optimizer updates')
    return dict(summarize=summarize(seconds), samples=TICKS*WORLDS, device='cpu',
        stand_in=True, physics_evidence=False, optimizer_steps=MEASURED_UPDATES)


def decide(collection, optimizer):
    derived = caps(collection['summarize'], optimizer['summarize'], collection['setup_seconds'])
    return dict(protocol=PROTOCOL, collection=collection, optimizer=optimizer, caps=derived,
        decision='throughput-measured-not-a-capability', training_admitted=False,
        checkpoint_admitted=False, learned_stance=False, physical_motion_authorized=False)


def output_path(source):
    host.supervisor.hex_id(source, 40)
    return host.ROOT/'artifacts/evaluations'/('stance-lean-throughput-'+source[:12])


def plan(source, inputs, deadline):
    require(type(deadline) is int and deadline > 0, 'explicit integer deadline')
    return dict(protocol=PROTOCOL, source=source, inputs=inputs, worlds=WORLDS, ticks=TICKS,
        warmup_updates=WARMUP_UPDATES, measured_updates=MEASURED_UPDATES,
        target_updates=TARGET_UPDATES, safety_factor=SAFETY,
        watchdog_margin_seconds=WATCHDOG_MARGIN_SECONDS, closeout_seconds=CLOSEOUT_SECONDS,
        parent_checkpoint_sha256=checkpoint.LEAN_PARENT_SHA256,
        parent_iteration=checkpoint.LEAN_PARENT_ITERATION, parent_file=checkpoint.LEAN_PARENT_FILE,
        child_seconds=PROBE_CHILD_SECONDS, service_seconds=PROBE_SERVICE_SECONDS,
        deadline_unix=deadline, optimizer_steps=0, checkpoint_admitted=False,
        training_admitted=False, learned_stance=False, physical_motion_authorized=False)


def prepare(source, deadline):
    """CPU-only preparation. Copies the pinned parent in as a read-only input."""
    check_window(deadline, launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only probe preparation')
    inputs = host.identity(source)
    raw = host.supervisor.file_bytes(parent_path(), limit=checkpoint.LIMIT)
    require(sha256(raw).hexdigest() == checkpoint.LEAN_PARENT_SHA256,
            'pinned lean-lesson parent export')
    root = host.supervisor.native._plain_path(output_path(source))
    root.mkdir(exist_ok=False)
    smoke.write_bytes(root/'parent.pt', raw)
    host.supervisor.write_json(root/'launch.json', plan(source, inputs, deadline))
    return dict(output=str(root), launch_sha256=host.digest(root/'launch.json'))


def checked(source, launch_sha):
    root = output_path(source)
    require(host.digest(root/'launch.json') == launch_sha, 'independent probe launch hash')
    value = host.supervisor.parse(host.supervisor.file_bytes(root/'launch.json'))
    require(value == plan(source, host.identity(source), value['deadline_unix']),
            'exact probe source/inputs/plan')
    return value


def load_parent(root):
    """Read-only load of the copied parent, hash checked before deserialization."""
    raw = host.supervisor.file_bytes(root/'parent.pt', limit=checkpoint.LIMIT)
    digest = sha256(raw).hexdigest()
    require(digest == checkpoint.LEAN_PARENT_SHA256, 'pinned lean-lesson parent export')
    identity = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)['identity']
    return checkpoint.load_lean_parent(raw, digest, identity)


def child(source, launch_sha, fd):
    smoke.inherited_lease(fd)
    launch = checked(source, launch_sha)
    check_window(launch['deadline_unix'], launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.is_available(),
            'explicit CUDA0 probe child')
    host.wait_idle()
    random.seed(523)
    np.random.seed(523)
    torch.manual_seed(523)
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
    env = WarpStanceRuntime(WORLDS, device='cuda:0')
    require(str(env.device) == str(env.wp_device) == 'cuda:0' and env.wp_device.is_cuda
            and env.forward_graph is None, 'actual probe CUDA physics; no graph')
    root = output_path(source)
    require(plant.describe(env.native) == plant.reference(), 'actual probe plant')
    before = sha256(host.supervisor.file_bytes(root/'parent.pt', limit=checkpoint.LIMIT)).hexdigest()
    parent = load_parent(root)
    collection = measure_collection(env, parent['actor'].eval())
    after = sha256(host.supervisor.file_bytes(root/'parent.pt', limit=checkpoint.LIMIT)).hexdigest()
    require(before == after == checkpoint.LEAN_PARENT_SHA256,
            'parent archive byte-identical after collection')
    require(env.forward_graph is None, 'no forward graph during the probe')
    host.supervisor.write_json(root/'collection.json', collection)


SERVICE_PROPERTIES = ('MainPID', 'RuntimeMaxUSec', 'KillMode', 'ActiveState')
# systemd's own rendering of `RuntimeMaxSec=960`. Pinned by test so the constant
# cannot silently drift away from PROBE_SERVICE_SECONDS.
SERVICE_RUNTIME_MAX = '16min'


def service_unit(source):
    return 'microduck-lean-throughput-'+source[:12]+'.service'


def service_state(unit):
    """Read the four properties as *values*.

    ``-p KEY --value`` is the repo-wide form, used identically by
    ``stance_forward_probe``, ``stance_throughput_probe`` and
    ``stance_training_smoke``. Without ``--value`` systemctl prints
    ``KEY=value``; comparing those raw strings to bare values is what made the
    first launch fail its own self-check while every property was in fact
    correct.
    """
    return {k: host.read('systemctl', '--user', 'show', unit, '-p', k, '--value')
            for k in SERVICE_PROPERTIES}


def check_service(source):
    """The unit must be the one systemd is timing, and this process must be it."""
    actual = service_state(service_unit(source))
    require(actual == dict(MainPID=str(os.getpid()), RuntimeMaxUSec=SERVICE_RUNTIME_MAX,
                           KillMode='control-group', ActiveState='active'),
            'independently timed probe service')
    return service_unit(source)


def supervise(source, launch_sha):
    launch = checked(source, launch_sha)
    check_window(launch['deadline_unix'], launching=True)
    root = output_path(source)
    check_service(source)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only probe supervisor')
    require({p.name for p in root.iterdir()} == {'launch.json', 'parent.pt'},
            'one fresh probe attempt')
    report = dict(protocol=PROTOCOL, launch_sha256=launch_sha, decision='failed',
        training_admitted=False, checkpoint_admitted=False)
    try:
        with host.supervisor.gpu_lease() as fd:
            report['idle_before'] = host.wait_idle()
            def guard():
                require(time.time()+CLOSEOUT_SECONDS < launch['deadline_unix'], 'closeout boundary')
                host.check_log(root/'child.log')
                require(host.identity(source) == launch['inputs'], 'live probe source drift')
            report['child'] = host.supervisor.supervised_stance_smoke(
                [str(host.ROOT/'.venv/bin/python'), '-m', MODULE, 'child', '--source', source,
                 '--launch-sha256', launch_sha, '--lock-fd', str(fd)], root/'child.log',
                cwd=host.ROOT, env=host.supervisor.child_environment(), lock_fd=fd, guard=guard)
            host.check_log(root/'child.log')
            collection = host.supervisor.parse(host.supervisor.file_bytes(root/'collection.json'))
            # Component B runs here, in the CPU-only supervisor, not in the CUDA child.
            optimizer = measure_optimizer(load_parent(root))
            host.supervisor.write_json(root/'optimizer.json', optimizer)
            decision = decide(collection, optimizer)
            host.supervisor.write_json(root/'decision.json', decision)
            report['result'] = decision
            report['decision'] = decision['decision']
            report['caps'] = decision['caps']
            report['idle_after'] = host.wait_idle()
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc),
                      error_notes=getattr(exc, '__notes__', []))
        raise
    finally:
        report['files'] = {p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        host.supervisor.write_json(root/'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'supervise', 'child'))
    parser.add_argument('--source', required=True)
    parser.add_argument('--deadline-unix', type=int)
    parser.add_argument('--launch-sha256')
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        print(canonical(prepare(args.source, args.deadline_unix)))
    elif args.mode == 'supervise':
        supervise(args.source, args.launch_sha256)
    else:
        child(args.source, args.launch_sha256, args.lock_fd)


if __name__ == '__main__':
    main()
