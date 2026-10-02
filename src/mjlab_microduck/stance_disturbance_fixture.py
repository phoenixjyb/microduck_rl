"""Separate one-Euler force-path fixture, not nominal stance or learned recovery.

Owns its own model/data. It never instantiates, subclasses or changes the
nominal WarpStanceRuntime, and has no actor, reward, reset or optimizer loop.
"""
from dataclasses import replace
import gc

import mjlab  # Task registration before the BAM-dependent adapters.
import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.sim.randomization import expand_model_fields
from mjlab.sim.sim_data import WarpBridge

from mjlab_microduck import stance_disturbance_contract as contract
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_forward_probe as forward_evidence
from mjlab_microduck.stance_contact_evidence import read_contacts
from mjlab_microduck.stance_control_state import BamStateCommit
from mjlab_microduck.stance_lesson_contract import JOINTS
from mjlab_microduck.stance_warp_integrator import EulerCandidateCommit
from mjlab_microduck.stance_warp_runtime import build_entity
from mjlab_microduck.football_flat_hold import place_on_floor
from mjlab_microduck.first_attempt_smoke import require

INPUT_FIELDS = ('ctrl', 'xfrc_applied', 'qfrc_applied')
PHASES = ('before', 'forced_pre', 'integrated', 'unforced_post')


def compiled_binding(model):
    return dict(selected_plant=plant.describe(model), nbody=model.nbody,
        body_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ''
                    for i in range(model.nbody)],
        body_name=contract.BODY_NAME, body_id=int(model.body(contract.BODY_NAME).id))


class SingleSubstepFixture:
    """One fresh, non-resettable physical input; any error permanently faults it."""
    def __init__(self, device='cpu'):
        require(device in ('cpu', 'cuda:0'), 'explicit CPU or sole CUDA0 fixture')
        self.n = 1
        self.device = torch.device(device)
        self.wp_device = wp.get_device(device)
        self.faulted = False
        self.used = False
        self.entity = build_entity()
        self.native = self.entity.compile()
        require((self.native.nq, self.native.nv, self.native.nu) == (21, 20, 14),
                'exact rigid full-robot topology')
        self.binding = compiled_binding(self.native)
        contract.plan('0'*40, self.binding)  # Validate resolved body table, not a guessed id.
        nd = mujoco.MjData(self.native)
        mujoco.mj_resetDataKeyframe(self.native, nd, self.native.key('init_state').id)
        nd.qpos[:7] = [0, 0, .2, 1, 0, 0, 0]
        nd.qvel[:] = 0
        nd.ctrl[:] = 0
        place_on_floor(self.native, nd)
        require(not nd.warning.number.any(), 'native fixture placement warning')
        with wp.ScopedDevice(self.wp_device):
            self.model = mjwarp.put_model(self.native)
            self.data = mjwarp.put_data(self.native, nd, nworld=1, nconmax=128, njmax=512)
            expand_model_fields(self.model, 1, ['dof_frictionloss', 'dof_damping'])
        require(not self.model.is_sparse and self.data.naccdmax == self.data.naconmax,
                'audited dense solver and full CCD capacity')
        self.entity.actuators[0].cfg = replace(self.entity.actuators[0].cfg,
            vin_range=(7.5, 7.5), vin_drop_gain_range=(.1, .1))
        self.entity.initialize(self.native, WarpBridge(self.model, nworld=1),
                               WarpBridge(self.data), device)
        require(len(self.entity.actuators) == 1, 'one owned motor group')
        self.actuator = self.entity.actuators[0]
        self.motor = BamStateCommit(self.actuator)
        live = torch.ones(1, dtype=torch.bool, device=self.device)
        self.motor.reset(live)
        qids = [int(self.native.joint(name).qposadr[0]) for name in JOINTS]
        pos = self._view('qpos')[:, qids]
        vel = self._view('qvel')[:, self.actuator._dof_ids]
        zeros = torch.zeros_like(pos)
        proposal = self.motor.compute(ActuatorCmd(pos.clone(), zeros, zeros, pos, vel), live)
        require(proposal['accepted'].all() and not proposal['rejected'].any()
                and not proposal['torque_nm'].any(), 'single zero-error BAM preparation')
        self._view('ctrl')[:, self.actuator._global_ctrl_ids] = proposal['torque_nm']
        self.frozen_ctrl = self._view('ctrl').detach().clone()
        self.frozen_fields = {k: v.detach().clone() for k, v in self.motor.fields.items()}
        self.integrator = EulerCandidateCommit(self.model, self.data)
        self._solve('zero-wrench')

    def _view(self, name):
        return wp.to_torch(getattr(self.data, name))

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        wp.synchronize_device(self.wp_device)

    def _applied(self, case):
        self._sync()
        contract.validate_applied(self._view('xfrc_applied').cpu().tolist(),
            self._view('qfrc_applied').cpu().tolist(), case,
            self.binding['nbody'], self.binding['body_id'])

    def _solve(self, case):
        self._applied(case)
        require(torch.equal(self._view('ctrl'), self.frozen_ctrl)
                and all(torch.equal(v, self.frozen_fields[k]) for k, v in self.motor.fields.items()),
                'unchanged prepared motor inputs before every solve')
        with wp.ScopedDevice(self.wp_device):
            mjwarp.forward(self.model, self.data)
        self._sync()
        self._applied(case)
        self.contacts = read_contacts(self.model, self.data)
        forward_evidence.output(self)  # Full finite/capacity/constraint/contact checks.

    def capture(self):
        self._sync()
        return dict(solved=forward_evidence.output(self),
            inputs={k: self._view(k).detach().cpu().clone() for k in INPUT_FIELDS},
            motor_fields={k: v.detach().cpu().clone() for k, v in self.motor.fields.items()})

    @torch.no_grad()
    def run(self, case):
        require(not self.faulted and not self.used, 'fresh nonfaulted one-use force fixture')
        self.used = True
        try:
            self._applied('zero-wrench')
            before = self.capture()
            xfrc, _ = contract.expected_wrench(case, self.binding['nbody'], self.binding['body_id'])
            self._view('xfrc_applied').copy_(torch.tensor(xfrc, dtype=torch.float32, device=self.device))
            self._solve(case)
            pre = self.capture()
            self.integrator.integrate(torch.ones(1, dtype=torch.bool, device=self.device))
            self._applied(case)
            integrated = self.capture()
            require(torch.equal(integrated['solved']['kinematics']['time'],
                                before['solved']['kinematics']['time']+contract.DT),
                    'exactly one Euler physical clock increment')
            # The pulse must NOT survive to the post-forward or any later solve.
            self._view('xfrc_applied').zero_()
            self._view('qfrc_applied').zero_()
            self._solve('zero-wrench')
            post = self.capture()
            for field in ('qpos', 'qvel', 'time', 'qacc_warmstart'):
                require(torch.equal(integrated['solved']['kinematics'][field],
                                    post['solved']['kinematics'][field]),
                        'post-forward cannot integrate '+field)
            require(all(torch.equal(before['inputs']['ctrl'], phase['inputs']['ctrl'])
                        and all(torch.equal(before['motor_fields'][k], phase['motor_fields'][k])
                                for k in before['motor_fields'])
                        for phase in (pre, integrated, post)), 'frozen matched motor inputs')
            return dict(case=case, before=before, forced_pre=pre, integrated=integrated,
                        unforced_post=post)
        except Exception:
            self.faulted = True
            raise


def capture_cases(source, device='cpu'):
    """Five fresh one-world inputs; no policy checkpoint or learned claim."""
    rows = []
    binding = None
    for name, _ in contract.CASES:
        torch.manual_seed(619)
        env = SingleSubstepFixture(device)
        require(binding is None or env.binding == binding, 'same compiled force plant')
        binding = env.binding
        rows.append(env.run(name))
        del env
        gc.collect()
    baseline = rows[0]['before']
    require(all(exact_tree(baseline, r['before']) for r in rows), 'identical prepared case inputs')
    for row in rows[1:]:
        delta = (row['forced_pre']['solved']['dynamics']['qacc']-
                 rows[0]['forced_pre']['solved']['dynamics']['qacc']).abs().max()
        require(float(delta) > 1e-5, 'declared force must affect solved acceleration')
    return dict(plan=contract.plan(source, binding), rows=rows,
        backend=dict(torch_device=str(device), warp_is_cuda=device == 'cuda:0'),
        **contract.NO_ADMISSION)


def exact_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact_tree(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(exact_tree(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def replay(value):
    """Fresh CPU model/solve/Euler replay, not merely a recorded-state scorer."""
    require(type(value) is dict and set(value) == {'plan', 'rows', 'backend'} | set(contract.NO_ADMISSION),
            'exact force fixture payload schema')
    require(value['backend'] in (dict(torch_device='cpu', warp_is_cuda=False),
                                dict(torch_device='cuda:0', warp_is_cuda=True)),
            'explicit force fixture backend')
    require(value['plan'] == contract.plan(value['plan']['source'], value['plan']['plant']),
            'exact force declaration')
    require(all(value[k] is False for k in contract.NO_ADMISSION),
            'non-admission flags')
    reference = capture_cases(value['plan']['source'], 'cpu')
    require(value['plan'] == reference['plan'], 'fresh compiled source/plant binding')
    require(len(value['rows']) == 5, 'complete five-case fixture')
    errors = {}
    def compare(a, b, path):
        require(type(a) is type(b), 'replay type '+path)
        if isinstance(a, torch.Tensor):
            require(a.device.type == 'cpu' and a.dtype == b.dtype and a.shape == b.shape,
                    'replay tensor layout '+path)
            if a.is_floating_point():
                require(torch.isfinite(a).all() and torch.allclose(a, b,
                    atol=contract.CHECK_TOL_ATOL, rtol=contract.CHECK_TOL_RTOL),
                    'independent force solve replay '+path)
                errors[path] = float((a.double()-b.double()).abs().max()) if a.numel() else 0.
            else:
                require(torch.equal(a, b), 'replay integer '+path)
        elif isinstance(a, dict):
            require(a.keys() == b.keys(), 'replay keys '+path)
            for k in a: compare(a[k], b[k], path+'/'+k)
        elif isinstance(a, list):
            require(len(a) == len(b), 'replay length '+path)
            for i, (x, y) in enumerate(zip(a, b)): compare(x, y, path+'/'+str(i))
        else:
            require(a == b, 'replay value '+path)
    compare(value['rows'], reference['rows'], 'rows')
    # External inputs are exact, even where solved float32 fields use tolerance.
    for row in value['rows']:
        for phase in PHASES:
            inputs = row[phase]['inputs']
            expected = row['case'] if phase in ('forced_pre', 'integrated') else 'zero-wrench'
            contract.validate_applied(inputs['xfrc_applied'].tolist(), inputs['qfrc_applied'].tolist(),
                expected, reference['plan']['plant']['nbody'], reference['plan']['plant']['body_id'])
    return dict(protocol=contract.PROTOCOL, force_path_replayed=True, cases=5, euler_steps=5,
        maximum_solved_field_error=max(errors.values(), default=0.),
        atol=contract.CHECK_TOL_ATOL, rtol=contract.CHECK_TOL_RTOL,
        cuda_initialized=torch.cuda.is_initialized(), **contract.NO_ADMISSION)
