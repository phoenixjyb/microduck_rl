"""Caller-owned B1-N Warp loop; not a registered PPO task or launch admission.

Entity/BAM initialization is real. Physics uses raw Warp arrays with explicit
Torch/Warp synchronization and eager forward by default. An opt-in captured
forward candidate preserves all checks; no installed-library changes are made.
The CPU device is the integration audit backend; CUDA needs its own retained test.
"""

from copy import deepcopy
from dataclasses import fields, replace

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.sim.randomization import expand_model_fields
from mjlab.sim.sim_data import WarpBridge

from mjlab_microduck.football_contact_fixture import ROBOT_XML, FEET
from mjlab_microduck.football_flat_hold import place_on_floor
from mjlab_microduck.stance_lesson_contract import JOINTS
from mjlab_microduck.stance_control_state import StanceActionDelay, BamStateCommit, mask_check
from mjlab_microduck.stance_contact_evidence import read_contacts, contact_summary, FirstTerminalContacts
from mjlab_microduck.stance_transition import PhysicsState, StanceTransition, DECIMATION, physical_failures
from mjlab_microduck.stance_warp_integrator import EulerCandidateCommit


def build_entity():
    from mjlab_microduck.robot.microduck_constants import HOME_FRAME, actuators

    def spec_fn():
        spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
        spec.worldbody.add_geom(name='hold_floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[2, 2, .1], friction=[.6, 0, 0], condim=3)
        spec.option.timestep = .002
        spec.option.gravity = [0, 0, -9.81]
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
        spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
        spec.option.iterations = 100
        spec.option.tolerance = 1e-8
        return spec

    # The owned FIFO supplies the single fixed three-step delay. Do not also
    # allocate/call the stock randomized get_command delay or double the delay.
    motor = replace(deepcopy(actuators), delay_min_lag=0, delay_max_lag=0,
                    target_names_expr=JOINTS)
    return EntityCfg(spec_fn=spec_fn, init_state=deepcopy(HOME_FRAME),
        articulation=EntityArticulationInfoCfg(actuators=(motor,),
                                              soft_joint_pos_limit_factor=.9)).build()


class WarpStanceRuntime:
    """Explicit first-terminal stepping/reset, with no auto-reset or optimizer.

    Owns all arrays and the Entity exclusively. Callers must not mutate them.
    Each tick returns owned boundary snapshots; callers retain full evaluation
    trajectories separately. reset() returns prior terminals before clearing
    selected rows. No CUDA/service/lease or checkpoint admission is implied.
    """

    def __init__(self, nworld=2, *, device='cpu'):
        if type(nworld) is not int or not 1 <= nworld <= 512:
            raise ValueError('one to 512 declared stance worlds required')
        self.n = nworld; self.device = torch.device(device)
        self.wp_device = wp.get_device(device)
        self.faulted = False
        self.forward_graph = None
        self.entity = build_entity()
        self.native = self.entity.compile()
        nd = mujoco.MjData(self.native)
        mujoco.mj_resetDataKeyframe(self.native, nd, self.native.key('init_state').id)
        nd.qpos[:7] = [0, 0, .2, 1, 0, 0, 0]
        nd.qvel[:] = 0; nd.ctrl[:] = 0
        self.initial_gaps = place_on_floor(self.native, nd)
        if nd.warning.number.any(): raise ValueError('native placement warning')
        if (self.native.nq, self.native.nv, self.native.nu) != (21, 20, 14):
            raise ValueError('exact rigid stance topology required')
        self.qids = torch.tensor([int(self.native.joint(n).qposadr[0]) for n in JOINTS], device=self.device)
        self.dofs = torch.tensor([int(self.native.joint(n).dofadr[0]) for n in JOINTS], device=self.device)
        if self.native.joint('trunk_base_freejoint').qposadr[0] != 0 or self.native.joint('trunk_base_freejoint').dofadr[0] != 0:
            raise ValueError('free-root address mismatch')
        self.floor = int(self.native.geom('hold_floor').id)
        self.feet = tuple(int(self.native.geom(n).id) for n in FEET)
        self.ranges = torch.tensor(self.native.jnt_range[[self.native.joint(n).id for n in JOINTS]],
                                   dtype=torch.float32, device=self.device)
        self.initial_qpos = torch.tensor(nd.qpos.copy(), dtype=torch.float32, device=self.device)
        self.nominal = self.initial_qpos[self.qids].clone()
        with wp.ScopedDevice(self.wp_device):
            self.model = mjwarp.put_model(self.native)
            # naccdmax defaults to the full shared contact capacity, not a smaller
            # CCD pool. Every accepted broadphase count must fit that same pool.
            self.data = mjwarp.put_data(self.native, nd, nworld=nworld, nconmax=128, njmax=512)
            expand_model_fields(self.model, nworld, ['dof_frictionloss', 'dof_damping'])
        if self.model.is_sparse: raise ValueError('this stance binding requires the audited dense solver')
        self.model_bridge = WarpBridge(self.model, nworld=nworld)
        self.data_bridge = WarpBridge(self.data)
        # Compile with the historical 8.2V ceiling, exactly as the native hold.
        # Fix only initialization values afterward, before normal initialize():
        # no random mechanics, tensor patching or change to the compiled ceiling.
        self.entity.actuators[0].cfg = replace(self.entity.actuators[0].cfg,
            vin_range=(7.5, 7.5), vin_drop_gain_range=(.1, .1))
        self.entity.initialize(self.native, self.model_bridge, self.data_bridge, device)
        if len(self.entity.actuators) != 1: raise ValueError('one owned BAM group required')
        self.actuator = self.entity.actuators[0]
        if tuple(self.actuator.target_names) != JOINTS or self.actuator.has_delay:
            raise ValueError('exact joint order and one caller-owned delay required')
        self.ctrl_ids = self.actuator._global_ctrl_ids.clone()
        if self.ctrl_ids.unique().numel() != 14: raise ValueError('unique stance controls')
        if self.data.naccdmax != self.data.naconmax: raise ValueError('full CCD capacity required')
        self.motor = BamStateCommit(self.actuator)
        self.delay = StanceActionDelay(self.nominal, self.ranges, nworld)
        self.integrator = EulerCandidateCommit(self.model, self.data)
        self.steps = torch.zeros(nworld, dtype=torch.long, device=self.device)
        self.live = torch.ones(nworld, dtype=torch.bool, device=self.device)
        self.terminal = [None]*nworld
        self.state = None; self._observations = None
        self.reset(self.live.clone())

    def _view(self, name):
        return wp.to_torch(getattr(self.data, name))

    def _sync(self):
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        wp.synchronize_device(self.wp_device)

    def _healthy(self):
        if self.faulted: raise RuntimeError('faulted stance runtime requires job closeout')

    def enable_forward_graph(self):
        """Explicit candidate path; default eager physics and all checks remain."""
        self._healthy()
        if self.forward_graph is not None: raise ValueError('forward graph already bound')
        try:
            from mjlab_microduck.stance_forward_graph import ForwardGraph
            self._sync()
            self.forward_graph = ForwardGraph(self.model, self.data, self.wp_device)
            self._sync()
        except Exception:
            self.faulted = True
            raise

    def _forward(self):
        self._sync()
        if self.forward_graph is None:
            with wp.ScopedDevice(self.wp_device): mjwarp.forward(self.model, self.data)
        else:
            self.forward_graph.run(self.model, self.data)
        self._sync()
        for name in ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'time', 'ctrl',
                     'qfrc_bias', 'qfrc_constraint', 'qfrc_actuator', 'cvel', 'xquat'):
            if not torch.isfinite(self._view(name)).all():
                raise ValueError('nonfinite solved stance field: '+name)
        if self._view('qfrc_applied').any() or self._view('xfrc_applied').any():
            raise ValueError('B1-N does not permit external assistance or pushes')
        self.contacts = read_contacts(self.model, self.data)
        # BAM's scatter uses DOF-friction IDs; validate before any BAM call.
        nefc = self._view('nefc')
        types = wp.to_torch(self.data.efc.type); ids = wp.to_torch(self.data.efc.id)
        active = torch.arange(ids.shape[1], device=self.device)[None] < nefc[:, None]
        friction = active & (types == int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF))
        if (friction & ((ids < 0) | (ids >= self.native.nv))).any():
            raise ValueError('invalid active friction DOF address')

    def _refresh(self, rows):
        """Cache only changed worlds; never replace a closed world's observation."""
        support, forbidden = contact_summary(self.contacts, self.n, floor_id=self.floor, foot_ids=self.feet)
        gravity = self.entity.data.projected_gravity_b
        angular = self.entity.data.root_link_ang_vel_b
        if not torch.isfinite(gravity).all() or not torch.allclose(
                torch.linalg.vector_norm(gravity, dim=1), torch.ones(self.n, device=self.device), atol=1e-5, rtol=0):
            raise ValueError('invalid projected gravity')
        q = self._view('qpos')[:, self.qids]; vel = self._view('qvel')
        hard = ((q < self.ranges[:, 0]) | (q > self.ranges[:, 1])).any(1)
        state = PhysicsState(torch.acos((-gravity[:, 2]).clamp(-1, 1)), vel[:, :3].clone(),
            self._view('qpos')[:, 2].clone(), support, self._view('qfrc_actuator')[:, self.dofs].clone(),
            vel[:, self.dofs].clone(), hard, forbidden, torch.zeros_like(rows))
        # Warp has no native warning.number array. Finite/capacity/ID checks are
        # explicit above; numerical GPU log supervision remains a launch gate.
        state.validate(self.n, self.device)
        actor = torch.cat((gravity, angular/5., (q-self.nominal)/.5,
                           state.joint_velocity/10., self.delay.correction/.2), dim=1)
        critic = torch.cat((actor, state.root_velocity, (state.height[:, None]-.12)/.1, support/10.), dim=1)
        if not torch.isfinite(critic).all(): raise ValueError('nonfinite stance observation')
        if self.state is None:
            self.state = PhysicsState(**{f.name: getattr(state, f.name).clone() for f in fields(state)})
            self._observations = dict(actor=actor.clone(), critic=critic.clone())
        else:
            for f in fields(state): getattr(self.state, f.name)[rows] = getattr(state, f.name)[rows]
            self._observations['actor'][rows] = actor[rows]
            self._observations['critic'][rows] = critic[rows]

    def observations(self):
        self._healthy()
        return {k: v.clone() for k, v in self._observations.items()}

    def snapshot(self):
        self._healthy()
        q = self._view('qpos')[:, self.qids]
        soft = (q-self.ranges.mean(1)).abs() > .45*(self.ranges[:, 1]-self.ranges[:, 0])
        return dict(physics_steps=self.steps.clone(), qpos=self._view('qpos').clone(),
            qvel=self._view('qvel').clone(), soft_limit_mask=soft,
            state=PhysicsState(**{f.name: getattr(self.state, f.name).clone() for f in fields(self.state)}),
            observation=self.observations())

    def _capture(self, tick, proposed=None):
        new = (tick.terminated | tick.timed_out) & torch.tensor(
            [r is None for r in self.terminal], device=self.device)
        if not new.any(): return
        ledger = FirstTerminalContacts(self.n)
        ledger.capture(state=self.state, steps=tick.episode_steps,
            qpos=self._view('qpos'), qvel=self._view('qvel'), contacts=self.contacts,
            terminated=new & tick.terminated, timed_out=new & tick.timed_out, proposed_torque=proposed)
        for i, record in enumerate(ledger.records()):
            if record is not None:
                record['observation'] = {k: v[i].detach().cpu().tolist() for k, v in self._observations.items()}
                self.terminal[i] = record

    @torch.no_grad()
    def reset(self, rows):
        """Explicit selective reset returns prior terminals for caller retention."""
        self._healthy()
        try:
            mask_check(rows, self.n, self.device)
            retained = deepcopy([r if bool(rows[i]) else None for i, r in enumerate(self.terminal)])
            if not rows.any(): return retained
            self._view('qpos')[rows] = self.initial_qpos
            for name in ('qvel', 'ctrl', 'time', 'qacc_warmstart', 'qacc', 'qfrc_applied', 'xfrc_applied'):
                self._view(name)[rows] = 0
            self.motor.reset(rows); self.delay.reset(rows)
            self.steps[rows] = 0; self.live[rows] = True
            self._forward(); self._refresh(rows)
            if (rows & physical_failures(self.state, self.steps)).any():
                raise ValueError('invalid stance reset')
            for i in rows.nonzero().flatten().tolist(): self.terminal[i] = None
            return retained
        except Exception:
            self.faulted = True
            raise

    def _control_snapshot(self):
        """Owned optional evidence; never included in the actor observation."""
        return {name: value.detach().clone() for name, value in dict(
            correction=self.delay.correction, target=self.delay.target, queue=self.delay.queue,
            previous=self.motor.previous, voltage=self.motor.voltage, kp=self.motor.kp,
            friction=self.motor.fields['dof_frictionloss'], damping=self.motor.fields['dof_damping'],
            ctrl=self._view('ctrl')[:, self.ctrl_ids]).items()}

    @torch.no_grad()
    def step(self, actions, *, capture_control=False):
        self._healthy()
        if not self.live.any(): raise RuntimeError('all stance worlds closed; explicit reset required')
        try:
            if type(capture_control) is not bool: raise ValueError('boolean control capture flag')
            evidence = dict(initial=self._control_snapshot(), proposals=[]) if capture_control else None
            correction, change = self.delay.set_actions(actions, self.live)
            if evidence is not None: evidence['after_action'] = self._control_snapshot()
            # Cached physical fields are current already; refresh the correction
            # portion for still-live worlds without performing an extra solve.
            self._observations['actor'][self.live, -10:] = correction[self.live]/.2
            self._observations['critic'][self.live, 34:44] = correction[self.live]/.2
            tick = StanceTransition(self.steps, correction, change, initial_live=self.live)
            boundaries = [self.snapshot()]
            for _ in range(DECIMATION):
                tick.reject_pre_step_state(self.state); self._capture(tick)
                if not tick.live.any(): break
                pos = self._view('qpos')[:, self.qids]; vel = self._view('qvel')[:, self.dofs]
                zeros = torch.zeros_like(pos)
                command = ActuatorCmd(self.delay.peek(), zeros, zeros, pos, vel)
                if evidence is not None:
                    captured = dict(before_steps=self.steps.clone(), live=tick.live.clone(),
                        command={k: getattr(command, k).detach().clone() for k in
                                 ('position_target', 'velocity_target', 'effort_target', 'pos', 'vel')})
                proposal = self.motor.compute(command, tick.live)
                rejected = tick.reject_proposed_torque(proposal['torque_nm'])
                if not torch.equal(rejected, proposal['rejected']) or not torch.equal(tick.live, proposal['accepted']):
                    raise ValueError('motor and physics live masks disagree')
                self._capture(tick, proposal['torque_nm'])
                accepted = tick.live.clone()
                if evidence is not None:
                    captured.update({k: v.detach().clone() for k, v in proposal.items()})
                if not accepted.any():
                    if evidence is not None:
                        captured['committed'] = self._control_snapshot()
                        evidence['proposals'].append(captured)
                    break
                ctrl = self._view('ctrl')
                ctrl[accepted.nonzero().flatten()[:, None], self.ctrl_ids[None]] = proposal['torque_nm'][accepted]
                self.delay.advance(accepted)
                if evidence is not None:
                    captured['committed'] = self._control_snapshot()
                    evidence['proposals'].append(captured)
                # Same order as native audit: motor -> new-control solve -> Euler
                # -> fresh post-step solve -> stops/reward. Closed physical arrays
                # and motor/FIFO rows do not commit any candidate updates.
                self._forward()
                if (self._view('qfrc_actuator')[:, self.dofs][accepted].abs() > .36).any():
                    raise ValueError('applied motor torque exceeds pre-step gate')
                old_time = self._view('time').clone()
                self.integrator.integrate(accepted)
                expected_time = torch.where(accepted, old_time+.002, old_time)
                if not torch.equal(self._view('time'), expected_time):
                    raise ValueError('physical clock and accepted substeps disagree')
                self._forward(); self._refresh(accepted)
                tick.advance(self.state)
                self.steps.copy_(tick.episode_steps)
                self._capture(tick)
                boundaries.append(self.snapshot())
            self.live.copy_(tick.live)
            result = tick.result()
            result.update(observation=self.observations(), boundaries=boundaries,
                          terminal_records=deepcopy(self.terminal), optimizer_launched=False)
            if evidence is not None:
                evidence['final'] = self._control_snapshot()
                result['control_evidence'] = evidence
            return result
        except Exception:
            self.faulted = True
            raise
