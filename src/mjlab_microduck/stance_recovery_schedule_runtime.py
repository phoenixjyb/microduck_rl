"""Sibling runtime for explicit finite per-row recovery schedules.

The inherited nominal ``step`` and its external-force refusal are untouched.
Only ``step_with_schedule`` can temporarily install a hash-bound declared
wrench; it always clears both complete force arrays before the inherited
unforced forward solve. This CPU-testable adapter is not learner/job admission.
"""

from copy import deepcopy
from hashlib import sha256

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.actuator.actuator import ActuatorCmd

from mjlab_microduck import stance_forward_probe as forward_evidence
from mjlab_microduck import stance_plant_evidence as plant_evidence
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck.stance_contact_evidence import read_contacts, FirstTerminalContacts
from mjlab_microduck.stance_transition import PhysicsState, StanceTransition, DECIMATION, physical_failures
from mjlab_microduck.stance_warp_integrator import EulerCandidateCommit
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
from mjlab_microduck.first_attempt_smoke import canonical


class ScheduledRecoveryRuntime(WarpStanceRuntime):
    """Fixed row-to-cell schedule, with pulse access only on an explicit method."""

    def __init__(self, declaration, *, device='cpu', solved_field_check='packed'):
        self.schedule_declaration = schedule.checked(declaration)
        self._schedule_sha256 = schedule.binding_sha256(self.schedule_declaration)
        super().__init__(self.schedule_declaration['worlds'], device=device,
                         solved_field_check=solved_field_check)

        # Bind the actual compiled model's ordered trunk body table, then keep
        # both public descriptors hash checked before any caller control changes.
        self.binding = dict(selected_plant=plant_evidence.describe(self.native),
            nbody=self.native.nbody,
            body_names=[mujoco.mj_id2name(self.native, mujoco.mjtObj.mjOBJ_BODY, i) or ''
                        for i in range(self.native.nbody)],
            body_name=baseline.force.BODY_NAME,
            body_id=int(self.native.body(baseline.force.BODY_NAME).id))
        baseline.force._binding(self.binding)
        self.plant_binding = deepcopy(self.binding)
        self._binding_sha256 = sha256(canonical(self.binding).encode()).hexdigest()
        self.scheduled_pulse_evidence = []

    def _scheduled_forward(self, steps, accepted):
        """Eager forced solve with exact full-array checks on both sides."""
        self._sync()
        try:
            if (steps.shape != (self.n,) or steps.dtype != torch.long or steps.device != self.device
                    or accepted.shape != (self.n,) or accepted.dtype != torch.bool
                    or accepted.device != self.device):
                raise ValueError('schedule counters and accepted mask must match runtime')
            steps_list = steps.detach().cpu().tolist()
            accepted_list = accepted.detach().cpu().tolist()
            expected_x, expected_q = schedule.expected_wrenches(self.schedule_declaration,
                steps_list, accepted_list, self.plant_binding['nbody'], self.plant_binding['body_id'])
            schedule.validate_wrenches(self._view('xfrc_applied').detach().cpu().tolist(),
                self._view('qfrc_applied').detach().cpu().tolist(), self.schedule_declaration,
                steps_list, accepted_list, self.plant_binding['nbody'], self.plant_binding['body_id'])
            if not torch.equal(self._view('xfrc_applied'), torch.as_tensor(
                    expected_x, dtype=self._view('xfrc_applied').dtype, device=self.device)):
                raise ValueError('scheduled external force differs from declared exact matrix')
            if not torch.equal(self._view('qfrc_applied'), torch.as_tensor(
                    expected_q, dtype=self._view('qfrc_applied').dtype, device=self.device)):
                raise ValueError('scheduled generalized force differs from exact zero matrix')
            if self.forward_graph is not None:
                raise ValueError('scheduled recovery requires the declared eager solver')
            with wp.ScopedDevice(self.wp_device):
                mjwarp.forward(self.model, self.data)
            self._sync()
            schedule.validate_wrenches(self._view('xfrc_applied').detach().cpu().tolist(),
                self._view('qfrc_applied').detach().cpu().tolist(), self.schedule_declaration,
                steps_list, accepted_list, self.plant_binding['nbody'], self.plant_binding['body_id'])
            self._check_solved_fields()
            nefc = self._view('nefc')
            if ((nefc < 0) | (nefc > self.data.njmax)).any():
                raise ValueError('active constraint capacity exceeded')
            if (self._view('solver_niter') > 100).any():
                raise ValueError('solver iteration ceiling exceeded')
            self.contacts = read_contacts(self.model, self.data)
            types = wp.to_torch(self.data.efc.type); ids = wp.to_torch(self.data.efc.id)
            active = torch.arange(ids.shape[1], device=self.device)[None] < nefc[:, None]
            friction = active & (types == int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF))
            if (friction & ((ids < 0) | (ids >= self.native.nv))).any():
                raise ValueError('invalid active friction DOF address')
        except Exception:
            self.faulted = True
            raise

    def _phase_capture(self):
        """Owned solved outputs and force/control inputs at one full-batch phase."""
        solved = forward_evidence.output(self)
        inputs = {name: self._view(name).detach().cpu().clone()
                  for name in ('ctrl', 'xfrc_applied', 'qfrc_applied')}
        return dict(solved=solved, inputs=inputs,
            motor_fields={name: value.detach().cpu().clone()
                          for name, value in self.motor.fields.items()})

    def _check_frozen_schedule_and_binding(self):
        checked = schedule.checked(self.schedule_declaration)
        if schedule.binding_sha256(checked) != self._schedule_sha256:
            raise ValueError('fixed recovery schedule changed after runtime construction')
        if (self.binding != self.plant_binding
                or sha256(canonical(self.binding).encode()).hexdigest() != self._binding_sha256
                or sha256(canonical(self.plant_binding).encode()).hexdigest() != self._binding_sha256):
            raise ValueError('compiled recovery plant/body binding changed')
        baseline.force._binding(self.binding)

    @torch.no_grad()
    def step_with_schedule(self, actions, *, capture_control=False):
        """Run one original-policy tick with only the fixed accepted-row schedule."""
        self._healthy()
        if not self.live.any():
            raise RuntimeError('all stance worlds closed; explicit reset required')
        step_evidence = []
        try:
            if type(capture_control) is not bool:
                raise ValueError('boolean control capture flag')
            self._check_frozen_schedule_and_binding()
            if self.forward_graph is not None:
                raise ValueError('scheduled recovery requires the declared eager solver')
            # Refuse external inputs before snapshots, BAM proposals, FIFO
            # changes, action update or any other motor/control mutation.
            if self._view('xfrc_applied').any() or self._view('qfrc_applied').any():
                raise ValueError('scheduled step requires zero applied-force arrays at entry')

            evidence = dict(initial=self._control_snapshot(), proposals=[]) if capture_control else None
            correction, change = self.delay.set_actions(actions, self.live)
            if evidence is not None:
                evidence['after_action'] = self._control_snapshot()
            self._observations['actor'][self.live, -10:] = correction[self.live]/.2
            self._observations['critic'][self.live, 34:44] = correction[self.live]/.2
            tick = StanceTransition(self.steps, correction, change, initial_live=self.live)
            boundaries = [self.snapshot()]

            for _ in range(DECIMATION):
                tick.reject_pre_step_state(self.state)
                self._capture(tick)
                if not tick.live.any():
                    break
                pos = self._view('qpos')[:, self.qids]
                vel = self._view('qvel')[:, self.dofs]
                zeros = torch.zeros_like(pos)
                command = ActuatorCmd(self.delay.peek(), zeros, zeros, pos, vel)
                captured = None
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
                before_steps = self.steps.clone()
                if captured is not None:
                    captured.update({k: v.detach().clone() for k, v in proposal.items()})
                if not accepted.any():
                    if captured is not None:
                        captured['committed'] = self._control_snapshot()
                        evidence['proposals'].append(captured)
                    break

                self._view('ctrl')[accepted.nonzero().flatten()[:, None], self.ctrl_ids[None]] = proposal['torque_nm'][accepted]
                self.delay.advance(accepted)
                if captured is not None:
                    captured['committed'] = self._control_snapshot()
                    evidence['proposals'].append(captured)

                steps_list = before_steps.detach().cpu().tolist()
                accepted_list = accepted.detach().cpu().tolist()
                x_expected, q_expected = schedule.expected_wrenches(self.schedule_declaration,
                    steps_list, accepted_list, self.plant_binding['nbody'], self.plant_binding['body_id'])
                x_tensor = torch.as_tensor(x_expected, dtype=self._view('xfrc_applied').dtype,
                                           device=self.device)
                q_tensor = torch.as_tensor(q_expected, dtype=self._view('qfrc_applied').dtype,
                                           device=self.device)
                if self._view('xfrc_applied').any() or self._view('qfrc_applied').any():
                    raise ValueError('applied-force arrays must be zero before schedule installation')
                active_window = torch.as_tensor(schedule.window_mask(
                    self.schedule_declaration, steps_list, accepted_list),
                    dtype=torch.bool, device=self.device)
                phase_active = bool(active_window.any())
                phases = {}
                old_time = self._view('time').clone()
                post_xfrc = post_qfrc = None
                try:
                    # Include installation itself in cleanup protection in case
                    # either full-array copy partially succeeds then raises.
                    self._view('xfrc_applied').copy_(x_tensor)
                    self._view('qfrc_applied').copy_(q_tensor)
                    self._scheduled_forward(before_steps, accepted)
                    if (self._view('qfrc_actuator')[:, self.dofs][accepted].abs() > .36).any():
                        raise ValueError('applied motor torque exceeds pre-step gate')
                    if phase_active:
                        phases['forced_pre'] = self._phase_capture()
                    self.integrator.integrate(accepted)
                    if phase_active:
                        phases['integrated'] = self._phase_capture()
                finally:
                    self._view('xfrc_applied').zero_()
                    self._view('qfrc_applied').zero_()
                    post_xfrc = self._view('xfrc_applied').detach().clone()
                    post_qfrc = self._view('qfrc_applied').detach().clone()

                expected_time = torch.where(accepted, old_time+.002, old_time)
                if not torch.equal(self._view('time'), expected_time):
                    raise ValueError('physical clock and accepted substeps disagree')
                # This inherited nominal guard runs only after both complete
                # external arrays were cleared by finally above.
                self._forward()
                if phase_active:
                    phases['unforced_post'] = self._phase_capture()
                self._refresh(accepted)
                tick.advance(self.state)
                self.steps.copy_(tick.episode_steps)
                self._capture(tick)
                boundaries.append(self.snapshot())
                step_evidence.append(dict(before_steps=before_steps,
                    accepted=accepted.detach().clone(), pre_xfrc=x_tensor.detach().clone(),
                    pre_qfrc=q_tensor.detach().clone(), post_xfrc=post_xfrc,
                    post_qfrc=post_qfrc, phases=phases if phase_active else None,
                    schedule_sha256=self._schedule_sha256, binding_sha256=self._binding_sha256))

            self.scheduled_pulse_evidence = step_evidence
            self.live.copy_(tick.live)
            result = tick.result()
            result.update(observation=self.observations(), boundaries=boundaries,
                terminal_records=deepcopy(self.terminal), optimizer_launched=False,
                scheduled_pulse_evidence=deepcopy(step_evidence))
            if evidence is not None:
                evidence['final'] = self._control_snapshot()
                result['control_evidence'] = evidence
            return result
        except Exception:
            self.faulted = True
            raise
