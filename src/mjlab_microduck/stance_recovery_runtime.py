"""Explicit, bounded D1 pulse collector layered beside nominal stance runtime.

This adapter has no learner role. Its inherited nominal ``step`` keeps the
nominal force guard; recovery pulses are possible only through
``step_with_pulse`` and remain bound to an ordered, unique case list.
"""

from copy import deepcopy
from hashlib import sha256

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.actuator.actuator import ActuatorCmd

from mjlab_microduck import stance_recovery_contract as contract
from mjlab_microduck import stance_forward_probe as forward_evidence
from mjlab_microduck import stance_plant_evidence as plant_evidence
from mjlab_microduck.stance_contact_evidence import read_contacts, contact_summary, FirstTerminalContacts
from mjlab_microduck.stance_control_state import mask_check
from mjlab_microduck.stance_transition import PhysicsState, StanceTransition, DECIMATION, physical_failures
from mjlab_microduck.stance_warp_integrator import EulerCandidateCommit
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
from mjlab_microduck.first_attempt_smoke import canonical


class RecoveryRuntime(WarpStanceRuntime):
    """A nominal runtime with one explicit bounded recovery-pulse collector."""

    def __init__(self, cases, *, device='cpu', solved_field_check='packed'):
        self.recovery_cases = contract.cases_checked(cases)
        self.cases = self.recovery_cases
        super().__init__(len(self.recovery_cases), device=device,
                         solved_field_check=solved_field_check)
        # Bind against the just-compiled model's actual ordered body table.
        self.binding = dict(
            selected_plant=plant_evidence.describe(self.native),
            nbody=self.native.nbody,
            body_names=[mujoco.mj_id2name(self.native, mujoco.mjtObj.mjOBJ_BODY, i) or ''
                        for i in range(self.native.nbody)],
            body_name=contract.force.BODY_NAME,
            body_id=int(self.native.body(contract.force.BODY_NAME).id),
        )
        self.plant_binding = self.binding
        contract.force._binding(self.binding)
        self._binding_sha256 = sha256(canonical(self.binding).encode()).hexdigest()
        self._declared_cases = self.cases
        self.pulse_evidence = []

    def _pulse_forward(self, steps, accepted):
        """Solve while permitting only the exact accepted rows' declared pulse."""
        self._sync()
        try:
            if (steps.shape != (self.n,) or steps.dtype != torch.long or steps.device != self.device
                    or accepted.shape != (self.n,) or accepted.dtype != torch.bool
                    or accepted.device != self.device):
                raise ValueError('pulse counters and accepted mask must match runtime')
            expected_x, expected_q = contract.expected_wrenches(
                self.recovery_cases, steps.detach().cpu().tolist(), accepted.detach().cpu().tolist(),
                self.plant_binding['nbody'], self.plant_binding['body_id'])
            contract.validate_wrenches(self._view('xfrc_applied').detach().cpu().tolist(),
                self._view('qfrc_applied').detach().cpu().tolist(), self.recovery_cases,
                steps.detach().cpu().tolist(), accepted.detach().cpu().tolist(),
                self.plant_binding['nbody'], self.plant_binding['body_id'])
            if not torch.equal(self._view('xfrc_applied'), torch.as_tensor(
                    expected_x, dtype=self._view('xfrc_applied').dtype, device=self.device)):
                raise ValueError('xfrc_applied differs from exact declared recovery input')
            if not torch.equal(self._view('qfrc_applied'), torch.as_tensor(
                    expected_q, dtype=self._view('qfrc_applied').dtype, device=self.device)):
                raise ValueError('qfrc_applied differs from exact zero recovery input')
            if self.forward_graph is not None:
                raise ValueError('recovery requires the declared eager solver')
            with wp.ScopedDevice(self.wp_device):
                mjwarp.forward(self.model, self.data)
            self._sync()
            # Revalidate the complete force tensors after the solver call.
            contract.validate_wrenches(self._view('xfrc_applied').detach().cpu().tolist(),
                self._view('qfrc_applied').detach().cpu().tolist(), self.recovery_cases,
                steps.detach().cpu().tolist(), accepted.detach().cpu().tolist(),
                self.plant_binding['nbody'], self.plant_binding['body_id'])
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
        """Owned complete solved output plus force/control inputs at a boundary."""
        solved = forward_evidence.output(self)
        inputs = {name: self._view(name).detach().cpu().clone()
                  for name in ('ctrl', 'xfrc_applied', 'qfrc_applied')}
        return dict(solved=solved, inputs=inputs,
            motor_fields={name: value.detach().cpu().clone()
                          for name, value in self.motor.fields.items()})

    @torch.no_grad()
    def step_with_pulse(self, actions, *, capture_control=False):
        """Run one original-policy tick, applying only the scheduled accepted-row pulse."""
        self._healthy()
        if not self.live.any():
            raise RuntimeError('all stance worlds closed; explicit reset required')
        step_pulse_evidence = []
        try:
            if type(capture_control) is not bool:
                raise ValueError('boolean control capture flag')
            if (self.cases != self._declared_cases or self.recovery_cases != self._declared_cases
                    or self.plant_binding != self.binding
                    or sha256(canonical(self.binding).encode()).hexdigest() != self._binding_sha256):
                raise ValueError('frozen recovery case/body binding changed')
            if self.forward_graph is not None:
                raise ValueError('recovery requires the declared eager solver')
            # Never overwrite or normalize an external force input. This check
            # precedes snapshots, BAM proposals, FIFO changes and action updates.
            if self._view('xfrc_applied').any() or self._view('qfrc_applied').any():
                raise ValueError('recovery step requires zero applied-force arrays at entry')
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
                # Exact full arrays for the accepted rows only; zero rows remain zero.
                x_expected, q_expected = contract.expected_wrenches(
                    self.recovery_cases, before_steps.detach().cpu().tolist(), accepted.detach().cpu().tolist(),
                    self.plant_binding['nbody'], self.plant_binding['body_id'])
                x_tensor = torch.as_tensor(x_expected, dtype=self._view('xfrc_applied').dtype, device=self.device)
                q_tensor = torch.as_tensor(q_expected, dtype=self._view('qfrc_applied').dtype, device=self.device)
                if self._view('xfrc_applied').any() or self._view('qfrc_applied').any():
                    raise ValueError('applied-force arrays must be zero before pulse installation')
                self._view('xfrc_applied').copy_(x_tensor)
                self._view('qfrc_applied').copy_(q_tensor)
                in_window = ((before_steps >= contract.ONSET_STEP) &
                             (before_steps < contract.ONSET_STEP+contract.PULSE_STEPS) & accepted)
                phases = {}
                old_time = self._view('time').clone()
                try:
                    self._pulse_forward(before_steps, accepted)
                    if (self._view('qfrc_actuator')[:, self.dofs][accepted].abs() > .36).any():
                        raise ValueError('applied motor torque exceeds pre-step gate')
                    if in_window.any():
                        phases['forced_pre'] = self._phase_capture()
                    self.integrator.integrate(accepted)
                    if in_window.any():
                        phases['integrated'] = self._phase_capture()
                finally:
                    # Clear our installed pulse after its integrated capture,
                    # or after any forced-solve/phase/Euler failure. No later
                    # solver call sees it, including a failed owned operation.
                    self._view('xfrc_applied').zero_()
                    self._view('qfrc_applied').zero_()
                    post_xfrc = self._view('xfrc_applied').detach().clone()
                    post_qfrc = self._view('qfrc_applied').detach().clone()
                expected_time = torch.where(accepted, old_time+.002, old_time)
                if not torch.equal(self._view('time'), expected_time):
                    raise ValueError('physical clock and accepted substeps disagree')
                # Nominal unforced solve and refresh occur only after the finally-clear.
                self._forward()
                if in_window.any():
                    phases['unforced_post'] = self._phase_capture()
                self._refresh(accepted)
                tick.advance(self.state)
                self.steps.copy_(tick.episode_steps)
                self._capture(tick)
                boundaries.append(self.snapshot())
                step_pulse_evidence.append(dict(before_steps=before_steps,
                    accepted=accepted.detach().clone(),
                    pre_xfrc=x_tensor.detach().clone(), pre_qfrc=q_tensor.detach().clone(),
                    post_xfrc=post_xfrc, post_qfrc=post_qfrc,
                    phases=phases if in_window.any() else None))
            self.pulse_evidence = step_pulse_evidence
            self.live.copy_(tick.live)
            result = tick.result()
            result.update(observation=self.observations(), boundaries=boundaries,
                          terminal_records=deepcopy(self.terminal), optimizer_launched=False,
                          pulse_evidence=deepcopy(step_pulse_evidence))
            if evidence is not None:
                evidence['final'] = self._control_snapshot()
                result['control_evidence'] = evidence
            return result
        except Exception:
            self.faulted = True
            raise
