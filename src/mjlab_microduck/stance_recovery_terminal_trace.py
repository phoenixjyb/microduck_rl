"""Record-only first-terminal stochastic trace for the scheduled B1 dose pair.

This sibling probe uses the immutable D1 actor/critic and a fresh CPU2 runtime.
It stops on the first natural terminal, records one explicit reset, and never
creates PPO storage or an optimizer. Passing a consistency score is not skill,
training, execution, or full-duration acceptance.
"""
import io
import math
import os
import time
from copy import deepcopy
from hashlib import sha256

import torch

from mjlab_microduck import stance_attempt_trace as attempt
from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_control_evidence as control
from mjlab_microduck import stance_cpu_replay_profile as profile
from mjlab_microduck import stance_disturbance_fixture as fixture
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_parent as parent
from mjlab_microduck import stance_recovery_ppo_trace as ppo_trace
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.stance_ppo import observations, timeout_rewards
from mjlab_microduck.stance_transition import (
    EPISODE_STEPS,
    PhysicsState,
    physical_failures,
)

PROTOCOL = "football-b1d-cpu-scheduled-first-terminal-trace-v1"
SEED, WORLDS, HORIZON = 653, 2, 250
WALL_LIMIT, LIMIT = 180.0, 128 * 1024 * 1024
WORLD_CELLS = ("zero-wrench", "+x-2n-20steps-t250")
PULSE_KEYS = {"before_steps", "accepted", "pre_xfrc", "pre_qfrc", "post_xfrc",
    "post_qfrc", "phases", "schedule_sha256", "binding_sha256"}


def _target_schedule(value):
    value = schedule.checked(value)
    require(value["split"] == "training" and value["stage"] == "dose"
            and value["worlds"] == WORLDS and tuple(value["cell_ids"]) == WORLD_CELLS,
            "exact two-row training dose schedule")
    zero, pulse = value["row_cells"]
    require(zero["onset_step"] == 500 and zero["duration_steps"] == 10
            and zero["force_world_newtons"] == [0.0, 0.0, 0.0]
            and pulse["onset_step"] == 250 and pulse["duration_steps"] == 20
            and pulse["force_world_newtons"] == [2.0, 0.0, 0.0],
            "fixed held dose pulse rows")
    return value


def _binding_matches(recorded, declaration, compiled_plant):
    profile.check_recorded(recorded["cpu_math_profile"])
    return canonical(recorded) == canonical(binding(declaration, compiled_plant,
                                                    recorded["cpu_math_profile"]))


def _check_policy_clock(clock, tick):
    attempt.tensor(clock["before"], (WORLDS,), torch.float32, "clock before")
    attempt.tensor(clock["after"], (WORLDS,), torch.float32, "clock after")
    attempt.tensor(clock["accepted_substeps"], (WORLDS,), torch.int64, "clock steps")
    expected = clock["before"].clone()
    for index in range(10):
        expected = torch.where(clock["accepted_substeps"] > index,
            expected + torch.full_like(expected, .002), expected)
    require(torch.equal(clock["accepted_substeps"], tick["executed_steps"])
            and torch.equal(clock["after"], expected),
            "per-policy physical time agrees with accepted Euler increments")


def _pulse_complete(windows, delivered, calls, checked):
    return (calls == HORIZON and checked == EPISODE_STEPS
            and windows == [10, 20] and delivered == [0, 20])


def _check_private_chain(initial, rows, final):
    require(torch.is_tensor(initial) and initial.dtype == torch.uint8
            and initial.device.type == "cpu", "private CPU RNG initial state")
    rng = initial.detach().cpu().clone()
    for row in rows:
        require(torch.equal(row["rng_before"], rng), "private RNG transition chain")
        require(torch.is_tensor(row["rng_after"]) and row["rng_after"].dtype == torch.uint8
                and row["rng_after"].device.type == "cpu", "private RNG recorded next state")
        rng = row["rng_after"].detach().cpu().clone()
    require(torch.equal(rng, final), "private RNG final state")
    return True


def _check_reset_siblings(before, after, done):
    untouched = ~done
    require(set(before["controls"]) == set(after["controls"]),
            "complete sibling control inventory")
    for key in ("qpos", "qvel", "time"):
        require(torch.equal(before[key][untouched], after[key][untouched]),
                "untouched sibling physical state preserved: "+key)
    require(torch.equal(before["frame"]["physics_steps"][untouched],
                        after["frame"]["physics_steps"][untouched])
            and torch.equal(before["live"][untouched], after["live"][untouched]),
            "selectively untouched sibling clocks and liveness")
    for name, pre in before["controls"].items():
        post = after["controls"][name]
        require(torch.equal(pre[untouched], post[untouched]),
                "untouched sibling control preserved: "+name)
    return True


def _check_terminal_frame(before, final_tick):
    attempt.validate_frame(before["frame"], WORLDS)
    require(_tree_equal(before["frame"], final_tick["boundaries"][-1]),
            "pre-reset frame exactly equals retained final physical boundary")
    require(torch.equal(before["live"], final_tick["live"]),
            "pre-reset liveness equals final physical tick")
    for key in ("qpos", "qvel"):
        require(torch.equal(before[key], before["frame"][key]),
                "pre-reset kinematics match retained frame: "+key)
    return True


def _check_reset_rows(after, done, initial_frame):
    attempt.validate_frame(after["frame"], WORLDS)
    require(torch.equal(after["frame"]["physics_steps"][done],
                        torch.zeros_like(after["frame"]["physics_steps"][done]))
            and after["live"][done].all(), "reset rows reopen at zero episode clocks")
    attempt.same_rows(initial_frame, after["frame"], done, observation=True)
    require(torch.equal(after["qpos"][done], initial_frame["qpos"][done])
            and torch.equal(after["qvel"][done], initial_frame["qvel"][done])
            and not after["time"][done].any(), "reset rows return to original kinematics and time")
    return True


def _reset_claims(done, records, physics_steps, full_claim, selective_claim):
    attempt.tensor(done, (WORLDS,), torch.bool, "reset semantics done mask")
    attempt.tensor(physics_steps, (WORLDS,), torch.int64, "reset semantics terminal clocks")
    require(type(records) is list and len(records) == WORLDS
            and type(full_claim) is bool and type(selective_claim) is bool,
            "typed reset semantics evidence")
    selective = bool(done.any() and (~done).any())
    full = bool(done.all() and all(record is not None and record["timed_out"]
                and not record["terminated"] for record in records)
                and (physics_steps == EPISODE_STEPS).all())
    require(selective_claim is selective,
            "selective reset requires staggered terminal rows")
    require(full_claim is full, "full timeout means both rows time out at full horizon")
    return full, selective


def binding(declaration, compiled_plant, cpu_math_profile):
    declaration = _target_schedule(declaration)
    require(type(compiled_plant) is dict and set(compiled_plant) ==
            {"selected_plant", "nbody", "body_names", "body_name", "body_id"},
            "actual compiled plant binding")
    baseline.force._binding(compiled_plant)
    profile.validate_receipt(cpu_math_profile)
    return {"protocol": PROTOCOL, "source": declaration["source"], "worlds": WORLDS,
        "capture_device": "cpu", "horizon": HORIZON, "max_policy_calls": HORIZON,
        "max_physics_steps": EPISODE_STEPS, "seed": SEED,
        "schedule_sha256": schedule.binding_sha256(declaration),
        "plant_sha256": sha256(canonical(compiled_plant).encode()).hexdigest(),
        "parent_checkpoint_sha256": baseline.CHECKPOINT_SHA256,
        "parent_checkpoint_identity": parent.expected_identity(),
        "parent_state_sha256": parent.PARENT_STATE_SHA256,
        "cpu_math_profile": deepcopy(cpu_math_profile)}


class TerminalTrace(attempt.FirstAttemptTrace):
    """Existing physical continuity codec with a distinct 250-call binding."""
    def __init__(self, value, initial, declaration, compiled_plant):
        require(canonical(value) == canonical(binding(declaration, compiled_plant,
                value["cpu_math_profile"])), "exact terminal trace binding")
        self.binding = deepcopy(value); self.n = WORLDS; self.faulted = False
        self._tensor_device = "cpu"
        frame = attempt.owned(initial); attempt.validate_frame(frame, WORLDS)
        require(str(initial["qpos"].device) == "cpu" and not frame["physics_steps"].any()
                and not physical_failures(PhysicsState(**frame["state"]),
                                          frame["physics_steps"]).any(),
                "fresh valid CPU2 initial frame")
        self.initial = frame; self.last = frame; self.ticks = []
        self.terminals = [None] * WORLDS

    def _initialize(self, *_):
        raise RuntimeError("terminal trace requires its sibling source binding")

    def append(self, result, actor_input, actions):
        require(len(self.ticks) < HORIZON, "bounded first-terminal call count")
        return super().append(result, actor_input, actions)


def _tree_equal(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left, right))
    if type(left) is dict or type(right) is dict:
        return (type(left) is dict and type(right) is dict and left.keys() == right.keys()
                and all(_tree_equal(left[k], right[k]) for k in left))
    if type(left) in (list, tuple) or type(right) in (list, tuple):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_tree_equal(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def _fresh(env, raw_parent):
    from mjlab_microduck.stance_recovery_schedule_runtime import (
        ScheduledRecoveryRuntime,
    )
    require(type(env) is ScheduledRecoveryRuntime and env.n == WORLDS
            and torch.device(env.device).type == "cpu" and not env.wp_device.is_cuda
            and env.forward_graph is None and env.solved_field_check == "packed",
            "fresh exact eager packed CPU2 scheduled runtime")
    require(env.live.dtype == torch.bool and env.live.shape == (WORLDS,) and env.live.all()
            and env.steps.dtype == torch.long and env.steps.shape == (WORLDS,)
            and not env.steps.any() and not env.faulted and env.terminal == [None, None]
            and env.scheduled_pulse_evidence == [], "fresh all-live physical clocks and runtime")
    require(torch.equal(env._view("qpos"), env.initial_qpos.expand(WORLDS, -1))
            and not env._view("qvel").any() and not env._view("time").any(),
            "fresh initial physical state and clocks")
    require(schedule.checked(env.schedule_declaration) == _target_schedule(env.schedule_declaration),
            "fixed declared dose schedule")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "CUDA-hidden CPU-only capture")
    require(type(raw_parent) is bytes and sha256(raw_parent).hexdigest() == baseline.CHECKPOINT_SHA256,
            "exact frozen parent bytes before load")


def _sample(actor, critic, actor_obs, critic_obs):
    obs = observations({"actor": actor_obs, "critic": critic_obs}, WORLDS)
    with torch.no_grad():
        action = actor(obs, stochastic_output=True)
        value = critic(obs).detach()
        params = actor.output_distribution_params
        log_prob = actor.distribution.log_prob(action).reshape(WORLDS, 1)
    mu, sigma = (p.detach().cpu().clone() for p in params)
    attempt.tensor(action, (WORLDS, 10), torch.float32, "raw Gaussian action")
    attempt.tensor(value, (WORLDS, 1), torch.float32, "pre-action critic")
    attempt.tensor(log_prob, (WORLDS, 1), torch.float32, "sampled action log probability")
    attempt.tensor(mu, (WORLDS, 10), torch.float32, "Gaussian mean")
    attempt.tensor(sigma, (WORLDS, 10), torch.float32, "Gaussian scale")
    require((sigma > 0).all(), "positive exact-parent Gaussian scale")
    return action.detach().cpu().clone(), value.cpu().clone(), log_prob.cpu().clone(), mu, sigma


def _runtime_failure_snapshot(env):
    """Best-effort owned state after a fault; never advances or repairs runtime."""
    try:
        return {"faulted": bool(env.faulted), "steps": env.steps.detach().cpu().clone(),
            "live": env.live.detach().cpu().clone(),
            "qpos": env._view("qpos").detach().cpu().clone(),
            "qvel": env._view("qvel").detach().cpu().clone(),
            "time": env._view("time").detach().cpu().clone(),
            "ctrl": env._view("ctrl").detach().cpu().clone(),
            "xfrc_applied": env._view("xfrc_applied").detach().cpu().clone(),
            "qfrc_applied": env._view("qfrc_applied").detach().cpu().clone(),
            "controls": attempt.owned(env._control_snapshot())}
    except Exception as error:  # noqa: BLE001 - diagnostic read only after fault
        return {"capture_error_type": type(error).__name__,
                "capture_error": str(error)[:300]}


@torch.no_grad()
def collect(env, raw_parent, *, deadline_monotonic, clock=time.monotonic):
    """Collect through the first natural terminal and make at most one reset.

    On transition/reset exceptions the returned artifact retains the last
    validated trace prefix and an explicitly unvalidated failure record.
    """
    _fresh(env, raw_parent)
    require(type(deadline_monotonic) in (int, float) and math.isfinite(deadline_monotonic)
            and clock() < deadline_monotonic <= clock() + WALL_LIMIT,
            "live 180-second bounded collection deadline")
    loaded = parent.load_parent(raw_parent)
    actor, critic = loaded["actor"], loaded["critic"]
    initial_hash = checkpoint.state_hash(checkpoint.states_of(actor, critic))
    require(initial_hash == parent.PARENT_STATE_SHA256
            and (actor.distribution.std_param > 0).all(), "exact unchanged parent actor and critic")
    declaration = _target_schedule(env.schedule_declaration)
    cpu_profile = profile.checked_receipt()
    b = binding(declaration, env.binding, cpu_profile)
    recorder = TerminalTrace(b, env.snapshot(), declaration, env.binding)
    control_ticks, pulse_ticks, policies, terminal_obs_rows = [], [], [], []
    caller_before = torch.random.get_rng_state().clone()
    initial_controls = attempt.owned(env._control_snapshot())
    initial_qpos = env._view("qpos").detach().cpu().clone()
    private = torch.Generator(device="cpu").manual_seed(SEED).get_state()
    private_before = private.clone()
    started = clock(); reason = "policy-call-limit"; failure = None
    terminal_policy = terminal_reset = unvalidated = None
    for _ in range(HORIZON):
        if clock() >= deadline_monotonic:
            reason = "deadline-exhausted"; break
        obs = env.observations()
        rng_before = private.clone()
        sample_error = None
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(private)
            try:
                action, value, log_prob, mu, sigma = _sample(
                    actor, critic, obs["actor"], obs["critic"])
            except Exception as error:  # noqa: BLE001 - preserve sampler failure as a typed partial trace
                sample_error = error
            finally:
                private = torch.random.get_rng_state().clone()
        if sample_error is not None:
            reason = "sampling-exception"
            failure = {"stage": "policy-sample", "error_type": type(sample_error).__name__,
                       "error": str(sample_error)[:500]}
            break
        row = {"rng_before": rng_before, "rng_after": private.clone(),
            "actor_input": obs["actor"].detach().cpu().clone(),
            "critic_input": obs["critic"].detach().cpu().clone(), "raw_action": action,
            "value": value, "log_prob": log_prob, "mu": mu, "sigma": sigma}
        time_before = env._view("time").detach().cpu().clone()
        try:
            result = env.step_with_schedule(action, capture_control=True)
        except Exception as error:  # noqa: BLE001 - retain simulator failure without retry
            unvalidated = {"stage": "scheduled-step", "policy": row,
                "error_type": type(error).__name__, "error": str(error)[:500],
                "runtime_after_failure": _runtime_failure_snapshot(env)}
            reason = "transition-exception"; failure = unvalidated; break
        try:
            recorder.append(result, row["actor_input"], row["raw_action"])
        except Exception as error:  # noqa: BLE001 - retain invalid transition without retry
            unvalidated = {"stage": "physical-retention", "result": attempt.owned(result), "policy": row,
                "error_type": type(error).__name__, "error": str(error)[:500]}
            reason = "transition-validation-failed"; failure = unvalidated; break
        policies.append(row)
        control_ticks.append(attempt.owned(result["control_evidence"]))
        pulse_ticks.append(attempt.owned(result["scheduled_pulse_evidence"]))
        terminal_obs_rows.append(attempt.owned(result["observation"]))
        done = result["terminated"] | result["timed_out"]
        after_time = env._view("time").detach().cpu().clone()
        row["clock"] = {"before": time_before, "after": after_time,
            "accepted_substeps": env.steps.detach().cpu().clone()
                - recorder.ticks[-1]["boundaries"][0]["physics_steps"]}
        if done.any():
            terminal_obs = {k: v.detach().cpu().clone() for k, v in result["observation"].items()}
            obs_td = observations(terminal_obs, WORLDS)
            terminal_value = critic(obs_td).detach().cpu().clone()
            rewards = timeout_rewards(result["reward"], result["terminated"],
                                      result["timed_out"], terminal_value)
            terminal_policy = {"terminal_observation": terminal_obs,
                "terminal_critic_value": terminal_value, "timeout_reward": rewards.detach().cpu().clone(),
                "done_mask": done.detach().cpu().clone()}
            pre_controls = attempt.owned(env._control_snapshot())
            pre = {"frame": attempt.owned(env.snapshot()), "controls": pre_controls,
                "live": env.live.detach().cpu().clone(),
                "terminal_records": attempt.owned(result["terminal_records"]),
                "qpos": env._view("qpos").detach().cpu().clone(),
                "qvel": env._view("qvel").detach().cpu().clone(),
                "time": env._view("time").detach().cpu().clone(),
                "xfrc_applied": env._view("xfrc_applied").detach().cpu().clone(),
                "qfrc_applied": env._view("qfrc_applied").detach().cpu().clone()}
            try:
                reset_records = env.reset(done)
            except Exception as error:  # noqa: BLE001 - preserve explicit reset failure
                terminal_reset = {"before": pre, "done": done.cpu().clone(),
                    "reset_calls": 1,
                    "reset_error": {"type": type(error).__name__, "message": str(error)[:500]}}
                unvalidated = {"stage": "reset", "result": attempt.owned(result)}
                unvalidated["runtime_after_failure"] = _runtime_failure_snapshot(env)
                reason = "reset-exception"
                failure = {"stage": "reset", "error": str(error)[:500],
                           "error_type": type(error).__name__}
                break
            post = {"frame": attempt.owned(env.snapshot()),
                "controls": attempt.owned(env._control_snapshot()),
                "live": env.live.detach().cpu().clone(),
                "qpos": env._view("qpos").detach().cpu().clone(),
                "qvel": env._view("qvel").detach().cpu().clone(),
                "time": env._view("time").detach().cpu().clone(),
                "xfrc_applied": env._view("xfrc_applied").detach().cpu().clone(),
                "qfrc_applied": env._view("qfrc_applied").detach().cpu().clone()}
            full_timeout = bool(done.all() and result["timed_out"].all()
                                and (env.steps == 0).all()
                                and (pre["frame"]["physics_steps"] == EPISODE_STEPS).all())
            selective = bool(done.any() and (~done).any())
            terminal_reset = {"before": pre, "done": done.cpu().clone(),
                "reset_calls": 1, "reset_records": attempt.owned(reset_records), "after": post,
                "initial_controls": initial_controls,
                "initial_qpos": initial_qpos,
                "full_timeout": full_timeout, "selective_reset": selective}
            reason = "first-natural-terminal"; break
    elapsed = float(clock() - started)
    caller_after = torch.random.get_rng_state().clone()
    if not torch.equal(caller_before, caller_after):
        reason = "caller-rng-changed"
        failure = {"stage": "rng-closeout", "error_type": "RuntimeError",
                   "error": "caller CPU RNG changed", "prior": failure}
    value = dict(protocol=PROTOCOL, binding=b, declaration=declaration,
        compiled_plant=deepcopy(env.binding),
        payload=attempt.owned({"binding": b, "initial": recorder.initial, "ticks": recorder.ticks}),
        policy_ticks=policies,
        control_evidence={"protocol": control.PROTOCOL, "binding": deepcopy(b), "ticks": control_ticks},
        scheduled_pulse_evidence=pulse_ticks, terminal_observations=terminal_obs_rows,
        terminal_policy=terminal_policy, terminal_reset=terminal_reset,
        unvalidated_transition=unvalidated, private_rng={"initial": private_before, "final": private},
        caller_rng={"initial": caller_before, "final": caller_after},
        model_state={"initial_sha256": initial_hash,
            "final_sha256": checkpoint.state_hash(checkpoint.states_of(actor, critic)),
            "unchanged": initial_hash == checkpoint.state_hash(checkpoint.states_of(actor, critic))},
        collection={"policy_ticks": len(policies), "elapsed_seconds": elapsed,
            "stop_reason": reason, "failure": failure},
        backend={"torch_device": "cpu", "warp_is_cuda": False},
        training_update_performed=False, whole_trajectory_physics_resimulated=False,
        thermal_model_applied=False, **baseline.FALSE_FLAGS)
    return attempt.owned(value)


def _check_pulses(value):
    """Validate every full-force phase with this protocol's own pulse windows."""
    payload, declaration, compiled = value["payload"], value["declaration"], value["compiled_plant"]
    ticks = payload["ticks"]; pulse_rows = value["scheduled_pulse_evidence"]
    control_rows = value["control_evidence"]["ticks"]
    require(len(ticks) == len(pulse_rows) == len(control_rows), "pulse/control tick coverage")
    windows = [0, 0]; delivered = [0, 0]; checked = 0
    for tick, pulses, ctl in zip(ticks, pulse_rows, control_rows):
        frames = tick["boundaries"]
        proposals = [p for p in ctl["proposals"] if p["accepted"].any()]
        require(len(pulses) == len(proposals) == len(frames)-1, "substep evidence alignment")
        for entry, before, after, proposal in zip(pulses, frames, frames[1:], proposals):
            require(set(entry) == PULSE_KEYS, "exact scheduled force entry schema")
            require(entry["schedule_sha256"] == schedule.binding_sha256(declaration)
                    and entry["binding_sha256"] == sha256(canonical(compiled).encode()).hexdigest(),
                    "force entry binding")
            attempt.tensor(entry["before_steps"], (WORLDS,), torch.int64, "force clocks")
            attempt.tensor(entry["accepted"], (WORLDS,), torch.bool, "accepted rows")
            for key, shape in (("pre_xfrc", (WORLDS, compiled["nbody"], 6)),
                               ("post_xfrc", (WORLDS, compiled["nbody"], 6)),
                               ("pre_qfrc", (WORLDS, 20)), ("post_qfrc", (WORLDS, 20))):
                attempt.tensor(entry[key], shape, torch.float32, "scheduled "+key)
            require(torch.equal(entry["before_steps"], before["physics_steps"])
                    and torch.equal(entry["accepted"], after["physics_steps"]
                                    - before["physics_steps"] == 1)
                    and torch.equal(entry["accepted"], proposal["accepted"]),
                    "force accepted-step ledger")
            steps, accepted = entry["before_steps"].tolist(), entry["accepted"].tolist()
            schedule.validate_wrenches(entry["pre_xfrc"].tolist(), entry["pre_qfrc"].tolist(),
                declaration, steps, accepted, compiled["nbody"], compiled["body_id"])
            schedule.validate_wrenches(entry["post_xfrc"].tolist(), entry["post_qfrc"].tolist(),
                declaration, steps, [False, False], compiled["nbody"], compiled["body_id"])
            active = schedule.window_mask(declaration, steps, accepted)
            for i, yes in enumerate(active):
                windows[i] += int(yes)
                delivered[i] += int(yes and any(declaration["row_cells"][i]["force_world_newtons"]))
            require((entry["phases"] is not None) == any(active), "all and only active force phases")
            if any(active):
                phases = entry["phases"]
                require(set(phases) == {"forced_pre", "integrated", "unforced_post"},
                        "complete forced/integrated/unforced phase set")
                ctrl = proposal["committed"]["ctrl"]
                ppo_trace._phase(phases["forced_pre"], compiled, declaration, steps, accepted, ctrl)
                ppo_trace._phase(phases["integrated"], compiled, declaration, steps, accepted, ctrl)
                ppo_trace._phase(phases["unforced_post"], compiled, declaration,
                                 steps, [False, False], ctrl)
                pre, integrated, post = (phases[k] for k in
                                         ("forced_pre", "integrated", "unforced_post"))
                for phase, frame in ((pre, before), (integrated, after), (post, after)):
                    for key in ("qpos", "qvel"):
                        require(torch.equal(phase["solved"]["kinematics"][key], frame[key]),
                                "force phase physical boundary")
                require(torch.equal(phases["integrated"]["solved"]["kinematics"]["time"],
                    phases["forced_pre"]["solved"]["kinematics"]["time"] +
                    torch.tensor(accepted, dtype=torch.float32) * .002),
                    "accepted substep time increments")
                require(fixture.exact_tree(integrated["solved"]["kinematics"],
                                           post["solved"]["kinematics"]),
                        "unforced post-solve does not integrate again")
                for key in ("dynamics", "solver", "contacts", "constraints"):
                    require(fixture.exact_tree(pre["solved"][key], integrated["solved"][key]),
                            "forced solve retained phase field: "+key)
                require(all(fixture.exact_tree(pre["motor_fields"], phase["motor_fields"])
                            for phase in (integrated, post)), "force phase motor fields unchanged")
            checked += 1
    complete = _pulse_complete(windows, delivered, len(ticks), checked)
    require(windows[0] <= 10 and windows[1] <= 20 and delivered[0] == 0
            and delivered[1] <= 20, "bounded declared force phase prefix")
    return {"checked_physics_steps": checked, "window_steps_per_row": windows,
        "delivered_nonzero_steps_per_row": delivered,
        # Prefix consistency and full protocol coverage are distinct claims:
        # a valid early-terminal trace may have every captured force phase
        # checked without reaching the complete declared pulse windows.
        "recorded_force_prefix_checked": True,
        "complete_force_phase_checks": complete}


def _check_pre_action_inputs(previous_frame, policy_row):
    """Bind the sampled policy inputs to the frame before this action.

    The runtime records the first physical boundary after set_actions(), which
    refreshes realized-correction observation features.  Those post-action
    features are not the observations from which this action was sampled.
    """
    observation = previous_frame["observation"]
    require(torch.equal(policy_row["actor_input"], observation["actor"])
            and torch.equal(policy_row["critic_input"], observation["critic"]),
            "policy inputs bind to pre-action observation")


def _check_terminal_policy(value, done, critic):
    policy = value["terminal_policy"]
    require(type(policy) is dict and set(policy) == {
        "terminal_observation", "terminal_critic_value", "timeout_reward", "done_mask"},
        "terminal policy and timeout bootstrap record")
    attempt.tensor(policy["terminal_critic_value"], (WORLDS, 1), torch.float32, "terminal critic")
    attempt.tensor(policy["timeout_reward"], (WORLDS,), torch.float32, "timeout reward")
    require(torch.equal(policy["done_mask"], done), "terminal critic uses exact done mask")
    final_tick = value["payload"]["ticks"][-1]
    require(torch.equal(policy["terminal_observation"]["actor"],
                        final_tick["boundaries"][-1]["observation"]["actor"])
            and torch.equal(policy["terminal_observation"]["critic"],
                            final_tick["boundaries"][-1]["observation"]["critic"]),
            "retained terminal observation matches pre-reset frame")
    terminal_obs = observations(policy["terminal_observation"], WORLDS)
    expected_value = critic(terminal_obs).detach().cpu()
    require(torch.equal(policy["terminal_critic_value"], expected_value),
            "terminal critic exactly recomputed from terminal observation")
    expected_reward = timeout_rewards(final_tick["reward"], final_tick["terminated"],
                                      final_tick["timed_out"], expected_value)
    require(torch.equal(policy["timeout_reward"], expected_reward),
            "single exact timeout bootstrap reward")


def _validate_terminal(value, critic):
    terminal = value["terminal_reset"]
    if terminal is None:
        require(value["terminal_policy"] is None, "no terminal policy without natural terminal")
        if value["payload"]["ticks"]:
            last = value["payload"]["ticks"][-1]
            require(not (last["terminated"] | last["timed_out"]).any()
                    and all(record is None for record in last["terminal_records"]),
                    "terminal reset evidence cannot be omitted after first natural terminal")
        return {"full_timeout_qualified": False, "timeout_reset_qualified": False,
                    "selective_reset_qualified": False}
    if "reset_error" in terminal:
        before = terminal.get("before")
        done = terminal.get("done")
        require(type(before) is dict and torch.is_tensor(done),
                "failed reset retains pre-reset terminal evidence")
        attempt.tensor(done, (WORLDS,), torch.bool, "reset failure done mask")
        require(terminal["reset_calls"] == 1, "one actual terminal reset attempt")
        require(torch.equal(done, torch.tensor(
            [r is not None for r in before["terminal_records"]], dtype=torch.bool)),
            "failed reset retains exact terminal rows")
        require(torch.equal(before["live"], ~done), "failed reset retains terminal live mask")
        _check_terminal_frame(before, value["payload"]["ticks"][-1])
        for i, record in enumerate(before["terminal_records"]):
            if record is not None:
                attempt.terminal_matches(record, before["frame"], i)
        _check_terminal_policy(value, done, critic)
        return {"full_timeout_qualified": False, "timeout_reset_qualified": False,
                    "selective_reset_qualified": False}
    require(type(terminal) is dict and {"before", "done", "reset_calls", "reset_records", "after",
            "initial_controls", "initial_qpos", "full_timeout", "selective_reset"}.issubset(terminal),
            "retained before/after explicit reset")
    require(terminal["reset_calls"] == 1, "one actual terminal reset")
    before, after, done = terminal["before"], terminal["after"], terminal["done"]
    attempt.tensor(done, (WORLDS,), torch.bool, "terminal done mask")
    records = before["terminal_records"]
    require(len(records) == WORLDS and torch.equal(done,
        torch.tensor([r is not None for r in records], dtype=torch.bool)), "terminal done/record alignment")
    require(torch.equal(before["live"], ~done), "natural terminal closes only done rows")
    expected_resets = [r if bool(done[i]) else None for i, r in enumerate(records)]
    require(_tree_equal(terminal["reset_records"], expected_resets), "one reset returns exact terminal rows")
    frame = before["frame"]
    _check_terminal_frame(before, value["payload"]["ticks"][-1])
    for i, record in enumerate(records):
        if record is not None:
            attempt.terminal_matches(record, frame, i)
    full, selective = _reset_claims(done, records, frame["physics_steps"],
                                    terminal["full_timeout"], terminal["selective_reset"])
    untouched = ~done
    _check_reset_siblings(before, after, done)
    _check_reset_rows(after, done, value["payload"]["initial"])
    attempt.tensor(terminal["initial_qpos"], (21,), torch.float32, "original reset qpos")
    require(torch.equal(terminal["initial_qpos"].expand(WORLDS, -1),
                        value["payload"]["initial"]["qpos"])
            and torch.equal(before["qpos"], frame["qpos"])
            and torch.equal(before["qvel"], frame["qvel"])
            and torch.equal(after["frame"]["qpos"], after["qpos"])
            and torch.equal(after["frame"]["qvel"], after["qvel"]),
            "reset and terminal kinematics bound to physical frames")
    nbody = value["compiled_plant"]["nbody"]
    for where, record in (("pre", before), ("post", after)):
        attempt.tensor(record["xfrc_applied"], (WORLDS, nbody, 6), torch.float32,
                       where+"-reset full external force")
        attempt.tensor(record["qfrc_applied"], (WORLDS, 20), torch.float32,
                       where+"-reset generalized force")
        require(not record["xfrc_applied"].any() and not record["qfrc_applied"].any(),
                "complete applied-force arrays are zero around reset")
    require(torch.equal(before["frame"]["physics_steps"][untouched],
                        after["frame"]["physics_steps"][untouched])
            and torch.equal(before["live"][untouched], after["live"][untouched]),
            "selectively untouched sibling clocks and liveness")
    for name in before["controls"]:
        post = after["controls"][name]
        require(torch.equal(post[done], terminal["initial_controls"][name][done]),
                "reset row control restored: "+name)
    require(set(terminal["initial_controls"]) == set(after["controls"]),
            "complete post-reset control inventory")
    require(value["control_evidence"]["ticks"] and
            _tree_equal(value["control_evidence"]["ticks"][0]["initial"],
                        terminal["initial_controls"]),
            "retained original controls match pre-collection state")
    require(torch.equal(before["time"], value["policy_ticks"][-1]["clock"]["after"]),
            "terminal time equals final per-call clock")
    _check_terminal_policy(value, done, critic)
    return {"full_timeout_qualified": full,
        "timeout_reset_qualified": full,
        "selective_reset_qualified": selective}


def _validate_collection(collection, has_terminal):
    require(set(collection) == {"policy_ticks", "elapsed_seconds", "stop_reason", "failure"}
            and type(collection["policy_ticks"]) is int
            and type(collection["elapsed_seconds"]) is float
            and math.isfinite(collection["elapsed_seconds"])
            and collection["elapsed_seconds"] >= 0
            and type(collection["stop_reason"]) is str,
            "typed collection outcome and elapsed time")
    allowed = {"first-natural-terminal", "policy-call-limit", "deadline-exhausted",
        "sampling-exception", "transition-exception", "transition-validation-failed",
        "reset-exception", "caller-rng-changed"}
    require(collection["stop_reason"] in allowed, "declared first-terminal stop reason")
    failure = collection["failure"]
    require(failure is None or type(failure) is dict, "typed retained collection failure")
    if failure is not None:
        require(type(failure.get("stage")) is str and type(failure.get("error")) is str
                and type(failure.get("error_type")) is str,
                "typed stage and error for failed collection")
    if collection["stop_reason"] == "first-natural-terminal":
        require(has_terminal and failure is None, "stop immediately after first natural terminal")
    else:
        require(not has_terminal or collection["stop_reason"] in {
            "reset-exception", "caller-rng-changed"},
            "terminal collection cannot continue or be relabeled")
    return collection["elapsed_seconds"] < WALL_LIMIT


@torch.no_grad()
def score(value, raw_parent, expected_schedule, expected_plant):
    required = {"protocol", "binding", "declaration", "compiled_plant", "payload", "policy_ticks",
        "control_evidence", "scheduled_pulse_evidence", "terminal_observations", "terminal_policy",
        "terminal_reset", "unvalidated_transition", "private_rng", "caller_rng", "model_state",
        "collection", "backend", "training_update_performed", "whole_trajectory_physics_resimulated",
        "thermal_model_applied"} | set(baseline.FALSE_FLAGS)
    require(type(value) is dict and set(value) == required and value["protocol"] == PROTOCOL,
            "exact first-terminal trace schema")
    require(all(value[k] is False for k in baseline.FALSE_FLAGS)
            and value["training_update_performed"] is False
            and value["whole_trajectory_physics_resimulated"] is False
            and value["thermal_model_applied"] is False, "record-only non-admission flags")
    require(value["backend"] == {"torch_device": "cpu", "warp_is_cuda": False},
            "actual CPU-only capture provenance")
    require(value["unvalidated_transition"] is None
            or type(value["unvalidated_transition"]) is dict,
            "retained explicitly unvalidated partial transition")
    require(type(value["private_rng"]) is dict
            and set(value["private_rng"]) == {"initial", "final"}
            and type(value["caller_rng"]) is dict
            and set(value["caller_rng"]) == {"initial", "final"}, "typed private/caller RNG records")
    require(all(torch.is_tensor(state) and state.dtype == torch.uint8 and state.device.type == "cpu"
                for state in (*value["private_rng"].values(), *value["caller_rng"].values())),
            "CPU private and caller RNG states")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "CUDA-hidden CPU consistency scorer")
    declaration = _target_schedule(expected_schedule)
    require(canonical(value["declaration"]) == canonical(declaration)
            and canonical(value["compiled_plant"]) == canonical(expected_plant),
            "expected fixed schedule and compiled plant")
    b = value["binding"]
    require(_binding_matches(b, declaration, expected_plant),
            "source, schedule, plant, parent and profile binding")
    require(type(raw_parent) is bytes and sha256(raw_parent).hexdigest()
            == baseline.CHECKPOINT_SHA256, "parent hash before deserialize")
    loaded = parent.load_parent(raw_parent)
    actor, critic = loaded["actor"], loaded["critic"]
    require(checkpoint.state_hash(checkpoint.states_of(actor, critic))
            == parent.PARENT_STATE_SHA256 == b["parent_state_sha256"], "exact frozen actor and critic")
    require(canonical(value["payload"]["binding"]) == canonical(b), "physical trace binding")
    recorder = TerminalTrace(b, value["payload"]["initial"], declaration, expected_plant)
    rng = value["private_rng"]["initial"].detach().cpu().clone()
    require(rng.dtype == torch.uint8 and torch.equal(rng,
        torch.Generator(device="cpu").manual_seed(SEED).get_state()), "fixed private sampler seed")
    require(type(value["payload"]["ticks"]) is list
            and type(value["policy_ticks"]) is list
            and len(value["payload"]["ticks"]) == len(value["policy_ticks"])
            == len(value["terminal_observations"]), "full trace/policy/observation inventory")
    previous_clock = torch.zeros(WORLDS, dtype=torch.float32)
    for idx, (tick, row) in enumerate(zip(value["payload"]["ticks"], value["policy_ticks"])):
        require(set(row) == {"rng_before", "rng_after", "actor_input", "critic_input",
            "raw_action", "value", "log_prob", "mu", "sigma", "clock"},
            "exact stochastic tick fields")
        require(torch.equal(row["rng_before"], rng), "private RNG transition chain before sample")
        # `recorder.last` is the pre-action snapshot.  In contrast,
        # tick.boundaries[0] is captured after runtime.set_actions() and may
        # contain newly realized correction values in its critic observation.
        _check_pre_action_inputs(recorder.last, row)
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(rng)
            action, critic_value, log_prob, mu, sigma = _sample(
                actor, critic, row["actor_input"], row["critic_input"])
            next_rng = torch.random.get_rng_state().clone()
        require(torch.equal(action, row["raw_action"]) and torch.equal(critic_value, row["value"])
                and torch.equal(log_prob, row["log_prob"]) and torch.equal(mu, row["mu"])
                and torch.equal(sigma, row["sigma"]) and torch.equal(next_rng, row["rng_after"]),
                "exact frozen-parent stochastic action/critic/Gaussian/private RNG")
        rng = next_rng
        require(tick["actor_input"].equal(row["actor_input"])
                and tick["actions"].equal(row["raw_action"])
                and torch.equal(value["terminal_observations"][idx]["actor"],
                                tick["boundaries"][-1]["observation"]["actor"])
                and torch.equal(value["terminal_observations"][idx]["critic"],
                                tick["boundaries"][-1]["observation"]["critic"]),
                "policy observations/action bound to physical transition")
        _check_policy_clock(row["clock"], tick)
        require(torch.equal(row["clock"]["before"], previous_clock),
                "policy-call clock continuity from initial zero")
        previous_clock = row["clock"]["after"]
        recorder.append(tick, row["actor_input"], row["raw_action"])
    _check_private_chain(value["private_rng"]["initial"], value["policy_ticks"],
                         value["private_rng"]["final"])
    require(torch.equal(value["caller_rng"]["initial"], value["caller_rng"]["final"]),
            "caller RNG unchanged")
    require(value["model_state"]["unchanged"] is True
            and value["model_state"]["initial_sha256"] == value["model_state"]["final_sha256"]
            == parent.PARENT_STATE_SHA256, "frozen parent unchanged")
    require(value["collection"]["policy_ticks"] == len(value["policy_ticks"])
            and len(value["policy_ticks"]) == len(value["payload"]["ticks"]) <= HORIZON,
            "one-policy-call-to-one-physical-tick ordering")
    if recorder.ticks:
        physical = plant.check_trace(value["payload"], expected_plant["selected_plant"])
        control_result = control.replay(value["control_evidence"], value["payload"],
                                          expected_plant["selected_plant"])
    else:
        physical = {"compiled_plant_checked": False}
        control_result = None
    pulses = _check_pulses(value)
    terminal = _validate_terminal(value, critic)
    within_cap = _validate_collection(value["collection"], value["terminal_reset"] is not None)
    qualifiers = {name: result and within_cap for name, result in terminal.items()}
    return dict(protocol=PROTOCOL, validated_policy_ticks=len(value["policy_ticks"]),
        collection=deepcopy(value["collection"]), physical=physical, control=control_result,
        force=pulses, **qualifiers, exact_policy_replay=True, private_rng=True, caller_rng=True,
        terminal_critic_bootstrap_exact=(value["terminal_policy"] is not None),
        compiled_plant_checked=bool(recorder.ticks), control_trace_checked=control_result is not None,
        physical_trace_checked=bool(recorder.ticks),
        collection_cap_respected=within_cap,
        recorded_force_prefix_checked=pulses["recorded_force_prefix_checked"],
        complete_force_phase_checks=pulses["complete_force_phase_checks"],
        whole_trajectory_physics_resimulated=False, thermal_model_applied=False,
        **baseline.FALSE_FLAGS)


def encode(value):
    out = io.BytesIO(); torch.save(attempt.owned(value), out); raw = out.getvalue()
    require(len(raw) <= LIMIT, "bounded terminal trace artifact")
    return raw


def verify(raw, expected_sha256, raw_parent, expected_schedule, expected_plant):
    require(type(raw) is bytes and len(raw) <= LIMIT
            and sha256(raw).hexdigest() == expected_sha256, "exact bounded raw trace hash")
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    return score(value, raw_parent, expected_schedule, expected_plant)
