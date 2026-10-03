"""Bounded CPU stochastic PPO transition trace for a fixed two-row schedule.

Sibling evidence only: one real 28-transition collection, no optimizer update,
whole-trajectory physics replay, GPU qualification, or execution admission.
"""
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import distribution
import io
import math
import os
import time

import torch

from mjlab_microduck import stance_attempt_trace as attempt
from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_control_evidence as control
from mjlab_microduck import stance_cpu_replay_profile as profile
from mjlab_microduck import stance_disturbance_fixture as fixture
from mjlab_microduck import stance_forward_probe as forward
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_parent as parent
from mjlab_microduck import stance_recovery_ppo_bridge as bridge_module
from mjlab_microduck import stance_recovery_policy_preparation as preparation
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck.stance_transition import PhysicsState, physical_failures
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = "football-b1d-cpu-scheduled-ppo-transition-trace-v1"
SEED, HORIZON, WALL_LIMIT, LIMIT = 653, 28, 120.0, 64 * 1024 * 1024
WORLD_CELLS = ("zero-wrench", "+x-2n-20steps-t250")
PULSE_KEYS = {"before_steps", "accepted", "pre_xfrc", "pre_qfrc", "post_xfrc",
              "post_qfrc", "phases", "schedule_sha256", "binding_sha256"}
POLICY_KEYS = {"rng_before", "rng_after", "actor_input", "raw_action", "value",
               "critic_input", "log_prob", "mu", "sigma", "distribution_params",
               "stored_reward", "done"}
STORAGE_KEYS = {"observations", "actions", "rewards", "dones", "values",
                "actions_log_prob", "distribution_params", "returns", "advantages"}


def _schema(actor, critic):
    return {group: {name: (tuple(value.shape), str(value.dtype))
                    for name, value in model.state_dict().items()}
            for group, model in (("actor", actor), ("critic", critic))}


def _state_hash(actor, critic):
    return checkpoint.state_hash(checkpoint.states_of(actor, critic))


def _elapsed_within_cap(value):
    require(type(value) is float and math.isfinite(value) and value >= 0,
            "finite nonnegative elapsed time")
    return value < WALL_LIMIT


def _accepted_complete(*, tick_count, storage_step, stop_reason, elapsed_seconds,
                       private_rng_closed, caller_rng_unchanged, terminals):
    within_cap = _elapsed_within_cap(elapsed_seconds)
    return (tick_count == HORIZON and storage_step == HORIZON
            and stop_reason == "transition-limit" and within_cap
            and private_rng_closed and caller_rng_unchanged
            and terminals == [None, None])


def _target_schedule(value):
    value = schedule.checked(value)
    require(value["split"] == "training" and value["stage"] == "dose"
            and value["worlds"] == 2 and tuple(value["cell_ids"]) == WORLD_CELLS,
            "exact two-row training dose schedule")
    zero, pulse = value["row_cells"]
    require(zero["onset_step"] == 500 and zero["duration_steps"] == 10
            and not any(zero["force_world_newtons"])
            and pulse["onset_step"] == 250 and pulse["duration_steps"] == 20
            and pulse["force_world_newtons"] == [2.0, 0.0, 0.0],
            "exact zero and +x 2 N 20-step onset-250 rows")
    return value


def binding(declaration, compiled_plant, cpu_profile):
    declaration = _target_schedule(declaration)
    require(type(compiled_plant) is dict and set(compiled_plant) ==
            {"selected_plant", "nbody", "body_names", "body_name", "body_id"},
            "actual compiled plant binding")
    baseline.force._binding(compiled_plant)
    profile.validate_receipt(cpu_profile)
    return dict(protocol=PROTOCOL, source=declaration["source"], worlds=2,
        capture_device="cpu", horizon=HORIZON,
        schedule_sha256=schedule.binding_sha256(declaration),
        plant_sha256=sha256(canonical(compiled_plant).encode()).hexdigest(),
        parent_checkpoint_sha256=baseline.CHECKPOINT_SHA256,
        parent_checkpoint_identity=parent.expected_identity(),
        parent_state_sha256=parent.PARENT_STATE_SHA256, learner_seed=SEED,
        ppo_config=deepcopy(bridge_module.CONFIG),
        ppo_source_sha256=deepcopy(preparation.PINS),
        cpu_math_profile=deepcopy(cpu_profile))


class ScheduledPPOTrace(attempt.FirstAttemptTrace):
    """Existing physical continuity validator with a sibling CPU2 binding."""
    def __init__(self, value, initial, declaration, compiled_plant):
        require(canonical(value) == canonical(binding(declaration, compiled_plant,
                value["cpu_math_profile"])), "exact stochastic scheduled binding")
        self.binding = deepcopy(value); self.n = 2; self.faulted = False
        self._tensor_device = "cpu"
        frame = attempt.owned(initial); attempt.validate_frame(frame, 2)
        require(str(initial["qpos"].device) == "cpu" and not frame["physics_steps"].any()
                and not physical_failures(PhysicsState(**frame["state"]),
                                          frame["physics_steps"]).any(),
                "fresh valid two-world CPU first attempt")
        self.initial = frame; self.last = frame; self.ticks = []
        self.terminals = [None, None]

    def _initialize(self, *_):
        raise RuntimeError("stochastic recovery trace requires its sibling binding")


def _storage(storage):
    require(storage.distribution_params is not None, "sampled Gaussian storage exists")
    return dict(observations={k: v.detach().cpu().clone()
                              for k, v in storage.observations.items()},
        actions=storage.actions.detach().cpu().clone(),
        rewards=storage.rewards.detach().cpu().clone(),
        dones=storage.dones.detach().cpu().clone(),
        values=storage.values.detach().cpu().clone(),
        actions_log_prob=storage.actions_log_prob.detach().cpu().clone(),
        distribution_params=[v.detach().cpu().clone() for v in storage.distribution_params],
        returns=storage.returns.detach().cpu().clone(),
        advantages=storage.advantages.detach().cpu().clone())


def _fresh(learner, raw_parent):
    from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
    env = learner.env
    require(type(env) is ScheduledRecoveryRuntime and env.n == 2
            and torch.device(env.device).type == "cpu" and not env.wp_device.is_cuda
            and env.forward_graph is None and env.solved_field_check == "packed",
            "fresh native eager CPU2 scheduled runtime")
    require(learner.synthetic_fixture is False and learner.learner_seed == SEED
            and learner.worlds == 2 and learner.phase == "empty" and not learner.faulted
            and learner.storage.step == 0 and learner.storage.num_transitions_per_env == HORIZON
            and type(learner.algorithm.optimizer) is torch.optim.Adam
            and not learner.algorithm.optimizer.state and learner.optimizer_steps == 0
            and learner.completed_updates == 0, "fresh no-update seed-653 bridge")
    require(sha256(raw_parent).hexdigest() == baseline.CHECKPOINT_SHA256
            and learner.initial_state_sha256 == parent.PARENT_STATE_SHA256
            and learner.parent_receipt["parent_checkpoint_sha256"] == baseline.CHECKPOINT_SHA256,
            "exact immutable parent bytes and loaded weights")
    require(schedule.checked(env.schedule_declaration) == _target_schedule(env.schedule_declaration)
            and env.live.all() and not env.steps.any(), "fresh fixed row schedule")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "CUDA-hidden CPU2 collection")


@torch.no_grad()
def collect(learner, raw_parent, *, deadline_monotonic, clock=time.monotonic):
    """Collect once; exceptions and early terminals produce rejected retainable prefixes."""
    _fresh(learner, raw_parent)
    require(clock() < deadline_monotonic, "CPU2 capture before deadline")
    declaration = _target_schedule(learner.schedule)
    cpu_profile = profile.checked_receipt()
    b = binding(declaration, learner.env.binding, cpu_profile)
    recorder = ScheduledPPOTrace(b, learner.env.snapshot(), declaration, learner.env.binding)
    controls = dict(protocol=control.PROTOCOL, binding=deepcopy(b), ticks=[])
    policies, pulse_rows, bridge_receipts, terminal_observations = [], [], [], []
    unvalidated_transition = None
    caller_initial = torch.random.get_rng_state().clone()
    private_initial = learner.private_rng_state.detach().cpu().clone()
    model_initial = _state_hash(learner.actor, learner.critic)
    started = clock(); reason = "transition-limit"; failure = None
    require(deadline_monotonic <= started + WALL_LIMIT, "120-second bounded CPU2 collection window")
    for _ in range(HORIZON):
        if clock() >= deadline_monotonic:
            reason = "wall-budget-exhausted"; break
        obs = learner_module_observations(learner)
        rng_before = learner.private_rng_state.detach().cpu().clone()
        try:
            result = learner.collect_one(capture_control=True)
        except Exception as error:
            failure = dict(stage="collect-one", error_type=type(error).__name__,
                           error=str(error)[:500])
            reason = "collection-exception"; break
        sampled = result["bridge_receipt"]["raw_actions"]
        cursor = learner.storage.step - 1
        require(cursor == len(policies), "single ordered storage cursor")
        params = [v.detach().cpu().clone() for v in learner.storage.distribution_params]
        row = dict(rng_before=rng_before,
            rng_after=learner.private_rng_state.detach().cpu().clone(),
            actor_input=obs["actor"].detach().cpu().clone(),
            critic_input=obs["critic"].detach().cpu().clone(),
            raw_action=sampled.detach().cpu().clone(),
            value=learner.storage.values[cursor].detach().cpu().clone(),
            log_prob=learner.storage.actions_log_prob[cursor].detach().cpu().clone(),
            mu=params[0][cursor].clone(), sigma=params[1][cursor].clone(),
            distribution_params=[v[cursor].clone() for v in params],
            stored_reward=learner.storage.rewards[cursor].detach().cpu().clone(),
            done=learner.storage.dones[cursor].detach().cpu().clone())
        try:
            recorder.append(result, row["actor_input"], row["raw_action"])
        except Exception as error:
            unvalidated_transition = dict(result=attempt.owned(result), policy=attempt.owned(row))
            failure = dict(stage="physical-retention", error_type=type(error).__name__,
                           error=str(error)[:500])
            reason = "collection-exception"; break
        policies.append(row)
        controls["ticks"].append(attempt.owned(result["control_evidence"]))
        pulse_rows.append(attempt.owned(result["scheduled_pulse_evidence"]))
        bridge_receipts.append(attempt.owned(result["bridge_receipt"]))
        terminal_observations.append(attempt.owned(result["observation"]))
        if (result["terminated"] | result["timed_out"]).any():
            reason = "unexpected-terminal"; break
    elapsed = float(clock()-started)
    if elapsed >= WALL_LIMIT and reason == "transition-limit":
        reason = "wall-budget-exhausted"
    caller_final = torch.random.get_rng_state().clone()
    if not torch.equal(caller_initial, caller_final):
        prior_failure = failure
        reason = "caller-rng-changed"
        failure = dict(stage="rng-closeout", error_type="RuntimeError",
                       error="caller CPU RNG changed", prior_failure=prior_failure)
    try:
        storage = _storage(learner.storage)
    except Exception as error:
        storage = None
        if failure is None:
            reason = "storage-retention-failed"
            failure = dict(stage="storage-retention", error_type=type(error).__name__,
                           error=str(error)[:500])
    model_final = _state_hash(learner.actor, learner.critic)
    accepted = (reason == "transition-limit" and len(policies) == HORIZON
                and elapsed < WALL_LIMIT and learner.storage.step == HORIZON
                and not learner.terminal_events)
    value = dict(protocol=PROTOCOL, binding=b, declaration=declaration,
        compiled_plant=deepcopy(learner.env.binding),
        payload=recorder.payload(), policy_ticks=policies, storage=storage,
        control_evidence=controls, scheduled_pulse_evidence=pulse_rows,
        bridge_receipts=bridge_receipts, terminal_observations=terminal_observations,
        unvalidated_transition=unvalidated_transition,
        private_rng=dict(initial=private_initial,
                         final=learner.private_rng_state.detach().cpu().clone()),
        caller_rng=dict(initial=caller_initial, final=caller_final),
        model_state=dict(initial_sha256=model_initial, final_sha256=model_final,
                         schema=_schema(learner.actor, learner.critic),
                         unchanged=model_initial == model_final),
        optimizer=dict(type="Adam", state_entries=len(learner.algorithm.optimizer.state),
            optimizer_steps=learner.optimizer_steps, updates=learner.completed_updates,
            storage_step=learner.storage.step, returns_uncomputed=True,
            advantages_uncomputed=True, storage_cleared=False),
        collection=dict(policy_ticks=len(policies), elapsed_seconds=elapsed,
            stop_reason=reason, accepted_complete=accepted, failure=failure),
        backend=dict(torch_device="cpu", warp_is_cuda=False),
        training_update_performed=False, whole_trajectory_physics_resimulated=False,
        thermal_model_applied=False, **baseline.FALSE_FLAGS)
    return attempt.owned(value)


def learner_module_observations(learner):
    # Use the exact pre-action observation captured before collect_one mutates it.
    return bridge_module.observations(learner.env.observations(), learner.worlds)


def _phase(value, compiled, declaration, steps, accepted, ctrl):
    require(type(value) is dict and set(value) == {"solved", "inputs", "motor_fields"},
            "complete two-world phase")
    forward.validate_output(value["solved"], 2)
    inputs = value["inputs"]
    require(set(inputs) == {"ctrl", "xfrc_applied", "qfrc_applied"}, "phase inputs")
    attempt.tensor(inputs["ctrl"], (2, 14), torch.float32, "phase ctrl")
    require(torch.equal(inputs["ctrl"], ctrl), "phase committed control")
    schedule.validate_wrenches(inputs["xfrc_applied"].tolist(), inputs["qfrc_applied"].tolist(),
        declaration, steps, accepted, compiled["nbody"], compiled["body_id"])
    require(set(value["motor_fields"]) == {"dof_frictionloss", "dof_damping"}, "phase motor fields")
    for name, item in value["motor_fields"].items():
        attempt.tensor(item, (2, 20), torch.float32, name)
        require((item >= 0).all(), "nonnegative motor fields")


def check_pulses(value):
    """Full-batch scheduled-force validation; never adapts the CPU1 scorer."""
    payload, declaration, compiled = value["payload"], value["declaration"], value["compiled_plant"]
    pulses, ticks, controls = (value["scheduled_pulse_evidence"], payload["ticks"],
                               value["control_evidence"]["ticks"])
    require(len(pulses) == len(ticks) == len(controls), "force/control tick coverage")
    windows, delivered, checked = [0, 0], [0, 0], 0
    for entries, tick, ctl in zip(pulses, ticks, controls):
        frames = tick["boundaries"]
        proposals = [p for p in ctl["proposals"] if p["accepted"].any()]
        require(len(entries) == len(frames)-1 == len(proposals), "accepted physical-step force coverage")
        for row, before, after, proposal in zip(entries, frames, frames[1:], proposals):
            require(type(row) is dict and set(row) == PULSE_KEYS, "exact n=2 force entry")
            require(row["schedule_sha256"] == schedule.binding_sha256(declaration)
                    and row["binding_sha256"] == sha256(canonical(compiled).encode()).hexdigest(),
                    "force schedule/plant binding")
            attempt.tensor(row["before_steps"], (2,), torch.int64, "force clocks")
            attempt.tensor(row["accepted"], (2,), torch.bool, "accepted mask")
            require(torch.equal(row["before_steps"], before["physics_steps"])
                    and torch.equal(row["accepted"], after["physics_steps"]-before["physics_steps"] == 1)
                    and torch.equal(row["accepted"], proposal["accepted"]), "accepted row clocks")
            steps, accepted = row["before_steps"].tolist(), row["accepted"].tolist()
            for key, shape in (("pre_xfrc", (2, compiled["nbody"], 6)),
                               ("post_xfrc", (2, compiled["nbody"], 6)),
                               ("pre_qfrc", (2, 20)), ("post_qfrc", (2, 20))):
                attempt.tensor(row[key], shape, torch.float32, key)
            schedule.validate_wrenches(row["pre_xfrc"].tolist(), row["pre_qfrc"].tolist(),
                declaration, steps, accepted, compiled["nbody"], compiled["body_id"])
            schedule.validate_wrenches(row["post_xfrc"].tolist(), row["post_qfrc"].tolist(),
                declaration, steps, [False, False], compiled["nbody"], compiled["body_id"])
            active = schedule.window_mask(declaration, steps, accepted)
            for index, yes in enumerate(active):
                windows[index] += int(yes)
                delivered[index] += int(yes and any(declaration["row_cells"][index]["force_world_newtons"]))
            require((row["phases"] is not None) == any(active), "full n=2 phases iff any row window")
            if any(active):
                phases = row["phases"]
                require(set(phases) == {"forced_pre", "integrated", "unforced_post"}, "all force phases")
                ctrl = proposal["committed"]["ctrl"]
                _phase(phases["forced_pre"], compiled, declaration, steps, accepted, ctrl)
                _phase(phases["integrated"], compiled, declaration, steps, accepted, ctrl)
                _phase(phases["unforced_post"], compiled, declaration, steps, [False, False], ctrl)
                pre, integrated, post = (phases[k] for k in ("forced_pre", "integrated", "unforced_post"))
                for name, frame in (("forced_pre", before), ("integrated", after), ("unforced_post", after)):
                    for key in ("qpos", "qvel"):
                        require(torch.equal(phases[name]["solved"]["kinematics"][key], frame[key]),
                                "phase physical boundary "+name+"/"+key)
                expected_time = torch.where(row["accepted"],
                    pre["solved"]["kinematics"]["time"]+.002,
                    pre["solved"]["kinematics"]["time"])
                require(torch.equal(integrated["solved"]["kinematics"]["time"], expected_time),
                        "accepted-row Euler clock")
                require(fixture.exact_tree(integrated["solved"]["kinematics"],
                                           post["solved"]["kinematics"]), "unforced post no integration")
                for key in ("dynamics", "solver", "contacts", "constraints"):
                    require(fixture.exact_tree(pre["solved"][key], integrated["solved"][key]),
                            "forced solve retained "+key)
                require(all(fixture.exact_tree(pre["motor_fields"], phase["motor_fields"])
                            for phase in (integrated, post)), "phase motor fields unchanged")
            checked += 1
    last_terminals = ticks[-1]["terminal_records"] if ticks else []
    full = (len(ticks) == HORIZON and len(last_terminals) == 2
            and all(record is None for record in last_terminals))
    if full:
        require(checked == 280 and windows == [0, 20] and delivered == [0, 20],
                "full 280-substep zero/+x schedule")
    return dict(checked_physics_steps=checked, window_steps_per_row=windows,
        delivered_nonzero_steps_per_row=delivered,
        complete_pulse_delivery=windows == [0, 20] and delivered == [0, 20],
        complete_full_batch_phase_checks=True, unforced_post_arrays_zero=True)


def _check_storage(value, cursor):
    storage, policies = value["storage"], value["policy_ticks"]
    require(type(storage) is dict and set(storage) == STORAGE_KEYS
            and len(policies) <= cursor <= HORIZON,
            "plain dict storage and cursor covering validated transitions")
    require(set(storage["observations"]) == {"actor", "critic"}
            and storage["observations"]["actor"].shape == (HORIZON, 2, 44)
            and storage["observations"]["critic"].shape == (HORIZON, 2, 50)
            and storage["actions"].shape == (HORIZON, 2, 10)
            and storage["values"].shape == storage["actions_log_prob"].shape == (HORIZON, 2, 1)
            and storage["rewards"].shape == storage["dones"].shape
                == storage["returns"].shape == storage["advantages"].shape == (HORIZON, 2, 1)
            and storage["dones"].dtype == torch.uint8
            and len(storage["distribution_params"]) == 2
            and all(v.shape == (HORIZON, 2, 10) for v in storage["distribution_params"]),
            "complete plain 28x2 PPO storage schema")
    for name, item in storage["observations"].items():
        attempt.tensor(item, item.shape, torch.float32, name)
    for item in storage["distribution_params"]:
        attempt.tensor(item, (HORIZON, 2, 10), torch.float32, "distribution parameters")
    for name in ("actions", "rewards", "values", "actions_log_prob", "returns", "advantages"):
        attempt.tensor(storage[name], storage[name].shape, torch.float32, name)
    attempt.tensor(storage["dones"], (HORIZON, 2, 1), torch.uint8, "dones")
    require(((storage["dones"] == 0) | (storage["dones"] == 1)).all(),
            "binary stored done masks")
    require(not storage["returns"].any() and not storage["advantages"].any(),
            "GAE untouched; no optimizer update")
    for t, row in enumerate(policies):
        require(torch.equal(storage["observations"]["actor"][t], row["actor_input"])
                and torch.equal(storage["observations"]["critic"][t], row["critic_input"])
                and torch.equal(storage["actions"][t], row["raw_action"])
                and torch.equal(storage["values"][t], row["value"])
                and torch.equal(storage["actions_log_prob"][t], row["log_prob"])
                and torch.equal(storage["rewards"][t], row["stored_reward"])
                and torch.equal(storage["dones"][t], row["done"]),
                "storage retains exact raw transition")
        for stored, captured in zip(storage["distribution_params"], row["distribution_params"]):
            require(torch.equal(stored[t], captured), "storage Gaussian distribution parameters")
    for name in ("actions", "rewards", "dones", "values", "actions_log_prob"):
        require(not storage[name][cursor:].any(), "unused storage remains zero: "+name)
    for name in ("actor", "critic"):
        require(not storage["observations"][name][cursor:].any(), "unused observations remain zero")
    for item in storage["distribution_params"]:
        require(not item[cursor:].any(), "unused Gaussian storage remains zero")


def _check_bridge_reward(receipt_reward, stored_reward):
    """Bind native vector rewards to RSL's column layout without broadcasting."""
    attempt.tensor(receipt_reward, (2,), torch.float32, "bridge vector reward")
    attempt.tensor(stored_reward, (2, 1), torch.float32, "stored column reward")
    require(torch.equal(receipt_reward, stored_reward[:, 0]),
            "bridge and stored reward values match exactly")
    return True


@torch.no_grad()
def score(value, checkpoint_raw, expected_schedule, expected_plant):
    required = {"protocol", "binding", "declaration", "compiled_plant", "payload",
        "policy_ticks", "storage", "control_evidence", "scheduled_pulse_evidence",
        "bridge_receipts", "terminal_observations", "unvalidated_transition",
        "private_rng", "caller_rng", "model_state", "optimizer", "collection",
        "backend", "training_update_performed", "whole_trajectory_physics_resimulated",
        "thermal_model_applied"} | set(baseline.FALSE_FLAGS)
    require(type(value) is dict and set(value) == required and value["protocol"] == PROTOCOL,
            "exact CPU2 stochastic trace schema")
    require(all(value[k] is False for k in baseline.FALSE_FLAGS)
            and value["training_update_performed"] is False
            and value["whole_trajectory_physics_resimulated"] is False
            and value["thermal_model_applied"] is False, "non-admitting no-update artifact")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "independent CUDA-hidden CPU scorer")
    declaration = _target_schedule(expected_schedule)
    require(canonical(value["declaration"]) == canonical(declaration)
            and canonical(value["compiled_plant"]) == canonical(expected_plant),
            "exact expected schedule and compiled plant")
    b = value["binding"]; profile.check_recorded(b["cpu_math_profile"])
    require(canonical(b) == canonical(binding(declaration, expected_plant, b["cpu_math_profile"])),
            "exact source, parent, profile, schedule, plant, seed binding")
    rsl_root = distribution("rsl-rl-lib").locate_file("rsl_rl")
    require({name: sha256((rsl_root/name).read_bytes()).hexdigest()
             for name in preparation.PINS} == b["ppo_source_sha256"],
            "independently pinned PPO/storage implementation bytes")
    require(type(checkpoint_raw) is bytes and sha256(checkpoint_raw).hexdigest()
            == baseline.CHECKPOINT_SHA256, "parent byte hash before load")
    loaded = parent.load_parent(checkpoint_raw)
    actor, critic = loaded["actor"], loaded["critic"]
    require(loaded["receipt"]["strict_actor_restore"] and loaded["receipt"]["strict_critic_restore"]
            and _state_hash(actor, critic) == parent.PARENT_STATE_SHA256 == b["parent_state_sha256"],
            "strict exact parent actor/critic replay models")
    require((actor.distribution.std_param > 0).all(), "positive exact-parent Gaussian scale")
    require(canonical(value["model_state"]["schema"]) == canonical(_schema(actor, critic))
            and value["model_state"]["initial_sha256"] == value["model_state"]["final_sha256"]
            == _state_hash(actor, critic) and value["model_state"]["unchanged"] is True,
            "model schema and weights unchanged")
    require(set(value["payload"]) == {"binding", "initial", "ticks"}
            and value["payload"]["binding"] == b
            and type(value["payload"]["ticks"]) is list
            and type(value["policy_ticks"]) is list
            and len(value["payload"]["ticks"]) == len(value["policy_ticks"]) <= HORIZON,
            "physical/policy tick alignment")
    require(len(value["bridge_receipts"]) == len(value["terminal_observations"])
            == len(value["policy_ticks"]), "retained bridge and terminal observations")
    recorder = ScheduledPPOTrace(b, value["payload"]["initial"], declaration, expected_plant)
    for index, (tick, policy) in enumerate(zip(value["payload"]["ticks"], value["policy_ticks"])):
        require(set(policy) == POLICY_KEYS and set(tick) == attempt.TICK_KEYS
                and torch.equal(tick["actor_input"], policy["actor_input"])
                and torch.equal(tick["actions"], policy["raw_action"])
                and value["bridge_receipts"][index]["storage_step"] == index
                and torch.equal(value["bridge_receipts"][index]["raw_actions"], policy["raw_action"])
                and _check_bridge_reward(value["bridge_receipts"][index]["reward"], policy["stored_reward"])
                and value["bridge_receipts"][index]["terminal_records"] == tick["terminal_records"]
                and type(value["bridge_receipts"][index]["reset_records"]) is list
                and len(value["bridge_receipts"][index]["reset_records"]) == 2
                and all(reset == terminal if terminal is not None else reset is None
                        for reset, terminal in zip(value["bridge_receipts"][index]["reset_records"],
                                                   tick["terminal_records"]))
                and torch.equal(value["terminal_observations"][index]["actor"],
                                tick["boundaries"][-1]["observation"]["actor"])
                and torch.equal(value["terminal_observations"][index]["critic"],
                                tick["boundaries"][-1]["observation"]["critic"])
                and torch.equal(policy["done"][:, 0].bool(),
                                tick["terminated"] | tick["timed_out"]),
                "action, bridge receipt, and terminal observation binding")
        recorder.append(tick, tick["actor_input"], tick["actions"])
    require(set(value["control_evidence"]) == {"protocol", "binding", "ticks"}
            and value["control_evidence"]["protocol"] == control.PROTOCOL
            and value["control_evidence"]["binding"] == b, "control trace binding")
    if recorder.ticks:
        plant_receipt = plant.check_trace(value["payload"], expected_plant["selected_plant"])
        control_receipt = control.replay(value["control_evidence"], value["payload"],
                                          expected_plant["selected_plant"])
        pulse_receipt = check_pulses(value)
    else:
        require(not value["control_evidence"]["ticks"] and not value["scheduled_pulse_evidence"],
                "no fabricated physical partial trace")
        plant_receipt = dict(compiled_plant_checked=False)
        control_receipt = None
        pulse_receipt = dict(checked_physics_steps=0, window_steps_per_row=[0, 0],
            delivered_nonzero_steps_per_row=[0, 0], complete_pulse_delivery=False,
            complete_full_batch_phase_checks=False, unforced_post_arrays_zero=False)
    require(type(value["optimizer"]) is dict and set(value["optimizer"]) == {
        "type", "state_entries", "optimizer_steps", "updates", "storage_step",
        "returns_uncomputed", "advantages_uncomputed", "storage_cleared"},
        "exact no-update receipt")
    optimizer = value["optimizer"]; cursor = optimizer["storage_step"]
    require(type(cursor) is int and optimizer == dict(type="Adam", state_entries=0, optimizer_steps=0, updates=0,
        storage_step=cursor, returns_uncomputed=True, advantages_uncomputed=True,
        storage_cleared=False), "empty Adam and uncleared unupdated storage")
    require(cursor >= len(value["policy_ticks"]), "storage cursor covers retained transitions")
    if value["storage"] is not None:
        _check_storage(value, cursor)
    require(value["unvalidated_transition"] is None or
            (type(value["unvalidated_transition"]) is dict
             and set(value["unvalidated_transition"]) == {"result", "policy"}
             and type(value["unvalidated_transition"]["result"]) is dict
             and type(value["unvalidated_transition"]["policy"]) is dict),
            "retained failed physical append evidence")
    require(type(value["caller_rng"]) is dict and set(value["caller_rng"]) == {"initial", "final"}
            and value["caller_rng"]["initial"].dtype == torch.uint8
            and value["caller_rng"]["final"].dtype == torch.uint8,
            "caller CPU RNG states retained")
    caller_unchanged = torch.equal(value["caller_rng"]["initial"], value["caller_rng"]["final"])
    require(value["backend"] == dict(torch_device="cpu", warp_is_cuda=False),
            "CPU-only artifact backend")
    require(type(value["private_rng"]) is dict and set(value["private_rng"]) == {"initial", "final"}
            and value["private_rng"]["initial"].dtype == torch.uint8
            and value["private_rng"]["final"].dtype == torch.uint8
            and value["private_rng"]["initial"].device.type == "cpu"
            and value["private_rng"]["final"].device.type == "cpu", "private CPU RNG retained")
    seeded_rng = torch.Generator(device="cpu").manual_seed(SEED).get_state()
    require(torch.equal(value["private_rng"]["initial"], seeded_rng),
            "private sampler begins at exact seed 653")
    live_rng = torch.random.get_rng_state().clone()
    for index, row in enumerate(value["policy_ticks"]):
        attempt.tensor(row["actor_input"], (2, 44), torch.float32, "sampled actor input")
        attempt.tensor(row["critic_input"], (2, 50), torch.float32, "sampled critic input")
        attempt.tensor(row["raw_action"], (2, 10), torch.float32, "raw Gaussian action")
        attempt.tensor(row["value"], (2, 1), torch.float32, "sampled critic value")
        attempt.tensor(row["log_prob"], (2, 1), torch.float32, "sample log probability")
        attempt.tensor(row["mu"], (2, 10), torch.float32, "Gaussian mean")
        attempt.tensor(row["sigma"], (2, 10), torch.float32, "Gaussian scale")
        attempt.tensor(row["stored_reward"], (2, 1), torch.float32, "stored reward")
        attempt.tensor(row["done"], (2, 1), torch.uint8, "stored done")
        require(torch.equal(row["rng_before"], value["private_rng"]["initial"] if index == 0
                else value["policy_ticks"][index-1]["rng_after"]), "private RNG chain")
        require(value["storage"] is not None, "storage provides critic replay observations")
        require(torch.equal(value["storage"]["observations"]["critic"][index],
                            row["critic_input"]), "stored critic input")
        obs = bridge_module.observations(dict(actor=row["actor_input"],
            critic=row["critic_input"]), 2)
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(row["rng_before"])
            sampled = actor(obs, stochastic_output=True).detach()
            val = critic(obs).detach()
            log_prob = actor.get_output_log_prob(sampled).detach().reshape(2, 1)
            params = [p.detach().clone() for p in actor.output_distribution_params]
            rng_after = torch.random.get_rng_state().clone()
        require((row["sigma"] > 0).all() and torch.equal(sampled, row["raw_action"])
                and torch.equal(val, row["value"])
                and torch.equal(log_prob, row["log_prob"]) and len(params) == 2
                and torch.equal(params[0], row["mu"]) and torch.equal(params[1], row["sigma"])
                and torch.equal(rng_after, row["rng_after"]), "independent stochastic replay")
    require(torch.equal(torch.random.get_rng_state(), live_rng), "scorer preserves caller RNG")
    last_rng = value["private_rng"]["initial"] if not value["policy_ticks"] else value["policy_ticks"][-1]["rng_after"]
    private_rng_closed = torch.equal(last_rng, value["private_rng"]["final"])
    collection = value["collection"]
    require(value["backend"] == dict(torch_device="cpu", warp_is_cuda=False)
            and set(collection) == {"policy_ticks", "elapsed_seconds", "stop_reason",
            "accepted_complete", "failure"}
            and collection["policy_ticks"] == len(recorder.ticks)
            and collection["stop_reason"] in ("transition-limit", "wall-budget-exhausted",
                "unexpected-terminal", "collection-exception", "caller-rng-changed",
                "storage-retention-failed")
            and type(collection["accepted_complete"]) is bool,
            "bounded status; early terminals rejected")
    within_cap = _elapsed_within_cap(collection["elapsed_seconds"])
    expected_complete = _accepted_complete(tick_count=len(recorder.ticks),
        storage_step=cursor, stop_reason=collection["stop_reason"],
        elapsed_seconds=collection["elapsed_seconds"], private_rng_closed=private_rng_closed,
        caller_rng_unchanged=caller_unchanged, terminals=recorder.terminals)
    require(collection["accepted_complete"] == expected_complete,
            "only complete in-budget unreset rollout qualifies")
    if collection["stop_reason"] in ("collection-exception", "caller-rng-changed",
                                     "storage-retention-failed"):
        require(type(collection["failure"]) is dict, "failure closeout receipt")
    else:
        require(collection["failure"] is None, "no fabricated failure for bounded/terminal stop")
    require((collection["stop_reason"] == "caller-rng-changed") == (not caller_unchanged),
            "caller RNG mutation is retained as a rejected failure")
    if collection["stop_reason"] != "collection-exception":
        require(private_rng_closed or collection["failure"] is not None,
                "private RNG closure or explicit failed attempt")
    return dict(protocol=PROTOCOL, collection=deepcopy(collection), storage_step=cursor,
        policy_replay_max_abs_error=0.0, private_rng_replayed=True, plant=plant_receipt,
        control=control_receipt, pulse=pulse_receipt,
        validated_policy_ticks=len(recorder.ticks),
        uncorroborated_storage_transitions=cursor-len(value["policy_ticks"]),
        private_rng_matches_final=private_rng_closed, caller_rng_unchanged=caller_unchanged,
        within_wall_cap=within_cap,
        complete_two_world_transition_qualification=collection["accepted_complete"],
        whole_trajectory_physics_resimulated=False, thermal_model_applied=False,
        **baseline.FALSE_FLAGS)


def encode(value):
    out = io.BytesIO(); torch.save(attempt.owned(value), out); raw = out.getvalue()
    require(0 < len(raw) <= LIMIT, "64 MiB CPU2 trace bound")
    return raw


def verify(raw, expected_sha256, checkpoint_raw, expected_schedule, expected_plant):
    require(type(raw) is bytes and 0 < len(raw) <= LIMIT
            and sha256(raw).hexdigest() == expected_sha256,
            "whole trace byte hash before tensor loading")
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    return score(value, checkpoint_raw, expected_schedule, expected_plant)
