"""One bounded recovery-PPO update and environment-free optimizer replay.

This sibling evidence protocol is integration-only. It neither qualifies the
policy nor resimulates physics; the input must be a separately verified,
complete native 28-by-2 no-update PPO trace.
"""

import io
import math
import os
from copy import deepcopy
from hashlib import sha256

import torch
from rsl_rl.storage import RolloutStorage

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_parent as parent
from mjlab_microduck import stance_recovery_policy_preparation as preparation
from mjlab_microduck import stance_recovery_ppo_bridge as bridge_module
from mjlab_microduck import stance_recovery_ppo_trace as source_trace
from mjlab_microduck.first_attempt_smoke import require
from mjlab_microduck.stance_ppo import CONFIG, observations

PROTOCOL = "football-b1d-cpu-scheduled-ppo-one-update-trace-v1"
HORIZON, WORLDS, UPDATE_STEPS = 28, 2, 20
LIMIT = 64 * 1024 * 1024
FALSE_FLAGS = dict(baseline.FALSE_FLAGS)


def _copy_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if type(value) is dict:
        return {key: _copy_tree(item) for key, item in value.items()}
    if type(value) is list:
        return [_copy_tree(item) for item in value]
    if type(value) is tuple:
        return tuple(_copy_tree(item) for item in value)
    return deepcopy(value)


def _tree_equal(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if type(left) is dict or type(right) is dict:
        return (
            type(left) is dict
            and type(right) is dict
            and left.keys() == right.keys()
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if type(left) in (list, tuple) or type(right) in (list, tuple):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _state_schema(actor, critic):
    return {
        group: {
            name: (tuple(value.shape), str(value.dtype))
            for name, value in model.state_dict().items()
        }
        for group, model in (("actor", actor), ("critic", critic))
    }


def _model_states(actor, critic):
    return {
        group: {
            name: value.detach().cpu().contiguous().clone()
            for name, value in model.state_dict().items()
        }
        for group, model in (("actor", actor), ("critic", critic))
    }


def _parameter_names(actor, critic):
    result = {}
    for group, model in (("actor", actor), ("critic", critic)):
        for name, parameter in model.named_parameters():
            key = group + "." + name
            require(key not in result, "unique named PPO parameter")
            result[key] = parameter
    return result


def _positive_gaussian(actor):
    scale = getattr(getattr(actor, "distribution", None), "std_param", None)
    require(
        torch.is_tensor(scale)
        and scale.device.type == "cpu"
        and torch.isfinite(scale).all()
        and (scale > 0).all(),
        "positive finite parent Gaussian scale",
    )


def _finite_models(actor, critic):
    for group, model in (("actor", actor), ("critic", critic)):
        for name, parameter in model.named_parameters():
            require(
                parameter.device.type == "cpu"
                and parameter.dtype == torch.float32
                and parameter.requires_grad
                and torch.isfinite(parameter).all(),
                "finite trainable CPU " + group + " parameter " + name,
            )


def _adam_state(actor, critic, optimizer):
    names = _parameter_names(actor, critic)
    require(
        type(optimizer) is torch.optim.Adam and len(optimizer.param_groups) == 1,
        "one actual Adam optimizer",
    )
    params = optimizer.param_groups[0]["params"]
    require(
        len(params) == len(names)
        and {id(p) for p in params} == {id(p) for p in names.values()},
        "Adam owns exactly the actor and critic parameters",
    )
    require(set(optimizer.state) == set(params), "complete per-parameter Adam state")
    state = {}
    for name, parameter in names.items():
        item = optimizer.state[parameter]
        require(
            type(item) is dict and set(item) == {"step", "exp_avg", "exp_avg_sq"},
            "complete named Adam moment schema",
        )
        step = item["step"]
        require(
            torch.is_tensor(step)
            and step.ndim == 0
            and step.dtype == torch.float32
            and step.device.type == "cpu"
            and step.item() == UPDATE_STEPS,
            "scalar CPU float32 Adam step count per parameter",
        )
        require(
            all(
                torch.is_tensor(item[key])
                and item[key].shape == parameter.shape
                and item[key].dtype == parameter.dtype
                and item[key].device.type == "cpu"
                and torch.isfinite(item[key]).all()
                for key in ("exp_avg", "exp_avg_sq")
            ),
            "finite typed Adam moments",
        )
        state[name] = {key: item[key].detach().cpu().clone() for key in item}
    return state


def _install_finite_hooks(algorithm):
    counts = {"pre": 0, "post": 0}

    def before_step(optimizer, _args, _kwargs):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                require(
                    parameter.grad is not None and torch.isfinite(parameter.grad).all(),
                    "finite complete gradients before Adam",
                )
        counts["pre"] += 1

    def after_step(optimizer, _args, _kwargs):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                require(torch.isfinite(parameter).all(), "finite parameter after Adam")
                item = optimizer.state.get(parameter)
                require(
                    type(item) is dict
                    and all(
                        torch.isfinite(value).all()
                        for value in item.values()
                        if torch.is_tensor(value)
                    ),
                    "finite Adam moments after step",
                )
        counts["post"] += 1

    handles = (
        algorithm.optimizer.register_step_pre_hook(before_step),
        algorithm.optimizer.register_step_post_hook(after_step),
    )
    return counts, handles


def _require_full_trace(raw, raw_parent, expected_sha256, schedule, compiled_plant):
    require(
        type(raw) is bytes
        and 0 < len(raw) <= source_trace.LIMIT
        and sha256(raw).hexdigest() == expected_sha256,
        "whole source trace hash before tensor loading",
    )
    result = source_trace.verify(
        raw, expected_sha256, raw_parent, schedule, compiled_plant
    )
    require(
        result["protocol"] == source_trace.PROTOCOL
        and result["complete_two_world_transition_qualification"] is True
        and result["validated_policy_ticks"] == HORIZON
        and result["storage_step"] == HORIZON
        and result["collection"]["accepted_complete"] is True
        and result["pulse"]["complete_pulse_delivery"] is True,
        "complete authenticated native no-update 28x2 trace required",
    )
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    require(
        value["backend"] == {"torch_device": "cpu", "warp_is_cuda": False}
        and value["training_update_performed"] is False
        and value["optimizer"]
        == {
            "type": "Adam",
            "state_entries": 0,
            "optimizer_steps": 0,
            "updates": 0,
            "storage_step": HORIZON,
            "returns_uncomputed": True,
            "advantages_uncomputed": True,
            "storage_cleared": False,
        },
        "unupdated CPU bridge trace provenance",
    )
    require(
        not any(row["done"].any() for row in value["policy_ticks"]),
        "no terminal rows in the source rollout",
    )
    return value, result


def _compare_storage(learner, traced):
    current = source_trace._storage(learner.storage)
    require(
        _tree_equal(current, traced["storage"]),
        "live learner storage exactly matches the authenticated trace",
    )


def _check_endpoint(env, traced):
    """Match live observation and Warp snapshot to owned final trace frames."""
    retained_endpoint = traced["terminal_observations"][-1]
    live_obs = observations(env.observations(), WORLDS)
    require(
        torch.equal(live_obs["actor"], retained_endpoint["actor"])
        and torch.equal(live_obs["critic"], retained_endpoint["critic"]),
        "live final post-transition observation matches retained trace endpoint",
    )
    final_frame = traced["payload"]["ticks"][-1]["boundaries"][-1]
    actual_frame = source_trace.attempt.owned(env.snapshot())
    require(
        _tree_equal(actual_frame, final_frame),
        "actual final Warp state exactly matches the retained physical trace frame",
    )
    return _copy_tree(retained_endpoint)


def capture(learner, raw_parent, raw_trace, trace_sha256):
    """Perform exactly one declared update on a real, already-filled bridge.

    The caller/supervisor owns runtime leases, deadlines, and persistence. This
    function performs no collection, reset, environment step, or retry.
    """
    from mjlab_microduck.stance_recovery_schedule_runtime import (
        ScheduledRecoveryRuntime,
    )

    require(
        type(learner) is bridge_module.RecoveryPPOBridge
        and learner.synthetic_fixture is False
        and type(learner.env) is ScheduledRecoveryRuntime,
        "native exact recovery bridge required for capture",
    )
    require(
        learner.env.n == WORLDS
        and torch.device(learner.env.device).type == "cpu"
        and not learner.env.wp_device.is_cuda
        and learner.env.forward_graph is None
        and learner.env.solved_field_check == "packed",
        "actual eager CPU2 Warp scheduled runtime backend",
    )
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CUDA-hidden CPU optimizer capture",
    )
    require(
        sha256(raw_parent).hexdigest() == baseline.CHECKPOINT_SHA256,
        "exact frozen parent bytes before any load",
    )
    traced, trace_score = _require_full_trace(
        raw_trace, raw_parent, trace_sha256, learner.schedule, learner.env.binding
    )
    require(
        learner.phase == "full"
        and learner.storage.step == HORIZON
        and not learner.faulted
        and learner.completed_updates == 0
        and learner.optimizer_steps == 0
        and not learner.algorithm.optimizer.state,
        "fresh full native bridge with empty Adam",
    )
    require(
        learner.learner_seed == source_trace.SEED
        and learner.worlds == WORLDS
        and learner.initial_state_sha256 == parent.PARENT_STATE_SHA256
        and learner.parent_receipt["parent_checkpoint_sha256"]
        == baseline.CHECKPOINT_SHA256
        and learner.parent_receipt["parent_identity"] == parent.expected_identity(),
        "unchanged exact parent preparation identity",
    )
    require(
        traced["binding"]["source"] == learner.schedule["source"]
        and traced["binding"]["parent_checkpoint_identity"]
        == parent.expected_identity()
        and traced["binding"]["parent_checkpoint_sha256"] == baseline.CHECKPOINT_SHA256
        and traced["binding"]["parent_state_sha256"] == learner.initial_state_sha256
        and traced["binding"]["schedule_sha256"] == learner.schedule_sha256,
        "trace/learner source and schedule binding",
    )
    require(
        traced["binding"]["ppo_config"] == bridge_module.CONFIG
        and traced["binding"]["ppo_source_sha256"] == preparation.PINS,
        "exact pinned PPO/storage implementation and configuration",
    )
    _compare_storage(learner, traced)
    require(
        torch.equal(learner.private_rng_state, traced["private_rng"]["final"]),
        "learner private RNG exactly follows the no-update trace",
    )
    retained_endpoint = _check_endpoint(learner.env, traced)
    require(
        checkpoint.state_hash(checkpoint.states_of(learner.actor, learner.critic))
        == traced["model_state"]["initial_sha256"]
        == learner.initial_state_sha256,
        "exact unchanged parent weights before update",
    )
    _finite_models(learner.actor, learner.critic)
    _positive_gaussian(learner.actor)

    caller_before = torch.random.get_rng_state().clone()
    private_before = learner.private_rng_state.detach().cpu().clone()
    returns = advantages = None
    counts, hooks = _install_finite_hooks(learner.algorithm)
    update_calls = learner.completed_updates
    steps_before = learner.optimizer_steps
    try:
        update_receipt = learner.update()
    finally:
        for handle in hooks:
            handle.remove()
    caller_after = torch.random.get_rng_state().clone()
    require(torch.equal(caller_before, caller_after), "update preserves caller CPU RNG")
    require(
        learner.completed_updates == update_calls + 1 == 1
        and learner.optimizer_steps - steps_before == UPDATE_STEPS
        and counts == {"pre": UPDATE_STEPS, "post": UPDATE_STEPS}
        and learner.storage.step == 0
        and learner.phase == "empty",
        "exactly one bridge update and twenty observed Adam steps",
    )
    require(
        update_receipt["completed_updates"] == 1
        and update_receipt["optimizer_steps"] == UPDATE_STEPS
        and update_receipt["storage_step"] == 0,
        "actual bridge update receipt",
    )
    returns = learner.storage.returns.detach().cpu().clone()
    advantages = learner.storage.advantages.detach().cpu().clone()
    require(
        torch.isfinite(returns).all() and torch.isfinite(advantages).all(),
        "finite captured GAE arrays",
    )
    _finite_models(learner.actor, learner.critic)
    _positive_gaussian(learner.actor)
    states = _model_states(learner.actor, learner.critic)
    state_sha = checkpoint.state_hash(states)
    require(state_sha == update_receipt["updated_state_sha256"], "updated state hash")
    adam = _adam_state(learner.actor, learner.critic, learner.algorithm.optimizer)
    require(
        len(adam)
        == sum(1 for _ in learner.actor.parameters())
        + sum(1 for _ in learner.critic.parameters()),
        "all Adam parameters retained",
    )
    private_after = learner.private_rng_state.detach().cpu().clone()
    require(
        not torch.equal(private_before, private_after),
        "private update minibatch RNG advanced",
    )
    binding = {
        "protocol": PROTOCOL,
        "source": traced["binding"]["source"],
        "source_trace_sha256": trace_sha256,
        "parent_checkpoint_sha256": baseline.CHECKPOINT_SHA256,
        "parent_checkpoint_identity": parent.expected_identity(),
        "parent_state_sha256": parent.PARENT_STATE_SHA256,
        "schedule_sha256": learner.schedule_sha256,
        "compiled_plant_sha256": traced["binding"]["plant_sha256"],
        "ppo_config": deepcopy(bridge_module.CONFIG),
        "ppo_source_sha256": deepcopy(preparation.PINS),
        "cpu_math_profile": deepcopy(traced["binding"]["cpu_math_profile"]),
        "learner_seed": source_trace.SEED,
        "worlds": WORLDS,
        "horizon": HORIZON,
        "capture_device": "cpu",
        "synthetic_fixture": False,
    }
    value = dict(
        protocol=PROTOCOL,
        binding=binding,
        backend={
            "torch_device": "cpu",
            "warp_is_cuda": False,
            "runtime_type": "ScheduledRecoveryRuntime",
            "runtime_device": str(learner.env.device),
            "solved_field_check": "packed",
            "forward_graph": None,
            "synthetic_fixture": False,
        },
        trace_score=_copy_tree(trace_score),
        trace_endpoint=_copy_tree(retained_endpoint),
        caller_rng={"before": caller_before, "after": caller_after},
        private_rng={"before": private_before, "after": private_after},
        gae={"returns": returns, "advantages": advantages},
        model_states=states,
        model_schema=_state_schema(learner.actor, learner.critic),
        parent_initial_state_sha256=learner.initial_state_sha256,
        updated_state_sha256=state_sha,
        adam=adam,
        optimizer={
            "type": "Adam",
            "state_entries": len(learner.algorithm.optimizer.state),
            "optimizer_steps": learner.optimizer_steps,
            "completed_updates": learner.completed_updates,
            "storage_step": learner.storage.step,
            "finite_gradient_hook_steps": counts["pre"],
            "finite_moment_hook_steps": counts["post"],
        },
        metrics=deepcopy(update_receipt["metrics"]),
        learner_update_receipt=_copy_tree(update_receipt),
        training_update_performed=True,
        optimizer_replay_performed=False,
        whole_trajectory_physics_resimulated=False,
        thermal_model_applied=False,
        cuda_initialized=False,
        optimizer_integration_only=True,
        execution_admitted=False,
        student_export_available=False,
        **FALSE_FLAGS,
    )
    return _copy_tree(value)


def _validate_capture(value, trace_value, trace_sha256, *, synthetic_fixture=False):
    required = {
        "protocol",
        "binding",
        "backend",
        "trace_score",
        "trace_endpoint",
        "caller_rng",
        "private_rng",
        "gae",
        "model_states",
        "model_schema",
        "parent_initial_state_sha256",
        "updated_state_sha256",
        "adam",
        "optimizer",
        "metrics",
        "learner_update_receipt",
        "training_update_performed",
        "optimizer_replay_performed",
        "whole_trajectory_physics_resimulated",
        "thermal_model_applied",
        "cuda_initialized",
        "optimizer_integration_only",
        "execution_admitted",
        "student_export_available",
    } | set(FALSE_FLAGS)
    require(
        type(value) is dict
        and set(value) == required
        and value["protocol"] == PROTOCOL,
        "exact one-update capture schema",
    )
    require(
        all(value[key] is False for key in FALSE_FLAGS)
        and value["training_update_performed"] is True
        and value["optimizer_replay_performed"] is False
        and value["whole_trajectory_physics_resimulated"] is False
        and value["thermal_model_applied"] is False
        and value["cuda_initialized"] is False
        and value["optimizer_integration_only"] is True
        and value["execution_admitted"] is False
        and value["student_export_available"] is False,
        "integration-only non-admitting evidence",
    )
    expected_backend = (
        {
            "torch_device": "cpu",
            "warp_is_cuda": False,
            "runtime_type": "SyntheticTestFixture",
            "runtime_device": "cpu",
            "solved_field_check": "synthetic",
            "forward_graph": None,
            "synthetic_fixture": True,
        }
        if synthetic_fixture
        else {
            "torch_device": "cpu",
            "warp_is_cuda": False,
            "runtime_type": "ScheduledRecoveryRuntime",
            "runtime_device": "cpu",
            "solved_field_check": "packed",
            "forward_graph": None,
            "synthetic_fixture": False,
        }
    )
    require(
        value["backend"] == expected_backend,
        "actual or explicitly synthetic CPU runtime provenance retained",
    )
    b = value["binding"]
    require(
        type(b) is dict
        and b
        == {
            "protocol": PROTOCOL,
            "source": trace_value["binding"]["source"],
            "source_trace_sha256": trace_sha256,
            "parent_checkpoint_sha256": baseline.CHECKPOINT_SHA256,
            "parent_checkpoint_identity": parent.expected_identity(),
            "parent_state_sha256": parent.PARENT_STATE_SHA256,
            "schedule_sha256": trace_value["binding"]["schedule_sha256"],
            "compiled_plant_sha256": trace_value["binding"]["plant_sha256"],
            "ppo_config": bridge_module.CONFIG,
            "ppo_source_sha256": preparation.PINS,
            "cpu_math_profile": trace_value["binding"]["cpu_math_profile"],
            "learner_seed": source_trace.SEED,
            "worlds": WORLDS,
            "horizon": HORIZON,
            "capture_device": "cpu",
            "synthetic_fixture": synthetic_fixture,
        },
        "exact immutable capture binding",
    )
    require(
        synthetic_fixture is True or b["synthetic_fixture"] is False,
        "native replay rejects synthetic optimizer fixtures",
    )
    require(
        _tree_equal(value["trace_score"], _copy_tree(value["trace_score"]))
        and value["trace_score"]["complete_two_world_transition_qualification"] is True,
        "complete source trace qualification",
    )
    require(
        _tree_equal(value["trace_endpoint"], trace_value["terminal_observations"][-1]),
        "exact final post-transition source endpoint",
    )
    require(
        value["parent_initial_state_sha256"] == parent.PARENT_STATE_SHA256
        and value["updated_state_sha256"]
        == checkpoint.state_hash(value["model_states"]),
        "exact parent and updated model identities",
    )
    require(
        set(value["model_states"]) == {"actor", "critic"}
        and set(value["model_schema"]) == {"actor", "critic"},
        "both model groups",
    )
    require(
        set(value["gae"]) == {"returns", "advantages"}
        and value["gae"]["returns"].shape
        == value["gae"]["advantages"].shape
        == (HORIZON, WORLDS, 1),
        "complete GAE arrays",
    )
    for name, tensor in value["gae"].items():
        require(
            torch.is_tensor(tensor)
            and tensor.device.type == "cpu"
            and tensor.dtype == torch.float32
            and torch.isfinite(tensor).all(),
            "finite GAE " + name,
        )
    for key in ("caller_rng", "private_rng"):
        require(
            type(value[key]) is dict
            and set(value[key]) == {"before", "after"}
            and all(
                torch.is_tensor(item)
                and item.device.type == "cpu"
                and item.dtype == torch.uint8
                for item in value[key].values()
            ),
            "typed " + key,
        )
    require(
        torch.equal(value["caller_rng"]["before"], value["caller_rng"]["after"])
        and not torch.equal(
            value["private_rng"]["before"], value["private_rng"]["after"]
        ),
        "caller RNG preserved and private RNG advanced",
    )
    require(
        type(value["optimizer"]) is dict
        and value["optimizer"]
        == {
            "type": "Adam",
            "state_entries": len(value["adam"]),
            "optimizer_steps": UPDATE_STEPS,
            "completed_updates": 1,
            "storage_step": 0,
            "finite_gradient_hook_steps": UPDATE_STEPS,
            "finite_moment_hook_steps": UPDATE_STEPS,
        },
        "one finite 20-step update",
    )
    require(
        type(value["metrics"]) is dict
        and value["metrics"]
        and all(
            type(number) in (float, int) and math.isfinite(number)
            for number in value["metrics"].values()
        ),
        "finite update metrics",
    )
    receipt = value["learner_update_receipt"]
    receipt_keys = {
        "protocol",
        "metrics",
        "completed_updates",
        "optimizer_steps",
        "parent_initial_state_sha256",
        "updated_state_sha256",
        "storage_step",
        "execution_admitted",
        "student_export_available",
    } | set(FALSE_FLAGS)
    require(
        type(receipt) is dict
        and set(receipt) == receipt_keys
        and receipt["protocol"] == bridge_module.PROTOCOL
        and _tree_equal(value["metrics"], receipt["metrics"])
        and receipt["completed_updates"] == 1
        and receipt["optimizer_steps"] == UPDATE_STEPS
        and receipt["parent_initial_state_sha256"]
        == value["parent_initial_state_sha256"]
        and receipt["updated_state_sha256"] == value["updated_state_sha256"]
        and receipt["storage_step"] == 0
        and receipt["execution_admitted"] is False
        and receipt["student_export_available"] is False
        and all(receipt[key] is False for key in FALSE_FLAGS),
        "exact actual bridge update receipt and non-admitting flags",
    )


def _replay_core(trace_value, capture_value, actor, critic):
    """Rebuild/update from checked plain rollout data; deliberately no env."""
    require(
        source_trace._state_hash(actor, critic) == parent.PARENT_STATE_SHA256,
        "offline replay starts from exact frozen parent",
    )
    _positive_gaussian(actor)
    schema = _state_schema(actor, critic)
    require(schema == capture_value["model_schema"], "parent model schema preserved")
    require(
        checkpoint.state_hash(checkpoint.states_of(actor, critic))
        == capture_value["parent_initial_state_sha256"],
        "unchanged parent initial state",
    )
    _finite_models(actor, critic)
    data = trace_value["storage"]
    dummy = observations(
        {"actor": torch.zeros(WORLDS, 44), "critic": torch.zeros(WORLDS, 50)}, WORLDS
    )
    caller_before_build = torch.random.get_rng_state().clone()
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        storage = RolloutStorage("rl", WORLDS, HORIZON, dummy, (10,), device="cpu")
        algorithm = bridge_module.RecoveryFinitePPO(
            actor, critic, storage, **CONFIG, device="cpu"
        )
    require(
        torch.equal(torch.random.get_rng_state(), caller_before_build),
        "offline storage/algorithm construction preserves caller RNG",
    )
    require(
        not algorithm.optimizer.state and storage.step == 0,
        "fresh empty offline Adam and public storage",
    )
    for index in range(HORIZON):
        transition = RolloutStorage.Transition()
        transition.observations = observations(
            {key: data["observations"][key][index] for key in ("actor", "critic")},
            WORLDS,
        )
        transition.actions = data["actions"][index].detach().cpu().clone()
        transition.rewards = data["rewards"][index].detach().cpu().clone()
        transition.dones = data["dones"][index].detach().cpu().clone()
        transition.values = data["values"][index].detach().cpu().clone()
        transition.actions_log_prob = (
            data["actions_log_prob"][index].detach().cpu().clone()
        )
        transition.distribution_params = tuple(
            tensor[index].detach().cpu().clone()
            for tensor in data["distribution_params"]
        )
        storage.add_transition(transition)
        require(storage.step == index + 1, "public ordered storage cursor")
    require(
        storage.step == HORIZON and not storage.dones.any(),
        "exact 28x2 nonterminal replay buffer",
    )
    endpoint = capture_value["trace_endpoint"]
    require(
        type(endpoint) is dict and set(endpoint) == {"actor", "critic"},
        "retained final endpoint",
    )
    last_obs = observations(endpoint, WORLDS)
    with torch.no_grad():
        algorithm.compute_returns(last_obs)
    returns = storage.returns.detach().cpu().clone()
    advantages = storage.advantages.detach().cpu().clone()
    require(
        torch.equal(returns, capture_value["gae"]["returns"])
        and torch.equal(advantages, capture_value["gae"]["advantages"]),
        "exact independently recomputed GAE arrays",
    )

    counts, handles = _install_finite_hooks(algorithm)
    private_before = trace_value["private_rng"]["final"].detach().cpu().clone()
    require(
        torch.equal(capture_value["private_rng"]["before"], private_before),
        "update begins at final no-update trace private RNG",
    )
    caller_before = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(private_before)
        try:
            metrics = algorithm.update()
            private_after = torch.random.get_rng_state().clone()
        finally:
            for handle in handles:
                handle.remove()
    caller_after = torch.random.get_rng_state().clone()
    require(
        torch.equal(caller_before, caller_after), "offline update preserves caller RNG"
    )
    require(
        counts == {"pre": UPDATE_STEPS, "post": UPDATE_STEPS} and storage.step == 0,
        "offline update executes exactly twenty Adam steps",
    )
    require(
        private_after.equal(capture_value["private_rng"]["after"]),
        "exact private minibatch RNG replay",
    )
    require(_tree_equal(metrics, capture_value["metrics"]), "exact PPO metric replay")
    _finite_models(actor, critic)
    _positive_gaussian(actor)
    states = _model_states(actor, critic)
    require(
        _tree_equal(states, capture_value["model_states"]),
        "exact actor/critic state replay",
    )
    require(
        checkpoint.state_hash(states) == capture_value["updated_state_sha256"],
        "updated actor/critic hash replay",
    )
    adam = _adam_state(actor, critic, algorithm.optimizer)
    require(
        _tree_equal(adam, capture_value["adam"]), "exact named Adam moment/step replay"
    )
    return dict(
        protocol=PROTOCOL,
        optimizer_replay=True,
        environment_created=False,
        physics_resimulated=False,
        exact_gae=True,
        exact_updated_model=True,
        exact_adam_state=True,
        exact_private_rng=True,
        exact_metrics=True,
        gradient_hook_steps=counts["pre"],
        moment_hook_steps=counts["post"],
        whole_trajectory_physics_resimulated=False,
        thermal_model_applied=False,
        cuda_initialized=False,
        optimizer_integration_only=True,
        **FALSE_FLAGS,
    )


def replay(raw_capture, expected_sha256, raw_trace, trace_sha256, raw_parent):
    """Independently repeat the one optimizer update without constructing an env."""
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CUDA-hidden CPU optimizer replay",
    )
    require(
        type(raw_capture) is bytes
        and 0 < len(raw_capture) <= LIMIT
        and sha256(raw_capture).hexdigest() == expected_sha256,
        "whole update artifact hash before tensor loading",
    )
    value = torch.load(io.BytesIO(raw_capture), map_location="cpu", weights_only=True)
    require(
        sha256(raw_parent).hexdigest() == baseline.CHECKPOINT_SHA256,
        "exact parent bytes before deserialization",
    )
    # The schedule/plant are bound by the retained trace itself. The trace scorer
    # receives those values as expected inputs, then rechecks their own semantics.
    require(
        type(raw_trace) is bytes
        and 0 < len(raw_trace) <= source_trace.LIMIT
        and sha256(raw_trace).hexdigest() == trace_sha256,
        "whole source trace hash before tensor loading",
    )
    preliminary = torch.load(
        io.BytesIO(raw_trace), map_location="cpu", weights_only=True
    )
    require(
        type(preliminary) is dict
        and type(preliminary.get("declaration")) is dict
        and type(preliminary.get("compiled_plant")) is dict,
        "retained trace binding inputs",
    )
    trace_value, trace_score = _require_full_trace(
        raw_trace,
        raw_parent,
        trace_sha256,
        preliminary["declaration"],
        preliminary["compiled_plant"],
    )
    _validate_capture(value, trace_value, trace_sha256)
    require(
        _tree_equal(value["trace_score"], trace_score),
        "capture retains the exact fresh source-trace score",
    )
    require(
        value["binding"]["schedule_sha256"]
        == source_trace.schedule.binding_sha256(trace_value["declaration"])
        and value["binding"]["compiled_plant_sha256"]
        == trace_value["binding"]["plant_sha256"],
        "replay declaration/compiled plant hashes",
    )
    loaded = parent.load_parent(raw_parent)
    require(
        loaded["receipt"]["strict_actor_restore"]
        and loaded["receipt"]["strict_critic_restore"]
        and loaded["receipt"]["parent_identity"] == parent.expected_identity()
        and loaded["receipt"]["parent_state_sha256"] == parent.PARENT_STATE_SHA256,
        "strict fresh actor/critic restoration for offline replay",
    )
    result = _replay_core(trace_value, value, loaded["actor"], loaded["critic"])
    result["source_trace_sha256"] = trace_sha256
    result["capture_sha256"] = expected_sha256
    result["source_trace_score"] = trace_score
    return result


def encode(value):
    out = io.BytesIO()
    torch.save(_copy_tree(value), out)
    raw = out.getvalue()
    require(0 < len(raw) <= LIMIT, "64 MiB update evidence bound")
    return raw
