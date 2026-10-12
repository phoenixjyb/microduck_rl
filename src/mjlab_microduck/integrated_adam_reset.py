"""Matched continuation with an actor/critic Adam-state-only reset arm.

Both arms restore the same parent models, normalizers, learner settings, and
optimizer. The reset arm clears only the restored Adam state mapping before
continuation; it does not recreate the optimizer or modify its parameter groups.
"""

import copy
import sys

import torch

from mjlab_microduck import integrated_straight_lesson as straight
from mjlab_microduck import integrated_training_budget_refinement as previous
from mjlab_microduck import integrated_motor_refinement as motor


base = previous.base
PROTOCOL = "integrated-adam-reset-continuation-v1"
ARMS = ("control", "reset")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 1301),
                                             ("benchmark", 256, 10, 1303))
         for arm in ARMS}
MODES.update(control=(256, 1000, 1307), reset=(256, 1000, 1307))
CAP_SECONDS = 2100
TRAINING_CAP_SECONDS = dict(control=2100, reset=2100)
PARENT, PARENT_SHA = previous.PARENT, previous.PARENT_SHA
PARENT_LEARNING_RATE = previous.PARENT_LEARNING_RATE


def _require(condition, message):
    base.require(condition, message)


def _optimizer_parameters(optimizer):
    _require(type(optimizer) is torch.optim.Adam, "canonical torch Adam optimizer")
    _require(isinstance(optimizer.param_groups, list) and optimizer.param_groups,
             "nonempty Adam parameter groups")
    parameters, seen = [], set()
    for group in optimizer.param_groups:
        _require(isinstance(group, dict) and isinstance(group.get("params"), list)
                 and group["params"], "well-formed Adam parameter group")
        _require(type(group.get("amsgrad", False)) is bool, "boolean Adam amsgrad option")
        for parameter in group["params"]:
            _require(isinstance(parameter, torch.nn.Parameter) and parameter.is_floating_point(),
                     "floating Adam parameter")
            _require(id(parameter) not in seen, "unique Adam parameter identities")
            seen.add(id(parameter))
            parameters.append((parameter, group))
    _require(parameters, "Adam owns model parameters")
    return parameters


def allowed_adam_step_devices(parameter_device, *, capturable=False, fused=False):
    """Match torch 2.9 loading: noncapturable counters keep their mapped device.

    Fresh noncapturable Adam normally creates CPU counters, but load_state_dict
    returns an existing step tensor unchanged. A checkpoint loaded onto CUDA
    therefore legitimately has CUDA counters even with capturable/fused off.
    Never move a restored tensor to satisfy this read-only validation.
    """
    return ((parameter_device,) if capturable or fused else
            (torch.device("cpu"), parameter_device))


def validate_populated_adam(optimizer):
    """Validate every canonical Adam slot before reporting or clearing it."""
    parameters = _optimizer_parameters(optimizer)
    state = optimizer.state
    _require(isinstance(state, dict), "Adam state mapping")
    _require(set(state.keys()) == {parameter for parameter, _ in parameters},
             "one populated Adam state per optimizer parameter")
    entries = 0
    for parameter, group in parameters:
        slots = state[parameter]
        expected = {"step", "exp_avg", "exp_avg_sq"}
        if group.get("amsgrad", False):
            expected.add("max_exp_avg_sq")
        _require(isinstance(slots, dict) and set(slots) == expected,
                 "canonical populated Adam slots")
        step = slots["step"]
        expected_step_dtype = (torch.float32 if group.get("fused", False) else
                               torch.float64 if torch.get_default_dtype() == torch.float64
                               else torch.float32)
        _require(isinstance(step, torch.Tensor) and step.shape == torch.Size([])
                 and step.dtype == expected_step_dtype and torch.isfinite(step).all().item()
                 and step.item() > 0 and step.item().is_integer(),
                 "finite positive integer-valued scalar Adam step")
        step_devices = allowed_adam_step_devices(parameter.device,
            capturable=group.get("capturable", False), fused=group.get("fused", False))
        _require(step.device in step_devices, "Adam step on supported restored device")
        for key in expected - {"step"}:
            value = slots[key]
            _require(isinstance(value, torch.Tensor) and value.shape == parameter.shape
                     and value.dtype == parameter.dtype and value.device == parameter.device
                     and torch.isfinite(value).all().item()
                     and (key != "exp_avg_sq" or bool((value >= 0).all())),
                     "finite shape/dtype/device-matched Adam moment " + key)
        entries += 1
    _require(set(state.keys()) <= {parameter for parameter, _ in parameters},
             "no unowned Adam state entries")
    return entries


def _slot_counts(state):
    counts = {}
    for slots in state.values():
        for name in slots:
            counts[name] = counts.get(name, 0) + 1
    return counts


def reset_adam_state(optimizer):
    """Clear only an existing, fully populated canonical torch Adam state.

    Returns before/after parameter-state entry counts. It deliberately rejects
    empty, partial, malformed, or non-Adam optimizers so a reset cannot conceal
    a failed parent restoration.
    """
    before = validate_populated_adam(optimizer)
    state_mapping = optimizer.state
    state_mapping.clear()
    _require(optimizer.state is state_mapping and len(optimizer.state) == 0,
             "same Adam state mapping cleared in place")
    return dict(before=before, after=len(optimizer.state))


def _clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return copy.deepcopy(value)


def _tree_equal(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.shape == right.shape and left.dtype == right.dtype
                and left.device == right.device and torch.equal(left, right))
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict)
                and left.keys() == right.keys()
                and all(_tree_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True)))
    return left == right


def _module_snapshot(module):
    _require(isinstance(module, torch.nn.Module), "actor/critic torch modules")
    return _clone_tree(module.state_dict())


def _capture_rng():
    snapshot = {"cpu": torch.random.get_rng_state().clone()}
    if torch.cuda.is_initialized():
        snapshot["cuda"] = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return snapshot


def _check_rng(snapshot):
    now = _capture_rng()
    _require(_tree_equal(snapshot, now), "torch CPU/CUDA RNG preserved")


def warm_load_for_mode(runner, env, mode):
    """Restore the exact parent first, optionally clear only Adam moments."""
    _require(mode in MODES, "known Adam reset mode")
    restored = previous.warm_load(runner, env)
    _require(restored.get("parent_sha256") == PARENT_SHA,
             "exact pinned parent hash restored")
    _require(restored.get("learning_rate") == PARENT_LEARNING_RATE,
             "original parent Adam learning rate")
    _require(restored.get("restored_models_normalizers_adam") is True,
             "parent models, normalizers, and Adam restored")

    algorithm = runner.alg
    optimizer = algorithm.optimizer
    models = {"actor": algorithm.actor, "critic": algorithm.critic}
    model_parameter_ids = {id(parameter) for model in models.values()
                           for parameter in model.parameters()}
    optimizer_parameter_ids = {id(parameter) for group in optimizer.param_groups
                               for parameter in group["params"]}
    _require(model_parameter_ids and optimizer_parameter_ids == model_parameter_ids,
             "Adam owns exactly actor and critic parameters")
    _require(all(type(group.get("lr")) in (int, float)
                 and group["lr"] == PARENT_LEARNING_RATE
                 for group in optimizer.param_groups),
             "all parent Adam group learning rates match the original Python rate")
    model_state = {name: _module_snapshot(model) for name, model in models.items()}
    learning_rate = algorithm.learning_rate
    parameter_identities = [[id(parameter) for parameter in group["params"]]
                            for group in optimizer.param_groups]
    group_options = [_clone_tree({key: value for key, value in group.items() if key != "params"})
                     for group in optimizer.param_groups]
    defaults = _clone_tree(optimizer.defaults)
    optimizer_object, state_mapping = optimizer, optimizer.state
    state_before = _clone_tree(optimizer.state)
    groups_object = optimizer.param_groups
    group_objects = list(optimizer.param_groups)
    rng_before = _capture_rng()
    slot_count_before = validate_populated_adam(optimizer)
    slot_counts_before = _slot_counts(optimizer.state)

    mode_arm = mode.split("-")[-1]
    if mode_arm == "reset":
        reset_counts = reset_adam_state(optimizer)
        _require(reset_counts["before"] == slot_count_before and reset_counts["after"] == 0,
                 "all restored Adam slots cleared")
    else:
        reset_counts = dict(before=slot_count_before, after=slot_count_before)

    _require(algorithm.optimizer is optimizer_object and optimizer.state is state_mapping,
             "same Adam optimizer and state mapping")
    _require(optimizer.param_groups is groups_object
             and all(current is original for current, original in zip(
                 optimizer.param_groups, group_objects, strict=True)),
             "same Adam parameter groups")
    _require([[id(parameter) for parameter in group["params"]]
              for group in optimizer.param_groups] == parameter_identities,
             "Adam parameter object identities preserved")
    _require(_tree_equal([_clone_tree({key: value for key, value in group.items()
                                      if key != "params"}) for group in optimizer.param_groups],
                         group_options), "Adam group options preserved")
    _require(_tree_equal(optimizer.defaults, defaults), "Adam defaults preserved")
    _require(algorithm.learning_rate == learning_rate == PARENT_LEARNING_RATE,
             "Python learner learning rate preserved")
    for name, model in models.items():
        _require(_tree_equal(model_state[name], model.state_dict()),
                 name + " parameters and normalizer buffers preserved")
    if mode_arm == "control":
        _require(_tree_equal(state_before, optimizer.state), "control Adam state preserved")
        _require(len(optimizer.state) == slot_count_before, "control Adam remains populated")
    else:
        _require(len(optimizer.state) == 0, "reset Adam state is empty before first PPO update")
    _check_rng(rng_before)

    result = dict(restored)
    result.update(parent_adam_restoration_verified=True,
                  restored_models_normalizers_adam=(mode_arm == "control"),
                  optimizer_initialization_kind=("restored_parent_adam_state" if mode_arm == "control"
                                                 else "restored_parent_adam_then_state_reset"),
                  adam_state_parameter_entries_before=slot_count_before,
                  adam_state_parameter_entries_after=len(optimizer.state),
                  adam_slot_counts_before=slot_counts_before,
                  adam_slot_counts_after=_slot_counts(optimizer.state),
                  adam_groups_preserved=True, adam_parameter_identities_preserved=True,
                  actor_critic_models_and_normalizers_preserved=True,
                  torch_rng_preserved=True, original_learning_rate=PARENT_LEARNING_RATE)
    return result


def prepare_config(mode):
    _require(mode in MODES, "known Adam reset mode")
    cfg, agent = previous.prepare_config("control")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


warm_load = previous.warm_load
decision = previous.decision


class MotorObserver(previous.MotorObserver):
    """Keep all inherited reward, KL, optimizer, and command census callbacks."""

    def __init__(self, mode):
        _require(mode in MODES, "known Adam reset observer mode")
        super().__init__("control")
        self.reset_mode = mode


# Keep the read-only held-out gait telemetry wrapper from the straight lesson.
evaluate_case = straight.evaluate_case


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
