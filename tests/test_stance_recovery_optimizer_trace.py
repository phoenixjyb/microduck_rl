"""Synthetic optimizer replay tests only; no native env or qualification claim."""

import io
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.storage import RolloutStorage

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_optimizer_trace as optimizer_trace
from mjlab_microduck import stance_recovery_parent as parent
from mjlab_microduck import stance_recovery_ppo_bridge as bridge
from mjlab_microduck import stance_recovery_ppo_trace as source_trace
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck.stance_ppo import CONFIG, observations
from mjlab_microduck.stance_transition import PhysicsState

SOURCE = "a" * 40
CELLS = ["zero-wrench", "+x-2n-20steps-t250"]
RAW_PARENT = b"synthetic parent bytes; not an authenticated checkpoint"


def _models():
    with torch.random.fork_rng(devices=[]):
        actor, critic = checkpoint.validate_identity(
            parent.expected_identity(), evaluation="lean-replication"
        )
    actor.train().requires_grad_(True)
    critic.train().requires_grad_(True)
    return actor, critic


def _fixture(monkeypatch):
    plant = {"synthetic_fixture": True}
    actor, critic = _models()
    state_sha = checkpoint.state_hash(checkpoint.states_of(actor, critic))
    # Synthetic test seam only: the production identity remains unchanged.
    monkeypatch.setattr(parent, "PARENT_STATE_SHA256", state_sha)
    monkeypatch.setattr(baseline, "CHECKPOINT_SHA256", sha256(RAW_PARENT).hexdigest())
    declaration = schedule.declaration(SOURCE, "dose", "training", CELLS)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    def load_parent(raw):
        assert raw == RAW_PARENT
        fresh_actor, fresh_critic = _models()
        return {
            "actor": fresh_actor,
            "critic": fresh_critic,
            "receipt": {
                "strict_actor_restore": True,
                "strict_critic_restore": True,
                "parent_identity": parent.expected_identity(),
                "parent_state_sha256": parent.PARENT_STATE_SHA256,
            },
        }

    monkeypatch.setattr(parent, "load_parent", load_parent)
    storage = {
        "observations": {
            "actor": torch.zeros(28, 2, 44),
            "critic": torch.zeros(28, 2, 50),
        },
        "actions": torch.zeros(28, 2, 10),
        "rewards": torch.zeros(28, 2, 1),
        "dones": torch.zeros(28, 2, 1, dtype=torch.uint8),
        "values": torch.zeros(28, 2, 1),
        "actions_log_prob": torch.zeros(28, 2, 1),
        "distribution_params": [torch.zeros(28, 2, 10), torch.full((28, 2, 10), 0.3)],
        "returns": torch.zeros(28, 2, 1),
        "advantages": torch.zeros(28, 2, 1),
    }
    storage["observations"]["actor"][:, :, 0] = torch.arange(28).view(28, 1) / 100
    storage["observations"]["critic"][:, :, :44] = storage["observations"]["actor"]
    storage["observations"]["critic"][:, :, -1] = 2.0
    storage["rewards"][:, :, 0] = torch.arange(28).view(28, 1) / 100 + 0.1
    for tick in range(28):
        obs = observations(
            {key: storage["observations"][key][tick] for key in ("actor", "critic")}, 2
        )
        with torch.no_grad():
            actor(obs, stochastic_output=True)
            storage["values"][tick] = critic(obs)
            params = actor.output_distribution_params
            storage["distribution_params"][0][tick] = params[0]
            storage["distribution_params"][1][tick] = params[1]
    private = torch.Generator(device="cpu").manual_seed(653).get_state()
    trace_value = {
        "protocol": source_trace.PROTOCOL,
        "binding": {
            "source": SOURCE,
            "parent_checkpoint_identity": parent.expected_identity(),
            "parent_checkpoint_sha256": baseline.CHECKPOINT_SHA256,
            "parent_state_sha256": parent.PARENT_STATE_SHA256,
            "schedule_sha256": schedule.binding_sha256(declaration),
            "plant_sha256": sha256(source_trace.canonical(plant).encode()).hexdigest(),
            "ppo_config": deepcopy(bridge.CONFIG),
            "ppo_source_sha256": deepcopy(bridge.preparation.PINS),
            "cpu_math_profile": {"synthetic_test_fixture": True},
        },
        "declaration": declaration,
        "compiled_plant": plant,
        "storage": storage,
        "policy_ticks": [
            {"done": torch.zeros(2, 1, dtype=torch.uint8)} for _ in range(28)
        ],
        "terminal_observations": [
            {"actor": torch.zeros(2, 44), "critic": torch.zeros(2, 50)}
            for _ in range(28)
        ],
        "private_rng": {"final": private},
        "backend": {"torch_device": "cpu", "warp_is_cuda": False},
        "training_update_performed": False,
        "synthetic_test_fixture": True,
        "optimizer": {
            "type": "Adam",
            "state_entries": 0,
            "optimizer_steps": 0,
            "updates": 0,
            "storage_step": 28,
            "returns_uncomputed": True,
            "advantages_uncomputed": True,
            "storage_cleared": False,
        },
    }
    trace_result = {
        "protocol": source_trace.PROTOCOL,
        "complete_two_world_transition_qualification": True,
        "validated_policy_ticks": 28,
        "storage_step": 28,
        "collection": {"accepted_complete": True},
        "pulse": {"complete_pulse_delivery": True},
    }
    monkeypatch.setattr(source_trace, "verify", lambda *_args: deepcopy(trace_result))
    out = io.BytesIO()
    torch.save(trace_value, out)
    raw_trace = out.getvalue()
    trace_sha = sha256(raw_trace).hexdigest()
    return trace_value, trace_result, declaration, plant, raw_trace, trace_sha, private


def _build_capture(monkeypatch):
    trace_value, trace_result, _declaration, _plant, raw_trace, trace_sha, private = (
        _fixture(monkeypatch)
    )
    actor, critic = _models()
    dummy = observations({"actor": torch.zeros(2, 44), "critic": torch.zeros(2, 50)}, 2)
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        storage = RolloutStorage("rl", 2, 28, dummy, (10,), device="cpu")
        algorithm = bridge.RecoveryFinitePPO(
            actor, critic, storage, **CONFIG, device="cpu"
        )
    data = trace_value["storage"]
    for index in range(28):
        transition = RolloutStorage.Transition()
        transition.observations = observations(
            {key: data["observations"][key][index] for key in ("actor", "critic")}, 2
        )
        transition.actions = data["actions"][index]
        transition.rewards = data["rewards"][index]
        transition.dones = data["dones"][index]
        transition.values = data["values"][index]
        transition.actions_log_prob = data["actions_log_prob"][index]
        transition.distribution_params = tuple(
            item[index] for item in data["distribution_params"]
        )
        storage.add_transition(transition)
    endpoint = trace_value["terminal_observations"][-1]
    algorithm.compute_returns(observations(endpoint, 2))
    gae = {
        name: getattr(storage, name).detach().cpu().clone()
        for name in ("returns", "advantages")
    }
    counts, hooks = optimizer_trace._install_finite_hooks(algorithm)
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(private)
        metrics = algorithm.update()
        private_after = torch.random.get_rng_state().clone()
    for hook in hooks:
        hook.remove()
    states = optimizer_trace._model_states(actor, critic)
    adam = optimizer_trace._adam_state(actor, critic, algorithm.optimizer)
    source_score = trace_result
    value = dict(
        protocol=optimizer_trace.PROTOCOL,
        binding={
            "protocol": optimizer_trace.PROTOCOL,
            "source": SOURCE,
            "source_trace_sha256": trace_sha,
            "parent_checkpoint_sha256": baseline.CHECKPOINT_SHA256,
            "parent_checkpoint_identity": parent.expected_identity(),
            "parent_state_sha256": parent.PARENT_STATE_SHA256,
            "schedule_sha256": trace_value["binding"]["schedule_sha256"],
            "compiled_plant_sha256": trace_value["binding"]["plant_sha256"],
            "ppo_config": deepcopy(bridge.CONFIG),
            "ppo_source_sha256": deepcopy(bridge.preparation.PINS),
            "cpu_math_profile": deepcopy(trace_value["binding"]["cpu_math_profile"]),
            "learner_seed": 653,
            "worlds": 2,
            "horizon": 28,
            "capture_device": "cpu",
            "synthetic_fixture": True,
        },
        backend={
            "torch_device": "cpu",
            "warp_is_cuda": False,
            "runtime_type": "SyntheticTestFixture",
            "runtime_device": "cpu",
            "solved_field_check": "synthetic",
            "forward_graph": None,
            "synthetic_fixture": True,
        },
        trace_score=deepcopy(source_score),
        trace_endpoint=deepcopy(endpoint),
        caller_rng={
            "before": torch.random.get_rng_state(),
            "after": torch.random.get_rng_state(),
        },
        private_rng={"before": private, "after": private_after},
        gae=gae,
        model_states=states,
        model_schema=optimizer_trace._state_schema(actor, critic),
        parent_initial_state_sha256=parent.PARENT_STATE_SHA256,
        updated_state_sha256=checkpoint.state_hash(states),
        adam=adam,
        optimizer={
            "type": "Adam",
            "state_entries": len(adam),
            "optimizer_steps": 20,
            "completed_updates": 1,
            "storage_step": 0,
            "finite_gradient_hook_steps": counts["pre"],
            "finite_moment_hook_steps": counts["post"],
        },
        metrics=metrics,
        learner_update_receipt={
            "protocol": bridge.PROTOCOL,
            "metrics": metrics,
            "completed_updates": 1,
            "optimizer_steps": 20,
            "parent_initial_state_sha256": parent.PARENT_STATE_SHA256,
            "updated_state_sha256": checkpoint.state_hash(states),
            "storage_step": 0,
            "execution_admitted": False,
            "student_export_available": False,
            **optimizer_trace.FALSE_FLAGS,
        },
        training_update_performed=True,
        optimizer_replay_performed=False,
        whole_trajectory_physics_resimulated=False,
        thermal_model_applied=False,
        cuda_initialized=False,
        optimizer_integration_only=True,
        execution_admitted=False,
        student_export_available=False,
        **optimizer_trace.FALSE_FLAGS,
    )
    out = optimizer_trace.encode(value)
    return out, sha256(out).hexdigest(), raw_trace, trace_sha, RAW_PARENT, trace_value


def test_synthetic_optimizer_replay_matches_gae_models_adam_rng_and_metrics(
    monkeypatch,
):
    """Exercises real CPU PPO math using explicitly synthetic evidence only."""
    raw, _digest, _raw_trace, trace_sha, _raw_parent, trace_value = _build_capture(
        monkeypatch
    )
    capture = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    optimizer_trace._validate_capture(
        capture, trace_value, trace_sha, synthetic_fixture=True
    )
    actor, critic = _models()
    result = optimizer_trace._replay_core(trace_value, capture, actor, critic)
    assert result["optimizer_replay"] is True
    assert result["environment_created"] is False
    assert result["physics_resimulated"] is False
    assert result["exact_gae"] and result["exact_updated_model"]
    assert (
        result["exact_adam_state"]
        and result["exact_private_rng"]
        and result["exact_metrics"]
    )
    assert all(result[name] is False for name in baseline.FALSE_FLAGS)


@pytest.mark.parametrize(
    "damage",
    [
        "gae",
        "endpoint",
        "adam",
        "rng",
        "count",
        "receipt_flag",
        "receipt_hash",
        "receipt_storage",
    ],
)
def test_synthetic_replay_rejects_tampered_update_evidence(monkeypatch, damage):
    raw, _, _raw_trace, trace_sha, _raw_parent, trace_value = _build_capture(
        monkeypatch
    )
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if damage == "gae":
        value["gae"]["returns"][0, 0, 0] += 1.0
    elif damage == "endpoint":
        value["trace_endpoint"]["critic"][0, 0] += 1.0
    elif damage == "adam":
        first = next(iter(value["adam"].values()))
        first["exp_avg"][0] += 1.0
    elif damage == "rng":
        value["private_rng"]["after"][0] ^= 1
    elif damage == "count":
        value["optimizer"]["finite_gradient_hook_steps"] -= 1
    elif damage == "receipt_flag":
        value["learner_update_receipt"]["training_admitted"] = True
    elif damage == "receipt_hash":
        value["learner_update_receipt"]["updated_state_sha256"] = "0" * 64
    else:
        value["learner_update_receipt"]["storage_step"] = 1
    with pytest.raises(ValueError):
        optimizer_trace._validate_capture(
            value, trace_value, trace_sha, synthetic_fixture=True
        )
        actor, critic = _models()
        optimizer_trace._replay_core(trace_value, value, actor, critic)
    tampered = optimizer_trace.encode(value)
    # The public native entry point must not treat this fixture as evidence.
    with pytest.raises(ValueError):
        optimizer_trace._validate_capture(
            torch.load(io.BytesIO(tampered), map_location="cpu", weights_only=True),
            trace_value,
            trace_sha,
        )


def test_incomplete_or_terminal_source_trace_is_refused(monkeypatch):
    trace_value, trace_result, _declaration, _plant, raw_trace, trace_sha, _private = (
        _fixture(monkeypatch)
    )
    trace_result["complete_two_world_transition_qualification"] = False
    with pytest.raises(ValueError, match="complete authenticated"):
        optimizer_trace._require_full_trace(
            raw_trace,
            RAW_PARENT,
            trace_sha,
            trace_value["declaration"],
            trace_value["compiled_plant"],
        )
    trace_result["complete_two_world_transition_qualification"] = True
    trace_value["policy_ticks"][3]["done"][1, 0] = 1
    # A tiny helper scorer reports an incomplete trace; terminal rows are also
    # rejected independently by the no-update source artifact validator.
    monkeypatch.setattr(
        source_trace,
        "verify",
        lambda *_args: {
            **trace_result,
            "validated_policy_ticks": 28,
            "storage_step": 28,
            "collection": {"accepted_complete": True},
            "pulse": {"complete_pulse_delivery": True},
        },
    )
    out = io.BytesIO()
    torch.save(trace_value, out)
    terminal_raw = out.getvalue()
    with pytest.raises(ValueError, match="no terminal rows"):
        optimizer_trace._require_full_trace(
            terminal_raw,
            RAW_PARENT,
            sha256(terminal_raw).hexdigest(),
            trace_value["declaration"],
            trace_value["compiled_plant"],
        )


def test_trace_hash_is_checked_before_tensor_deserialization(monkeypatch):
    _fixture(monkeypatch)
    with pytest.raises(ValueError, match="whole source trace hash"):
        optimizer_trace._require_full_trace(
            b"not a tensor archive", RAW_PARENT, "0" * 64, None, None
        )


def test_compare_storage_uses_real_public_rollout_storage_and_source_helper():
    obs = observations({"actor": torch.zeros(2, 44), "critic": torch.zeros(2, 50)}, 2)
    storage = RolloutStorage("rl", 2, 28, obs, (10,), device="cpu")
    transition = RolloutStorage.Transition()
    transition.observations = obs
    transition.actions = torch.zeros(2, 10)
    transition.rewards = torch.zeros(2)
    transition.dones = torch.zeros(2, dtype=torch.uint8)
    transition.values = torch.zeros(2, 1)
    transition.actions_log_prob = torch.zeros(2, 1)
    transition.distribution_params = (torch.zeros(2, 10), torch.full((2, 10), 0.3))
    storage.add_transition(transition)
    learner = SimpleNamespace(storage=storage)
    traced = {"storage": source_trace._storage(storage)}

    optimizer_trace._compare_storage(learner, traced)
    traced["storage"]["actions"][0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="live learner storage"):
        optimizer_trace._compare_storage(learner, traced)


def _endpoint_fixture():
    obs = {"actor": torch.zeros(2, 44), "critic": torch.zeros(2, 50)}
    obs["critic"][:, :44] = obs["actor"]
    state = PhysicsState(
        tilt=torch.zeros(2),
        root_velocity=torch.zeros(2, 3),
        height=torch.full((2,), 0.12),
        support=torch.ones(2, 2),
        torque=torch.zeros(2, 14),
        joint_velocity=torch.zeros(2, 14),
        hard_limit=torch.zeros(2, dtype=torch.bool),
        forbidden_contact=torch.zeros(2, dtype=torch.bool),
        warning=torch.zeros(2, dtype=torch.bool),
    )
    live_frame = {
        "physics_steps": torch.full((2,), 280, dtype=torch.long),
        "qpos": torch.zeros(2, 20),
        "qvel": torch.zeros(2, 20),
        "soft_limit_mask": torch.zeros(2, 20, dtype=torch.bool),
        "state": state,
        "observation": {key: value.clone() for key, value in obs.items()},
    }
    retained_frame = source_trace.attempt.owned(live_frame)
    env = SimpleNamespace(
        observations=lambda: {key: value.clone() for key, value in obs.items()},
        snapshot=lambda: live_frame,
    )
    traced = {
        "terminal_observations": [{key: value.clone() for key, value in obs.items()}],
        "payload": {"ticks": [{"boundaries": [retained_frame]}]},
    }
    return env, traced


def test_endpoint_helper_normalizes_physics_state_dataclass_to_owned_dict():
    env, traced = _endpoint_fixture()
    endpoint = optimizer_trace._check_endpoint(env, traced)
    assert torch.equal(endpoint["actor"], torch.zeros(2, 44))
    assert type(traced["payload"]["ticks"][0]["boundaries"][0]["state"]) is dict


@pytest.mark.parametrize(
    "damage", ["kinematics", "snapshot_observation", "terminal_observation"]
)
def test_endpoint_helper_rejects_tampered_physical_or_observation_endpoint(damage):
    env, traced = _endpoint_fixture()
    if damage == "kinematics":
        traced["payload"]["ticks"][0]["boundaries"][0]["qpos"][0, 0] += 1.0
    elif damage == "snapshot_observation":
        traced["payload"]["ticks"][0]["boundaries"][0]["observation"]["critic"][
            0, 0
        ] += 1.0
    else:
        traced["terminal_observations"][0]["actor"][0, 0] += 1.0
    with pytest.raises(ValueError):
        optimizer_trace._check_endpoint(env, traced)


@pytest.mark.parametrize(
    "bad_step",
    [torch.tensor([20.0]), torch.tensor(20, dtype=torch.int64)],
    ids=["non-scalar", "wrong-dtype"],
)
def test_adam_step_requires_scalar_cpu_float32(bad_step):
    actor, critic = _models()
    parameters = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(parameters, lr=CONFIG["learning_rate"])
    for parameter in parameters:
        optimizer.state[parameter] = {
            "step": bad_step.clone(),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    with pytest.raises(ValueError, match="scalar CPU float32 Adam step"):
        optimizer_trace._adam_state(actor, critic, optimizer)
