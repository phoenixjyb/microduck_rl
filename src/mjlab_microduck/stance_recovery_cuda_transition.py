"""Source-only CUDA64 transition collector; no returns, updates or admission.

This sibling preserves the frozen CPU bridge. A future independently supervised
probe must authenticate preparation, physical schedule delivery and raw records.
Calling this class, or passing its synthetic source tests, does not qualify any
CUDA physics, natural terminal, selective reset or learned capability.
"""

from copy import deepcopy

import torch
from tensordict import TensorDict

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_recovery_cuda_policy_preparation as preparation
from mjlab_microduck import stance_recovery_cuda_rng_scope as rng_scope
from mjlab_microduck import stance_recovery_cuda_shadow_sampling as sampling
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck import stance_training_smoke as training_smoke
from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
from mjlab_microduck.stance_transition import DECIMATION, EPISODE_STEPS

PROTOCOL = "football-b1d-cuda64-transition-source-contract-v1"
WORLDS, HORIZON, DEVICE = 64, 28, "cuda:0"
FALSE_FLAGS = dict(preparation.FALSE_FLAGS)


def _tensor(value, shape, label, *, dtype=torch.float32):
    require(
        torch.is_tensor(value)
        and tuple(value.shape) == shape
        and value.dtype == dtype
        and value.device.type == "cuda"
        and value.device.index == 0,
        "typed CUDA0 transition " + label,
    )
    require(
        not value.is_floating_point() or torch.isfinite(value).all(),
        "finite CUDA0 transition " + label,
    )


def _observations(value):
    require(set(value) == {"actor", "critic"}, "exact CUDA observation groups")
    for name, width in (("actor", 44), ("critic", 50)):
        _tensor(value[name], (WORLDS, width), name + " observations")
    require(
        torch.equal(value["actor"], value["critic"][:, :44]),
        "exact CUDA actor/critic observation prefix",
    )
    return TensorDict(
        {name: item.detach().clone() for name, item in value.items()}, [WORLDS]
    )


def _equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and torch.equal(left, right)
        )
    if type(left) is dict or type(right) is dict:
        return (
            type(left) is type(right) is dict
            and set(left) == set(right)
            and all(_equal(left[key], right[key]) for key in left)
        )
    if type(left) in (list, tuple) or type(right) in (list, tuple):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _check_stock_step(algorithm):
    method = getattr(algorithm, "process_env_step", None)
    require(
        type(algorithm) is preparation.PPO
        and getattr(method, "__self__", None) is algorithm
        and getattr(method, "__func__", None) is preparation.PPO.process_env_step,
        "exact pinned stock PPO.process_env_step bound method",
    )


class CudaTransitionCollector:
    """At most 28 stock no-update transitions with fault-latched reset ordering.

    There is no synthetic-fixture argument or public device/profile bypass.
    The supervising caller, not this class, authenticates CPU provenance and
    owns GPU idle checks, resource caps and immutable evidence retention.
    """

    def __init__(self, prepared, env, *, lease_fd):
        require(type(lease_fd) is int and lease_fd >= 0, "inherited shared lease")
        training_smoke.inherited_lease(lease_fd)  # Before CUDA queries/operations.
        require(type(prepared) is dict, "typed prepared CUDA policy")
        require(
            set(prepared)
            == {
                "actor",
                "critic",
                "algorithm",
                "storage",
                "receipt",
                "private_cuda_generator",
                "private_cuda_rng_state",
                "caller_rng_states",
            },
            "exact prepared CUDA policy fields",
        )
        self.source, self.seed, state = sampling._validate_receipt(
            deepcopy(prepared["receipt"]), prepared
        )
        sampling._check_cuda0_ready()
        preparation._check_rsl_sources()
        checkpoint.runtime_check()
        self.actor, self.critic, self.algorithm, self.storage = (
            prepared[name] for name in ("actor", "critic", "algorithm", "storage")
        )
        preparation._check_policy(self.actor, self.critic, self.algorithm, self.storage)
        preparation._check_storage(self.storage)
        sampling._transition_empty(self.algorithm.transition)
        sampling._check_stock_act(self.algorithm)
        _check_stock_step(self.algorithm)
        require(
            type(env) is ScheduledRecoveryRuntime
            and env.n == WORLDS
            and str(env.device) == DEVICE
            and not env.faulted,
            "exact healthy scheduled CUDA64 runtime",
        )
        self.env, self.lease_fd = env, lease_fd
        self.declaration = schedule.checked(env.schedule_declaration)
        require(
            self.declaration["source"] == self.source
            and self.declaration["worlds"] == WORLDS,
            "exact source-bound CUDA64 row declaration",
        )
        self.schedule_hash = schedule.binding_sha256(self.declaration)
        _tensor(env.initial_qpos, (21,), "initial reset qpos")
        self.initial_qpos = env.initial_qpos.detach().clone()
        self.initial_controls = deepcopy(env._control_snapshot())
        self.expected_steps = env.steps.detach().clone()
        _tensor(self.expected_steps, (WORLDS,), "initial clocks", dtype=torch.long)
        require(not self.expected_steps.any(), "fresh zero-clock CUDA64 runtime")
        _tensor(env.live, (WORLDS,), "initial live mask", dtype=torch.bool)
        require(env.live.all(), "all fresh CUDA64 rows live")
        self.scope = rng_scope.CudaPrivateRngScope(self.seed, state, lease_fd=lease_fd)
        require(
            self.scope.receipt["source"] == self.source, "prepared RNG source binding"
        )
        self.phase, self.faulted, self.last_record = "empty", False, None
        self.transitions = 0
        self._healthy()

    def _healthy(self):
        require(not self.faulted, "faulted CUDA collector cannot retry")
        training_smoke.inherited_lease(self.lease_fd)
        self.scope._verify_context()
        sampling._check_stock_act(self.algorithm)
        _check_stock_step(self.algorithm)
        preparation._check_policy(self.actor, self.critic, self.algorithm, self.storage)
        require(
            checkpoint.state_hash(checkpoint.states_of(self.actor, self.critic))
            == preparation.parent.PARENT_STATE_SHA256,
            "unchanged actual D1 parent during no-update collection",
        )
        require(
            not self.env.faulted
            and canonical(schedule.checked(self.env.schedule_declaration))
            == canonical(self.declaration)
            and self.env._schedule_sha256 == self.schedule_hash,
            "unchanged healthy scheduled runtime and fixed row map",
        )
        require(
            torch.equal(self.env.initial_qpos, self.initial_qpos),
            "unchanged reset qpos",
        )
        _tensor(self.env.steps, (WORLDS,), "entry clocks", dtype=torch.long)
        _tensor(self.env.live, (WORLDS,), "entry live mask", dtype=torch.bool)
        require(
            torch.equal(self.env.steps, self.expected_steps)
            and self.env.live.all()
            and ((self.env.steps >= 0) & (self.env.steps < EPISODE_STEPS)).all(),
            "no stale, injected or unreset entry row clocks",
        )

    @torch.no_grad()
    def collect_one(self, *, capture_control=False):
        """Store the actual pre-reset transition before any selective reset."""
        require(not self.faulted, "faulted CUDA collector cannot retry")
        try:
            self._healthy()
            require(type(capture_control) is bool, "boolean control capture")
            step = self.storage.step
            require(
                type(step) is int
                and 0 <= step < HORIZON
                and step == self.transitions
                and self.phase in ("empty", "collecting"),
                "bounded no-update CUDA collection phase",
            )
            sampling._transition_empty(self.algorithm.transition)
            before_steps = self.env.steps.detach().clone()
            obs = _observations(self.env.observations())
            obs_saved = {key: value.clone() for key, value in obs.items()}
            private_state_before = self.scope.state
            with self.scope.scope():
                action = self.algorithm.act(obs)
            scope_receipt = self.scope.receipt
            require(
                not torch.equal(private_state_before, self.scope.state)
                and scope_receipt["scope_count"] == self.transitions + 1
                and scope_receipt["scope_active"] is False
                and scope_receipt["faulted"] is False
                and scope_receipt["caller_cpu_rng_preserved"] is True
                and scope_receipt["caller_cuda_rng_preserved"] is True,
                "one advancing private draw with preserved caller streams",
            )
            _tensor(action, (WORLDS, 10), "raw policy actions")
            action_saved = action.detach().clone()
            transition = self.algorithm.transition
            _tensor(transition.values, (WORLDS, 1), "pre-action critic values")
            _tensor(
                transition.actions_log_prob, (WORLDS,), "raw action log probabilities"
            )
            require(
                torch.equal(transition.actions, action_saved), "unclipped PPO action"
            )
            require(
                type(transition.distribution_params) is tuple
                and len(transition.distribution_params) == 2,
                "stock Gaussian distribution parameters",
            )
            mean, std = transition.distribution_params
            _tensor(mean, (WORLDS, 10), "Gaussian action mean")
            _tensor(std, (WORLDS, 10), "Gaussian action scale")
            require((std > 0).all(), "positive Gaussian action scale")
            distribution = (mean.clone(), std.clone())
            pre_values, pre_log_prob = (
                transition.values.clone(),
                transition.actions_log_prob.clone(),
            )
            result = self.env.step_with_schedule(
                action_saved.clone(), capture_control=capture_control
            )
            runtime_result_before_reset = deepcopy(result)
            require(
                type(result) is dict and result.get("optimizer_launched") is False,
                "actual pre-reset no-optimizer transition result",
            )
            terminal_obs = _observations(result["observation"])
            for name, dtype in (
                ("reward", torch.float32),
                ("terminated", torch.bool),
                ("timed_out", torch.bool),
                ("live", torch.bool),
                ("episode_steps", torch.long),
                ("executed_steps", torch.long),
            ):
                _tensor(result[name], (WORLDS,), name, dtype=dtype)
            terminated, timed_out = (
                result["terminated"].clone(),
                result["timed_out"].clone(),
            )
            require(not (terminated & timed_out).any(), "exclusive failure and timeout")
            done = terminated | timed_out
            clocks, executed = result["episode_steps"], result["executed_steps"]
            require(
                torch.equal(result["live"], ~done)
                and torch.equal(result["live"], self.env.live)
                and torch.equal(clocks, self.env.steps)
                and torch.equal(clocks, before_steps + executed)
                and ((executed >= 0) & (executed <= DECIMATION)).all()
                and (clocks <= EPISODE_STEPS).all()
                and (clocks[timed_out] == EPISODE_STEPS).all()
                and (executed[~done] == DECIMATION).all(),
                "pre-reset live, bounded substep and natural timeout clock binding",
            )
            records = deepcopy(result["terminal_records"])
            require(
                type(records) is list and len(records) == WORLDS, "terminal row ledger"
            )
            for row, record in enumerate(records):
                require(
                    type(record) is dict if bool(done[row]) else record is None,
                    "terminal record exactly for each done row",
                )
            terminal_values = torch.zeros_like(result["reward"][:, None])
            if timed_out.any():
                # Only these rows bootstrap; critic receives pre-reset inputs.
                selected = TensorDict(
                    {
                        key: value[timed_out].clone()
                        for key, value in terminal_obs.items()
                    },
                    [int(timed_out.sum())],
                )
                values = self.critic(selected).detach()
                _tensor(values, (int(timed_out.sum()), 1), "timeout terminal values")
                terminal_values[timed_out] = values
            raw_reward = result["reward"].clone()
            reward = raw_reward + self.algorithm.gamma * terminal_values[:, 0]
            _tensor(reward, (WORLDS,), "exactly-once learner reward target")
            require(
                torch.equal(action, action_saved)
                and torch.equal(transition.actions, action_saved)
                and all(torch.equal(obs[key], obs_saved[key]) for key in obs_saved),
                "runtime cannot alias raw actions or pre-action observations",
            )
            pre_reset = {
                name: self.env._view(name).detach().clone()
                for name in ("qpos", "qvel", "time")
            }
            for name, shape in (
                ("qpos", (WORLDS, 21)),
                ("qvel", (WORLDS, 20)),
                ("time", (WORLDS,)),
            ):
                _tensor(pre_reset[name], shape, "pre-reset physical " + name)
            pre_controls = deepcopy(self.env._control_snapshot())
            after_controls = deepcopy(pre_controls)
            pre_clocks = self.env.steps.clone()
            # Empty metadata is deliberate: stock time_outs bootstraps the wrong
            # pre-action value and would add a second timeout bonus.
            self.algorithm.process_env_step(terminal_obs, reward, done, {})
            require(self.storage.step == step + 1, "one and only one stored transition")
            for name, expected in (
                ("actions", action_saved),
                ("values", pre_values),
                ("actions_log_prob", pre_log_prob[:, None]),
                ("rewards", reward[:, None]),
                ("dones", done[:, None]),
            ):
                require(
                    torch.equal(getattr(self.storage, name)[step], expected),
                    "stored raw " + name,
                )
            require(
                all(
                    torch.equal(self.storage.observations[key][step], obs_saved[key])
                    for key in obs_saved
                ),
                "storage retains pre-action rather than terminal/reset observations",
            )
            resets = [None] * WORLDS
            if done.any():
                resets = self.env.reset(done.clone())
                require(
                    _equal(resets, records),
                    "reset returns retained exact terminal ledger",
                )
                for name, before in pre_reset.items():
                    require(
                        torch.equal(before[~done], self.env._view(name)[~done]),
                        "untouched physical " + name,
                    )
                require(
                    torch.equal(
                        self.env._view("qpos")[done],
                        self.initial_qpos.expand(int(done.sum()), -1),
                    )
                    and not self.env._view("qvel")[done].any()
                    and not self.env._view("time")[done].any(),
                    "reset physical rows restore exact initial qpos and zero qvel/time",
                )
                require(
                    torch.equal(self.env.steps[~done], pre_clocks[~done])
                    and not self.env.steps[done].any()
                    and self.env.live.all(),
                    "selective reset clock and live-mask isolation",
                )
                after_controls = deepcopy(self.env._control_snapshot())
                require(
                    set(after_controls) == set(pre_controls),
                    "same reset control schema",
                )
                for name, before in pre_controls.items():
                    after = after_controls[name]
                    if (
                        torch.is_tensor(before)
                        and before.ndim
                        and before.shape[0] == WORLDS
                    ):
                        require(
                            torch.equal(before[~done], after[~done])
                            and torch.equal(
                                self.initial_controls[name][done], after[done]
                            ),
                            "selective reset preserves/restores control " + name,
                        )
                    else:
                        require(
                            _equal(before, after), "unchanged global control " + name
                        )
            next_obs = _observations(self.env.observations())
            require(
                all(
                    torch.equal(next_obs[key][~done], terminal_obs[key][~done])
                    for key in next_obs.keys()
                ),
                "untouched observation rows preserved across reset",
            )
            self.expected_steps = self.env.steps.detach().clone()
            self._healthy()
            self.transitions += 1
            self.phase = "full" if self.storage.step == HORIZON else "collecting"
            record = {
                "receipt": {
                    "protocol": PROTOCOL,
                    "source": self.source,
                    "learner_seed": self.seed,
                    "storage_step": step,
                    "worlds": WORLDS,
                    "device": DEVICE,
                    "schedule_sha256": self.schedule_hash,
                    "optimizer_steps": 0,
                    "transition_bridge_qualified": False,
                    "native_terminal_qualified": False,
                    "native_selective_reset_qualified": False,
                    **FALSE_FLAGS,
                },
                "pre_action_observations": obs_saved,
                "raw_actions": action_saved,
                "pre_action_values": pre_values,
                "raw_actions_log_prob": pre_log_prob,
                "distribution_params": distribution,
                "terminal_observations": {
                    key: value.clone() for key, value in terminal_obs.items()
                },
                "timeout_terminal_values": terminal_values.clone(),
                "environment_reward": raw_reward,
                "learner_reward": reward.clone(),
                "terminated": terminated,
                "timed_out": timed_out,
                "pre_reset_steps": pre_clocks,
                "post_reset_steps": self.env.steps.detach().clone(),
                "pre_reset_kinematics": pre_reset,
                "post_reset_kinematics": {
                    name: self.env._view(name).detach().clone() for name in pre_reset
                },
                "pre_reset_controls": pre_controls,
                "post_reset_controls": after_controls,
                "initial_reset_controls": deepcopy(self.initial_controls),
                "terminal_records": records,
                "reset_records": deepcopy(resets),
                "next_observations": {
                    key: value.clone() for key, value in next_obs.items()
                },
                "runtime_result_before_reset": runtime_result_before_reset,
                "private_rng_state_before": private_state_before,
                "private_rng_state": self.scope.state,
                "rng_scope_receipt": deepcopy(self.scope.receipt),
            }
            self.last_record = deepcopy(record)
            return record
        except BaseException:
            self.faulted, self.phase = True, "faulted"
            raise
