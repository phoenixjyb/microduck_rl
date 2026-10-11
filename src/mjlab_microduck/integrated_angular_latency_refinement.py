"""Matched actor angular-velocity observation latency refinement.

Only the actor ``base_ang_vel`` delay ceiling changes: the control keeps its
0/1-step DelayBuffer and the fresh arm sets the ceiling to zero. In mjlab a
zero ceiling omits the DelayBuffer entirely, so this is not a sampled-zero
latency arm and does not support a pure causal claim.
"""

import sys
from collections.abc import Mapping

import torch

from mjlab_microduck import integrated_training_budget_refinement as previous
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck import observation_timing_capture as capture

base = previous.base
PROTOCOL = "integrated-angular-latency-continuation-v1"
ARMS = ("control", "fresh")
MODES = {f"{phase}-{arm}": (worlds, updates, seed)
         for phase, worlds, updates, seed in (("smoke", 64, 5, 1193),
                                               ("benchmark", 256, 10, 1201))
         for arm in ARMS}
MODES.update(control=(256, 1000, 1213), fresh=(256, 1000, 1213))
CAP_SECONDS = 2100
TRAINING_CAP_SECONDS = {arm: CAP_SECONDS for arm in ARMS}
PARENT, PARENT_SHA = previous.PARENT, previous.PARENT_SHA
PARENT_LEARNING_RATE = previous.PARENT_LEARNING_RATE


def prepare_config(mode):
    base.require(type(mode) is str and mode in MODES, "known angular latency mode")
    cfg, agent = previous.prepare_config("control")
    worlds, updates, seed = MODES[mode]
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    angular = cfg.observations["actor"].terms["base_ang_vel"]
    base.require(angular.delay_min_lag == 0 and angular.delay_max_lag == 1,
                 "inherited actor angular 0/1-step delay baseline")
    if mode.endswith("-fresh") or mode == "fresh":
        angular.delay_max_lag = 0
    agent.max_iterations = updates
    agent.experiment_name, agent.run_name = PROTOCOL, mode
    return cfg, agent


warm_load = previous.warm_load
evaluate_case = previous.evaluate_case
decision = previous.decision


def _delay_contract(delay, name, worlds, min_lag, max_lag, update_period):
    base.require(delay is not None
                 and (delay.min_lag, delay.max_lag, delay.update_period) ==
                 (min_lag, max_lag, update_period)
                 and delay.per_env is True and delay.per_env_phase is True
                 and delay.hold_prob == 0., f"pinned live {name} delay contract")
    lag = delay.current_lags
    history = delay._buffer
    length = history.current_length
    base.require(delay.is_initialized and lag.shape == length.shape == (worlds,)
                 and lag.dtype == length.dtype == torch.int64
                 and bool(((lag >= min_lag) & (lag <= max_lag)
                           & (length >= 1) & (length <= max_lag + 1)).all()),
                 f"initialized live {name} delay state")
    return lag, length


def inspect_latency(env, actor, mode):
    """Read cached actor and real buffers without advancing compute or RNG."""
    from mjlab.utils.noise import UniformNoiseCfg
    from mjlab.envs import mdp as base_mdp
    from mjlab_microduck.tasks import mdp

    base.require(type(mode) is str and mode in MODES, "known angular latency observer mode")
    base.require(type(env.num_envs) is int and env.num_envs > 0
                 and type(env.common_step_counter) is int and env.common_step_counter >= 0,
                 "positive live worlds and nonnegative control clock")
    worlds = env.num_envs
    manager = env.observation_manager
    expected_terms = capture.TERMS
    expected_dims = capture.DIMS
    base.require(tuple(manager.active_terms["actor"]) == expected_terms
                 and tuple(manager.group_obs_term_dim["actor"]) == expected_dims
                 and manager.group_obs_concatenate["actor"]
                 and manager.cfg["actor"].enable_corruption,
                 "pinned raw 61D actor observation layout")
    base.require(isinstance(actor, torch.Tensor) and actor.shape == (worlds, 61)
                 and base.finite_tree(actor)
                 and isinstance(manager._obs_buffer, Mapping)
                 and torch.equal(actor, manager._obs_buffer["actor"]),
                 "already-computed raw actor cache")
    command = env.command_manager.get_command("twist")
    base.require(command.shape == (worlds, 3) and base.finite_tree(command)
                 and torch.equal(actor[:, 48:51], command),
                 "cached actor twist command identity")

    terms = {name: manager.get_term_cfg("actor", name)
             for name in ("base_ang_vel", "projected_gravity", "joint_vel")}
    angular = terms["base_ang_vel"]
    max_angular_lag = 0 if mode.endswith("-fresh") or mode == "fresh" else 1
    base.require(angular.func is mdp.base_ang_vel_imu_misaligned
                 and angular.params == {"max_angle_deg": 6.}
                 and angular.scale is None and angular.clip is None and angular.history_length == 0
                 and (angular.delay_min_lag, angular.delay_max_lag,
                      angular.delay_update_period) == (0, max_angular_lag, 64)
                 and angular.delay_per_env is True and angular.delay_per_env_phase is True
                 and angular.delay_hold_prob == 0.
                 and isinstance(angular.noise, UniformNoiseCfg)
                 and (angular.noise.n_min, angular.noise.n_max, angular.noise.operation)
                 == (-.03, .03, "add"),
                 "pinned effective actor angular observation config")
    for name, min_lag, max_lag, period in (("projected_gravity", 0, 1, 64),
                                           ("joint_vel", 1, 1, 0)):
        term = terms[name]
        expected_func = (mdp.projected_gravity_imu_misaligned if name == "projected_gravity"
                         else base_mdp.joint_vel_rel)
        expected_params = ({"max_angle_deg": 6.} if name == "projected_gravity" else None)
        expected_noise = ((-.01, .01) if name == "projected_gravity" else (-.25, .25))
        base.require(term.func is expected_func
                     and (term.params == expected_params if expected_params is not None else
                          getattr(term.params.get("asset_cfg"), "name", None) == "robot"
                          and getattr(term.params.get("asset_cfg"), "joint_names", None)
                          == (r"^(?!passive_).*",))
                     and term.scale is None and term.clip is None
                     and isinstance(term.noise, UniformNoiseCfg)
                     and (term.noise.n_min, term.noise.n_max, term.noise.operation)
                     == (*expected_noise, "add")
                     and term.delay_min_lag == min_lag and term.delay_max_lag == max_lag
                     and term.delay_update_period == period and term.delay_per_env is True
                     and term.delay_per_env_phase is True and term.delay_hold_prob == 0.
                     and term.history_length == 0,
                     f"unchanged {name} effective delay config")

    delays = manager._group_obs_term_delay_buffer.get("actor")
    base.require(isinstance(delays, Mapping), "live actor delay-buffer map")
    expected_buffers = {"projected_gravity", "joint_vel"}
    if max_angular_lag:
        expected_buffers.add("base_ang_vel")
    base.require(set(delays) == expected_buffers,
                 "only terms with positive delay ceilings have live buffers")
    for name, min_lag, max_lag, period in (("projected_gravity", 0, 1, 64),
                                           ("joint_vel", 1, 1, 0)):
        lag, length = _delay_contract(delays[name], name, worlds, min_lag, max_lag, period)
        base.require(bool((lag == min_lag).all()) if min_lag == max_lag else True,
                     f"live {name} fixed delay state")

    if max_angular_lag:
        delay = delays["base_ang_vel"]
        lag, length = _delay_contract(delay, "base_ang_vel", worlds, 0, 1, 64)
        effective = torch.minimum(lag, length - 1).clamp_min(0)
        # A read-only indexed lookup: no compute(), sampling, history append, or noise.
        selected = delay._buffer[effective]
        base.require(selected.shape == (worlds, 3) and torch.equal(actor[:, :3], selected),
                     "control raw angular cache equals its effective delayed sample")
        lag1 = int((effective == 1).sum())
    else:
        # mjlab omits the DelayBuffer entirely at max_lag == 0; bypassing its
        # append/compute path also changes RNG consumption, so classify separately.
        effective = torch.zeros(worlds, dtype=torch.long, device=actor.device)
        lag1 = 0
    return {"lag0": worlds - lag1, "lag1": lag1}


class MotorObserver(previous.MotorObserver):
    """Keep inherited reward telemetry and add read-only effective-lag counts."""

    def __init__(self, mode):
        base.require(type(mode) is str and mode in MODES, "known angular latency mode")
        super().__init__("control")
        # Preserve `self.mode == 'control'` for inherited signed angular
        # objective schedules; keep the experiment arm in a distinct field.
        self.latency_mode = mode
        self.pending_latency = None
        self.latency_steps = 0
        self.latency_counts = {"lag0": 0, "lag1": 0}
        self.latency_cumulative = {"lag0": 0, "lag1": 0}
        self.latency_worlds = None
        self.first_common_step = None
        self.last_pre_common_step = None

    def before(self, env):
        super().before(env)
        base.require(self.pending_latency is None, "previous angular latency sample consumed")
        base.require(type(env.common_step_counter) is int and env.common_step_counter >= 0
                     and (self.last_pre_common_step is None
                          or env.common_step_counter == self.last_pre_common_step + 1),
                     "consecutive pre-action common-step latency reads")
        if self.first_common_step is None:
            self.first_common_step = env.common_step_counter
        manager = env.observation_manager
        buffer = getattr(manager, "_obs_buffer", None)
        actor = buffer.get("actor") if isinstance(buffer, Mapping) else None
        counts = inspect_latency(env, actor, self.latency_mode)
        base.require(self.latency_worlds in (None, env.num_envs), "stable latency world count")
        self.latency_worlds = env.num_envs
        self.last_pre_common_step = env.common_step_counter
        self.pending_latency = (env.common_step_counter, counts)

    def step(self, env, sample, extras):
        super().step(env, sample, extras)
        base.require(self.pending_latency is not None
                     and env.common_step_counter == self.pending_latency[0] + 1,
                     "one consumed action step per latency snapshot")
        for key, count in self.pending_latency[1].items():
            self.latency_counts[key] += count
            self.latency_cumulative[key] += count
        self.latency_steps += 1
        self.pending_latency = None

    def update(self, env):
        row = super().update(env)
        worlds = env.num_envs
        base.require(self.pending_latency is None and self.latency_steps == 24
                     and sum(self.latency_counts.values()) == 24 * worlds
                     and sum(self.latency_cumulative.values())
                     == (env.common_step_counter - self.first_common_step) * worlds,
                     "complete 24-step effective angular latency census")
        row.update(angular_latency_protocol="cached-effective-angular-delay-v1",
                   angular_effective_lag_counts=dict(self.latency_counts),
                   angular_effective_lag_cumulative=dict(self.latency_cumulative),
                   angular_delay_buffer_present=(self.latency_mode.endswith("-control")
                       or self.latency_mode == "control"),
                   angular_zero_lag_mechanism=("sampled-delay-buffer" if self.latency_mode.endswith("-control")
                       or self.latency_mode == "control" else "no-delay-buffer-at-zero-ceiling"))
        self.latency_steps = 0
        self.latency_counts = {"lag0": 0, "lag1": 0}
        return row


if __name__ == "__main__":
    motor.main(sys.modules[__name__])
