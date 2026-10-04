"""CPU consistency replay of retained CUDA64 physical/control/phase records.

This sibling does not expand historical capture protocols or execute a solver.
An independently retained pre-action initial snapshot is required. Compiling the
CPU plant checks its selected contract, not CUDA execution provenance. Actor,
CUDA RNG, BAM outputs and model/optimizer state still require separate evidence.
"""

from copy import deepcopy
from hashlib import sha256
import os

import torch

from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_control_evidence as control
from mjlab_microduck import stance_cpu_replay_profile as profile
from mjlab_microduck import stance_disturbance_fixture as fixture
from mjlab_microduck import stance_forward_probe as forward
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_recovery_cuda_record_archive as archive
from mjlab_microduck import stance_recovery_cuda_transition as transition
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.stance_transition import PhysicsState, physical_failures

PROTOCOL = "football-b1d-cuda64-record-consistency-source-v1"
N = 64
PHASES = ("forced_pre", "integrated", "unforced_post")


def _compiled_reference():
    # No runtime, Warp allocation, CUDA initialization or simulation step.
    return fixture.compiled_binding(plant.build_entity().compile())


class RecordTrace(trace.FirstAttemptTrace):
    """CPU-only rehydration; never a CUDA capture constructor."""

    def __init__(self, binding, initial):
        self.binding = deepcopy(binding)
        self.n = N
        self.faulted = False
        self._tensor_device = "cpu"
        frame = trace.owned(initial)
        trace.validate_frame(frame, N)
        require(
            not frame["physics_steps"].any()
            and not physical_failures(
                PhysicsState(**frame["state"]), frame["physics_steps"]
            ).any(),
            "fresh valid independently retained initial frame",
        )
        self.initial = frame
        self.last = frame
        self.ticks = []
        self.terminals = [None] * N

    def _initialize(self, *_):
        raise RuntimeError("record replay requires its separate source constructor")


def _phase(value, compiled, declaration, steps, accepted, committed):
    require(
        type(value) is dict and set(value) == {"solved", "inputs", "motor_fields"},
        "complete CUDA64 retained phase",
    )
    forward.validate_output(value["solved"], N)
    inputs = value["inputs"]
    require(
        type(inputs) is dict
        and set(inputs) == {"ctrl", "xfrc_applied", "qfrc_applied"},
        "complete CUDA64 phase inputs",
    )
    for key, shape in (
        ("ctrl", (N, 14)),
        ("xfrc_applied", (N, compiled["nbody"], 6)),
        ("qfrc_applied", (N, 20)),
    ):
        trace.tensor(inputs[key], shape, torch.float32, key)
    require(
        torch.equal(inputs["ctrl"], committed["ctrl"]),
        "retained phase committed control",
    )
    schedule.validate_wrenches(
        inputs["xfrc_applied"].tolist(),
        inputs["qfrc_applied"].tolist(),
        declaration,
        steps,
        accepted,
        compiled["nbody"],
        compiled["body_id"],
    )
    fields = value["motor_fields"]
    require(
        type(fields) is dict and set(fields) == {"dof_frictionloss", "dof_damping"},
        "complete CUDA64 retained motor fields",
    )
    for name, key in (("dof_frictionloss", "friction"), ("dof_damping", "damping")):
        trace.tensor(fields[name], (N, 20), torch.float32, name)
        require(
            (fields[name] >= 0).all() and torch.equal(fields[name], committed[key]),
            "retained phase committed motor field " + name,
        )


def _check_phases(records, declaration, compiled):
    windows = 0
    for record in records:
        raw = record["runtime_result_before_reset"]
        entries = raw["scheduled_pulse_evidence"]
        frames = raw["boundaries"]
        proposals = [
            p for p in raw["control_evidence"]["proposals"] if p["accepted"].any()
        ]
        require(
            len(entries) == len(frames) - 1 == len(proposals),
            "complete CUDA64 phase boundary coverage",
        )
        for entry, before, after, proposal in zip(
            entries, frames, frames[1:], proposals
        ):
            steps = entry["before_steps"].tolist()
            accepted = entry["accepted"].tolist()
            require(
                (entry["phases"] is not None)
                == any(schedule.window_mask(declaration, steps, accepted)),
                "exact CUDA64 solved phase-window coverage",
            )
            if entry["phases"] is None:
                continue
            phases = entry["phases"]
            require(
                type(phases) is dict and set(phases) == set(PHASES),
                "three CUDA64 phases",
            )
            for name in PHASES:
                _phase(
                    phases[name],
                    compiled,
                    declaration,
                    steps,
                    accepted if name != "unforced_post" else [False] * N,
                    proposal["committed"],
                )
            pre, integrated, post = [phases[name] for name in PHASES]
            for name, frame in (
                ("forced_pre", before),
                ("integrated", after),
                ("unforced_post", after),
            ):
                for key in ("qpos", "qvel"):
                    require(
                        torch.equal(
                            phases[name]["solved"]["kinematics"][key], frame[key]
                        ),
                        "retained phase physical boundary " + name + "/" + key,
                    )
            kin = pre["solved"]["kinematics"]
            require(
                torch.allclose(
                    kin["time"],
                    entry["before_steps"].to(torch.float32) * 0.002,
                    rtol=1e-5,
                    atol=1e-6,
                ),
                "retained phase time bound to actual row clocks",
            )
            delta = entry["accepted"].to(torch.float32) * 0.002
            require(
                torch.equal(
                    integrated["solved"]["kinematics"]["time"], kin["time"] + delta
                ),
                "accepted-only CUDA64 Euler clock increment",
            )
            require(
                fixture.exact_tree(
                    integrated["solved"]["kinematics"], post["solved"]["kinematics"]
                ),
                "unforced post cannot integrate",
            )
            for key in ("dynamics", "solver", "contacts", "constraints"):
                require(
                    fixture.exact_tree(pre["solved"][key], integrated["solved"][key]),
                    "integrated fields retain pre-Euler solve " + key,
                )
            windows += 1
    return dict(
        phase_entries=windows,
        full_phase_consistency_checked=True,
        solver_independently_reexecuted=False,
        bam_independently_recomputed=False,
    )


def check(
    value, initial_frame, expected_schedule, compiled_plant, *, seed, cpu_profile
):
    """Check CPU-owned records; caller authenticates whole artifact bytes first.

    This API cannot qualify native collection, policy training or a skill. Its
    receipts must not be substituted for an independently supervised GPU probe.
    """
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CUDA-hidden record consistency replay",
    )
    archive._owned_tree(initial_frame, allow_cuda=False, clone=False)
    archive._owned_tree(cpu_profile, allow_cuda=False, clone=False)
    archived = archive.check(value, expected_schedule, compiled_plant, seed=seed)
    profile.check_recorded(cpu_profile)
    require(
        canonical(compiled_plant) == canonical(_compiled_reference()),
        "independently CPU-compiled selected plant binding",
    )
    binding = dict(
        protocol=PROTOCOL,
        source=expected_schedule["source"],
        worlds=N,
        capture_device="cuda:0",
        learner_seed=seed,
        schedule_sha256=schedule.binding_sha256(expected_schedule),
        plant_sha256=sha256(canonical(compiled_plant).encode()).hexdigest(),
        cpu_math_profile=deepcopy(cpu_profile),
    )
    recorder = RecordTrace(binding, initial_frame)
    controls = dict(protocol=control.PROTOCOL, binding=deepcopy(binding), ticks=[])
    for record in value["records"]:
        raw = record["runtime_result_before_reset"]
        recorder.append(
            raw, record["pre_action_observations"]["actor"], record["raw_actions"]
        )
        controls["ticks"].append(trace.owned(raw["control_evidence"]))
    payload = recorder.payload()
    plant_receipt = plant.check_trace(payload, compiled_plant["selected_plant"])
    control_receipt = control.replay(
        controls, payload, compiled_plant["selected_plant"]
    )
    phase_receipt = _check_phases(value["records"], expected_schedule, compiled_plant)
    return dict(
        protocol=PROTOCOL,
        binding=binding,
        archive=archived,
        plant=plant_receipt,
        control=control_receipt,
        phases=phase_receipt,
        trajectory_continuity_checked=True,
        whole_trajectory_physics_resimulated=False,
        native_transition_qualified=False,
        actor_independently_replayed=False,
        cuda_rng_math_replayed=False,
        caller_rng_states_independently_checked=False,
        model_and_optimizer_states_independently_checked=False,
        natural_timeout_qualified=False,
        selective_reset_qualified=False,
        **transition.FALSE_FLAGS,
    )
