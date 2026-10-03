"""CPU-owned CUDA64 collector records, not native replay or launch admission.

This source sibling checks byte ownership, raw transition arithmetic and explicit
per-row force exposure. It does not independently replay the actor, CUDA random
draws, BAM, the solver, physics or an optimizer. A future supervised native probe
must retain and authenticate those separate inputs and execution boundaries.
"""

from collections.abc import Mapping
from hashlib import sha256
import io
import math
import os
import re

import torch

from mjlab_microduck import stance_attempt_trace as attempt
from mjlab_microduck import stance_control_evidence as control
from mjlab_microduck import stance_recovery_cuda_rng_scope as rng
from mjlab_microduck import stance_recovery_cuda_shadow_sampling as sampling
from mjlab_microduck import stance_recovery_cuda_transition as transition
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck import stance_training_smoke as training
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = "football-b1d-cuda64-record-archive-source-v1"
LIMIT = 256 * 1024**2
TENSOR_BUDGET = 128 * 1024**2
NODE_BUDGET = 250_000
RECORD_KEYS = set(
    """receipt pre_action_observations raw_actions pre_action_values
    raw_actions_log_prob distribution_params terminal_observations timeout_terminal_values
    environment_reward learner_reward terminated timed_out pre_reset_steps post_reset_steps
    pre_reset_kinematics post_reset_kinematics pre_reset_controls post_reset_controls
    initial_reset_controls terminal_records reset_records next_observations
    runtime_result_before_reset private_rng_state_before private_rng_state rng_scope_receipt""".split()
)
PULSE_KEYS = set(
    """before_steps accepted pre_xfrc pre_qfrc post_xfrc post_qfrc
    phases schedule_sha256 binding_sha256""".split()
)


def _owned_tree(value, *, allow_cuda, clone=True):
    budget = {"nodes": 0, "bytes": 0}

    def copy(item, depth=0):
        budget["nodes"] += 1
        require(depth <= 32 and budget["nodes"] <= NODE_BUDGET, "bounded archive tree")
        if torch.is_tensor(item):
            require(
                item.layout == torch.strided
                and item.dtype
                in (
                    torch.float32,
                    torch.float64,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                    torch.bool,
                ),
                "dense supported archive tensor",
            )
            require(
                item.device.type == "cpu"
                or (
                    allow_cuda and item.device.type == "cuda" and item.device.index == 0
                ),
                "CPU archive or leased CUDA0 capture tensor",
            )
            budget["bytes"] += item.numel() * item.element_size()
            require(budget["bytes"] <= TENSOR_BUDGET, "bounded archive tensor bytes")
            require(
                not item.is_floating_point() or torch.isfinite(item).all(),
                "finite archive tensor",
            )
            return item.detach().cpu().clone() if clone else item
        if isinstance(item, Mapping):
            require(all(type(key) is str for key in item), "string archive keys")
            return {key: copy(val, depth + 1) for key, val in item.items()}
        if type(item) in (list, tuple):
            result = [copy(val, depth + 1) for val in item]
            return tuple(result) if type(item) is tuple else result
        require(
            item is None or type(item) in (str, bool, int, float), "safe archive leaf"
        )
        require(type(item) is not float or math.isfinite(item), "finite archive scalar")
        return item

    return copy(value)


def retain_record_cpu(record, *, lease_fd):
    """Check the inherited lease before any tensor transfer; own tuples too."""
    require(type(lease_fd) is int and lease_fd >= 0, "inherited capture lease")
    training.inherited_lease(lease_fd)
    require(
        type(record) is dict and set(record) == RECORD_KEYS, "exact collector record"
    )
    return _owned_tree(record, allow_cuda=True)


def _tensor(value, shape, dtype, label):
    attempt.tensor(value, shape, dtype, label)


def _observations(value, label):
    require(
        type(value) is dict and set(value) == {"actor", "critic"}, label + " groups"
    )
    for name, width in (("actor", 44), ("critic", 50)):
        _tensor(value[name], (64, width), torch.float32, label + " " + name)
    require(torch.equal(value["actor"], value["critic"][:, :44]), label + " prefix")


def _state(value):
    require(
        torch.is_tensor(value)
        and value.device.type == "cpu"
        and value.dtype == torch.uint8
        and value.ndim == 1
        and value.numel() > 0,
        "raw CPU private state",
    )
    return sampling._state_digest(value)


def _binding(declaration, compiled):
    declaration = schedule.checked(declaration)
    require(declaration["worlds"] == 64, "exact 64-row archive schedule")
    require(
        type(compiled) is dict
        and set(compiled)
        == {"selected_plant", "nbody", "body_names", "body_name", "body_id"},
        "compiled plant descriptor",
    )
    transition.preparation.baseline.force._binding(compiled)
    return schedule.binding_sha256(declaration), sha256(
        canonical(compiled).encode()
    ).hexdigest()


def _pulse_audit(records, declaration, compiled):
    """Audit complete full-array force records, never solver/physics replay."""
    schedule_sha, plant_sha = _binding(declaration, compiled)
    nbody, body = compiled["nbody"], compiled["body_id"]
    windows, delivered = [set() for _ in range(64)], [set() for _ in range(64)]
    phase_count = 0
    for record in records:
        raw = record["runtime_result_before_reset"]
        entries = raw["scheduled_pulse_evidence"]
        frames = raw["boundaries"]
        ctl = raw["control_evidence"]
        proposals = [p for p in ctl["proposals"] if p["accepted"].any()]
        require(
            type(entries) is list and len(entries) == len(frames) - 1 == len(proposals),
            "one force entry per accepted physical boundary",
        )
        for entry, before, after, proposal in zip(
            entries, frames, frames[1:], proposals
        ):
            require(
                type(entry) is dict
                and set(entry) == PULSE_KEYS
                and entry["schedule_sha256"] == schedule_sha
                and entry["binding_sha256"] == plant_sha,
                "exact force entry and source/plant binding",
            )
            _tensor(entry["before_steps"], (64,), torch.int64, "force clocks")
            _tensor(entry["accepted"], (64,), torch.bool, "force accepted mask")
            require(
                torch.equal(entry["before_steps"], before["physics_steps"])
                and torch.equal(
                    after["physics_steps"] - before["physics_steps"],
                    entry["accepted"].long(),
                )
                and torch.equal(entry["accepted"], proposal["accepted"]),
                "force clock/commit binding",
            )
            attempt.same_rows(before, after, ~entry["accepted"], observation=True)
            clocks, accepted = (
                entry["before_steps"].tolist(),
                entry["accepted"].tolist(),
            )
            for name, shape in (
                ("pre_xfrc", (64, nbody, 6)),
                ("post_xfrc", (64, nbody, 6)),
                ("pre_qfrc", (64, 20)),
                ("post_qfrc", (64, 20)),
            ):
                _tensor(entry[name], shape, torch.float32, name)
            schedule.validate_wrenches(
                entry["pre_xfrc"].tolist(),
                entry["pre_qfrc"].tolist(),
                declaration,
                clocks,
                accepted,
                nbody,
                body,
            )
            schedule.validate_wrenches(
                entry["post_xfrc"].tolist(),
                entry["post_qfrc"].tolist(),
                declaration,
                clocks,
                [False] * 64,
                nbody,
                body,
            )
            active = schedule.window_mask(declaration, clocks, accepted)
            require(
                (entry["phases"] is not None) == any(active),
                "exact force-phase coverage",
            )
            if any(active):
                require(
                    type(entry["phases"]) is dict
                    and set(entry["phases"])
                    == {"forced_pre", "integrated", "unforced_post"},
                    "all three retained force phases",
                )
                phase_count += 1
            for row, yes in enumerate(active):
                if yes:
                    require(
                        clocks[row] not in windows[row],
                        "no duplicated per-row force substep",
                    )
                    windows[row].add(clocks[row])
                    if any(declaration["row_cells"][row]["force_world_newtons"]):
                        delivered[row].add(clocks[row])
    rows = []
    for row, cell in enumerate(declaration["row_cells"]):
        expected = set(
            range(cell["onset_step"], cell["onset_step"] + cell["duration_steps"])
        )
        nonzero = any(cell["force_world_newtons"])
        reached = windows[row] == expected
        rows.append(
            dict(
                world_id=row,
                cell_id=cell["id"],
                nonzero=nonzero,
                expected_window_steps=len(expected),
                observed_window_steps=len(windows[row]),
                delivered_nonzero_steps=len(delivered[row]),
                window_status="complete"
                if reached
                else "partial"
                if windows[row]
                else "not-reached",
                complete_nonzero_delivery=bool(
                    nonzero and reached and delivered[row] == expected
                ),
            )
        )
    nonzero_rows = [row for row in rows if row["nonzero"]]
    return dict(
        rows=rows,
        phase_entries=phase_count,
        complete_nonzero_pulse_delivery=bool(
            nonzero_rows
            and all(row["complete_nonzero_delivery"] for row in nonzero_rows)
        ),
        solver_phases_replayed=False,
        solved_phase_payloads_replayed=False,
        unforced_post_arrays_zero=True,
    )


def check(value, expected_schedule, expected_plant, *, seed):
    """CPU-only raw-record consistency; supplied metadata does not attest CUDA."""
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CUDA-hidden archive checker",
    )
    # Inspect every nested value, including fields outside the arithmetic audit.
    # This pass makes no copies and never permits a CUDA tensor transfer.
    _owned_tree(value, allow_cuda=False, clone=False)
    require(
        type(value) is dict
        and set(value)
        == {"protocol", "declaration", "compiled_plant", "learner_seed", "records"}
        | set(transition.FALSE_FLAGS),
        "exact non-admitting archive schema",
    )
    require(
        type(seed) is int
        and seed in transition.preparation.SEEDS
        and type(value["learner_seed"]) is int
        and value["learner_seed"] == seed
        and value["protocol"] == PROTOCOL
        and all(value[key] is False for key in transition.FALSE_FLAGS),
        "fixed archive seed/protocol and false capability flags",
    )
    expected_schedule = schedule.checked(expected_schedule)
    schedule_sha, _ = _binding(expected_schedule, expected_plant)
    require(
        canonical(value["declaration"]) == canonical(expected_schedule)
        and canonical(value["compiled_plant"]) == canonical(expected_plant),
        "independently selected archive binding",
    )
    records = value["records"]
    require(
        type(records) is list and 1 <= len(records) <= 28, "one to 28 ordered records"
    )
    steps = torch.zeros(64, dtype=torch.long)
    previous_rng = None
    initial_rng_sha = None
    previous_observations = previous_controls = previous_kinematics = None
    initial_controls = initial_qpos = None
    terminal_seen = False
    for index, record in enumerate(records):
        require(not terminal_seen, "archive stops at first observed terminal")
        require(
            type(record) is dict and set(record) == RECORD_KEYS,
            "exact collector record",
        )
        receipt = record["receipt"]
        require(
            type(receipt) is dict
            and set(receipt)
            == {
                "protocol",
                "source",
                "learner_seed",
                "storage_step",
                "worlds",
                "device",
                "schedule_sha256",
                "optimizer_steps",
                "transition_bridge_qualified",
                "native_terminal_qualified",
                "native_selective_reset_qualified",
            }
            | set(transition.FALSE_FLAGS),
            "exact transition receipt",
        )
        require(
            receipt["protocol"] == transition.PROTOCOL
            and receipt["source"] == expected_schedule["source"]
            and type(receipt["learner_seed"]) is int
            and receipt["learner_seed"] == seed
            and type(receipt["storage_step"]) is int
            and receipt["storage_step"] == index
            and type(receipt["worlds"]) is int
            and receipt["worlds"] == 64
            and receipt["device"] == "cuda:0"
            and receipt["schedule_sha256"] == schedule_sha
            and type(receipt["optimizer_steps"]) is int
            and receipt["optimizer_steps"] == 0
            and all(
                receipt[key] is False
                for key in (
                    "transition_bridge_qualified",
                    "native_terminal_qualified",
                    "native_selective_reset_qualified",
                    *transition.FALSE_FLAGS,
                )
            ),
            "source-bound ordered no-update receipt",
        )
        for group in (
            "pre_action_observations",
            "terminal_observations",
            "next_observations",
        ):
            _observations(record[group], group)
        for name, shape in (
            ("raw_actions", (64, 10)),
            ("pre_action_values", (64, 1)),
            ("raw_actions_log_prob", (64,)),
            ("timeout_terminal_values", (64, 1)),
            ("environment_reward", (64,)),
            ("learner_reward", (64,)),
        ):
            _tensor(record[name], shape, torch.float32, name)
        for name in ("terminated", "timed_out"):
            _tensor(record[name], (64,), torch.bool, name)
        require(
            not record["timed_out"].any(),
            "natural timeout unreachable in fresh 28-tick capture",
        )
        require(
            not record["timeout_terminal_values"].any()
            and torch.equal(record["environment_reward"], record["learner_reward"]),
            "no spurious timeout bonus",
        )
        params = record["distribution_params"]
        require(
            type(params) in (list, tuple) and len(params) == 2,
            "mean and standard deviation",
        )
        for item in params:
            _tensor(item, (64, 10), torch.float32, "Gaussian parameters")
        mean, std = params
        require((std > 0).all(), "positive Gaussian standard deviation")
        log_prob = (
            -0.5
            * ((record["raw_actions"].double() - mean.double()) / std.double()).square()
            - std.double().log()
            - 0.5 * math.log(2 * math.pi)
        ).sum(-1)
        require(
            torch.allclose(
                record["raw_actions_log_prob"].double(), log_prob, atol=2e-4, rtol=0
            ),
            "raw action Gaussian log-probability consistency",
        )
        before, after = record["private_rng_state_before"], record["private_rng_state"]
        before_sha, after_sha = _state(before), _state(after)
        require(
            before.shape == after.shape
            and not torch.equal(before, after)
            and (previous_rng is None or torch.equal(previous_rng, before)),
            "continuous advancing private state",
        )
        scope = record["rng_scope_receipt"]
        require(
            type(scope) is dict
            and set(scope)
            == {
                "protocol",
                "source",
                "seed",
                "device",
                "state_bytes",
                "initial_state_sha256",
                "private_state_sha256",
                "scope_count",
                "caller_cpu_rng_preserved",
                "caller_cuda_rng_preserved",
                "scope_active",
                "faulted",
                "native_cuda_rng_scope_qualified",
                "native_cuda_sampling_qualified",
            }
            | set(transition.FALSE_FLAGS)
            and scope.get("protocol") == rng.PROTOCOL
            and scope.get("source") == expected_schedule["source"]
            and type(scope.get("seed")) is int
            and scope["seed"] == seed
            and scope.get("device") == "cuda:0"
            and type(scope.get("scope_count")) is int
            and scope["scope_count"] == index + 1
            and scope.get("private_state_sha256") == after_sha
            and type(scope.get("state_bytes")) is int
            and scope["state_bytes"] == after.numel()
            and scope["initial_state_sha256"] == (initial_rng_sha or before_sha)
            and scope.get("scope_active") is False
            and scope.get("faulted") is False
            and scope.get("caller_cpu_rng_preserved") is True
            and scope.get("caller_cuda_rng_preserved") is True
            and all(
                scope.get(key) is False
                for key in (
                    "native_cuda_rng_scope_qualified",
                    "native_cuda_sampling_qualified",
                    *transition.FALSE_FLAGS,
                )
            ),
            "non-admitting private scope receipt",
        )
        require(
            re.fullmatch(r"[0-9a-f]{64}", before_sha) is not None,
            "private state digest",
        )
        previous_rng = after
        initial_rng_sha = scope["initial_state_sha256"]
        raw = record["runtime_result_before_reset"]
        require(
            type(raw) is dict
            and set(raw)
            == {
                "reward",
                "terminated",
                "timed_out",
                "episode_steps",
                "executed_steps",
                "live",
                "term_sums",
                "observation",
                "boundaries",
                "terminal_records",
                "optimizer_launched",
                "control_evidence",
                "scheduled_pulse_evidence",
            }
            and raw["optimizer_launched"] is False,
            "whole runtime result with mandatory controls and forces",
        )
        for source, retained in (
            ("reward", "environment_reward"),
            ("terminated", "terminated"),
            ("timed_out", "timed_out"),
            ("episode_steps", "pre_reset_steps"),
        ):
            require(
                torch.equal(raw[source], record[retained]),
                "whole raw result matches " + retained,
            )
        _tensor(raw["executed_steps"], (64,), torch.long, "executed substeps")
        _tensor(raw["live"], (64,), torch.bool, "pre-reset live")
        _tensor(record["pre_reset_steps"], (64,), torch.long, "pre-reset clocks")
        _tensor(record["post_reset_steps"], (64,), torch.long, "post-reset clocks")
        done = record["terminated"]
        require(
            torch.equal(raw["live"], ~done)
            and torch.equal(record["pre_reset_steps"], steps + raw["executed_steps"])
            and ((raw["executed_steps"] >= 0) & (raw["executed_steps"] <= 10)).all()
            and (raw["executed_steps"][~done] == 10).all()
            and (record["pre_reset_steps"] <= 280).all()
            and torch.equal(
                record["post_reset_steps"],
                torch.where(done, 0, record["pre_reset_steps"]),
            ),
            "bounded actual row clocks and reset mask",
        )
        frames = raw["boundaries"]
        require(
            type(frames) is list and 1 <= len(frames) <= 11,
            "complete per-substep boundaries",
        )
        for frame in frames:
            attempt.validate_frame(frame, 64)
        require(
            torch.equal(frames[0]["physics_steps"], steps)
            and torch.equal(frames[-1]["physics_steps"], record["pre_reset_steps"]),
            "frame entry/exit clocks",
        )
        ctl = raw["control_evidence"]
        require(
            type(ctl) is dict
            and set(ctl) == {"initial", "after_action", "proposals", "final"}
            and type(ctl["proposals"]) is list
            and 0 <= len(ctl["proposals"]) <= 10,
            "whole control evidence",
        )
        for name in ("initial", "after_action", "final"):
            control.state(ctl[name], 64)
        boundary_index = 0
        recorded_rejections = torch.zeros(64, dtype=torch.bool)
        for proposal in ctl["proposals"]:
            require(
                type(proposal) is dict
                and set(proposal)
                == {
                    "before_steps",
                    "live",
                    "command",
                    "torque_nm",
                    "accepted",
                    "rejected",
                    "committed",
                },
                "exact motor proposal fields",
            )
            _tensor(proposal["before_steps"], (64,), torch.int64, "proposal clocks")
            for name in ("live", "accepted", "rejected"):
                _tensor(proposal[name], (64,), torch.bool, "proposal " + name)
            _tensor(proposal["torque_nm"], (64, 14), torch.float32, "proposed torque")
            require(
                torch.equal(
                    proposal["accepted"] | proposal["rejected"], proposal["live"]
                )
                and not (proposal["accepted"] & proposal["rejected"]).any(),
                "disjoint exhaustive proposal masks",
            )
            require(
                proposal["live"].any()
                and torch.equal(
                    proposal["before_steps"], frames[boundary_index]["physics_steps"]
                ),
                "proposal current physical boundary",
            )
            require(
                torch.equal(
                    control.available(frames[boundary_index], proposal["live"]),
                    proposal["live"],
                ),
                "proposal excludes already failed physical rows",
            )
            require(
                torch.equal(
                    proposal["rejected"],
                    proposal["live"] & (proposal["torque_nm"].abs().amax(1) > 0.36),
                ),
                "proposal recorded torque rejection mask",
            )
            recorded_rejections |= proposal["rejected"]
            if proposal["accepted"].any():
                boundary_index += 1
                require(
                    boundary_index < len(frames), "accepted proposal physical boundary"
                )
            require(
                type(proposal["command"]) is dict
                and set(proposal["command"]) == control.COMMAND,
                "exact motor command fields",
            )
            for name in control.COMMAND:
                _tensor(
                    proposal["command"][name],
                    (64, 14),
                    torch.float32,
                    "command " + name,
                )
            control.state(proposal["committed"], 64)
        require(
            boundary_index == len(frames) - 1, "complete proposal boundary coverage"
        )
        recorded_failures = recorded_rejections.clone()
        for frame in frames:
            recorded_failures |= attempt.physical_failures(
                attempt.PhysicsState(**frame["state"]), frame["physics_steps"]
            )
        require(
            torch.equal(done, recorded_failures),
            "termination matches retained physical/proposed failures",
        )
        for name in (
            "pre_reset_controls",
            "post_reset_controls",
            "initial_reset_controls",
        ):
            require(type(record[name]) is dict, "typed retained " + name)
            control.state(record[name], 64)
        require(
            transition._equal(record["pre_reset_controls"], ctl["final"]),
            "pre-reset controls match raw final controls",
        )
        if index == 0:
            initial_controls = record["initial_reset_controls"]
            initial_qpos = frames[0]["qpos"]
            require(
                transition._equal(ctl["initial"], initial_controls),
                "fresh initial controls",
            )
        else:
            require(
                transition._equal(ctl["initial"], previous_controls)
                and transition._equal(
                    record["initial_reset_controls"], initial_controls
                )
                and transition._equal(
                    record["pre_action_observations"], previous_observations
                ),
                "continuous observation and control records",
            )
            for key in ("qpos", "qvel"):
                require(
                    torch.equal(frames[0][key], previous_kinematics[key]),
                    "continuous retained physical " + key,
                )
        require(
            torch.allclose(
                record["pre_action_observations"]["actor"][:, 34:],
                ctl["initial"]["correction"] / 0.2,
                atol=1e-6,
                rtol=1e-5,
            ),
            "pre-action previous correction",
        )
        _observations(raw["observation"], "raw terminal observation")
        require(
            transition._equal(record["terminal_observations"], raw["observation"])
            and transition._equal(
                record["terminal_observations"], frames[-1]["observation"]
            ),
            "raw terminal observation matches last boundary",
        )
        for name in ("pre_reset_kinematics", "post_reset_kinematics"):
            physical = record[name]
            require(
                type(physical) is dict and set(physical) == {"qpos", "qvel", "time"},
                "exact retained physical fields",
            )
            for key, shape in (("qpos", (64, 21)), ("qvel", (64, 20)), ("time", (64,))):
                _tensor(physical[key], shape, torch.float32, name + " " + key)
        pre, post = record["pre_reset_kinematics"], record["post_reset_kinematics"]
        for key in ("qpos", "qvel"):
            require(
                torch.equal(pre[key], frames[-1][key]), "pre-reset last boundary " + key
            )
        require(
            torch.allclose(
                pre["time"],
                record["pre_reset_steps"].float() * 0.002,
                atol=2e-6,
                rtol=1e-5,
            ),
            "fresh physical clock arithmetic",
        )
        for key in pre:
            require(
                torch.equal(pre[key][~done], post[key][~done]),
                "unchanged live physical " + key,
            )
        require(
            torch.equal(post["qpos"][done], initial_qpos[done])
            and not post["qvel"][done].any()
            and not post["time"][done].any(),
            "reset physical rows",
        )
        for key in control.SHAPES:
            require(
                torch.equal(
                    record["pre_reset_controls"][key][~done],
                    record["post_reset_controls"][key][~done],
                )
                and torch.equal(
                    initial_controls[key][done],
                    record["post_reset_controls"][key][done],
                ),
                "retained reset control rows " + key,
            )
        for key in ("actor", "critic"):
            require(
                torch.equal(
                    record["terminal_observations"][key][~done],
                    record["next_observations"][key][~done],
                ),
                "unchanged live observation rows",
            )
        require(type(raw["term_sums"]) is dict, "reward term fields")
        for term in raw["term_sums"].values():
            _tensor(term, (64,), torch.float32, "raw reward term")
        for name in ("terminal_records", "reset_records"):
            require(
                type(record[name]) is list and len(record[name]) == 64,
                "retained terminal/reset ledger",
            )
        require(
            transition._equal(record["terminal_records"], raw["terminal_records"])
            and transition._equal(record["terminal_records"], record["reset_records"]),
            "whole raw terminal/reset ledger copies",
        )
        for row, terminal in enumerate(record["terminal_records"]):
            if done[row]:
                require(
                    type(terminal) is dict, "observed terminal requires retained facts"
                )
                attempt.terminal_matches(terminal, frames[-1], row)
            else:
                require(terminal is None, "no fabricated live-row terminal")
        previous_observations, previous_controls = (
            record["next_observations"],
            record["post_reset_controls"],
        )
        previous_kinematics = post
        terminal_seen = bool(done.any())
        steps = record["post_reset_steps"]
    pulses = _pulse_audit(records, expected_schedule, expected_plant)
    return dict(
        protocol=PROTOCOL,
        records=len(records),
        final_steps=steps.tolist(),
        complete_short_window=bool(len(records) == 28 and not terminal_seen),
        pulse=pulses,
        typed_raw_transition_consistency_checked=True,
        whole_trajectory_physics_resimulated=False,
        physics_control_solver_replayed=False,
        cuda_rng_math_replayed=False,
        native_transition_qualified=False,
        natural_timeout_qualified=False,
        selective_reset_qualified=False,
        caller_rng_states_independently_checked=False,
        model_and_optimizer_states_independently_checked=False,
        **transition.FALSE_FLAGS,
    )


def encode(records, declaration, compiled_plant, *, seed):
    """CPU-only archive; the caller persists exact bytes and supplies trusted pins."""
    value = _owned_tree(
        dict(
            protocol=PROTOCOL,
            declaration=declaration,
            compiled_plant=compiled_plant,
            learner_seed=seed,
            records=records,
            **transition.FALSE_FLAGS,
        ),
        allow_cuda=False,
    )
    score = check(value, declaration, compiled_plant, seed=seed)

    class BoundedBuffer(io.BytesIO):
        cap_exceeded = False

        def write(self, data):
            if self.tell() + len(data) > LIMIT:
                self.cap_exceeded = True
                raise ValueError("bounded serialized archive")
            return super().write(data)

    buffer = BoundedBuffer()
    try:
        torch.save(value, buffer)
    except Exception as error:
        # PyTorch's zip finalizer can mask a failed write with an unexpected-pos
        # RuntimeError; retain a deterministic cap refusal in that case only.
        if buffer.cap_exceeded:
            raise ValueError("bounded serialized archive") from error
        raise
    raw = buffer.getvalue()
    return raw, dict(sha256=sha256(raw).hexdigest(), bytes=len(raw), score=score)


def verify(raw, expected_sha256, declaration, compiled_plant, *, seed):
    """Verify exact whole bytes BEFORE CPU weights-only decoding and checking."""
    require(type(raw) is bytes and 0 < len(raw) <= LIMIT, "bounded serialized archive")
    require(
        type(expected_sha256) is str
        and re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        and sha256(raw).hexdigest() == expected_sha256,
        "whole archive byte hash",
    )
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    return check(value, declaration, compiled_plant, seed=seed)
