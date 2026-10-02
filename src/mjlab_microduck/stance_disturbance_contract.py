"""Pure declaration and validation for five one-substep force fixtures.

This namespace is separate from nominal-stance scoring. The fixture has no
actor, action, checkpoint, optimizer, recovery acceptance, or training role.
"""

from __future__ import annotations

import hashlib
import json
import math
import re


PROTOCOL = "football-b1d-single-substep-force-fixture-v1"
FRAME = "world"
ORDER = "force-then-torque"
APPLICATION = "body-inertial-com"
BODY_NAME = "trunk_base"
WORLD_COUNT = 1
NQ = 21
NV = 20
NU = 14
DT = 0.002
CHECK_TOL_ATOL = 1e-5
CHECK_TOL_RTOL = 1e-4
SERVICE_SECONDS = 180
CHILD_SECONDS = 120
CLOSEOUT_SECONDS = 180
CPU_QUAL_SECONDS = 30
GPU_CAPTURE_RESERVE = 60
CHILD_CLOSEOUT_RESERVE = 10
CUTOFF = 1790985600

# Each wrench is ordered [Fx, Fy, Fz, Tx, Ty, Tz].
CASES = (
    ("zero-wrench", (0, 0, 0, 0, 0, 0)),
    ("+x", (2, 0, 0, 0, 0, 0)),
    ("-x", (-2, 0, 0, 0, 0, 0)),
    ("+y", (0, 2, 0, 0, 0, 0)),
    ("-y", (0, -2, 0, 0, 0, 0)),
)

NO_ADMISSION = {
    "recovery_accepted": False,
    "training_admitted": False,
    "learned_stance_accepted": False,
    "football_balance_accepted": False,
    "physical_motion_authorized": False,
}


def _binding(plant):
    required = {"nbody", "body_names", "body_name", "body_id"}
    if type(plant) is not dict or not required.issubset(plant):
        raise ValueError("compiled plant descriptor with body binding required")
    nbody, body_id = plant["nbody"], plant["body_id"]
    names, name = plant["body_names"], plant["body_name"]
    if type(nbody) is not int or nbody < 2:
        raise ValueError("exact positive compiled body count required")
    if type(body_id) is not int or not 0 < body_id < nbody:
        raise ValueError("exact non-world compiled body id required")
    if type(names) not in (list, tuple) or len(names) != nbody or any(type(x) is not str for x in names):
        raise ValueError("compiled body names must bind every body")
    if type(name) is not str or name != BODY_NAME:
        raise ValueError("unique trunk_base body binding required")
    if names[body_id] != name or names.count(name) != 1:
        raise ValueError("compiled body name/id mismatch")
    return nbody, body_id


def plan(source, plant):
    """Return a canonical, source-bound fixture declaration without runtime work."""
    if type(source) is not str or re.fullmatch(r"[0-9a-f]{40}", source) is None:
        raise ValueError("lowercase 40-character source revision required")
    nbody, body_id = _binding(plant)
    try:
        encoded_plant = json.dumps(plant, sort_keys=True, separators=(",", ":"),
                               allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("plant descriptor must have canonical finite JSON values") from exc
    canonical_plant = json.loads(encoded_plant)
    plant_sha256 = hashlib.sha256(encoded_plant).hexdigest()
    identity = json.dumps({"source": source, "plant_sha256": plant_sha256},
                          sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    result = {
        "protocol": PROTOCOL,
        "source": source,
        "plant": canonical_plant,
        "plant_sha256": plant_sha256,
        "source_plant_sha256": hashlib.sha256(identity).hexdigest(),
        "worlds": WORLD_COUNT,
        "nq": NQ,
        "nv": NV,
        "nu": NU,
        "dt": DT,
        "cases": [{"name": name, "wrench": list(wrench)} for name, wrench in CASES],
        "frame": FRAME,
        "wrench_order": ORDER,
        "application": APPLICATION,
        "pre_forward_forced_solves_per_case": 1,
        "euler_steps_per_case": 1,
        "post_forward_unforced_captures_per_case": 1,
        "clear_all_applied_forces_after_euler": True,
        "same_reset_and_prepared_motor_fields_each_case": True,
        "zero_error_bam_motor_fields": True,
        "frozen_ctrl": True,
        "policy_inferences": 0,
        "optimizer_steps": 0,
        "actor": False,
        "actions": False,
        "checkpoint": False,
        "perception": False,
        "child_seconds": CHILD_SECONDS,
        "service_seconds": SERVICE_SECONDS,
        "closeout_seconds": CLOSEOUT_SECONDS,
        "cpu_qualification_seconds": CPU_QUAL_SECONDS,
        "gpu_capture_reserve_seconds": GPU_CAPTURE_RESERVE,
        "child_closeout_reserve_seconds": CHILD_CLOSEOUT_RESERVE,
        "cutoff_unix": CUTOFF,
        "replay_atol": CHECK_TOL_ATOL,
        "replay_rtol": CHECK_TOL_RTOL,
        "force_fixture_only": True,
        **NO_ADMISSION,
    }
    return result


def _case_wrench(case):
    if type(case) is not str:
        raise ValueError("declared force case name required")
    for name, wrench in CASES:
        if case == name:
            return wrench
    raise ValueError("undeclared force case")


def expected_wrench(case, nbody, body_id):
    """Build exact one-world xfrc and zero generalized-force list payloads."""
    wrench = _case_wrench(case)
    if type(nbody) is not int or nbody < 2 or type(body_id) is not int or not 0 < body_id < nbody:
        raise ValueError("compiled non-world body binding required")
    xfrc = [[[0, 0, 0, 0, 0, 0] for _ in range(nbody)]]
    xfrc[0][body_id] = list(wrench)
    qfrc = [[0] * NV]
    return xfrc, qfrc


def _numeric_tree(value, shape, label):
    if not shape:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(label + " must contain finite non-boolean numbers")
        return value
    if type(value) is not list or len(value) != shape[0]:
        raise ValueError(label + " shape mismatch")
    for item in value:
        _numeric_tree(item, shape[1:], label)


def validate_applied(xfrc, qfrc, case, nbody, body_id):
    """Reject any undeclared, malformed, non-finite, or generalized applied force."""
    expected_xfrc, expected_qfrc = expected_wrench(case, nbody, body_id)
    _numeric_tree(xfrc, (WORLD_COUNT, nbody, 6), "xfrc_applied")
    _numeric_tree(qfrc, (WORLD_COUNT, NV), "qfrc_applied")
    if xfrc != expected_xfrc:
        raise ValueError("xfrc_applied differs from the declared body wrench")
    if qfrc != expected_qfrc:
        raise ValueError("qfrc_applied must be exact zero")
    return True
