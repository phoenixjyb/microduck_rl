"""Rescore retained matched evaluations for a straight-lesson stage exit.

This is a curriculum stage gate only. It does not establish policy acceptance,
simulator qualification, or physical motion authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

from mjlab_microduck import integrated_training_budget_refinement as foundation
from mjlab_microduck import integrated_joint_limit_refinement as jointlimit
from mjlab_microduck import integrated_straight_lesson as recipe

base = foundation.base
PROTOCOL = "straight-lesson-stage-exit-v1"
ARMS = ("control", "straight")
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_RE = re.compile(r"[0-9a-f]{40}\Z")
CASES = tuple(foundation.base.CASES)
SEEDS = tuple(foundation.base.SEEDS)
CASE_LABELS = ("idle", "straight", "negative-arc", "positive-arc")
WORLD_COUNT = 8
JOINT = jointlimit.JOINT


def _need(condition: bool, message: str) -> None:
    base.require(condition, message)


def _identity(row: Mapping) -> tuple[int, float, float]:
    return row["seed"], row["speed"], row["yaw"]


def _validate_aggregate(aggregate: Mapping) -> tuple[dict[str, list], dict[str, str]]:
    _need(isinstance(aggregate, Mapping), "retained aggregate mapping")
    _need(aggregate.get("protocol") == recipe.PROTOCOL, "exact straight-lesson recipe protocol")
    _need(isinstance(aggregate.get("source"), str)
          and SOURCE_RE.fullmatch(aggregate["source"]) is not None, "40-hex source revision")
    _need(aggregate.get("selection") == "both final999 only; unchanged held-out protocol",
          "both final999 checkpoints use unchanged held-out protocol")
    for flag in ("policy_acceptance", "simulator_qualified", "physical_motion_authorized"):
        _need(aggregate.get(flag) is False, f"aggregate {flag} remains false")
    cases = aggregate.get("cases")
    hashes = aggregate.get("checkpoint_sha256")
    decisions = aggregate.get("decisions")
    _need(isinstance(cases, Mapping) and set(cases) == set(ARMS), "exact control/straight cases")
    _need(isinstance(hashes, Mapping) and set(hashes) == set(ARMS), "exact control/straight checkpoint hashes")
    _need(isinstance(decisions, Mapping) and set(decisions) == set(ARMS), "exact control/straight canonical decisions")

    normalized_cases: dict[str, list] = {}
    normalized_hashes: dict[str, str] = {}
    seen_hashes: set[str] = set()
    expected = [(seed, *case) for seed in SEEDS for case in CASES]
    for arm in ARMS:
        checkpoint = hashes[arm]
        _need(isinstance(checkpoint, str) and HASH_RE.fullmatch(checkpoint) is not None,
              f"{arm} checkpoint SHA-256")
        _need(checkpoint not in seen_hashes, "distinct control and straight checkpoint hashes")
        seen_hashes.add(checkpoint)
        rows = cases[arm]
        _need(type(rows) is list and len(rows) == len(expected), f"{arm} complete ordered evaluation matrix")
        for index, (row, identity) in enumerate(zip(rows, expected, strict=True)):
            _need(isinstance(row, Mapping), f"{arm} row {index} mapping")
            _need(_identity(row) == identity, f"{arm} row {index} exact seed/case order")
            _need(row.get("checkpoint_sha256") == checkpoint,
                  f"{arm} row {index} checkpoint hash matches aggregate")
            _need(type(row.get("complete")) is bool
                  and type(row.get("worlds")) is int and row["worlds"] == WORLD_COUNT
                  and type(row.get("steps")) is int and 0 < row["steps"] <= 240,
                  f"{arm} row {index} fixed world and step shape")
            _need((row["complete"] and row["steps"] == 240)
                  or (not row["complete"] and bool(row.get("terminal_worlds"))),
                  f"{arm} row {index} complete/terminal consistency")
            terminals = row.get("terminal_worlds")
            _need(type(terminals) is list and len(terminals) == len(set(terminals))
                  and all(type(world) is int and 0 <= world < WORLD_COUNT for world in terminals),
                  f"{arm} row {index} terminal world indices")
            for metric in ("speed_mae", "yaw_mae"):
                values = row.get(metric)
                if values is None:
                    _need(not row["complete"], f"{arm} row {index} complete {metric} coverage")
                else:
                    _need(type(values) is list and len(values) == WORLD_COUNT
                          and all(type(value) in (int, float) and value >= 0 for value in values),
                          f"{arm} row {index} finite nonnegative {metric} coverage")
            _need(type(row.get("torque_p99")) in (int, float) and row["torque_p99"] >= 0
                  and type(row.get("rated_speed_exceed_fraction")) in (int, float)
                  and 0 <= row["rated_speed_exceed_fraction"] <= 1,
                  f"{arm} row {index} motor metric shape")
        normalized_cases[arm] = rows
        normalized_hashes[arm] = checkpoint

    return normalized_cases, normalized_hashes


def _case_label(index: int) -> str:
    return CASE_LABELS[index % len(CASES)]


def _evidence(seed: int, speed: float, yaw: float, world: int | None,
              metric: str, value, limit, *, arm: str, case: str) -> dict:
    return dict(arm=arm, seed=seed, case=case, speed=speed, yaw=yaw,
                world=world, metric=metric, value=value, limit=limit)


def _row_gate_evidence(row: Mapping, arm: str, index: int, *, include_tracking=True) -> list[dict]:
    """Return original foundation gate failures with world-level evidence."""
    seed, speed, yaw = _identity(row)
    case = _case_label(index)
    failures: list[dict] = []
    if not row["complete"]:
        failures.append(_evidence(seed, speed, yaw, None, "complete", False, True,
                                  arm=arm, case=case))
        return failures

    # The canonical scorer validates the complete schema and coverage before
    # this function is called. Keep its exact thresholds here for attributable
    # evidence, including per-world values rather than only maxima.
    speed_limit = .03 if speed == 0 else .10
    yaw_limit = .10
    if include_tracking:
        for world, value in enumerate(row["speed_mae"]):
            if value > speed_limit:
                failures.append(_evidence(seed, speed, yaw, world, "speed_mae", value,
                                          f"<={speed_limit}", arm=arm, case=case))
        for world, value in enumerate(row["yaw_mae"]):
            if value > yaw_limit:
                failures.append(_evidence(seed, speed, yaw, world, "yaw_mae", value,
                                          f"<={yaw_limit}", arm=arm, case=case))
    if row["torque_p99"] > .60:
        failures.append(_evidence(seed, speed, yaw, None, "torque_p99", row["torque_p99"],
                                  "<=0.60", arm=arm, case=case))
    if row["rated_speed_exceed_fraction"] != 0:
        failures.append(_evidence(seed, speed, yaw, None, "rated_speed_exceed_fraction",
                                  row["rated_speed_exceed_fraction"], "==0", arm=arm, case=case))
    diagnostics = row["response_diagnostics"]
    joint_index = diagnostics["joint_columns"].index(JOINT)
    minimum = diagnostics["min_distance_to_hard_stop_rad_by_joint"]
    proximity = diagnostics["hard_stop_proximity_fraction_by_joint"]
    if min(minimum) < -.005:
        joint = min(range(len(minimum)), key=minimum.__getitem__)
        failures.append(_evidence(seed, speed, yaw, None,
            f"min_distance_to_hard_stop_rad:{diagnostics['joint_columns'][joint]}", minimum[joint],
            ">=-0.005", arm=arm, case=case))
    if proximity[joint_index] > .01:
        failures.append(_evidence(seed, speed, yaw, None,
            f"hard_stop_proximity_fraction:{JOINT}", proximity[joint_index],
            "<=0.01", arm=arm, case=case))
    if row["terminal_worlds"]:
        failures.append(_evidence(seed, speed, yaw, row["terminal_worlds"][0],
            "terminal_worlds", row["terminal_worlds"], "empty", arm=arm, case=case))
    return failures


def evaluate(aggregate: Mapping) -> dict:
    """Validate and rescore a retained two-arm aggregate without training."""
    cases, checkpoint_hashes = _validate_aggregate(aggregate)
    full_foundation_decisions = {}
    for arm in ARMS:
        # This is the existing qpos-refinement alias to the original full
        # foundation decision. Calling it preserves the canonical predicate.
        recomputed = foundation.decision(cases[arm])
        _need(aggregate["decisions"][arm] == recomputed,
              f"{arm} stored canonical decision matches rescore")
        full_foundation_decisions[arm] = recomputed

    treatment = cases["straight"]
    stage_indices = [seed_index * len(CASES) + case_index
                     for seed_index in range(len(SEEDS)) for case_index in (0, 1)]
    arc_indices = [seed_index * len(CASES) + case_index
                   for seed_index in range(len(SEEDS)) for case_index in (2, 3)]
    stage_failures = [failure for index in stage_indices
                      for failure in _row_gate_evidence(treatment[index], "straight", index)]
    arc_safety_failures = [failure for index in arc_indices
                           for failure in _row_gate_evidence(treatment[index], "straight", index,
                                                             include_tracking=False)]

    arc_retention_failures: list[dict] = []
    for index in arc_indices:
        control_row, treatment_row = cases["control"][index], treatment[index]
        # Incomplete episodes are already an unchanged safety-gate failure.
        # They cannot be compared as if missing metric arrays were valid data.
        if not control_row["complete"] or not treatment_row["complete"]:
            seed, speed, yaw = _identity(treatment_row)
            arc_retention_failures.append(_evidence(
                seed, speed, yaw, None, "paired_complete",
                dict(control=control_row["complete"], straight=treatment_row["complete"]),
                "both true for per-world comparison", arm="straight", case=_case_label(index)))
            continue
        seed, speed, yaw = _identity(treatment_row)
        case = _case_label(index)
        for metric in ("speed_mae", "yaw_mae"):
            for world, (control_value, treatment_value) in enumerate(
                    zip(control_row[metric], treatment_row[metric], strict=True)):
                if treatment_value > control_value:
                    arc_retention_failures.append(_evidence(
                        seed, speed, yaw, world, metric + "_vs_control",
                        treatment_value, f"<={control_value}", arm="straight", case=case))

    stage_ready = not stage_failures and not arc_safety_failures and not arc_retention_failures
    return dict(protocol=PROTOCOL, source=aggregate["source"],
        input_protocol=aggregate["protocol"], selection=aggregate["selection"],
        decision="ready-for-declared-turn-lesson" if stage_ready else "straight-lesson-not-ready",
        stage_exit=not stage_failures,
        arc_safety=not arc_safety_failures,
        arc_retention=not arc_retention_failures,
        stage_exit_failures=stage_failures,
        arc_safety_failures=arc_safety_failures,
        arc_retention_failures=arc_retention_failures,
        full_foundation_decisions=full_foundation_decisions,
        checkpoint_sha256=checkpoint_hashes,
        policy_acceptance=False, simulator_qualified=False, physical_motion_authorized=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    raw = args.input.read_bytes()
    aggregate = json.loads(raw)
    result = evaluate(aggregate)
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    base.write_new(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
