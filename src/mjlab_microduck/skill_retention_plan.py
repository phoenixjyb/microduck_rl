"""CPU-only coverage and byte-reference audit, never a behavioral admission.

The fixed catalog describes required *future* retention work. A reference to
bytes does not prove that those bytes implement a protocol or satisfy a gate.
No policies, simulators, YAML constructors or evaluation reports are executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from mjlab_microduck.skill_compatibility import (
    _require, _verify_reference, compare_descriptors, verify_descriptor,
)


SCHEMA = "skill-retention-plan-v1"
SKILLS = ("foundation", "stop-recovery", "obstacle", "hop")
SKILL_FIELDS = ("descriptor", "original_protocol", "retention_procedure", "retention_report")
TRANSITIONS = tuple(f"{source}->{target}" for source in SKILLS for target in SKILLS
                    if source != target)
# Directed requests are not an instruction to switch directly: a procedure may
# explicitly refuse a request or route it through a tested stance/landing state.
CASES = (
    "fresh-command-and-action-before-first-apply",
    "actuator-delay-and-internal-state-handoff",
    "action-history-and-processed-cache-handoff",
    "observation-history-and-normalizer-handoff",
    "stop-from-motion-and-stable-stance",
    "restart-from-stop-and-speed-reacquisition",
    "interruption-before-takeoff",
    "interruption-in-air-landing-before-stop",
    "interruption-during-landing-and-recovery",
    "stale-structured-sensor-and-recovery",
    "missing-structured-sensor-and-recovery",
    "invalid-structured-sensor-and-recovery",
    "avoidance-slowdown-inside-maneuver-zone",
    "nominal-speed-before-and-after-avoidance",
    "per-placement-speed-bin-and-worst-seed-results",
    "independent-training-and-untouched-confirmation-seeds",
)
SEMANTIC_GAPS = (
    "original-protocol-content-and-unchanged-numerical-gates-not-verified",
    "descriptor-to-effective-runtime-binding-not-verified",
    "closed-loop-per-skill-retention-not-verified",
    "directed-transition-interruption-and-sensor-behavior-not-verified",
)


def empty_plan() -> dict:
    """Explicitly unfilled plan; absence is not a synthetic evidence reference."""
    return dict(schema=SCHEMA, skills={}, transitions={}, cases={})


def _mapping(value: object, allowed: tuple, label: str) -> dict:
    _require(type(value) is dict and set(value).issubset(allowed),
             f"{label}: unknown fields or non-mapping")
    return value


def check_plan(root: Path, plan: object) -> dict:
    """Audit a quiescent root, returning missing references in fixed order.

Omitted known entries and explicit null mean missing. Malformed, unknown or
tampered supplied entries raise ValueError instead of becoming benign gaps.
Even complete reference coverage leaves all behavioral/launch flags false.
"""
    _require(type(plan) is dict and set(plan) == {"schema", "skills", "transitions", "cases"},
             "exact plan fields required")
    _require(plan["schema"] == SCHEMA, "unsupported plan schema")
    root = Path(root).resolve(strict=True)
    _require(root.is_dir(), "artifact root must be a directory")
    skills = _mapping(plan["skills"], SKILLS, "skills")
    transitions = _mapping(plan["transitions"], TRANSITIONS, "transitions")
    cases = _mapping(plan["cases"], CASES, "cases")
    gaps, verified = [], dict(skills={}, transitions={}, cases={})

    def reference(value, label):
        if value is None:
            gaps.append(label)
            return None
        return _verify_reference(root, label, value)

    for skill in SKILLS:
        entry = _mapping(skills.get(skill, {}), SKILL_FIELDS, f"skills.{skill}")
        descriptor = entry.get("descriptor")
        if descriptor is None:
            gaps.append(f"skills.{skill}.descriptor")
        else:
            descriptor = verify_descriptor(root, descriptor)
        record = dict(descriptor=descriptor)
        for field in SKILL_FIELDS[1:]:
            record[field] = reference(entry.get(field), f"skills.{skill}.{field}")
        if descriptor is not None and record["original_protocol"] is not None:
            original = record["original_protocol"]
            bound = descriptor["evaluation_protocol"]
            _require((original["sha256"], original["bytes"]) == (bound["sha256"], bound["bytes"]),
                     f"skills.{skill}: original protocol differs from descriptor")
        verified["skills"][skill] = record

    for section, keys, entries in (("transitions", TRANSITIONS, transitions),
                                   ("cases", CASES, cases)):
        for name in keys:
            entry = _mapping(entries.get(name, {}), ("procedure", "report"), f"{section}.{name}")
            verified[section][name] = {
                field: reference(entry.get(field), f"{section}.{name}.{field}")
                for field in ("procedure", "report")
            }

    compatibility = {}
    for name in TRANSITIONS:
        source, target = name.split("->")
        left = verified["skills"][source]["descriptor"]
        right = verified["skills"][target]["descriptor"]
        if left is None or right is None:
            compatibility[name] = dict(decision="unknown-missing-descriptor", mismatches=None)
        else:
            comparison = compare_descriptors(root, left, root, right)
            compatibility[name] = {field: comparison[field] for field in ("decision", "mismatches")}

    return dict(
        schema=SCHEMA,
        canonical_plan_sha256=hashlib.sha256(json.dumps(
            plan, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest(),
        implementation_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("skill_compatibility.py"))
        },
        decision="incomplete-reference-coverage" if gaps else "reference-coverage-only",
        missing_references=gaps,
        reference_coverage_complete=not gaps,
        verified=verified,
        declared_static_compatibility=compatibility,
        unresolved_semantic_gates=list(SEMANTIC_GAPS),
        original_numerical_gates_verified=False,
        descriptor_to_runtime_binding_verified=False,
        behavioral_retention_verified=False,
        safe_stop_verified=False,
        policy_acceptance=False,
        transition_authorized=False,
        gpu_collection_authorized=False,
        training_authorized=False,
        physical_motion_authorized=False,
    )


def load_plan(path: Path) -> dict:
    """Reject ambiguous duplicate keys and non-JSON numeric constants."""
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    with path.open("rb") as stream:
        payload = stream.read(1024 * 1024 + 1)
    _require(len(payload) <= 1024 * 1024, "plan exceeds 1 MiB")
    return json.loads(payload, object_pairs_hook=pairs, parse_constant=constant)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check_plan(args.root, load_plan(args.plan))
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        # Never overwrite a plan, checkpoint or previous audit. Caller supplies
        # an existing audit directory outside immutable campaign payload roots.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        print(f"retention_plan_decision={result['decision']}")
        print(f"missing_references={len(result['missing_references'])}")


if __name__ == "__main__":
    main()
