"""Descriptive same-rollout accounting and replicate ranges, never promotion."""

import re

from mjlab_microduck import foundation_motor_experiment as exp
from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.motor_trace_audit import audit_motor_trace, JOINTS
from mjlab_microduck.rollout_repeatability import differences

PROTOCOL = "f1m-descriptive-motor-replicates-v1"
ORDER = ("parent-1", "control-1", "motor-1", "motor-2", "control-2", "parent-2")
CHECKPOINTS = {
    "parent": exp.MODEL_SHA,
    "control": "cf63952d0975b120f14f92b7cb9b7d5f8b2d6f5e46b5db8f214c63b1c77df83a",
    "motor": "dcef0f75a3b3bccd0f026de34510aca793f888d10ff8aa4569e74ee4201a18fb",
}
CLAIMS = dict(policy_acceptance=False, historical_replay_identity=False,
              causal_effect_established=False, physical_motion_authorized=False,
              training_admitted=False, curriculum_promotion=False)
GROUP_METRICS = (
    "body_forward_mean", "route_forward_mean", "cross_route_abs_mean", "heading_abs_max",
    "pre_reset_torque_p99", "pre_reset_squared_utilization_mean",
    "pre_reset_mechanical_abs_power_mean_w", "pre_reset_soft_limit_fraction",
)
FINGERPRINT_KEYS = {"environment", "deterministic_algorithms", "cudnn_benchmark", "cudnn_deterministic",
    "cudnn_enabled", "matmul_fp32_precision", "cudnn_fp32_precision", "default_dtype", "threads",
    "interop_threads", "cuda_initialized", "python_hash_probe"}
EXPECTED_ENV = dict(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONHASHSEED="503",
                    CUBLAS_WORKSPACE_CONFIG=None, NVIDIA_TF32_OVERRIDE=None, CUDA_LAUNCH_BLOCKING=None)


def identity(case):
    require(case in ORDER, "predeclared case, no extra replicates")
    arm, replica = case.split("-")
    return dict(protocol=PROTOCOL, arm=arm, replicate=int(replica), seed=503,
                raw_motor_history=True, **CLAIMS)


def measure_case(report, *, source, case):
    """Reconstruct this report only; never equate it with a historical rollout."""
    canonical(report)
    require(isinstance(source, str) and re.fullmatch("[0-9a-f]{40}", source) is not None
            and report["source"] == source, "exact recording source")
    require(canonical(report["measurement_identity"]) == canonical(identity(case)), "descriptive scope and case identity")
    for key in ("policy_acceptance", "training_admitted", "physical_motion_authorized", "reopens_recovery_ab"):
        require(report[key] is False, "raw report must not claim admission")
    exp.validate_controller_trace(report, True, protocol=exp.PROTOCOL, seeds=(503,),
                                  checkpoint_sha=CHECKPOINTS[case.split("-")[0]])
    analysis = audit_motor_trace(report)
    require(not report["safety_failures"] and report["sample_steps"] == 400
            and analysis["decision"] == "timing-measurement-only", "absolute-safe complete timing coverage")
    for name in ("all", "settled"):
        require(0 <= report["groups"][name]["pre_reset_torque_p99"] <= .60,
                "unchanged absolute pre-reset torque gate")
    return dict(protocol=PROTOCOL, case=case, source=source, raw_motor_summary_verified=True,
                analysis=analysis, **CLAIMS)


def observed_range(values):
    require(len(values) == 2 and all(type(v) in (int, float) for v in values), "two numeric replicates")
    canonical(values)
    return dict(replicates=list(values), minimum=min(values), maximum=max(values), spread=max(values)-min(values))


def range_contrast(reference, candidate):
    """Observed extrema only: not a confidence interval or independent seeds."""
    a, b = observed_range(reference), observed_range(candidate)
    lower, upper = b["minimum"]-a["maximum"], b["maximum"]-a["minimum"]
    return dict(reference=a, candidate=b, all_pair_deltas=[y-x for x in reference for y in candidate],
                delta_minimum=lower, delta_maximum=upper,
                observed_direction="higher" if lower > 0 else "lower" if upper < 0 else "overlapping-or-equal",
                confidence_interval=False, causal_effect_established=False)


def reconcile_derived_analysis(recorded, recomputed):
    """Cross-CPU derived means only, at the existing raw-summary precision.

    Raw reports, hashes, quantiles, indices, counts, labels and decisions do not
    receive this tolerance. This never replaces historical replay identity.
    """
    canonical([recorded,recomputed])
    require(all(r[k] is False for r in (recorded,recomputed) for k in CLAIMS), "descriptive scope only")
    delta=differences(recorded,recomputed)
    path=re.compile(r"/analysis/joints/("+"|".join(JOINTS)+r")/(settled_squared_load_share|"
                    r"squared_load_other|squared_load_when_route_below_027|one_second_bins/[0-5]/squared_utilization_mean)")
    for row in delta:
        require("missing" not in row and path.fullmatch(row["path"]) is not None
                and type(row["first"]) is float and type(row["second"]) is float
                and abs(row["first"]-row["second"]) <= 1e-9, "derived-analysis mismatch outside existing reconstruction precision")
    return dict(exact_match=not delta, differing_float_leaves=len(delta),
                max_absolute_delta=max((abs(r["first"]-r["second"]) for r in delta),default=0.),
                derived_mean_absolute_tolerance=1e-9, differences=delta, **CLAIMS)


def describe_dataset(records, *, source):
    canonical(records)
    require([r["case"] for r in records] == list(ORDER), "exact six-case balanced order, no selection or omission")
    by_case = {}
    reference_states = records[0]["fingerprint"]
    for record in records:
        case, report, states = record["case"], record["report"], record["fingerprint"]
        measure_case(report, source=source, case=case)
        require(set(states) == {"before", "after"}, "entry and exit fingerprints")
        for phase in ("before", "after"):
            state = states[phase]
            require(set(state) == FINGERPRINT_KEYS
                    and state["environment"] == EXPECTED_ENV
                    and type(state["python_hash_probe"]) is int
                    and state["cuda_initialized"] is (phase == "after")
                    and canonical(state) == canonical(reference_states[phase]), "matched effective process fingerprints")
        require(states["before"]["python_hash_probe"] == states["after"]["python_hash_probe"], "stable process hash probe")
        by_case[case] = report
    def contrast(get):
        return {reference: range_contrast([get(by_case[f"{reference}-{i}"]) for i in (1,2)],
                                          [get(by_case[f"motor-{i}"]) for i in (1,2)])
                for reference in ("parent", "control")}
    groups = {name: {key: contrast(lambda r: r["groups"][name][key]) for key in GROUP_METRICS}
              for name in ("all", "settled")}
    joints = {joint: contrast(lambda r: r["groups"]["settled"]["pre_reset_joint_p99"][joint]) for joint in JOINTS}
    # Preserve every original admission failure for each fresh replica. These
    # descriptive data do not reopen the historical five-case decision.
    paired = {str(i): exp.paired_decision(*(by_case[f"{arm}-{i}"] for arm in ("parent","control","motor")))
              for i in (1,2)}
    return dict(protocol=PROTOCOL, source=source, decision="descriptive-replicates-only", cases=list(ORDER),
                groups=groups, settled_joint_torque=joints, unchanged_gate_readouts=paired,
                process_replicates_per_arm=2, distinct_physics_seeds=1, optimizer_updates=0,
                inferential_statistics=False, historical_rejection_reopened=False, **CLAIMS)
