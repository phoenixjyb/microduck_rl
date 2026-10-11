import hashlib
import json
from pathlib import Path

import pytest

from mjlab_microduck import integrated_observation_timing_audit as audit
from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck import integrated_low_yaw_rehearsal_refinement as recipe


def put_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return path


def case_rows(arm, checkpoint_sha):
    joints = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
              "left_ankle", "neck_pitch", "head_pitch", "head_yaw", "head_roll",
              "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
              "right_ankle"]
    rows = []
    for seed in base.SEEDS:
        for speed, yaw in base.CASES:
            rows.append(dict(
                seed=seed, speed=speed, yaw=yaw, worlds=8, steps=240, complete=True,
                terminal_worlds=[], checkpoint_sha256=checkpoint_sha,
                speed_mae=[.01] * 8, yaw_mae=[.01] * 8, torque_p99=.2,
                soft_limit_fraction=.01, rated_speed_exceed_fraction=0.,
                mean_abs_mechanical_power_w=1.5, thermal_load_proxy=.1,
                response_diagnostics=dict(
                    protocol="signed-body-response-and-joint-load-v1", steps=240,
                    hard_stop_margin_rad=.05, joint_columns=joints,
                    min_distance_to_hard_stop_rad_by_joint=[.1] * 14,
                    hard_stop_proximity_fraction_by_joint=[0.] * 14,
                ),
            ))
    return rows


def fixture_bundle(tmp_path, monkeypatch):
    root = tmp_path / "retained-input"
    monkeypatch.setattr(audit, "INPUT", root)
    checkpoint_shas = {}
    cases = {}
    decisions = {}
    for arm in audit.ARMS:
        contract = motor.training_contract(recipe, arm)
        checkpoint = motor.checkpoint_path(root, arm, contract)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(("checkpoint-" + arm).encode())
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        checkpoint_shas[arm] = digest
        put_json(root / arm / "launch.json", dict(
            source=audit.INPUT_SOURCE, protocol=recipe.PROTOCOL, mode=arm,
            runtime=audit.RUNTIME, parent_sha256=recipe.PARENT_SHA,
        ))
        put_json(root / arm / "result.json", dict(
            source=audit.INPUT_SOURCE, protocol=recipe.PROTOCOL,
            status="training-complete-not-accepted", mode=arm,
            updates=contract["updates"], worlds=contract["worlds"], seed=contract["seed"],
            steps=contract["updates"] * 24,
            transitions=contract["updates"] * 24 * contract["worlds"],
            warm_start={"parent_sha256": recipe.PARENT_SHA},
            checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest,
        ))
        cases[arm] = case_rows(arm, digest)
        decisions[arm] = recipe.decision(cases[arm])
        for index, row in enumerate(cases[arm]):
            seed = base.SEEDS[index // 4]
            put_json(root / "evaluate" / f"{arm}-s{seed}-c{index % 4}.json", row)

    plan = motor.evaluation_plan(recipe)
    prior = dict(
        source=audit.INPUT_SOURCE, protocol=recipe.PROTOCOL,
        selection=plan["selection"], final_iterations=plan["final_iterations"],
        policy_acceptance=False, simulator_qualified=False, physical_motion_authorized=False,
        cases=cases, decisions=decisions, checkpoint_sha256=checkpoint_shas,
        paired_differences=motor.paired_differences(
            cases[audit.ARMS[0]], cases[audit.ARMS[1]], label=audit.ARMS[1] + "_minus_control"),
    )
    result_path = put_json(root / "evaluate" / "result.json", prior)
    put_json(root / "evaluate" / "launch.json", dict(
        source=audit.INPUT_SOURCE, protocol=recipe.PROTOCOL, mode="evaluate",
        runtime=audit.RUNTIME, parent_sha256=recipe.PARENT_SHA,
    ))
    raw = result_path.read_bytes()
    return dict(root=root, prior=prior, cases=cases, checkpoint_shas=checkpoint_shas,
                sha=hashlib.sha256(raw).hexdigest(), result_path=result_path)


def rewrite_aggregate(bundle):
    put_json(bundle["result_path"], bundle["prior"])
    bundle["sha"] = hashlib.sha256(bundle["result_path"].read_bytes()).hexdigest()


def test_audit_accepts_exact_final499_retained_full_two_arm_matrix(tmp_path, monkeypatch):
    bundle = fixture_bundle(tmp_path, monkeypatch)

    validated = audit.validated_input(bundle["sha"], audit.RUNTIME)

    assert validated["final_iterations"] == {arm: 499 for arm in audit.ARMS}
    assert validated["checkpoint_sha256"] == bundle["checkpoint_shas"]
    assert set(validated["cases"]) == set(validated["decisions"]) == set(audit.ARMS)
    assert all(len(validated["cases"][arm]) == 12 for arm in audit.ARMS)
    assert validated["paired_differences"] == bundle["prior"]["paired_differences"]


@pytest.mark.parametrize("mutation", [
    "wrong-final", "wrong-source", "wrong-runtime", "wrong-parent",
    "case-order", "decision", "paired-difference", "drop-final-iteration",
])
def test_audit_rejects_aggregate_identity_or_protocol_mutations(tmp_path, monkeypatch, mutation):
    bundle = fixture_bundle(tmp_path, monkeypatch)
    prior = bundle["prior"]
    if mutation == "wrong-final":
        prior["final_iterations"]["rehearsal"] = 498
    elif mutation == "wrong-source":
        prior["source"] = "f" * 40
    elif mutation == "wrong-runtime":
        put_json(bundle["root"] / "evaluate" / "launch.json", dict(
            source=audit.INPUT_SOURCE, protocol=recipe.PROTOCOL, mode="evaluate",
            runtime={**audit.RUNTIME, "mjlab": "999"}, parent_sha256=recipe.PARENT_SHA,
        ))
    elif mutation == "wrong-parent":
        put_json(bundle["root"] / "rehearsal" / "launch.json", dict(
            source=audit.INPUT_SOURCE, protocol=recipe.PROTOCOL, mode="rehearsal",
            runtime=audit.RUNTIME, parent_sha256="0" * 64,
        ))
    elif mutation == "case-order":
        prior["cases"]["control"][0], prior["cases"]["control"][1] = (
            prior["cases"]["control"][1], prior["cases"]["control"][0])
    elif mutation == "decision":
        prior["decisions"]["control"]["failures"].append(["changed"])
    elif mutation == "paired-difference":
        prior["paired_differences"][0]["rehearsal_minus_control"]["torque_p99"] += 1.
    else:
        prior.pop("final_iterations")
    rewrite_aggregate(bundle)

    with pytest.raises((ValueError, KeyError)):
        audit.validated_input(bundle["sha"], audit.RUNTIME)


@pytest.mark.parametrize("mutation", ["receipt-updates", "receipt-transitions", "receipt-checkpoint", "case-file"])
def test_audit_rejects_receipt_checkpoint_and_per_case_drift(tmp_path, monkeypatch, mutation):
    bundle = fixture_bundle(tmp_path, monkeypatch)
    root = bundle["root"]
    if mutation == "receipt-updates":
        receipt = json.loads((root / "control" / "result.json").read_text())
        receipt["updates"] -= 1
        put_json(root / "control" / "result.json", receipt)
    elif mutation == "receipt-transitions":
        receipt = json.loads((root / "rehearsal" / "result.json").read_text())
        receipt["transitions"] -= 1
        put_json(root / "rehearsal" / "result.json", receipt)
    elif mutation == "receipt-checkpoint":
        checkpoint = root / "control" / "model_499.pt"
        checkpoint.write_bytes(b"changed weights")
    else:
        row = bundle["cases"]["rehearsal"][0]
        row["yaw"] = .123
        put_json(root / "evaluate" / "rehearsal-s839-c0.json", row)

    with pytest.raises((ValueError, KeyError)):
        audit.validated_input(bundle["sha"], audit.RUNTIME)


def test_retain_case_writes_raw_terminal_trace_before_summary_validation(tmp_path):
    checkpoint_sha = "a" * 64
    speed, yaw = base.CASES[0]
    original = dict(seed=audit.SEED, speed=speed, yaw=yaw, worlds=8,
                    checkpoint_sha256=checkpoint_sha)
    observed = dict(
        seed=audit.SEED, speed=speed, yaw=yaw, worlds=8, steps=1,
        complete=False, terminal_worlds=[2], checkpoint_sha256=checkpoint_sha,
        observation_timing_trace=dict(steps=1, checkpoint_sha256=checkpoint_sha,
                                      samples=[{"partial": True}]),
    )
    output = tmp_path / "cases"
    output.mkdir()

    with pytest.raises((ValueError, KeyError)):
        audit.retain_case(output, "control", 0, observed, original, checkpoint_sha)

    raw = output / f"control-s{audit.SEED}-c0.json"
    assert raw.is_file()
    retained = json.loads(raw.read_text())
    assert retained["observed_case"]["observation_timing_trace"] == observed["observation_timing_trace"]
    assert not (output / f"control-s{audit.SEED}-c0-summary.json").exists()
