"""CPU-only contracts for the one-case packed evaluation probe.

The probe is a measured handoff, not a registered judged evaluator or skill
gate. No GPU, service, policy episode, or full evaluation is launched here.
"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_evaluation_bundle as bundle
from mjlab_microduck import stance_evaluation_worker as worker
from mjlab_microduck import stance_lean_evaluation as evaluation
from mjlab_microduck import stance_lean_lesson as lean
from mjlab_microduck import stance_packed_evaluation_probe as probe
from mjlab_microduck import stance_solved_field_check as checker
from mjlab_microduck import stance_wsl_qualification as qualification


SOURCE = "a" * 40
TRAINING_SOURCE = "d" * 40
RUNTIME_SHA = "b" * 64
CHECKPOINT_SHA = "c" * 64
LAUNCH_SHA = "e" * 64


def select_wsl(monkeypatch):
    monkeypatch.setattr(probe.host.execution, "PROFILE",
                        probe.host.execution.select(probe.host.execution.WSL))


def retained_training(seed=577):
    checkpoints = []
    for iteration in lean.CHECKPOINTS:
        identity = dict(protocol=checkpoint.PROTOCOL, source=TRAINING_SOURCE,
            runtime_sha256="1" * 64, training_launch_sha256="2" * 64,
            purpose=lean.PACKED_REPLICATION["purpose"], training_seed=seed,
            worlds=lean.WORLDS, iteration=iteration, initial_state_sha256="3" * 64,
            parent_checkpoint_sha256=checkpoint.LEAN_PARENT_SHA256,
            architecture=deepcopy(checkpoint.ARCHITECTURE))
        checkpoints.append(dict(file=f"model_{iteration}.pt", sha256="f" * 64,
                                identity=identity))
    return dict(source=TRAINING_SOURCE, report_sha256="4" * 64, checkpoints=checkpoints)


def valid_measurements():
    return dict(policy_ticks=evaluation.POLICY_TICKS,
        stop_reason=evaluation.VALID_STOP_REASON,
        checkpoint_iteration=evaluation.PROBE_ITERATION,
        evaluation_seed=evaluation.PROBE_SEED, cases_projected=evaluation.CASES,
        prelude_seconds=10.0, env_seconds=2.0, case_seconds=3.0)


def test_packed_probe_is_not_registered_as_full_evaluation_or_training_loader():
    assert evaluation.PACKED_PROBE is probe.DECLARATION
    assert evaluation.PACKED_PROBE["iterations"] == lean.CHECKPOINTS
    assert evaluation.PACKED_PROBE["trace_protocol"] == trace.PACKED_PROBE_PROTOCOL
    assert evaluation.PACKED_PROBE["label"] not in evaluation.EVALUATIONS
    assert evaluation.PACKED_PROBE["label"] not in evaluation.TRAINING_LOADERS
    with pytest.raises(ValueError, match="declared judged continuation"):
        evaluation.evaluation_of(evaluation.PACKED_PROBE["label"])
    with pytest.raises(ValueError, match="only declared judged continuations"):
        evaluation.plan(SOURCE, {}, RUNTIME_SHA, retained_training(), 1234,
                        "probe", evaluation.PACKED_PROBE)
    assert bundle.LOADERS[trace.PACKED_PROBE_PROTOCOL] is checkpoint.load_lean_replication_evaluation
    assert set(bundle.LOADERS) - {trace.PACKED_PROBE_PROTOCOL} == {
        trace.PROTOCOL, trace.EAGER_PROTOCOL, trace.LEAN_PROTOCOL, trace.LEAN_REPLICATION_PROTOCOL
    }


@pytest.mark.parametrize("bad_seed", [541, 547, 557, 576, 578, 587, 593, True])
def test_probe_plan_requires_one_declared_training_seed(monkeypatch, bad_seed):
    select_wsl(monkeypatch)
    with pytest.raises(ValueError, match="matched seed/worlds|declared learner seed|fixed first training seed"):
        probe.plan(SOURCE, {}, RUNTIME_SHA, retained_training(bad_seed),
                   qualification.PACKED_CUTOFF - 1)


def test_plan_binds_only_fixed255_seed541_packed_probe(monkeypatch):
    select_wsl(monkeypatch)
    launch = probe.plan(SOURCE, {"fixture": True}, RUNTIME_SHA,
                        retained_training(), qualification.PACKED_CUTOFF - 1)
    assert launch["protocol"] == probe.PROTOCOL
    assert launch["mode"] == "probe"
    assert launch["cases_required"] == 1
    assert launch["attempts_required"] == evaluation.WORLDS
    assert len(launch["cases"]) == 1
    case = launch["cases"][0]
    binding = case["binding"]
    assert case["name"] == "packed-probe-255-seed-541"
    assert binding["protocol"] == trace.PACKED_PROBE_PROTOCOL
    assert binding["checkpoint_iteration"] == 255
    assert binding["evaluation_seed"] == 541
    assert binding["worlds"] == evaluation.WORLDS == 128
    assert binding["capture_device"] == "cuda:0"
    assert binding["solved_field_check"] == "packed"
    assert binding["checker_sha256"] == sha256(Path(checker.__file__).read_bytes()).hexdigest()
    assert launch["solved_field_check"] == "packed"
    assert launch["checker_sha256"] == binding["checker_sha256"]
    assert launch["child_timeout_seconds"] == 600
    assert launch["service_timeout_seconds"] == 960
    assert launch["closeout_seconds"] == 600 and launch["watchdog_margin_seconds"] == 60
    assert launch["memory_max_bytes"] == 6 * 1024**3
    assert launch["cpu_quota_per_sec_usec"] == 2_000_000 and launch["nice"] == 10
    assert launch["full_evaluation_enabled"] is False
    for key in ("checkpoint_admitted", "learned_stance_accepted",
                "football_balance_accepted", "physical_motion_authorized",
                "forward_graph", "tilt_gate_relaxed"):
        assert launch[key] is False


@pytest.mark.parametrize("damage", ["wrong_seed", "wrong_iteration", "wrong_file", "wrong_source", "wrong_hash", "wrong_worlds"])
def test_plan_rejects_changed_checkpoint_source_seed_and_final_iteration(monkeypatch, damage):
    select_wsl(monkeypatch)
    retained = retained_training()
    last = retained["checkpoints"][-1]
    if damage == "wrong_seed": last["identity"]["training_seed"] = 579
    elif damage == "wrong_iteration": last["identity"]["iteration"] = 254
    elif damage == "wrong_file": last["file"] = "model_254.pt"
    elif damage == "wrong_source": last["identity"]["source"] = "9" * 40
    elif damage == "wrong_hash": last["sha256"] = "not-a-digest"
    else: last["identity"]["worlds"] = lean.WORLDS + 1
    with pytest.raises((ValueError, KeyError)):
        probe.plan(SOURCE, {}, RUNTIME_SHA, retained, qualification.PACKED_CUTOFF - 1)


def test_plan_requires_exact_common_checkpoint_coverage(monkeypatch):
    select_wsl(monkeypatch)
    retained = retained_training()
    retained["checkpoints"].pop(1)
    with pytest.raises(ValueError, match="all four common checkpoints"):
        probe.plan(SOURCE, {}, RUNTIME_SHA, retained, qualification.PACKED_CUTOFF - 1)


@pytest.mark.parametrize("deadline", [True, 10.0, qualification.PACKED_CUTOFF + 1])
def test_plan_rejects_malformed_or_late_cutoff(deadline, monkeypatch):
    select_wsl(monkeypatch)
    with pytest.raises(ValueError):
        probe.plan(SOURCE, {}, RUNTIME_SHA, retained_training(), deadline)


def test_probe_path_and_service_are_seed_isolated():
    seed = probe.PROBE_TRAINING_SEED
    assert probe.output_path(SOURCE, seed).name == (
        f"stance-wsl-packed-eval-probe-{SOURCE[:12]}-seed-{seed}")
    assert probe.service_name(SOURCE, seed) == (
        f"microduck-wsl-packed-eval-probe-{SOURCE[:12]}-seed-{seed}.service")
    for other_seed in (587, 593, 571):
        with pytest.raises(ValueError, match="fixed first training seed"):
            probe.output_path(SOURCE, other_seed)
        with pytest.raises(ValueError, match="fixed first training seed"):
            probe.service_name(SOURCE, other_seed)


@pytest.mark.parametrize("operation", ["prepare", "checked"])
@pytest.mark.parametrize("seed", [587, 593])
def test_nonfirst_replication_seeds_are_refused_before_probe_io(operation, seed):
    with pytest.raises(ValueError, match="fixed first training seed"):
        if operation == "prepare":
            probe.prepare(SOURCE, TRAINING_SOURCE, seed, qualification.PACKED_CUTOFF - 1)
        else:
            probe.checked(SOURCE, seed, LAUNCH_SHA)


@pytest.mark.parametrize("bad", [None, "linux", "other"])
def test_probe_requires_the_exact_wsl_profile(monkeypatch, bad):
    profile = dict(probe.host.execution.PROFILE)
    profile["name"] = bad
    monkeypatch.setattr(probe.host.execution, "PROFILE", profile)
    with pytest.raises(ValueError, match="exact packed probe WSL profile"):
        probe.check_window(qualification.PACKED_CUTOFF - 1)


def test_probe_window_enforces_cutoff_freshness_and_full_closeout(monkeypatch):
    select_wsl(monkeypatch)
    monkeypatch.setattr(probe.time, "time", lambda: 1_000.0)
    probe.check_window(2_561)
    probe.check_window(2_621, launching=True)
    with pytest.raises(ValueError, match="whole packed probe plus closeout"):
        probe.check_window(2_620, launching=True)
    with pytest.raises(ValueError, match="fresh bounded probe window"):
        probe.check_window(999)
    with pytest.raises(ValueError, match="authorized October 1 cutoff"):
        probe.check_window(qualification.PACKED_CUTOFF + 1)
    with pytest.raises(ValueError, match="fresh bounded probe window"):
        probe.check_window(0)


def test_probe_child_command_and_parser_round_trip():
    argv = probe.child_command(SOURCE, probe.PROBE_TRAINING_SEED, LAUNCH_SHA, 17, 123.5)
    assert argv[:3] == [str(probe.host.ROOT / ".venv/bin/python"), "-m", probe.MODULE]
    args = probe.parser().parse_args(argv[3:])
    assert args.mode == "child" and args.source == SOURCE and args.seed == probe.PROBE_TRAINING_SEED
    assert args.launch_sha256 == LAUNCH_SHA and args.lock_fd == 17
    assert args.started_monotonic == 123.5
    with pytest.raises(ValueError, match="fixed first training seed"):
        probe.child_command(SOURCE, 587, LAUNCH_SHA, 17, 123.5)
    with pytest.raises(ValueError):
        probe.child_command(SOURCE, probe.PROBE_TRAINING_SEED, LAUNCH_SHA, 17, float("nan"))


def test_measurement_projection_accounts_for_replay_and_nonrepeating_overhead():
    measurements = valid_measurements()
    # Measured components total 10 + 2 + 3 + replay 4 = 19 seconds. The four
    # unattributed seconds are conservatively repeated in every projected case.
    result = probe.derive(measurements, 4.0, 23.0)
    projected = 10.0 + evaluation.CASES * (2.0 + 3.0 + 4.0 + 4.0)
    assert result["cases"] == evaluation.CASES == 12
    assert result["attempts"] == evaluation.ATTEMPTS == 1536
    assert result["supervisor_replay_seconds"] == 4.0
    assert result["observed_probe_seconds"] == 23.0
    assert result["unattributed_overhead_seconds"] == 4.0
    assert result["overhead_projection_count"] == evaluation.CASES
    assert result["predicted_seconds"] == projected
    assert result["service_seconds"] == __import__("math").ceil(1.25 * projected)
    assert result["child_seconds"] == result["service_seconds"] - 60
    assert result["full_evaluation_enabled"] is False
    assert result["learned_stance_accepted"] is False
    assert result["football_balance_accepted"] is False
    assert result["physical_motion_authorized"] is False


@pytest.mark.parametrize("damage", ["short", "stop_reason", "iteration", "seed", "projection"])
def test_measurement_projection_requires_exact_complete_fixed_case(damage):
    measurements = valid_measurements()
    if damage == "short": measurements["policy_ticks"] -= 1
    elif damage == "stop_reason": measurements["stop_reason"] = "wall-budget-exhausted"
    elif damage == "iteration": measurements["checkpoint_iteration"] = 254
    elif damage == "seed": measurements["evaluation_seed"] = 547
    else: measurements["cases_projected"] -= 1
    with pytest.raises(ValueError): probe.derive(measurements, 4.0, 23.0)


@pytest.mark.parametrize("damage", ["replay", "observed", "too_small", "missing", "zero", "nan"])
def test_measurement_projection_rejects_bad_timing_components(damage):
    measurements = valid_measurements()
    replay, observed = 4.0, 23.0
    if damage == "replay": replay = float("inf")
    elif damage == "observed": observed = float("nan")
    elif damage == "too_small": observed = 18.5
    elif damage == "missing": del measurements["case_seconds"]
    elif damage == "zero": measurements["env_seconds"] = 0.0
    else: measurements["prelude_seconds"] = float("nan")
    with pytest.raises(ValueError): probe.derive(measurements, replay, observed)


def test_packed_binding_is_a_strict_distinct_trace_contract(monkeypatch):
    select_wsl(monkeypatch)
    binding = probe.plan(SOURCE, {}, RUNTIME_SHA, retained_training(),
                         qualification.PACKED_CUTOFF - 1)["cases"][0]["binding"]
    trace.validate_binding(binding)
    bad = deepcopy(binding)
    bad["solved_field_check"] = "legacy"
    with pytest.raises(ValueError, match="packed probe checker binding"):
        trace.validate_binding(bad)
    bad = deepcopy(binding)
    bad["checker_sha256"] = "bad"
    with pytest.raises(ValueError, match="packed probe checker binding"):
        trace.validate_binding(bad)
    bad = deepcopy(binding)
    bad["evaluation_seed"] = 547
    with pytest.raises(ValueError, match="fixed packed probe case"):
        trace.validate_binding(bad)
    bad = deepcopy(binding)
    bad["checkpoint_iteration"] = 192
    with pytest.raises(ValueError, match="predeclared checkpoint iteration"):
        trace.validate_binding(bad)
    bad = deepcopy(binding)
    bad["worlds"] = 64
    with pytest.raises(ValueError, match="fixed packed probe case"):
        trace.validate_binding(bad)


def test_legacy_trace_binding_shape_and_allowlist_remain_unchanged():
    binding = dict(protocol=trace.LEAN_REPLICATION_PROTOCOL, source=SOURCE,
        runtime_sha256=RUNTIME_SHA, checkpoint_sha256=CHECKPOINT_SHA,
        launch_sha256=LAUNCH_SHA, checkpoint_iteration=64, evaluation_seed=541,
        worlds=128, capture_device="cuda:0")
    trace.validate_binding(binding)
    assert trace.ITERATIONS[trace.LEAN_REPLICATION_PROTOCOL] == lean.CHECKPOINTS
    with pytest.raises(ValueError, match="exact trace binding fields"):
        trace.validate_binding({**binding, "solved_field_check": "packed",
                               "checker_sha256": "f" * 64})


@pytest.mark.parametrize("damage", [None, "mode", "device", "graph", "checker_hash"])
def test_worker_checks_actual_packed_eager_cuda_runtime_and_checker(damage):
    binding = dict(protocol=trace.PACKED_PROBE_PROTOCOL,
                   checker_sha256=sha256(Path(checker.__file__).read_bytes()).hexdigest())
    env = SimpleNamespace(solved_field_check="packed", forward_graph=None,
                          wp_device=SimpleNamespace(is_cuda=True))
    if damage == "mode": env.solved_field_check = "legacy"
    elif damage == "device": env.wp_device.is_cuda = False
    elif damage == "graph": env.forward_graph = object()
    elif damage == "checker_hash": binding["checker_sha256"] = "0" * 64
    if damage is None:
        assert worker.check_packed_runtime(env, binding) is None
    else:
        with pytest.raises(ValueError): worker.check_packed_runtime(env, binding)


def test_worker_leaves_all_legacy_trace_runtime_checks_unchanged():
    binding = dict(protocol=trace.LEAN_REPLICATION_PROTOCOL)
    assert worker.check_packed_runtime(SimpleNamespace(), binding) is None


def test_training_archive_hash_refusal_precedes_tensor_loader(tmp_path, monkeypatch):
    seed = 577
    root = tmp_path / "packed-training"
    root.mkdir()
    launch_bytes = (json.dumps(dict(protocol=lean.PACKED_REPLICATION["protocol"],
        source=TRAINING_SOURCE, seed=seed, solved_field_check="packed"), sort_keys=True) + "\n").encode()
    model_bytes = b"model 255"
    (root / "launch.json").write_bytes(launch_bytes)
    (root / "model_255.pt").write_bytes(model_bytes)
    report = dict(decision=lean.PACKED_REPLICATION["decision"],
        launch_sha256=LAUNCH_SHA, child={"returncode": 0}, files={
            "launch.json": sha256(launch_bytes).hexdigest(),
            "model_255.pt": "0" * 64})
    report_bytes = (json.dumps(report, sort_keys=True) + "\n").encode()
    (root / "report.json").write_bytes(report_bytes)
    monkeypatch.setattr(evaluation.lean, "output_path", lambda *_args: root)
    monkeypatch.setattr(evaluation.files, "file_bytes", lambda path, **_kwargs: (
        report_bytes if Path(path).name == "report.json" else Path(path).read_bytes()))
    monkeypatch.setattr(evaluation.host, "digest", lambda path: sha256(Path(path).read_bytes()).hexdigest())
    monkeypatch.setattr(evaluation.lean, "verify_completed",
                        lambda *_args, **_kwargs: pytest.fail("hash mismatch must stop before receipt replay"))
    monkeypatch.setattr(checkpoint, "load_lean_replication_evaluation",
                        lambda *_args: pytest.fail("archive must authenticate before tensor load"))
    with pytest.raises(ValueError, match="immutable training archive hash: model_255.pt"):
        evaluation.training_inputs(probe.DECLARATION, seed, TRAINING_SOURCE)


def test_training_inputs_authenticates_complete_replication_before_selected_loads(tmp_path, monkeypatch):
    seed = 577
    root = tmp_path / "packed-training"
    root.mkdir()
    launch_bytes = (json.dumps(dict(protocol=lean.PACKED_REPLICATION["protocol"],
        source=TRAINING_SOURCE, seed=seed, solved_field_check="packed"), sort_keys=True) + "\n").encode()
    (root / "launch.json").write_bytes(launch_bytes)
    saved_common = []
    files = {"launch.json": sha256(launch_bytes).hexdigest()}
    for iteration in lean.CHECKPOINTS:
        raw = f"model-{iteration}".encode()
        name = f"model_{iteration}.pt"
        (root / name).write_bytes(raw)
        files[name] = sha256(raw).hexdigest()
        identity = dict(source=TRAINING_SOURCE, iteration=iteration, training_seed=seed)
        saved_common.append(dict(file=name, sha256=files[name], identity=identity))
    report = dict(protocol=lean.PACKED_REPLICATION["protocol"], seed=seed,
        purpose=lean.PACKED_REPLICATION["purpose"],
        decision=lean.PACKED_REPLICATION["decision"],
        launch_sha256=sha256(launch_bytes).hexdigest(),
                  child={"returncode": 0}, files=files)
    report_bytes = (json.dumps(report, sort_keys=True) + "\n").encode()
    (root / "report.json").write_bytes(report_bytes)
    monkeypatch.setattr(evaluation.lean, "output_path", lambda *_args: root)
    monkeypatch.setattr(evaluation.files, "file_bytes", lambda path, **_kwargs: (
        report_bytes if Path(path).name == "report.json" else Path(path).read_bytes()))
    monkeypatch.setattr(evaluation.host, "digest", lambda path: sha256(Path(path).read_bytes()).hexdigest())
    all_checkpoints = [dict(file=f"model_{iteration}.pt", sha256="f" * 64,
                            identity={"iteration": iteration}) for iteration in range(-1, lean.UPDATES)]
    by_iteration = {item["identity"]["iteration"]: item for item in all_checkpoints}
    by_iteration.update({item["identity"]["iteration"]: item for item in saved_common})
    monkeypatch.setattr(evaluation.lean, "verify_completed", lambda *_args, **_kwargs: {
        "checkpoints": list(by_iteration.values())})
    loads = []
    monkeypatch.setattr(checkpoint, "load_lean_replication_evaluation",
                        lambda raw, digest, identity: loads.append((raw, digest, identity)) or "loaded")
    retained = evaluation.training_inputs(probe.DECLARATION, seed, TRAINING_SOURCE)
    assert retained["source"] == TRAINING_SOURCE
    assert retained["report_sha256"] == sha256(report_bytes).hexdigest()
    assert [item["identity"]["iteration"] for item in retained["checkpoints"]] == list(lean.CHECKPOINTS)
    assert len(loads) == len(lean.CHECKPOINTS)
    assert all(digest == sha256(raw).hexdigest() for raw, digest, _identity in loads)


def _retained_probe_fixture(tmp_path, monkeypatch):
    select_wsl(monkeypatch)
    root = tmp_path / "retained-probe"
    root.mkdir()
    retained = retained_training()
    runtime_raw = b'{"runtime":"fixture"}'
    launch = probe.plan(SOURCE, {"host": "fixture"}, sha256(runtime_raw).hexdigest(),
                        retained, qualification.PACKED_CUTOFF - 1)
    launch_raw = (json.dumps(launch, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (root / "launch.json").write_bytes(launch_raw)
    (root / "runtime.json").write_bytes(runtime_raw)
    measurements = valid_measurements()
    measurement_raw = (json.dumps(measurements, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (root / "measurements.json").write_bytes(measurement_raw)
    (root / "child.log").write_bytes(b"mocked complete child")
    names = ("launch.json", "runtime.json", "measurements.json", "child.log")
    report = dict(protocol=probe.PROTOCOL, launch_sha256=sha256(launch_raw).hexdigest(),
        decision="probe-measured-full-evaluation-disabled",
        child={"pid": 1234, "returncode": 0, "elapsed_s": 10.0,
               "samples": [{"sample": 1}]},
        optimizer_steps=0, full_evaluation_enabled=False, checkpoint_admitted=False,
        learned_stance_accepted=False, football_balance_accepted=False,
        physical_motion_authorized=False,
        measurements_sha256=sha256(measurement_raw).hexdigest(),
        probe={"measurements": measurements},
        timing=probe.derive(measurements, 4.0, 23.0),
        files={name: sha256((root / name).read_bytes()).hexdigest() for name in names})
    report_raw = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (root / "report.json").write_bytes(report_raw)

    def replay(_root, replay_launch, launch_sha, measurements_sha, declaration):
        assert replay_launch == launch
        assert launch_sha == report["launch_sha256"]
        assert declaration is probe.DECLARATION
        raw = (_root / "measurements.json").read_bytes()
        assert sha256(raw).hexdigest() == measurements_sha
        return {"measurements": json.loads(raw)}

    monkeypatch.setattr(evaluation, "verify_probe", replay)
    return root, launch, report


def _rewrite_report(root, report):
    raw = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (root / "report.json").write_bytes(raw)
    return sha256(raw).hexdigest()


@pytest.mark.parametrize("damage", [
    "missing_pid", "missing_returncode", "missing_elapsed", "missing_samples", "extra_key",
    "pid_zero", "pid_negative", "pid_bool", "pid_string", "returncode_bool",
    "returncode_nonzero", "elapsed_int", "elapsed_bool", "elapsed_zero", "elapsed_negative",
    "elapsed_nan", "elapsed_inf", "elapsed_over_cap", "samples_nonlist",
])
def test_verify_retained_requires_exact_bounded_child_receipt(tmp_path, monkeypatch, damage):
    root, _launch, report = _retained_probe_fixture(tmp_path, monkeypatch)
    changed = deepcopy(report)
    child = changed["child"]
    if damage == "missing_pid": child.pop("pid")
    elif damage == "missing_returncode": child.pop("returncode")
    elif damage == "missing_elapsed": child.pop("elapsed_s")
    elif damage == "missing_samples": child.pop("samples")
    elif damage == "extra_key": child["extra"] = "not-declared"
    elif damage == "pid_zero": child["pid"] = 0
    elif damage == "pid_negative": child["pid"] = -1
    elif damage == "pid_bool": child["pid"] = True
    elif damage == "pid_string": child["pid"] = "1234"
    elif damage == "returncode_bool": child["returncode"] = False
    elif damage == "returncode_nonzero": child["returncode"] = 1
    elif damage == "elapsed_int": child["elapsed_s"] = 120
    elif damage == "elapsed_bool": child["elapsed_s"] = True
    elif damage == "elapsed_zero": child["elapsed_s"] = 0.0
    elif damage == "elapsed_negative": child["elapsed_s"] = -0.1
    elif damage == "elapsed_nan": child["elapsed_s"] = float("nan")
    elif damage == "elapsed_inf": child["elapsed_s"] = float("inf")
    elif damage == "elapsed_over_cap": child["elapsed_s"] = probe.CHILD_SECONDS + 0.001
    else: child["samples"] = "not-a-list"
    report_sha = _rewrite_report(root, changed)
    assert report_sha == sha256((root / "report.json").read_bytes()).hexdigest()
    with pytest.raises((ValueError, TypeError, KeyError)):
        probe.verify_retained(root, changed["launch_sha256"], report_sha)


def test_verify_retained_accepts_exact_positive_child_receipt_at_cap(tmp_path, monkeypatch):
    root, _launch, report = _retained_probe_fixture(tmp_path, monkeypatch)
    report["child"]["elapsed_s"] = float(probe.CHILD_SECONDS)
    report["timing"] = probe.derive(report["probe"]["measurements"], 4.0,
                                   float(probe.CHILD_SECONDS) + 10.0)
    report["decision"] = "timing-rejected-no-full-evaluation"
    report_sha = _rewrite_report(root, report)
    result = probe.verify_retained(root, report["launch_sha256"], report_sha)
    assert result == report["timing"]


@pytest.mark.parametrize("damage", ["child_outlasts_service", "observed_exceeds_service_cap"])
def test_verify_retained_refuses_contradictory_service_timing(tmp_path, monkeypatch, damage):
    root, _launch, report = _retained_probe_fixture(tmp_path, monkeypatch)
    if damage == "child_outlasts_service":
        report["child"]["elapsed_s"] = report["timing"]["observed_probe_seconds"] + 0.001
    else:
        report["timing"] = probe.derive(report["probe"]["measurements"], 4.0,
                                       float(probe.SERVICE_SECONDS) + 0.001)
        report["decision"] = "timing-rejected-no-full-evaluation"
    report_sha = _rewrite_report(root, report)
    with pytest.raises(ValueError, match="consistent retained child and service timing bounds"):
        probe.verify_retained(root, report["launch_sha256"], report_sha)


@pytest.mark.parametrize("damage", ["report_hash", "launch_hash"])
def test_verify_retained_refuses_wrong_independent_report_or_launch_hash(tmp_path, monkeypatch, damage):
    root, launch, report = _retained_probe_fixture(tmp_path, monkeypatch)
    report_sha = sha256((root / "report.json").read_bytes()).hexdigest()
    launch_sha = report["launch_sha256"]
    if damage == "report_hash": report_sha = "0" * 64
    else: launch_sha = "1" * 64
    with pytest.raises(ValueError, match="independent retained packed probe hashes"):
        probe.verify_retained(root, launch_sha, report_sha)


@pytest.mark.parametrize("damage", ["decision", "returncode", "optimizer", "full_eval",
                                     "checkpoint", "learned", "football", "motion"])
def test_verify_retained_refuses_wrong_decision_child_or_capability_flags(
        tmp_path, monkeypatch, damage):
    root, _launch, report = _retained_probe_fixture(tmp_path, monkeypatch)
    changed = deepcopy(report)
    if damage == "decision": changed["decision"] = "probe-incomplete"
    elif damage == "returncode": changed["child"]["returncode"] = 1
    elif damage == "optimizer": changed["optimizer_steps"] = 1
    elif damage == "full_eval": changed["full_evaluation_enabled"] = True
    elif damage == "checkpoint": changed["checkpoint_admitted"] = True
    elif damage == "learned": changed["learned_stance_accepted"] = True
    elif damage == "football": changed["football_balance_accepted"] = True
    else: changed["physical_motion_authorized"] = True
    report_sha = _rewrite_report(root, changed)
    with pytest.raises(ValueError):
        probe.verify_retained(root, changed["launch_sha256"], report_sha)


@pytest.mark.parametrize("damage", ["topfile_hash", "extra_inventory", "missing_inventory"])
def test_verify_retained_refuses_top_level_file_hash_or_inventory_drift(
        tmp_path, monkeypatch, damage):
    root, _launch, report = _retained_probe_fixture(tmp_path, monkeypatch)
    changed = deepcopy(report)
    if damage == "topfile_hash": changed["files"]["child.log"] = "0" * 64
    elif damage == "extra_inventory":
        (root / "unreported.txt").write_text("extra")
    else: changed["files"].pop("child.log")
    report_sha = _rewrite_report(root, changed)
    with pytest.raises(ValueError):
        probe.verify_retained(root, changed["launch_sha256"], report_sha)


@pytest.mark.parametrize("damage", ["probe_report", "incomplete_measurement", "timing"])
def test_verify_retained_replays_probe_and_rederives_timing(
        tmp_path, monkeypatch, damage):
    root, _launch, report = _retained_probe_fixture(tmp_path, monkeypatch)
    changed = deepcopy(report)
    if damage == "probe_report":
        changed["probe"]["measurements"]["env_seconds"] += 1.0
    elif damage == "incomplete_measurement":
        measurements = valid_measurements()
        measurements["policy_ticks"] -= 1
        raw = (json.dumps(measurements, sort_keys=True, separators=(",", ":")) + "\n").encode()
        (root / "measurements.json").write_bytes(raw)
        changed["measurements_sha256"] = sha256(raw).hexdigest()
        changed["files"]["measurements.json"] = sha256(raw).hexdigest()
        changed["probe"] = {"measurements": measurements}
    else:
        changed["timing"]["predicted_seconds"] += 1.0
    report_sha = _rewrite_report(root, changed)
    with pytest.raises(ValueError):
        probe.verify_retained(root, changed["launch_sha256"], report_sha)


def test_verify_retained_accepts_faithful_cpu_replay_and_recomputed_timing_only(
        tmp_path, monkeypatch):
    root, _launch, report = _retained_probe_fixture(tmp_path, monkeypatch)
    timing = probe.verify_retained(root, report["launch_sha256"],
                                   sha256((root / "report.json").read_bytes()).hexdigest())
    assert timing == report["timing"]
    assert timing["full_evaluation_enabled"] is False
    assert timing["learned_stance_accepted"] is False
    assert timing["football_balance_accepted"] is False
    assert timing["physical_motion_authorized"] is False


def test_verify_retained_rejects_a_misreported_oversize_decision(tmp_path, monkeypatch):
    root, _launch, report = _retained_probe_fixture(tmp_path, monkeypatch)
    measurements = valid_measurements()
    measurements["case_seconds"] = 600.0
    raw = (json.dumps(measurements, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (root / "measurements.json").write_bytes(raw)
    report["measurements_sha256"] = sha256(raw).hexdigest()
    report["files"]["measurements.json"] = sha256(raw).hexdigest()
    report["probe"] = {"measurements": measurements}
    report["timing"] = probe.derive(measurements, 4.0, 620.0)
    assert report["timing"]["fits_declared_wsl_window"] is False
    report["decision"] = "probe-measured-full-evaluation-disabled"
    report_sha = _rewrite_report(root, report)
    with pytest.raises(ValueError, match="retained decision agrees"):
        probe.verify_retained(root, report["launch_sha256"], report_sha)


def test_verify_retained_accepts_oversize_measurement_only_as_timing_rejection(tmp_path, monkeypatch):
    root, _launch, report = _retained_probe_fixture(tmp_path, monkeypatch)
    measurements = valid_measurements()
    measurements["case_seconds"] = 600.0
    raw = (json.dumps(measurements, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (root / "measurements.json").write_bytes(raw)
    report["measurements_sha256"] = sha256(raw).hexdigest()
    report["files"]["measurements.json"] = sha256(raw).hexdigest()
    report["probe"] = {"measurements": measurements}
    report["timing"] = probe.derive(measurements, 4.0, 620.0)
    report["decision"] = "timing-rejected-no-full-evaluation"
    report_sha = _rewrite_report(root, report)
    verified = probe.verify_retained(root, report["launch_sha256"], report_sha)
    assert verified["fits_declared_wsl_window"] is False
    assert verified["full_evaluation_enabled"] is False


@pytest.mark.parametrize("started", [None, float("nan"), 12.0, 11.0, 12])
def test_child_rejects_missing_nonfinite_future_or_nonfloat_monotonic_start_before_lease(
        monkeypatch, started):
    monkeypatch.setattr(probe.time, "monotonic", lambda: 11.0)
    monkeypatch.setattr(probe.smoke, "inherited_lease",
                        lambda *_: pytest.fail("invalid start must fail before lease validation"))
    monkeypatch.setattr(probe, "checked", lambda *_: pytest.fail("invalid start must fail before input reads"))
    with pytest.raises(ValueError, match="same-clock pre-exec monotonic start"):
        probe.child(SOURCE, probe.PROBE_TRAINING_SEED, LAUNCH_SHA, 17, started)


def test_child_uses_same_clock_float_before_lease_and_input_checks(monkeypatch):
    calls = []
    monkeypatch.setattr(probe.time, "monotonic", lambda: 11.0)
    monkeypatch.setattr(probe.smoke, "inherited_lease", lambda fd: calls.append(("lease", fd)))
    def checked(*_):
        calls.append(("checked",))
        raise RuntimeError("passed preflight")
    monkeypatch.setattr(probe, "checked", checked)
    with pytest.raises(RuntimeError, match="passed preflight"):
        probe.child(SOURCE, probe.PROBE_TRAINING_SEED, LAUNCH_SHA, 17, 10.0)
    assert calls == [("lease", 17), ("checked",)]


def test_packed_probe_supervisor_uses_fixed_wsl_600_second_child_wrapper(monkeypatch):
    wrapper = probe.files.supervised_packed_evaluation_probe
    calls = []
    monkeypatch.setattr(probe.files.execution, "PROFILE",
                        probe.files.execution.select(probe.files.execution.WSL))
    monkeypatch.setattr(probe.files, "_timed_process",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or {"mocked": True})
    result = wrapper(["mock-child"], Path("mock.log"), cwd=Path("/tmp"),
                     env={}, lock_fd=7, guard=lambda: None)
    assert result == {"mocked": True}
    assert calls[0][1]["timeout"] == 600
    assert calls[0][1]["lock_fd"] == 7
    monkeypatch.setattr(probe.files.execution, "PROFILE",
                        probe.files.execution.select(probe.files.execution.DEFAULT))
    with pytest.raises(ValueError, match="explicit WSL"):
        wrapper(["mock-child"], Path("mock.log"), cwd=Path("/tmp"), env={}, lock_fd=7)
