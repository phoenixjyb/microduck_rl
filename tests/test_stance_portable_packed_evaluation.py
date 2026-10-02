"""CPU-only contracts for the bounded portable packed matrix declaration."""
from copy import deepcopy
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_cpu_replay_profile as profile
from mjlab_microduck import stance_historical_replication_auth as auth
from mjlab_microduck import stance_lean_evaluation as evaluation
from mjlab_microduck import stance_lean_replication_campaign as campaign
from mjlab_microduck import stance_portable_packed_evaluation as full


SOURCE = "a" * 40
RUNTIME_SHA = "b" * 64
CHECKER_SHA = "c" * 64
TRAINING_SOURCE = auth.TRAINING_SOURCE


def select_wsl(monkeypatch):
    monkeypatch.setattr(full.host.execution, "PROFILE",
                        full.host.execution.select(full.host.execution.WSL))


def retained_for(seed):
    checkpoints = []
    for iteration in full.ITERATIONS:
        checkpoints.append({
            "file": f"model_{iteration}.pt",
            "sha256": f"{iteration % 10}" * 64,
            "identity": {
                "training_seed": seed,
                "iteration": iteration,
                "training_launch_sha256": "d" * 64,
                "runtime_sha256": "e" * 64,
            },
        })
    return {"source": TRAINING_SOURCE, "report_sha256": "f" * 64,
            "checkpoints": checkpoints}


def authentications():
    result = {}
    for seed in auth.SEEDS:
        retained = retained_for(seed)
        result[str(seed)] = {
            "training_seed": seed,
            "retained_training": retained,
            "archive_files": {"launch.json": "d" * 64, "runtime.json": "e" * 64},
        }
    return result


def make_plan(monkeypatch):
    select_wsl(monkeypatch)
    monkeypatch.setattr(full.auth, "validate", lambda *_args: None)
    monkeypatch.setattr(full, "pinned_caps", lambda: caps())
    return full.plan(SOURCE,
        {"source": SOURCE, "execution_profile": full.host.execution.select(full.host.execution.WSL)},
        RUNTIME_SHA, authentications(), full.NOT_BEFORE + 20_000, CHECKER_SHA, caps())


def caps():
    # Exact completed R4 measurements, transcribed from its retained report.
    measurements = dict(policy_ticks=evaluation.POLICY_TICKS,
        stop_reason=evaluation.VALID_STOP_REASON,
        checkpoint_iteration=evaluation.PROBE_ITERATION,
        evaluation_seed=evaluation.PROBE_SEED, cases_projected=evaluation.CASES,
        prelude_seconds=21.741637519095093,
        env_seconds=2.2584270699881017,
        case_seconds=193.14527830597945)
    return full.derive_caps(measurements, 9.91598303313367, 261.755154005019)


def test_caps_rederive_measured_matrix_budget_and_authentication_reserve():
    result = caps()
    assert result["cases"] == full.CASES == 36
    assert result["attempts"] == full.ATTEMPTS == 4608
    assert result["service_seconds"] == full.SERVICE_SECONDS == 11728
    assert result["child_seconds"] == full.CHILD_SECONDS == 10981
    assert result["historical_authentication_calls"] == 12
    assert result["historical_authentication_reserve_seconds"] == 12 * auth.SECONDS
    assert result["repeating_unit_seconds"] == pytest.approx(240.0135164859239)
    assert result["unattributed_overhead_seconds"] == pytest.approx(34.69382807682268)
    assert result["predicted_seconds"] == pytest.approx(9382.228231012356)
    assert result["parent_replay_reserve_seconds"] == 507
    assert result["parent_authentication_reserve_seconds"] == 180
    assert result["closeout_seconds"] == 600 and result["watchdog_margin_seconds"] == 60
    assert result["runtime_max"] == "3h 15min 28s"


def test_plan_is_ordered_36_cases_from_three_archives_and_four_checkpoints(monkeypatch):
    launch = make_plan(monkeypatch)
    assert launch["training_source"] == TRAINING_SOURCE
    assert launch["training_seeds"] == list(auth.SEEDS)
    assert launch["checkpoint_iterations"] == list(full.ITERATIONS)
    assert launch["evaluation_seeds"] == list(trace.SEEDS)
    assert launch["cases_required"] == 36 and launch["attempts_required"] == 4608
    expected = [(training_seed, iteration, evaluation_seed)
                for training_seed in auth.SEEDS
                for iteration in full.ITERATIONS
                for evaluation_seed in trace.SEEDS]
    actual = [(case["binding"]["training_seed"],
               case["binding"]["checkpoint_iteration"],
               case["binding"]["evaluation_seed"]) for case in launch["cases"]]
    assert actual == expected
    assert len({case["name"] for case in launch["cases"]}) == 36
    assert all(case["binding"]["protocol"] == trace.PORTABLE_FULL_PROTOCOL
               and case["binding"]["worlds"] == 128
               and case["binding"]["capture_device"] == "cuda:0"
               and case["binding"]["solved_field_check"] == "packed"
               and case["binding"]["cpu_math_profile"] == profile.expected_receipt()
               for case in launch["cases"])
    assert launch["actor_device"] == "cpu" and launch["physics_device"] == "cuda:0"
    assert launch["optimizer_steps"] == 0 and launch["forward_graph"] is False
    for key in full.FALSE_FLAGS:
        assert launch[key] is False


def test_run_cases_allocates_ordered_fresh_packed_envs_and_binds_exact_inputs(
        tmp_path, monkeypatch):
    launch = make_plan(monkeypatch)
    root = tmp_path / "run-cases"
    root.mkdir()
    source_cases = launch["cases"]
    allocated = []
    class FakeRuntime:
        def __init__(self, worlds, *, device, solved_field_check):
            self.worlds = worlds
            self.device = device
            self.solved_field_check = solved_field_check
            self.forward_graph = None
            self.wp_device = SimpleNamespace(is_cuda=True)
            allocated.append(self)

        def reset(self, *_args, **_kwargs):
            pytest.fail("one case must not reset or reuse a packed runtime")

    fake_module = SimpleNamespace(WarpStanceRuntime=FakeRuntime)
    monkeypatch.setitem(sys.modules, "mjlab_microduck.stance_warp_runtime", fake_module)
    expected_cp = {case["checkpoint"]["file"]: ("checkpoint:" + case["checkpoint"]["file"]).encode()
                   for case in source_cases}
    monkeypatch.setattr(full.files, "file_bytes", lambda path: expected_cp[path.name])
    launch_encoder = full.evaluation.bundle.launch_bytes
    encoded_bindings = []
    def encode(binding, identity):
        encoded_bindings.append((binding, identity))
        return launch_encoder(binding, identity)
    monkeypatch.setattr(full.evaluation.bundle, "launch_bytes", encode)

    worker_calls = []
    def evaluate(directory, env, **kwargs):
        worker_calls.append((directory, env, kwargs))
        assert env.forward_graph is None and env.wp_device.is_cuda
        return dict(manifest_sha256="a" * 64,
            collection={"policy_ticks": evaluation.POLICY_TICKS},
            score={"complete_attempts": evaluation.WORLDS, "numerical_passes": 122})
    monkeypatch.setattr(full.evaluation.worker, "evaluate_owned_case", evaluate)

    seed_calls = []
    monkeypatch.setattr(full.random, "seed", lambda seed: seed_calls.append(("python", seed)))
    monkeypatch.setattr(full.np.random, "seed", lambda seed: seed_calls.append(("numpy", seed)))
    monkeypatch.setattr(full.torch, "manual_seed", lambda seed: seed_calls.append(("torch", seed)))
    runtime = b"exact compiled runtime fixture"
    deadline = full.time.monotonic() + 120
    started = full.time.monotonic()
    scores, receipts = full.run_cases(launch, root, runtime, deadline, started_monotonic=started)

    assert len(allocated) == len(worker_calls) == len(receipts) == len(scores) == full.CASES
    assert len({id(env) for env in allocated}) == full.CASES
    assert all((env.worlds, env.device, env.solved_field_check) == (128, "cuda:0", "packed")
               and env.forward_graph is None for env in allocated)
    expected_seeds = [case["binding"]["evaluation_seed"] for case in source_cases]
    assert seed_calls == [(kind, seed) for seed in expected_seeds
                          for kind in ("python", "numpy", "torch")]
    assert [binding["evaluation_seed"] for binding, _ in encoded_bindings] == expected_seeds
    assert [Path(directory).name for directory, _, _ in worker_calls] == [c["name"] for c in source_cases]
    for case, (_, env, kwargs), (binding, identity) in zip(source_cases, worker_calls, encoded_bindings):
        assert env in allocated
        assert kwargs["binding"] == case["binding"]
        assert kwargs["checkpoint_identity"] == case["checkpoint"]["identity"] == identity
        assert kwargs["checkpoint_raw"] == expected_cp[case["checkpoint"]["file"]]
        assert kwargs["runtime_raw"] == runtime
        assert kwargs["launch_raw"] == launch_encoder(binding, identity)
        assert kwargs["deadline_monotonic"] == deadline
        assert kwargs["policy_tick_limit"] == evaluation.POLICY_TICKS
    assert list(scores) == [case["name"] for case in source_cases]
    assert [receipt["case"] for receipt in receipts] == [case["name"] for case in source_cases]


def test_original_seed577_packed_probe_namespace_remains_separate(monkeypatch):
    from mjlab_microduck import stance_packed_evaluation_probe as probe

    assert probe.PROBE_TRAINING_SEED == 577
    assert probe.evaluation.PROBE_ITERATION == 255
    assert probe.evaluation.PROBE_SEED == 541
    assert probe.PROTOCOL != full.PROTOCOL
    assert trace.PORTABLE_PROBE_PROTOCOL != trace.PORTABLE_FULL_PROTOCOL
    assert probe.output_path(SOURCE, 577) != full.output_path(SOURCE)
    assert probe.service_name(SOURCE, 577) != full.service_name(SOURCE)


def test_fixed_window_latest_start_and_cutoff():
    start = full.NOT_BEFORE
    needed = full.SERVICE_SECONDS + full.CLOSEOUT_SECONDS + full.MARGIN_SECONDS
    assert needed == 12388
    full.check_window(start + needed + 1, launching=True, now=start)
    with pytest.raises(ValueError, match="whole full matrix plus closeout"):
        full.check_window(start + needed, launching=True, now=start)
    with pytest.raises(ValueError, match="active separately declared"):
        full.check_window(start + needed + 1, now=start - 1)
    with pytest.raises(ValueError, match="active separately declared"):
        full.check_window(full.CUTOFF, now=full.CUTOFF)
    with pytest.raises(ValueError, match="explicit full evaluation deadline"):
        full.check_window(full.CUTOFF + 1, now=start)


@pytest.mark.parametrize("mode", ["prepare", "supervise", "child"])
def test_parser_accepts_only_declared_modes_and_round_trips_command_fields(mode):
    argv = [mode, "--source", SOURCE]
    if mode == "prepare":
        argv += ["--deadline-unix", str(full.CUTOFF - 1)]
    elif mode == "supervise":
        argv += ["--launch-sha256", "d" * 64]
    else:
        argv += ["--launch-sha256", "d" * 64, "--lock-fd", "17",
                 "--started-monotonic", "123.5"]
    args = full.parser().parse_args(argv)
    assert args.mode == mode and args.source == SOURCE
    with pytest.raises(SystemExit):
        full.parser().parse_args(["unexpected", "--source", SOURCE])


@pytest.mark.parametrize("argv", [
    ["prepare", "--source", SOURCE],
    ["prepare", "--source", SOURCE, "--deadline-unix", "100", "--launch-sha256", "d" * 64],
    ["supervise", "--source", SOURCE],
    ["supervise", "--source", SOURCE, "--launch-sha256", "d" * 64, "--deadline-unix", "100"],
    ["child", "--source", SOURCE, "--launch-sha256", "d" * 64, "--lock-fd", "17"],
    ["child", "--source", SOURCE, "--launch-sha256", "d" * 64, "--lock-fd", "17",
     "--started-monotonic", "123.5", "--deadline-unix", "100"],
])
def test_main_rejects_missing_or_cross_mode_arguments(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", [full.MODULE, *argv])
    for name in ("prepare", "supervise", "child"):
        monkeypatch.setattr(full, name, lambda *_args, **_kwargs: pytest.fail("invalid CLI reached handler"))
    with pytest.raises(SystemExit) as error:
        full.main()
    assert error.value.code == 2


@pytest.mark.parametrize("started", [True, float("nan"), 100.0, 0.5])
def test_child_watchdog_rejects_invalid_or_expired_start_before_lease(started, monkeypatch):
    monkeypatch.setattr(full.time, "monotonic", lambda: 100.0 if started == 100.0 else full.CHILD_SECONDS + 1.0)
    monkeypatch.setattr(full.evaluation.smoke, "inherited_lease",
                        lambda _fd: pytest.fail("invalid child start must fail before lease use"))
    with pytest.raises(ValueError, match="same-clock bounded child start"):
        full.child(SOURCE, "d" * 64, 17, started)


def test_child_command_rejects_invalid_lease_and_clock_values():
    with pytest.raises(ValueError, match="owned full matrix lease and start"):
        full.child_command(SOURCE, "d" * 64, True, 123.5)
    with pytest.raises(ValueError, match="owned full matrix lease and start"):
        full.child_command(SOURCE, "d" * 64, 17, float("nan"))


def test_child_command_and_supervisor_keep_child_timeout_watchdog_and_guard(
        tmp_path, monkeypatch):
    launch = make_plan(monkeypatch)
    launch_sha = "d" * 64
    command = full.child_command(SOURCE, launch_sha, 17, 123.5)
    args = full.parser().parse_args(command[3:])
    assert command[:3] == [str(full.host.ROOT / ".venv/bin/python"), "-m", full.MODULE]
    assert args.mode == "child" and args.source == SOURCE
    assert args.launch_sha256 == launch_sha and args.lock_fd == 17
    assert args.started_monotonic == 123.5

    root = tmp_path / "supervised"
    root.mkdir()
    (root / "launch.json").write_text("{}\n")
    (root / "runtime.json").write_bytes(b"runtime")
    monkeypatch.setattr(full, "check_service", lambda _source: None)
    monkeypatch.setattr(full, "checked", lambda _source, _sha: launch)
    monkeypatch.setattr(full, "check_window", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(full, "output_path", lambda _source: root)
    monkeypatch.setattr(full.host, "identity", lambda _source: launch["inputs"])
    monkeypatch.setattr(full.host, "check_log", lambda _path: None)
    monkeypatch.setattr(full.host, "wait_idle", lambda: {"idle": True})
    monkeypatch.setattr(full.host, "digest", lambda _path: "e" * 64)
    monkeypatch.setattr(full.files, "child_environment", lambda: {"CUDA_VISIBLE_DEVICES": "0"})
    monkeypatch.setattr(full, "verify", lambda *_args: {"decision": "lean-replication-passed"})
    checks = []
    def timed_process(argv, log_path, **kwargs):
        checks.append((argv, kwargs))
        Path(log_path).write_text("mock child completed\n")
        kwargs["guard"]()
        return {"pid": 4321, "returncode": 0, "elapsed_s": 12.0, "samples": [{"mock": True}]}
    monkeypatch.setattr(full.files, "_timed_process", timed_process)

    @contextmanager
    def lease():
        yield 17
    monkeypatch.setattr(full.files, "gpu_lease", lease)
    monotonic_values = iter((100.0, 200.0, 300.0))
    monkeypatch.setattr(full.time, "monotonic", lambda: next(monotonic_values))
    full.supervise(SOURCE, launch_sha)

    assert len(checks) == 1
    child_argv, options = checks[0]
    parsed = full.parser().parse_args(child_argv[3:])
    assert parsed.mode == "child" and parsed.source == SOURCE
    assert parsed.launch_sha256 == launch_sha and parsed.lock_fd == 17
    assert parsed.started_monotonic == 200.0
    assert options["timeout"] == full.CHILD_SECONDS
    assert options["cwd"] == full.host.ROOT
    assert options["lock_fd"] == 17
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert callable(options["guard"])
    report = json.loads((root / "report.json").read_text())
    assert report["child"]["returncode"] == 0
    assert report["decision"] == "lean-replication-passed"
    assert report["observed_service_seconds"] == 200.0


def test_summary_requires_all_unique_scores_and_ordered_case_matrix(monkeypatch):
    launch = make_plan(monkeypatch)
    score_map = {case["name"]: {} for case in launch["cases"]}
    with pytest.raises(ValueError, match="all 36 scores required"):
        full.summarize(launch, {key: value for key, value in score_map.items()
                                if key != launch["cases"][-1]["name"]})

    one_seed = [case for case in launch["cases"] if case["binding"]["training_seed"] == auth.SEEDS[0]]
    seed_scores = {case["name"]: {} for case in one_seed}
    duplicate = deepcopy(one_seed)
    duplicate[-1]["name"] = duplicate[-2]["name"]
    with pytest.raises(ValueError, match="unique case names"):
        evaluation.summarize_cases(duplicate, seed_scores, full.DECLARATION)
    reordered = deepcopy(one_seed)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="exact ordered checkpoint/seed matrix"):
        evaluation.summarize_cases(reordered, seed_scores, full.DECLARATION)


@pytest.mark.parametrize("damage", ["incomplete", "duplicate", "reordered"])
def test_verify_rejects_incomplete_duplicate_or_reordered_full_case_inventory(
        tmp_path, monkeypatch, damage):
    launch = make_plan(monkeypatch)
    names = [case["name"] for case in launch["cases"]]
    if damage == "incomplete":
        names.pop()
    elif damage == "duplicate":
        names[-1] = names[-2]
    else:
        names[0], names[1] = names[1], names[0]
    comparison = dict(protocol=full.PROTOCOL,
        launch_sha256=sha256((full.canonical(launch) + "\n").encode()).hexdigest(),
        cases=[{"case": name} for name in names])
    raw = (json.dumps(comparison, sort_keys=True, separators=(",", ":")) + "\n").encode()
    root = tmp_path / "comparison"
    root.mkdir()
    (root / "comparison.json").write_bytes(raw)
    with pytest.raises(ValueError, match="ordered 36-case inventory"):
        full.verify(root, launch, sha256(raw).hexdigest())


def write_retained_report(tmp_path, monkeypatch, *, changes=None):
    root = tmp_path / "retained"
    root.mkdir()
    launch_raw = b"{}\n"
    (root / "launch.json").write_bytes(launch_raw)
    pid = 4321
    wsl = full.host.execution.select(full.host.execution.WSL)
    services = set(full.files.SERVICES) | {"user:" + key for key in full.files.SERVICES}
    sample = dict(services={key: "inactive" for key in services},
        compute_pids=[pid], temperature_c=20, memory_used_mib=1024, memory_free_mib=8192)
    decision = campaign.DECISIONS[0]
    launch_sha = sha256(launch_raw).hexdigest()
    report = dict(protocol=full.PROTOCOL, launch_sha256=launch_sha, decision=decision,
        optimizer_steps=0, child=dict(pid=pid, returncode=0, elapsed_s=10.0, samples=[sample]),
        observed_service_seconds=12.0, files={"launch.json": sha256(launch_raw).hexdigest()},
        comparison_sha256="e" * 64, summary={"decision": decision}, **full.FALSE_FLAGS)
    if changes:
        changes(report)
    report_raw = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (root / "report.json").write_bytes(report_raw)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(full.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(full.profile, "checked_receipt", lambda: None)
    monkeypatch.setattr(full.host.execution, "PROFILE", wsl)
    monkeypatch.setattr(full, "verify", lambda *_args: report["summary"])
    return root, launch_sha, sha256(report_raw).hexdigest()


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda report: report["child"].update(returncode=True), "bounded retained full child success"),
    (lambda report: report["child"].update(samples=[]), "bounded retained full child success"),
    (lambda report: report["child"]["samples"][0].update(compute_pids=[999]),
     "recorded full child ownership"),
    (lambda report: report["child"].update(elapsed_s=full.CHILD_SECONDS + 0.1),
     "bounded retained full child success"),
    (lambda report: report.update(observed_service_seconds=9.0),
     "consistent retained full service bounds"),
    (lambda report: report["child"]["samples"][0]["services"].pop(next(iter(report["child"]["samples"][0]["services"]))),
     "recorded full child ownership"),
])
def test_retained_child_receipt_rejects_malformed_ownership_and_timing(
        tmp_path, monkeypatch, mutate, message):
    root, launch_sha, report_sha = write_retained_report(
        tmp_path, monkeypatch, changes=mutate)
    with pytest.raises(ValueError, match=message):
        full.verify_retained(root, launch_sha, report_sha)


def test_retained_child_receipt_accepts_consistent_cpu_only_replay(tmp_path, monkeypatch):
    root, launch_sha, report_sha = write_retained_report(tmp_path, monkeypatch)
    summary = full.verify_retained(root, launch_sha, report_sha)
    assert summary == {"decision": campaign.DECISIONS[0]}


@pytest.mark.parametrize(("passes", "expected"), [
    ({577: 64, 587: 128, 593: 255}, "lean-replication-passed"),
    ({577: 64, 587: None, 593: None}, "lean-replication-seed-dependent"),
    ({577: None, 587: None, 593: None}, "lean-replication-rejected"),
])
def test_replication_decision_all_some_none_allows_different_passing_iterations(
        monkeypatch, passes, expected):
    launch = make_plan(monkeypatch)
    def summarize_for_seed(cases, scores, declaration):
        seed = cases[0]["binding"]["training_seed"]
        passing_iteration = passes[seed]
        iterations = {case["binding"]["checkpoint_iteration"] for case in cases}
        if passing_iteration is not None:
            assert passing_iteration in iterations
            # Each historical seed may pass at a different common checkpoint.
            return {"decision": declaration["passed"]}
        return {"decision": declaration["rejected"]}

    monkeypatch.setattr(evaluation, "summarize_cases", summarize_for_seed)
    scores = {case["name"]: {"fixture": True} for case in launch["cases"]}
    summary = full.summarize(launch, scores)
    assert summary["decision"] == expected
    assert summary["passing_seeds"] == [seed for seed in auth.SEEDS if passes[seed] is not None]
    assert summary["nominal_replication_numerical_gate_passed"] is (expected == "lean-replication-passed")
