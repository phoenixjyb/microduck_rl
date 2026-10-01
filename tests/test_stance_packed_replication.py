"""Declaration and wiring checks for the packed learner replication.

No learner job is launched here. These tests cover the static, source-bound
plan and refusal contracts only, not training, timing, or capability admission.
"""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from mjlab_microduck import stance_lean_lesson as lean
from mjlab_microduck import stance_solved_field_check as field_checker
from mjlab_microduck import stance_wsl_qualification as qualification


SOURCE = "a" * 40
LAUNCH = "b" * 64
RUNTIME = "c" * 64


def select_wsl(monkeypatch):
    monkeypatch.setattr(lean.host.execution, "PROFILE",
                        lean.host.execution.select(lean.host.execution.WSL))


def test_packed_declaration_is_a_narrow_replication_variant():
    packed = lean.PACKED_REPLICATION
    assert packed["seeds"] == lean.REPLICATION["seeds"]
    assert packed["purpose"] == lean.REPLICATION["purpose"]
    assert packed["label"] == "lean-replication-packed"
    assert packed["protocol"] == "football-b1n-lean-replication-packed-v1"
    assert packed["directory"] == "stance-lean-replication-packed-"
    assert packed["service"] == "microduck-lean-replication-packed-"
    assert packed["solved_field_check"] == "packed"
    assert lean.declaration_of(packed["label"]) is packed


def test_only_exact_wsl_selects_packed_and_historical_declarations_stay_legacy(monkeypatch):
    monkeypatch.setattr(lean.host.execution, "PROFILE",
                        lean.host.execution.select(lean.host.execution.DEFAULT))
    with pytest.raises(ValueError, match="exact WSL profile"):
        lean.checker_of(lean.PACKED_REPLICATION)
    assert lean.checker_of(lean.LESSON) == "legacy"
    assert lean.checker_of(lean.REPLICATION) == "legacy"
    select_wsl(monkeypatch)
    assert lean.checker_of(lean.PACKED_REPLICATION) == "packed"
    assert lean.checker_of(lean.REPLICATION) == "legacy"


def test_qualification_routes_packed_mode_explicitly_but_preserves_legacy_default(monkeypatch):
    select_wsl(monkeypatch)
    calls = []
    monkeypatch.setattr(qualification, "verify",
                        lambda *args: calls.append(args) or {"qualification": args})
    assert lean.qualification_of(SOURCE, lean.PACKED_REPLICATION) == {
        "qualification": (SOURCE, "packed")}
    assert lean.qualification_of(SOURCE, lean.REPLICATION) == {
        "qualification": (SOURCE,)}
    assert calls == [(SOURCE, "packed"), (SOURCE,)]


def test_declared_seed_set_is_exact_and_other_seeds_refuse():
    assert lean.PACKED_REPLICATION["seeds"] == (577, 587, 593)
    for seed in (577, 587, 593):
        assert lean.seed_of(lean.PACKED_REPLICATION, seed) == seed
    for seed in (571, 576, 578, 594, True):
        with pytest.raises(ValueError, match="declared learner seed"):
            lean.seed_of(lean.PACKED_REPLICATION, seed)


def test_packed_and_legacy_replication_paths_are_separate_and_lesson_path_unchanged():
    packed = lean.PACKED_REPLICATION
    seed = packed["seeds"][0]
    assert lean.output_path(SOURCE).name == "stance-lean-lesson-" + SOURCE[:12]
    assert lean.service_name(SOURCE) == "microduck-lean-lesson-" + SOURCE[:12] + ".service"
    assert lean.output_path(SOURCE, lean.REPLICATION, seed).name == (
        f"stance-lean-replication-{SOURCE[:12]}-seed-{seed}")
    assert lean.service_name(SOURCE, lean.REPLICATION, seed) == (
        f"microduck-lean-replication-{SOURCE[:12]}-seed-{seed}.service")
    assert lean.output_path(SOURCE, packed, seed).name == (
        f"stance-lean-replication-packed-{SOURCE[:12]}-seed-{seed}")
    assert lean.service_name(SOURCE, packed, seed) == (
        f"microduck-lean-replication-packed-{SOURCE[:12]}-seed-{seed}.service")


def test_packed_plan_binds_checker_cutoff_and_resource_bounds(monkeypatch):
    select_wsl(monkeypatch)
    monkeypatch.setattr(lean.host.execution, "training_budget", lambda: {
        "child_seconds": 4320, "service_seconds": 4380,
        "runtime_max": "1h 13min", "max_window_seconds": 7200})
    monkeypatch.setattr(lean, "qualification_of", lambda *_: {"qualified": True})
    deadline = qualification.PACKED_CUTOFF - 1
    packed = lean.plan(SOURCE, {"host": "fixture"}, RUNTIME, deadline,
                       lean.PACKED_REPLICATION, 577)
    assert packed["protocol"] == lean.PACKED_REPLICATION["protocol"]
    assert packed["purpose"] == lean.REPLICATION["purpose"]
    assert packed["seed"] == 577
    assert packed["host_qualification"] == {"qualified": True}
    assert packed["solved_field_check"] == "packed"
    assert packed["checker_sha256"] == lean.host.digest(field_checker.__file__)
    assert packed["deadline_unix"] < qualification.PACKED_CUTOFF
    assert packed["child_timeout_seconds"] == 4320
    assert packed["service_timeout_seconds"] == 4380
    assert packed["memory_max_bytes"] == 6 * 1024**3
    assert packed["cpu_quota_per_sec_usec"] == 2_000_000
    assert packed["nice"] == 10
    for key in ("learned_stance", "physical_motion_authorized", "pilot_parent_authorized"):
        assert packed[key] is False
    with pytest.raises(ValueError, match="authorized cutoff"):
        lean.plan(SOURCE, {}, RUNTIME, qualification.PACKED_CUTOFF + 1,
                  lean.PACKED_REPLICATION, 577)


def test_default_lesson_and_legacy_replication_plan_remain_without_packed_fields(monkeypatch):
    select_wsl(monkeypatch)
    monkeypatch.setattr(lean, "qualification_of", lambda *_: {"legacy": True})
    monkeypatch.setattr(lean.host.execution, "training_budget", lambda: {
        "child_seconds": 4320, "service_seconds": 4380,
        "runtime_max": "1h 13min", "max_window_seconds": 7200})
    lesson = lean.plan(SOURCE, {}, RUNTIME, 1)
    repl = lean.plan(SOURCE, {}, RUNTIME, 1, lean.REPLICATION, 577)
    assert "solved_field_check" not in lesson and "checker_sha256" not in lesson
    assert "solved_field_check" not in repl and "checker_sha256" not in repl
    assert lesson["protocol"] == lean.PROTOCOL
    assert lesson["seed"] == lean.SEED
    assert repl["protocol"] == lean.REPLICATION["protocol"]
    assert repl["seed"] == 577


def test_packed_child_command_round_trips_job_seed_and_bound_ids():
    seed = 587
    argv = lean.child_command(SOURCE, LAUNCH, 19, lean.PACKED_REPLICATION, seed)
    args = lean.parser().parse_args(argv[3:])
    assert args.mode == "child"
    assert args.job == "lean-replication-packed"
    assert args.seed == seed
    assert args.source == SOURCE and args.launch_sha256 == LAUNCH and args.lock_fd == 19
    assert "--job" in argv and "--seed" in argv
    assert "--solved-field-check" not in argv


@pytest.mark.parametrize(("bridge_mode", "should_pass"), [("legacy", False), ("packed", True)])
def test_update_loop_requires_the_actual_packed_bridge_before_collection(
        monkeypatch, tmp_path, bridge_mode, should_pass):
    select_wsl(monkeypatch)
    monkeypatch.setattr(lean, "UPDATES", 1)
    monkeypatch.setattr(lean, "STEPS", 1)
    learner = object.__new__(lean.LeanStanceLearner)
    learner.n = lean.WORLDS
    learner.updates = 0
    learner.parent_checkpoint_sha256 = lean.checkpoint.LEAN_PARENT_SHA256
    learner.restored_fixture_only = False
    learner.seed = 577
    calls = []
    learner.collect_one = lambda _bridge: calls.append("collect") or {"tick": True}
    learner.update = lambda: calls.append("update") or {"metrics": {"loss": 0.0}}
    bridge = SimpleNamespace(n=lean.WORLDS,
                             env=SimpleNamespace(solved_field_check=bridge_mode))
    monkeypatch.setattr(lean, "identity", lambda *_args, **_kwargs: {"fixture": True})
    monkeypatch.setattr(lean.smoke, "save_weights", lambda *_args, **_kwargs: {"saved": True})
    monkeypatch.setattr(lean.smoke, "tick_evidence", lambda *_args, **_kwargs: {"tick": True})
    monkeypatch.setattr(lean.supervisor, "write_json", lambda *_args, **_kwargs: None)
    if should_pass:
        exports = lean.run_updates(learner, bridge, tmp_path, SOURCE, LAUNCH, RUNTIME,
                                  deadline=10**12, declaration=lean.PACKED_REPLICATION)
        assert exports == [{"saved": True}, {"saved": True}]
        assert calls == ["collect", "update"]
    else:
        with pytest.raises(ValueError, match="actual packed learner runtime"):
            lean.run_updates(learner, bridge, tmp_path, SOURCE, LAUNCH, RUNTIME,
                             deadline=10**12, declaration=lean.PACKED_REPLICATION)
        assert calls == []


def test_rejected_packed_source_qualification_prevents_prepare_artifact_allocation(
        monkeypatch, tmp_path):
    select_wsl(monkeypatch)
    monkeypatch.setattr(lean, "check_window", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lean.host, "identity", lambda _source: {"fixture": True})
    monkeypatch.setattr(lean.host, "ROOT", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    calls = []

    def reject(source, mode="legacy"):
        calls.append((source, mode))
        raise ValueError("packed source qualification rejected")

    monkeypatch.setattr(qualification, "verify", reject)
    with pytest.raises(ValueError, match="packed source qualification rejected"):
        lean.prepare(SOURCE, qualification.PACKED_CUTOFF - 1, lean.PACKED_REPLICATION, 577)
    assert calls == [(SOURCE, "packed")]
    assert not (tmp_path / "artifacts").exists()


def _completed_fixture(tmp_path, monkeypatch, *, mode="packed", checker_hash="d" * 64):
    # Avoid checkpoint deserialization/training: verify_completed only checks the
    # initial export on this zero-update synthetic boundary fixture.
    monkeypatch.setattr(lean, "UPDATES", 0)
    monkeypatch.setattr(lean, "parent_bytes", lambda _root: b"parent")
    learner = SimpleNamespace(initial_hash="parent-state-hash")
    monkeypatch.setattr(lean, "started_learner_from", lambda *_args, **_kwargs: learner)
    monkeypatch.setattr(lean.checkpoint, "fresh_models", lambda _seed: (object(), object()))
    monkeypatch.setattr(lean.checkpoint, "states_of", lambda *_models: "fresh-model-states")
    monkeypatch.setattr(lean.checkpoint, "state_hash", lambda states: (
        "fresh-state-hash" if states == "fresh-model-states" else "parent-state-hash"))
    monkeypatch.setattr(lean.checkpoint, "validate_states", lambda *_args: None)
    meta = {"fixture": "initial identity"}
    monkeypatch.setattr(lean, "identity", lambda *_args, **_kwargs: meta)
    blob = b"synthetic initial weights"
    path = tmp_path / "initial.pt"
    path.write_bytes(blob)
    monkeypatch.setattr(lean.supervisor, "file_bytes", lambda _path, **_kwargs: blob)
    monkeypatch.setattr(lean.torch, "load", lambda *_args, **_kwargs: {
        "identity": meta, "states": {"fixture": True}})
    saved = {"file": "initial.pt", "sha256": sha256(blob).hexdigest(), "identity": meta}
    result = dict(protocol=lean.PACKED_REPLICATION["protocol"], launch_sha256=LAUNCH,
        completed_updates=0, checkpoints=[saved], physics_device="cuda:0", learner_device="cpu",
        seed=577, worlds=lean.WORLDS, purpose=lean.PACKED_REPLICATION["purpose"],
        common_checkpoints=list(lean.CHECKPOINTS),
        parent_checkpoint_sha256=lean.checkpoint.LEAN_PARENT_SHA256,
        forward_graph=False, optimizer_state_restored=False, simulation_resume_authorized=False,
        pilot_parent_authorized=False, learned_stance=False, physical_motion_authorized=False,
        weight_initialized=True, solved_field_check=mode, checker_sha256=checker_hash)
    root_files = {"launch.json", "runtime.json", "parent.pt", "child.log", "completed.json", "initial.pt"}
    for name in root_files - {"initial.pt"}:
        (tmp_path / name).touch()
    monkeypatch.setattr(lean.supervisor, "parse", lambda _data: result)
    launch = {"solved_field_check": "packed", "checker_sha256": checker_hash,
              "runtime_sha256": RUNTIME}
    return result, launch


@pytest.mark.parametrize("damage", ["mode", "checker_hash"])
def test_completed_packed_receipt_must_match_launch_mode_and_checker(tmp_path, monkeypatch, damage):
    result, launch = _completed_fixture(tmp_path, monkeypatch)
    if damage == "mode": result["solved_field_check"] = "legacy"
    else: result["checker_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="completed packed checker receipt"):
        lean.verify_completed(tmp_path, SOURCE, LAUNCH, launch, lean.PACKED_REPLICATION, 577)


def test_matching_packed_completed_receipt_is_accepted_without_admission(tmp_path, monkeypatch):
    result, launch = _completed_fixture(tmp_path, monkeypatch)
    accepted = lean.verify_completed(tmp_path, SOURCE, LAUNCH, launch,
                                     lean.PACKED_REPLICATION, 577)
    assert accepted is result
    assert accepted["solved_field_check"] == launch["solved_field_check"] == "packed"
    assert accepted["checker_sha256"] == launch["checker_sha256"]
    assert accepted["learned_stance"] is False
    assert accepted["physical_motion_authorized"] is False
