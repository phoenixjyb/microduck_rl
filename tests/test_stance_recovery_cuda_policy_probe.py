"""Portable source checks for the non-admitting CUDA64 preparation runner.

Positive tensor fixtures in this file exercise CPU validators only.  They are
synthetic consistency tests and are not CUDA constructor/native qualification.
"""

import hashlib
import io
import os
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from mjlab_microduck import stance_ppo
from mjlab_microduck import stance_recovery_cuda_policy_probe as probe

SOURCE = "a" * 40
ID = "b" * 32


def service(mode="supervise"):
    seconds = (
        probe.SUPERVISOR_SECONDS if mode == "supervise" else probe.CLOSEOUT_SECONDS
    )
    memory = (
        probe.SUPERVISOR_MEMORY_BYTES
        if mode == "supervise"
        else probe.CLOSEOUT_MEMORY_BYTES
    )
    return {
        "MainPID": "1234",
        "ActiveState": "active",
        "RuntimeMaxUSec": f"{seconds // 60}min",
        "MemoryMax": str(memory),
        "CPUQuotaPerSecUSec": "2s",
        "Nice": "10",
        "KillMode": "control-group",
        "InvocationID": ID,
    }


def launch_record():
    return {
        "protocol": probe.PROTOCOL,
        "source": SOURCE,
        "output_name": probe.output_path(SOURCE).name,
        "preparation_protocol": probe.preparation.PROTOCOL,
        "parent_checkpoint_sha256": probe.baseline.CHECKPOINT_SHA256,
        "parent_state_sha256": probe.preparation.parent.PARENT_STATE_SHA256,
        "parent_identity": probe.preparation.parent.expected_identity(),
        "native_prerequisites": {
            "preflight_failure_binding": {},
            "source": {"source": SOURCE},
            "gap_inventory": {"case-0.pt": {"sha256": "c" * 64, "bytes": 1}},
            "gap_receipts": {"launch_sha256": "d" * 64},
            "current_context": {"source_identity": {"source": SOURCE}},
        },
        "cpu_parent_receipt_canonical_sha256": "e" * 64,
        "cpu_parent_binding": probe.preparation.cpu_parent_binding(SOURCE, "e" * 64),
        "learner_seeds": list(probe.SEEDS),
        "worlds": probe.WORLDS,
        "horizon": probe.HORIZON,
        "supervisor_service_seconds": probe.SUPERVISOR_SECONDS,
        "child_timeout_seconds": probe.CHILD_SECONDS,
        "closeout_service_seconds": probe.CLOSEOUT_SECONDS,
        "launch_reserve_seconds": probe.LAUNCH_RESERVE_SECONDS,
        "margin_seconds": probe.MARGIN_SECONDS,
        "supervisor_memory_bytes": probe.SUPERVISOR_MEMORY_BYTES,
        "closeout_memory_bytes": probe.CLOSEOUT_MEMORY_BYTES,
        "cpu_quota": probe.CPU_QUOTA,
        "nice": probe.NICE,
        "kill_mode": probe.KILL_MODE,
        "raw_bytes_limit": probe.RAW_LIMIT,
        "log_bytes_limit": probe.LOG_LIMIT,
        "sequential_children": True,
        "cpu_parent_provenance_authenticated_by_supervisor": True,
        "cuda_child_claims_cpu_provenance_authenticated": False,
        "optimizer_steps": 0,
        "training_updates": 0,
        "simulator_resets": 0,
        "student_exports": 0,
        "action_sampling": False,
        "return_computation": False,
        "rollout_collection": False,
        "campaign_window": probe.window.declaration(),
        "service_properties": service(),
        **probe.PREPARATION_FALSE_FLAGS,
    }


def test_namespace_and_fixed_preparation_budget_are_literal():
    assert probe.output_path(SOURCE).name == probe.OUTPUT_PREFIX + SOURCE[:12]
    assert probe.SEEDS == (653, 659)
    assert (probe.WORLDS, probe.HORIZON) == (64, 28)
    assert (
        probe.SUPERVISOR_SECONDS,
        probe.CHILD_SECONDS,
        probe.CLOSEOUT_SECONDS,
        probe.MARGIN_SECONDS,
        probe.LAUNCH_RESERVE_SECONDS,
    ) == (360, 120, 180, 60, 600)
    assert (probe.SUPERVISOR_MEMORY_BYTES, probe.CLOSEOUT_MEMORY_BYTES) == (
        3 * 1024**3,
        2 * 1024**3,
    )
    assert (probe.LOG_LIMIT, probe.RAW_LIMIT) == (1024**2, 8 * 1024**2)
    assert len(probe.FALSE_FLAGS) == 8
    assert all(value is False for value in probe.PREPARATION_FALSE_FLAGS.values())


def test_recorded_service_cap_requires_mode_memory_invocation_and_control_group():
    assert probe._recorded_service_properties(service(), "supervise")
    assert probe._recorded_service_properties(service("closeout"), "closeout")
    for field, value in (
        ("RuntimeMaxUSec", "5min"),
        ("MemoryMax", str(2 * 1024**3)),
        ("CPUQuotaPerSecUSec", "4s"),
        ("Nice", "0"),
        ("KillMode", "process"),
        ("InvocationID", "not-an-invocation"),
    ):
        altered = service()
        altered[field] = value
        with pytest.raises(ValueError):
            probe._recorded_service_properties(altered, "supervise")


def test_live_service_reader_checks_exact_invocation_and_rejects_cap_drift(monkeypatch):
    expected = service()
    expected["MainPID"] = str(os.getpid())
    unit = probe.service_name(SOURCE, "supervise")

    def host_read(*args):
        if "list-units" in args:
            return f"{unit} loaded active running test"
        field = args[args.index("-p") + 1]
        return expected[field]

    monkeypatch.setattr(probe.host, "read", host_read)
    assert probe.service_properties(SOURCE, "supervise") == expected
    expected["MemoryMax"] = str(probe.CLOSEOUT_MEMORY_BYTES)
    with pytest.raises(ValueError, match="exact independently capped"):
        probe.service_properties(SOURCE, "supervise")


def test_launch_record_rejects_resampling_parallelism_and_any_update():
    assert probe._validate_launch_record(SOURCE, launch_record())
    mutations = (
        {"learner_seeds": [659, 653]},
        {"learner_seeds": [653, 653]},
        {"sequential_children": False},
        {"worlds": 2},
        {"horizon": 27},
        {"optimizer_steps": 1},
        {"training_updates": 1},
        {"simulator_resets": 1},
        {"training_admitted": True},
        {"finite_optimizer_step_qualified": True},
        {"campaign_window": {}},
    )
    for changes in mutations:
        altered = launch_record()
        altered.update(changes)
        with pytest.raises(ValueError):
            probe._validate_launch_record(SOURCE, altered)


def test_json_decoder_rejects_duplicate_keys_and_nonfinite_numbers():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        probe.parse_json(b'{"source":"x","source":"y"}')
    with pytest.raises(ValueError):
        probe.parse_json(b'{"elapsed":NaN}')
    assert probe.parse_json(b'{"ok":true}') == {"ok": True}


def test_exclusive_retention_rejects_overwrite_and_symlink(tmp_path):
    path = tmp_path / "evidence.json"
    raw = b'{"source":"fixture"}\n'
    assert probe._write_exclusive(path, raw, 128) == hashlib.sha256(raw).hexdigest()
    assert probe._read_file(path, 128) == raw
    with pytest.raises(FileExistsError):
        probe._write_exclusive(path, b"replacement", 128)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="regular non-symlink"):
        probe._read_file(link, 128)
    with pytest.raises(ValueError, match="bounded retained bytes"):
        probe._write_exclusive(tmp_path / "oversized", b"x" * 129, 128)


def zero_storage():
    return {
        "observations": {
            "actor": torch.zeros(
                (probe.HORIZON, probe.WORLDS, 44), dtype=torch.float32
            ),
            "critic": torch.zeros(
                (probe.HORIZON, probe.WORLDS, 50), dtype=torch.float32
            ),
        },
        "actions": torch.zeros((probe.HORIZON, probe.WORLDS, 10), dtype=torch.float32),
        "rewards": torch.zeros((probe.HORIZON, probe.WORLDS, 1), dtype=torch.float32),
        "dones": torch.zeros((probe.HORIZON, probe.WORLDS, 1), dtype=torch.uint8),
        "values": torch.zeros((probe.HORIZON, probe.WORLDS, 1), dtype=torch.float32),
        "actions_log_prob": torch.zeros(
            (probe.HORIZON, probe.WORLDS, 1), dtype=torch.float32
        ),
        "returns": torch.zeros((probe.HORIZON, probe.WORLDS, 1), dtype=torch.float32),
        "advantages": torch.zeros(
            (probe.HORIZON, probe.WORLDS, 1), dtype=torch.float32
        ),
        "distribution_params": None,
        "saved_hidden_state_a": None,
        "saved_hidden_state_c": None,
    }


def test_storage_validator_is_cpu_only_exact_and_zero():
    assert probe._check_storage(zero_storage())
    altered = zero_storage()
    altered["actions"][0, 0, 0] = 1
    with pytest.raises(ValueError, match="finite zero CPU snapshot actions"):
        probe._check_storage(altered)
    altered = zero_storage()
    altered["dones"] = altered["dones"].to(torch.float32)
    with pytest.raises(ValueError, match="finite zero CPU snapshot dones"):
        probe._check_storage(altered)
    altered = zero_storage()
    altered["observations"]["actor"] = torch.empty(
        (probe.HORIZON, probe.WORLDS, 44), device="meta"
    )
    with pytest.raises(ValueError, match="finite zero CPU snapshot actor observations"):
        probe._check_storage(altered)


def rng_fixture():
    rng = {
        name: torch.tensor([1, 2, 3], dtype=torch.uint8)
        for name in ("cpu_before", "cpu_after", "cuda_before", "cuda_after")
    }
    hashes = {
        name: hashlib.sha256(value.numpy().tobytes()).hexdigest()
        for name, value in rng.items()
    }
    receipt = {
        "caller_rng_state_sha256": hashes,
        "caller_cpu_rng_unchanged": True,
        "caller_cuda_rng_unchanged": True,
        "private_cuda_rng_state_sha256": "",
        "private_cuda_generator_device": "cuda:0",
        "private_cuda_rng_connected_to_sampler": False,
    }
    return rng, hashes, receipt


def test_caller_rng_validator_rejects_relabelled_bytes_and_advanced_stream():
    rng, hashes, receipt = rng_fixture()
    assert (
        probe._check_caller_rng(rng, {"caller_rng_state_sha256": hashes}, receipt)
        == hashes
    )
    altered = deepcopy(rng)
    altered["cuda_after"] = torch.tensor([9, 2, 3], dtype=torch.uint8)
    with pytest.raises(ValueError):
        probe._check_caller_rng(altered, {"caller_rng_state_sha256": hashes}, receipt)
    relabelled = deepcopy(rng)
    relabelled["cpu_before"] = relabelled["cpu_before"].to(torch.float32)
    with pytest.raises(ValueError, match="four raw CPU/CUDA caller RNG"):
        probe._check_caller_rng(
            relabelled, {"caller_rng_state_sha256": hashes}, receipt
        )
    relabelled = deepcopy(rng)
    relabelled["cuda_before"] = torch.empty(3, device="meta", dtype=torch.uint8)
    with pytest.raises(ValueError, match="four raw CPU/CUDA caller RNG"):
        probe._check_caller_rng(
            relabelled, {"caller_rng_state_sha256": hashes}, receipt
        )


def test_private_generator_state_is_separate_and_hash_bound():
    state = torch.tensor([8, 9, 4], dtype=torch.uint8)
    state_hash = hashlib.sha256(state.numpy().tobytes()).hexdigest()
    metadata = {"private_cuda_rng_state_sha256": state_hash}
    receipt = {
        "private_cuda_rng_state_sha256": state_hash,
        "private_cuda_generator_device": "cuda:0",
        "private_cuda_rng_connected_to_sampler": False,
    }
    summary = {"private_cuda_rng_state_sha256": state_hash}
    assert probe._check_private_rng(state, metadata, receipt, summary) == state_hash
    receipt["private_cuda_rng_connected_to_sampler"] = True
    with pytest.raises(ValueError):
        probe._check_private_rng(state, metadata, receipt, summary)


def test_empty_optimizer_union_config_and_pins_are_exact_source_checks():
    actor = {"weight": torch.zeros(1)}
    critic = {"weight": torch.zeros(1)}
    metadata = {
        "actor_parameter_names": ["weight"],
        "critic_parameter_names": ["weight"],
        "optimizer_parameter_names": ["actor.weight", "critic.weight"],
        "optimizer_parameter_count": 2,
    }
    group = {
        "params": [0, 1],
        "lr": stance_ppo.CONFIG["learning_rate"],
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0,
    }
    assert probe._check_optimizer_state(
        {"state": {}, "param_groups": [group]}, metadata, actor, critic
    )
    broken = deepcopy(metadata)
    broken["optimizer_parameter_names"] = ["actor.weight", "actor.weight"]
    with pytest.raises(ValueError):
        probe._check_optimizer_state(
            {"state": {}, "param_groups": [group]}, broken, actor, critic
        )


def verifier_test_root():
    native = probe.execution.ROOT / "artifacts/tools" / probe.VERIFIER_DIRECTORY
    mirror = (
        Path(__file__).resolve().parents[1]
        / "artifacts/retained"
        / "gentle-gap-closed-bb7c059d5dda.n3GlnP/tools"
        / probe.VERIFIER_DIRECTORY
    )
    root = native if native.is_dir() else mirror
    if not root.is_dir():
        assert not probe.execution.ROOT.is_dir(), (
            "native evidence is mandatory on its authorized host"
        )
        pytest.skip(
            "optional retained verifier is unavailable outside the authorized evidence hosts"
        )
    return root


def test_external_verifier_reader_authenticates_only_its_real_seven_files():
    root = verifier_test_root()
    result = probe._check_external_verifier(root, check_live=False)
    assert result["receipt_sha256"] == probe.VERIFIER_RECEIPT_SHA256
    assert len(result["journal_inventory"]) == 6
    assert {item.name for item in root.iterdir()} == {
        "receipt.json",
        "run-manager.log",
        "run-process.log",
        "tests-manager.log",
        "tests-process.log",
        "closeout-manager.log",
        "closeout-process.log",
    }


def test_external_verifier_reader_rejects_tampered_journal(tmp_path):
    source_root = verifier_test_root()
    for path in source_root.iterdir():
        shutil.copyfile(path, tmp_path / path.name)
    with (tmp_path / "run-process.log").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="pinned verifier journal bytes"):
        probe._check_external_verifier(tmp_path, check_live=False)


def test_live_gpu_monitor_samples_are_bounded_and_source_only():
    samples = [
        {
            "elapsed_seconds": 0.1,
            "child_pid": 1234,
            "sample": {
                "services": {
                    name: "inactive"
                    for name, _ in probe.execution.service_commands(
                        probe.campaign.SERVICES
                    )
                },
                "compute_pids": [1234],
                "temperature_c": 30,
                "memory_used_mib": 838,
                "memory_free_mib": 6144,
            },
        }
    ]
    children = [{"seed": seed, "live_gpu_samples": samples} for seed in probe.SEEDS]
    assert probe._check_live_gpu_samples(children)
    bad = deepcopy(children)
    bad[0]["live_gpu_samples"][0]["sample"]["compute_pids"].append(5678)
    with pytest.raises(ValueError, match="retained live GPU ownership"):
        probe._check_live_gpu_samples(bad)


@pytest.fixture
def payload_fixture(monkeypatch):
    """Real CPU RSL objects, synthetic CUDA labels; never native qualification."""
    preparation = probe.preparation
    actor, critic = probe.checkpoint.fresh_models(577)
    states = probe.checkpoint.states_of(actor, critic)
    synthetic_state_sha = probe.checkpoint.state_hash(states)
    monkeypatch.setattr(preparation.parent, "PARENT_STATE_SHA256", synthetic_state_sha)
    obs = preparation.TensorDict(
        {"actor": torch.zeros(64, 44), "critic": torch.zeros(64, 50)}, [64]
    )
    storage = preparation.RolloutStorage("rl", 64, 28, obs, (10,), device="cpu")
    algorithm = preparation.PPO(
        actor, critic, storage, **stance_ppo.CONFIG, device="cpu"
    )
    rng, rng_hashes, _ = rng_fixture()
    private = torch.tensor([8, 9, 4], dtype=torch.uint8)
    cpu_receipt = {"cpu_math_profile": probe.gap.base.profile.expected_receipt()}
    cpu_sha = "e" * 64
    binding = preparation.cpu_parent_binding(SOURCE, cpu_sha)
    source_identity = {"source": SOURCE, "synthetic_test_only": True}
    child_receipt = {
        "protocol": preparation.PROTOCOL,
        "source": SOURCE,
        "source_identity": source_identity,
        "parent_checkpoint_sha256": probe.baseline.CHECKPOINT_SHA256,
        "parent_identity": preparation.parent.expected_identity(),
        "parent_state_sha256": synthetic_state_sha,
        "cpu_parent_receipt_sha256": cpu_sha,
        "caller_binding": binding,
        "cpu_parent_receipt": cpu_receipt,
        "cpu_provenance_authentication_required": True,
        "native_cpu_provenance_authenticated_by_this_function": False,
        "learner_seed": 653,
        "worlds": 64,
        "horizon": 28,
        "device": "cuda:0",
        "actor_device": "cuda:0",
        "critic_device": "cuda:0",
        "state_sha256_before_transfer": synthetic_state_sha,
        "state_sha256_after_transfer": synthetic_state_sha,
        "actor_parameter_tensors": sum(1 for _ in actor.parameters()),
        "critic_parameter_tensors": sum(1 for _ in critic.parameters()),
        "ppo_config": deepcopy(stance_ppo.CONFIG),
        "ppo_source_sha256": deepcopy(stance_ppo.PINS),
        "storage_step": 0,
        "storage_empty": True,
        "optimizer_type": "Adam",
        "optimizer_state_entries": 0,
        "optimizer_empty": True,
        "private_cuda_generator_device": "cuda:0",
        "private_cuda_rng_state_sha256": probe.digest(private.numpy().tobytes()),
        "private_cuda_rng_connected_to_sampler": False,
        "caller_cpu_rng_unchanged": True,
        "caller_cuda_rng_unchanged": True,
        "caller_rng_state_sha256": rng_hashes,
        "cuda_initialized": True,
        "simulator_created": False,
        "rollout_collected": False,
        "optimizer_steps": 0,
        "training_update_performed": False,
        "finite_optimizer_step_qualified": False,
        "transition_bridge_qualified": False,
        "schedule_installed": False,
        "student_export_available": False,
        "training_job_predeclared": False,
        "execution_admitted": False,
        **probe.FALSE_FLAGS,
    }
    raw, metadata = probe._child_payload(
        {
            "actor": actor,
            "critic": critic,
            "algorithm": algorithm,
            "storage": storage,
            "receipt": child_receipt,
            "caller_rng_states": rng,
            "private_cuda_rng_state": private,
        },
        "d" * 64,
    )
    launch = {
        "source": SOURCE,
        "native_prerequisites": {
            "current_context": {
                "source_identity": source_identity,
                "cpu_math_profile": cpu_receipt["cpu_math_profile"],
            }
        },
        "cpu_parent_receipt_canonical_sha256": cpu_sha,
        "cpu_parent_receipt_file_sha256": "f" * 64,
        "cpu_parent_binding": binding,
    }
    summary = {
        "protocol": probe.PROTOCOL + ":child-summary-v1",
        "source": SOURCE,
        "launch_sha256": "d" * 64,
        "seed": 653,
        "payload_sha256": probe.digest(raw),
        "payload_bytes": len(raw),
        "metadata": metadata,
        "caller_rng_state_sha256": rng_hashes,
        "private_cuda_rng_state_sha256": child_receipt["private_cuda_rng_state_sha256"],
        "cpu_parent_receipt_file_sha256": "f" * 64,
        "cpu_parent_receipt_canonical_sha256": cpu_sha,
        "cuda_child_claims_cpu_provenance_authenticated": False,
        "action_sampling": False,
        "return_computation": False,
        "rollout_collection": False,
        "optimizer_steps": 0,
        "simulator_resets": 0,
        "training_update_performed": False,
        **probe.PREPARATION_FALSE_FLAGS,
    }
    return raw, summary, launch, cpu_receipt


def test_complete_payload_roundtrip_is_cpu_consistency_only(payload_fixture):
    raw, summary, launch, cpu_receipt = payload_fixture
    result = probe._score_payload(raw, summary, launch, "d" * 64, cpu_receipt, 653)
    assert result["storage_zero"] and result["optimizer_empty"]
    assert result["cuda_math_replayed"] is False
    assert all(result[key] is False for key in probe.PREPARATION_FALSE_FLAGS)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol", "wrong"),
        ("cuda_initialized", False),
        ("private_cuda_rng_connected_to_sampler", True),
        ("optimizer_steps", 1),
        ("training_job_predeclared", True),
        ("native_cpu_provenance_authenticated_by_this_function", True),
        *((key, True) for key in probe.FALSE_FLAGS),
    ],
)
def test_full_payload_rejects_rehashed_bad_receipt(payload_fixture, field, value):
    raw, summary, launch, cpu_receipt = payload_fixture
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    payload["metadata"]["raw_preparation_receipt"][field] = value
    stream = io.BytesIO()
    torch.save(payload, stream)
    raw = stream.getvalue()
    summary.update(
        payload_sha256=probe.digest(raw),
        payload_bytes=len(raw),
        metadata=payload["metadata"],
    )
    with pytest.raises(ValueError):
        probe._score_payload(raw, summary, launch, "d" * 64, cpu_receipt, 653)


@pytest.mark.parametrize(
    "field", ["rng", "adam", "storage", "parameter_subset", "receipt_schema"]
)
def test_full_payload_rejects_corruption_even_after_rehash(payload_fixture, field):
    raw, summary, launch, cpu_receipt = payload_fixture
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    if field == "rng":
        payload["caller_rng_states"]["cuda_after"][0] = 99
    elif field == "adam":
        payload["optimizer_state_dict"]["state"][0] = {"step": torch.tensor(1.0)}
    elif field == "storage":
        payload["storage"]["actions"][0, 0, 0] = 1.0
    elif field == "parameter_subset":
        metadata["actor_parameter_names"] = metadata["actor_parameter_names"][:1]
    else:
        metadata["raw_preparation_receipt"]["unreviewed_extra_key"] = False
    stream = io.BytesIO()
    torch.save(payload, stream)
    raw = stream.getvalue()
    summary.update(
        payload_sha256=probe.digest(raw), payload_bytes=len(raw), metadata=metadata
    )
    with pytest.raises(ValueError):
        probe._score_payload(raw, summary, launch, "d" * 64, cpu_receipt, 653)


@pytest.mark.parametrize("scenario", ["success", "oversize", "timeout"])
def test_streaming_child_enforces_owned_timeout_and_log_cap(
    monkeypatch, tmp_path, scenario
):
    """Ordinary CPU subprocess fixtures; no lease or CUDA qualification."""
    real_popen = probe.subprocess.Popen
    process = []
    scripts = {
        "success": "import time; print('fixture subprocess, not native CUDA', flush=True); time.sleep(.2)",
        "oversize": "import os; os.write(1, b'x' * (2 * 1024**2))",
        "timeout": "import time; print('retained before timeout', flush=True); time.sleep(60)",
    }

    def fixture_popen(command, **kwargs):
        assert "mjlab_microduck.stance_recovery_cuda_policy_probe" in command
        kwargs["cwd"] = tmp_path
        child = real_popen([probe.sys.executable, "-c", scripts[scenario]], **kwargs)
        process.append(child)
        return child

    monkeypatch.setattr(probe.subprocess, "Popen", fixture_popen)
    monkeypatch.setattr(probe, "CHILD_SECONDS", 1)
    monkeypatch.setattr(
        probe.campaign, "live_gpu", lambda pid: {"synthetic_cpu_process": pid}
    )
    probe.write_json(tmp_path / "seed-653.json", {"seed": 653, "synthetic_only": True})
    fd = os.open(tmp_path / "fixture-inherited-fd", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if scenario == "success":
            result = probe._run_child(
                SOURCE, "d" * 64, 653, fd, dict(os.environ), tmp_path
            )
            assert result["synthetic_only"] and result["live_gpu_samples"]
        else:
            with pytest.raises((ValueError, TimeoutError)):
                probe._run_child(SOURCE, "d" * 64, 653, fd, dict(os.environ), tmp_path)
        assert len(process) == 1 and process[0].poll() is not None
        assert (tmp_path / "seed-653.log").stat().st_size <= probe.LOG_LIMIT
        if scenario == "timeout":
            assert (
                b"retained before timeout" in (tmp_path / "seed-653.log").read_bytes()
            )
        with pytest.raises(FileExistsError):
            probe._run_child(SOURCE, "d" * 64, 653, fd, dict(os.environ), tmp_path)
    finally:
        os.close(fd)


def test_independent_closeout_requires_successful_original_terminal_service(
    monkeypatch,
):
    state = {
        "MainPID": "0",
        "ActiveState": "inactive",
        "NRestarts": "0",
        "ExecMainStatus": "0",
        "Result": "success",
        "InvocationID": ID,
    }
    monkeypatch.setattr(probe.host, "read", lambda *command: state[command[-2]])
    assert probe._completed_run_service(SOURCE, ID) == state
    state["InvocationID"] = ""  # Successful transient-unit GC is not a new invocation.
    assert probe._completed_run_service(SOURCE, ID) == state
    for key, value in (
        ("Result", "timeout"),
        ("ExecMainStatus", "1"),
        ("NRestarts", "1"),
        ("MainPID", "1234"),
        ("InvocationID", "c" * 32),
    ):
        original = state[key]
        state[key] = value
        with pytest.raises(ValueError):
            probe._completed_run_service(SOURCE, ID)
        state[key] = original


def test_cpu_validator_returns_binding_not_none(monkeypatch):
    """Regression for the retained failed native wrapper assertion."""
    expected = probe.preparation.cpu_parent_binding(SOURCE, "e" * 64)
    calls = []

    def validator(receipt, digest, binding, *, source):
        calls.append((receipt, digest, binding, source))
        return deepcopy(expected)

    monkeypatch.setattr(probe.preparation, "validate_cpu_parent_receipt", validator)
    assert (
        probe._validated_cpu_binding(SOURCE, {"synthetic": True}, "e" * 64) == expected
    )
    assert len(calls) == 1
    monkeypatch.setattr(
        probe.preparation, "validate_cpu_parent_receipt", lambda *a, **k: None
    )
    with pytest.raises(ValueError, match="actual validated CPU binding"):
        probe._validated_cpu_binding(SOURCE, {}, "e" * 64)


def test_preflight_has_its_own_fixed_source_namespace_and_service_caps():
    assert probe.MODE_SECONDS["preflight"] == 180
    assert probe.MODE_MEMORY["preflight"] == 2 * 1024**3
    assert probe.preflight_path(SOURCE).name == "cuda64-prerequisites-" + SOURCE[:12]
    assert (
        probe.service_name(SOURCE, "preflight")
        == "microduck-cuda64-policy-preflight-" + SOURCE[:12] + ".service"
    )
    assert probe.FAILED_PREFLIGHT_SOURCE == "7b234e6fb49d2f8cfb93a6383d24d418109c74bd"
    assert probe.FAILED_PREFLIGHT_INVOCATION == "cdc81099961d4da49d5e2be06f3b987d"


def test_preserved_native_preflight_failure_is_authenticated_not_relabelled(
    monkeypatch, tmp_path
):
    native = (
        probe.execution.ROOT / "artifacts/tools/cuda64-preflight-failure-7b234e6fb49d"
    )
    mirror = (
        Path(__file__).resolve().parents[1]
        / "artifacts/retained"
        / "cuda64-preflight-failure-7b234e6fb49d.cq9z31"
    )
    original = native if native.is_dir() else mirror
    if not original.is_dir():
        assert not probe.execution.ROOT.is_dir(), (
            "original failed native receipt is mandatory on the host"
        )
        pytest.skip(
            "optional failure evidence is unavailable outside its authorized hosts"
        )
    root = tmp_path / "artifacts/tools/cuda64-preflight-failure-7b234e6fb49d"
    root.mkdir(parents=True)
    for name in ("receipt.json", "journal.log"):
        shutil.copyfile(original / name, root / name)
    state = {
        "MainPID": "0",
        "ActiveState": "failed",
        "Result": "exit-code",
        "ExecMainStatus": "1",
        "NRestarts": "0",
        "InvocationID": probe.FAILED_PREFLIGHT_INVOCATION,
    }
    monkeypatch.setattr(probe.execution, "ROOT", tmp_path)
    monkeypatch.setattr(probe.host, "read", lambda *command: state[command[-2]])
    result = probe._previous_preflight_failure_binding()
    assert result["receipt_sha256"] == probe.FAILED_PREFLIGHT_RECEIPT_SHA256
    assert result["service"]["ActiveState"] == "failed"
    state["ActiveState"] = "inactive"
    with pytest.raises(ValueError, match="original sixth failed"):
        probe._previous_preflight_failure_binding()
    state["ActiveState"] = "failed"
    with (root / "journal.log").open("ab") as stream:
        stream.write(b"relabelled")
    with pytest.raises(ValueError, match="whole original preflight failure"):
        probe._previous_preflight_failure_binding()
