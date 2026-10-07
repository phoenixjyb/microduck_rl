"""Synthetic whole-evidence receiver negatives; not native CUDA evidence."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab_microduck import stance_com_coupled_receiver as r
from mjlab_microduck import stance_com_coupled_probe as p

SOURCE = "a" * 40


def fixture(tmp_path):
    # Authenticate the retained native descriptor before using it in synthetic packets.
    prior = Path(
        "artifacts/evaluations/com-entry-run-cbcc4d88ca3f/child.json"
    ).read_bytes()
    assert (
        sha256(prior).hexdigest()
        == "56370eb3546f19a395a89cf864fc83e25d0690413b6bfbcee459b54b151941b9"
    )
    descriptor = json.loads(prior)["compiled_descriptor"]
    chunks = []
    for name, shape in r.frames.FRAME_FIELDS:
        value = np.zeros((64, *shape), dtype="<f4")
        if name == "qpos":
            value[:] = np.asarray(descriptor["initial_qpos"], dtype="<f4")
        chunks.append(value.tobytes())
    packet = b"".join(chunks) * 2
    entry = b"\0" * r.ENTRY_BYTES
    state_cpu = {
        str(seed): torch.Generator(device="cpu")
        .manual_seed(seed)
        .get_state()
        .numpy()
        .tobytes()
        for seed in (673, 977)
    }
    states = {
        key: state_cpu["673" if key.startswith("caller") else "977"]
        if "cpu" in key
        else (677 if key.startswith("caller") else 983).to_bytes(8, "little")
        + b"\0" * 8
        for key in r.RNG_KEYS
    }
    rng = b"".join(states[k] for k in r.RNG_KEYS)
    metadata = {
        k: dict(bytes=len(v), sha256=sha256(v).hexdigest()) for k, v in states.items()
    }
    binding = dict(
        source=SOURCE,
        branch="feat/athletics-obstacle-curriculum",
        tree="c" * 40,
        leaf_count=640,
        leaves_sha256="d" * 64,
    )
    packages = dict(
        versions=r.old.VERSIONS,
        python="3.12.13",
        architecture="x86_64",
        python_trees=r.PACKAGE_TREES,
    )
    live = dict(
        r.old.SERVICE_CAPS,
        MainPID="100",
        InvocationID="1" * 32,
        ActiveState="active",
        SubState="running",
    )
    terminal = dict(
        r.old.SERVICE_CAPS,
        MainPID="0",
        InvocationID="1" * 32,
        ActiveState="active",
        SubState="exited",
        Result="success",
        ExecMainStatus="0",
    )
    test_caps = {**r.old.SERVICE_CAPS, "LimitFSIZE": "67108864"}
    test_terminal = dict(
        test_caps,
        MainPID="0",
        InvocationID="2" * 32,
        ActiveState="active",
        SubState="exited",
        Result="success",
        ExecMainStatus="0",
    )
    test_receipt = dict(
        protocol=r.DATA_PROTOCOL + ":tests",
        source_binding=binding,
        packages=packages,
        tests=r.EXPECTED_TESTS,
        test_files=p.test_files(),
        retained_descriptor_fixture=dict(
            path=p.DESCRIPTOR_FIXTURE,
            bytes=9948,
            sha256=p.DESCRIPTOR_FIXTURE_SHA256,
        ),
        flags=r.old.FLAGS,
        service_properties=dict(
            test_caps,
            MainPID="99",
            InvocationID="2" * 32,
            ActiveState="active",
            SubState="running",
        ),
        test_environment={
            "CUDA_VISIBLE_DEVICES": "",
            "MICRODUCK_STANCE_PROFILE": "wsl-10098-20260930",
            "ATEN_CPU_CAPABILITY": "default",
            "MKL_CBWR": "COMPATIBLE",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        },
    )
    test_raw = r.old.canonical(test_receipt)
    host = dict(
        machine=r.old.MACHINE,
        gpu=r.old.GPU,
        driver="595.95",
        driver_model="WDDM",
        temperature_c=30,
        used_mib=640,
        free_mib=23500,
        processes=[],
    )
    declaration = dict(
        protocol=r.DATA_PROTOCOL + ":declaration",
        source=SOURCE,
        source_binding=binding,
        packages=packages,
        flags=r.old.FLAGS,
        service_properties=live,
        host=host,
        expected_tests=r.EXPECTED_TESTS,
        tests=dict(
            count=r.EXPECTED_TESTS,
            files=63,
            source_binding=binding,
            receipt_sha256=sha256(test_raw).hexdigest(),
            terminal=test_terminal,
        ),
    )
    cases = {}
    files = {
        "tests.receipt.json": test_raw,
        "tests.terminal.json": r.old.canonical(test_terminal),
        "declaration.json": r.old.canonical(declaration),
        "child.log": b"synthetic fixture, not CUDA execution\n",
    }
    for case in r.frames.CASES:
        mode = "original" if case.startswith("original") else "serial"
        calls = [
            dict(
                call_index=i,
                worlds=64,
                bytes=r.ENTRY_BYTES,
                original_launches=11,
                sha256=sha256(entry).hexdigest(),
                device="cuda:0",
                stream=0,
                boundary="after-init-before-first-accumulation",
                initialization_and_accumulation_output_alias=True,
                readback_and_device_sync_perturb_timing=True,
                smooth_sha256="63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f",
                forward_sha256="c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3",
                subtree_com_layout=dict(
                    object_id=1,
                    ptr=2,
                    shape=[64, 16],
                    strides=[192, 12],
                    dtype="vec3f",
                    contiguous=True,
                ),
            )
            for i in (0, 1)
        ]
        com = dict(
            protocol="microduck-com-actual-entry-capture-v1:native-coupled-control",
            status="complete",
            fault_type=None,
            parents=list(r.old.PARENTS),
            reversed_levels=[
                [6, 15],
                [5, 10, 14],
                [4, 9, 13],
                [3, 8, 12],
                [2, 7, 11],
                [1],
                [0],
            ],
            flags=r.old.FLAGS,
            mode=mode,
            dispatched_original_kernel_counts=[13 if mode == "serial" else 11] * 2,
            split_body_ids=[2, 7, 11] if mode == "serial" else [],
            complete_kernel_count_matches=True,
            weighted_boundary="after-accumulation-before-original-division",
            weighted_boundary_readback_perturbs_timing=True,
            weighted_sha256=[sha256(entry).hexdigest()] * 2,
            calls=calls,
        )
        crb = dict(
            protocol="football-b1d-crb-runtime-control-v1",
            status="complete",
            mode="serial",
            fault_type=None,
            constructor_forward_covered=True,
            runtime_bound=True,
            initialization_timing_changed=True,
            max_forward_calls=2,
            topology_id_snapshots=2,
            singleton_child_arrays_allocated=3,
            constructor_forward_calls=1,
            forward_calls=2,
            original_level_launch_requests=14,
            parent_zero_noop_level_requests=4,
            actual_accumulation_launches=18,
            split_child_launches=6,
            dense_qM_launches=2,
            smooth_source_sha256=calls[0]["smooth_sha256"],
            forward_source_sha256=calls[0]["forward_sha256"],
            flags={
                k: False
                for k in (
                    "native_qualified",
                    "original_pair_accepted",
                    "cause_proven",
                    "full_window_passed",
                    "training_authorized",
                    "physical_result_accepted",
                )
            },
        )
        cases[case] = dict(
            mode=mode,
            compiled_descriptor=descriptor,
            ntendon=0,
            fixed_state_unchanged=True,
            physics_steps=0,
            com_scope=com,
            crb_scope=crb,
            rng_metadata=metadata,
        )
        files.update(
            {
                f"{case}.frames.bin": packet,
                f"{case}.entries.bin": entry * 2,
                f"{case}.weighted.bin": entry * 2,
                f"{case}.rng.bin": rng,
            }
        )
    child = dict(
        protocol=r.DATA_PROTOCOL + ":child",
        source=SOURCE,
        source_binding=binding,
        packages=packages,
        flags=r.old.FLAGS,
        owner_pid=100,
        child_pid=101,
        child_ppid=100,
        declaration_sha256=sha256(files["declaration.json"]).hexdigest(),
        case_order=list(r.frames.CASES),
        cases=cases,
        integration_calls=0,
        physics_steps=0,
        graph_created=False,
        actor_model_created=False,
        optimizer_created=False,
        storage_created=False,
    )
    files["child.json"] = r.old.canonical(child)
    report = dict(
        protocol=r.DATA_PROTOCOL + ":report",
        source=SOURCE,
        source_binding=binding,
        flags=r.old.FLAGS,
        returncode=0,
        gpu_child_observed=True,
        observed_child_pid=101,
        observed_child_ppid=100,
        child_sha256=sha256(files["child.json"]).hexdigest(),
        monitor=[
            dict(
                elapsed=1.0,
                child_pid=101,
                host={
                    **host,
                    "used_mib": 1024,
                    "free_mib": 23000,
                    "processes": [
                        dict(
                            pid=101,
                            memory_mib=None,
                            memory_status="unavailable-wddm",
                            raw_memory="N/A",
                        )
                    ],
                },
            )
        ],
    )
    files["report.json"] = r.old.canonical(report)
    for name, raw in files.items():
        (tmp_path / name).write_bytes(raw)
    return files, r.old.canonical(terminal)


def anchors(files):
    return {
        name: dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
        for name, raw in files.items()
    }


def verify(path, files, terminal):
    return r.verify_directory(
        path, anchors(files), terminal, sha256(terminal).hexdigest(), source=SOURCE
    )


def rewrite(path, files, *, child=None, report=None, declaration=None, test=None):
    if test is not None:
        files["tests.receipt.json"] = r.old.canonical(test)
        if declaration is None:
            declaration = json.loads(files["declaration.json"])
        declaration["tests"]["receipt_sha256"] = sha256(
            files["tests.receipt.json"]
        ).hexdigest()
    if declaration is not None:
        files["declaration.json"] = r.old.canonical(declaration)
        if child is None:
            child = json.loads(files["child.json"])
        child["declaration_sha256"] = sha256(files["declaration.json"]).hexdigest()
    if child is not None:
        files["child.json"] = r.old.canonical(child)
        if report is None:
            report = json.loads(files["report.json"])
        report["child_sha256"] = sha256(files["child.json"]).hexdigest()
    if report is not None:
        files["report.json"] = r.old.canonical(report)
    for name, raw in files.items():
        (path / name).write_bytes(raw)


def test_whole_synthetic_inventory_recomputes_all_rows(tmp_path):
    files, terminal = fixture(tmp_path)
    result = verify(tmp_path, files, terminal)
    assert result["decision"] == "fresh-coupled-control-exact"
    assert (
        result["actual_initialized_entries_exact"] is True
        and result["serial_weighted_arithmetic_exact"] is True
    )
    assert (
        sum(len(row["fields"]) for row in result["frame_comparison"]["comparisons"])
        == 204
    )
    assert len(result["receiver_files"]) == 22 and all(
        v is False for v in result["flags"].values()
    )


@pytest.mark.parametrize(
    "name", ["serial1.frames.bin", "child.json", "tests.receipt.json"]
)
def test_all_artifact_hashes_precede_any_json_or_array_parse(
    tmp_path, monkeypatch, name
):
    files, terminal = fixture(tmp_path)
    bad = anchors(files)
    bad[name]["sha256"] = "0" * 64
    monkeypatch.setattr(
        r.old, "_json", lambda *_: pytest.fail("JSON parsed before all hashes")
    )
    monkeypatch.setattr(
        r.np,
        "frombuffer",
        lambda *_a, **_k: pytest.fail("array decoded before all hashes"),
    )
    with pytest.raises(ValueError, match="whole external"):
        r.verify_directory(
            tmp_path, bad, terminal, sha256(terminal).hexdigest(), source=SOURCE
        )


@pytest.mark.parametrize(
    "key",
    [
        "integration_calls",
        "physics_steps",
        "graph_created",
        "actor_model_created",
        "optimizer_created",
        "storage_created",
    ],
)
def test_false_and_zero_admission_fields_cannot_be_coerced(tmp_path, key):
    files, terminal = fixture(tmp_path)
    child = json.loads(files["child.json"])
    child[key] = False if key.endswith("calls") or key == "physics_steps" else 0
    rewrite(tmp_path, files, child=child)
    with pytest.raises(ValueError):
        verify(tmp_path, files, terminal)


@pytest.mark.parametrize(
    "field",
    ["mode", "dispatch-count", "alias", "compiled-model", "crb-count", "rng-length"],
)
def test_native_case_semantics_are_not_self_reported_acceptance(tmp_path, field):
    files, terminal = fixture(tmp_path)
    child = json.loads(files["child.json"])
    row = child["cases"]["serial1"]
    if field == "mode":
        row["mode"] = "original"
    if field == "dispatch-count":
        row["com_scope"]["dispatched_original_kernel_counts"] = [11, 11]
    if field == "alias":
        row["com_scope"]["calls"][1]["subtree_com_layout"]["ptr"] += 1
    if field == "compiled-model":
        row["compiled_descriptor"]["initial_qpos"][2] += 1
    if field == "crb-count":
        row["crb_scope"]["actual_accumulation_launches"] = 14
    if field == "rng-length":
        row["rng_metadata"]["private_cpu_start"]["bytes"] -= 1
    rewrite(tmp_path, files, child=child)
    with pytest.raises(ValueError):
        verify(tmp_path, files, terminal)


def test_constant_wrong_serial_weighted_output_remains_negative(tmp_path):
    files, terminal = fixture(tmp_path)
    child = json.loads(files["child.json"])
    values = np.ones((2, 64, 16, 3), dtype="<f4")
    raw = values.tobytes()
    for case in ("serial0", "serial1"):
        files[f"{case}.weighted.bin"] = raw
        child["cases"][case]["com_scope"]["weighted_sha256"] = [
            sha256(v.tobytes()).hexdigest() for v in values
        ]
    rewrite(tmp_path, files, child=child)
    result = verify(tmp_path, files, terminal)
    assert (
        result["decision"] == "fresh-coupled-control-negative"
        and result["serial_weighted_arithmetic_exact"] is False
    )
    assert result["weighted_comparisons"]["serial0"][0]["bit_mismatch_scalars"] == 3072


def test_later_bias_negative_cannot_be_waived_by_exact_weighted_sums(tmp_path):
    files, terminal = fixture(tmp_path)
    value = np.frombuffer(files["serial1.frames.bin"], dtype="<f4").copy()
    offset = r.frames.FRAME_BYTES // 4 + sum(
        64 * int(np.prod(s) if s else 1) for n, s in r.frames.FRAME_FIELDS[:8]
    )
    value[offset] = 1
    files["serial1.frames.bin"] = value.tobytes()
    rewrite(tmp_path, files)
    result = verify(tmp_path, files, terminal)
    assert (
        result["serial_weighted_arithmetic_exact"] is True
        and result["decision"] == "fresh-coupled-control-negative"
    )
    rows = result["frame_comparison"]["comparisons"]
    assert (
        next(
            row
            for row in rows
            if row["left"] == "serial0"
            and row["right"] == "serial1"
            and row["boundary"] == "eager_forward"
        )["fields"]["qfrc_bias"]["bit_mismatch_scalars"]
        == 1
    )


@pytest.mark.parametrize(
    "change",
    [
        "failed",
        "same-invocation",
        "gpu-file-cap",
        "subset-count",
        "wrong-environment",
        "different-files",
        "wrong-descriptor",
        "bad-invocation",
    ],
)
def test_whole_cpu_prerequisite_is_separate_completed_and_exact(tmp_path, change):
    files, terminal = fixture(tmp_path)
    declaration = json.loads(files["declaration.json"])
    test = json.loads(files["tests.receipt.json"])
    if change == "failed":
        declaration["tests"]["terminal"]["Result"] = "failed"
    if change == "same-invocation":
        declaration["tests"]["terminal"]["InvocationID"] = "1" * 32
    if change == "gpu-file-cap":
        test["service_properties"]["LimitFSIZE"] = "1048576"
    if change == "subset-count":
        test["tests"] -= 1
    if change == "wrong-environment":
        test["test_environment"]["CUDA_VISIBLE_DEVICES"] = "0"
    if change == "different-files":
        test["test_files"][0] = "tests/unrelated.py"
    if change == "wrong-descriptor":
        test["retained_descriptor_fixture"]["sha256"] = "0" * 64
    if change == "bad-invocation":
        declaration["tests"]["terminal"]["InvocationID"] = "not-an-invocation"
        test["service_properties"]["InvocationID"] = "not-an-invocation"
    rewrite(tmp_path, files, test=test, declaration=declaration)
    with pytest.raises(ValueError):
        verify(tmp_path, files, terminal)


@pytest.mark.parametrize(
    "change",
    [
        "foreign-pid",
        "missing-child",
        "temperature",
        "global-memory",
        "zero-per-pid-memory",
    ],
)
def test_actual_gpu_ownership_temperature_and_memory_are_gates(tmp_path, change):
    files, terminal = fixture(tmp_path)
    report = json.loads(files["report.json"])
    sample = report["monitor"][0]["host"]
    if change == "foreign-pid":
        sample["processes"][0]["pid"] = 102
    if change == "missing-child":
        sample["processes"] = []
    if change == "temperature":
        sample["temperature_c"] = 75
    if change == "global-memory":
        sample["used_mib"] = 13000
    if change == "zero-per-pid-memory":
        sample["processes"][0].update(
            memory_mib=0, memory_status="reported", raw_memory="0"
        )
    rewrite(tmp_path, files, report=report)
    with pytest.raises(ValueError):
        verify(tmp_path, files, terminal)


def test_extra_file_and_symlink_are_not_authenticated_prefixes(tmp_path):
    files, terminal = fixture(tmp_path)
    (tmp_path / "extra").write_bytes(b"x")
    with pytest.raises(ValueError, match="inventory"):
        verify(tmp_path, files, terminal)
    (tmp_path / "extra").unlink()
    name = "serial1.frames.bin"
    (tmp_path / name).unlink()
    (tmp_path / name).symlink_to(tmp_path / "serial0.frames.bin")
    with pytest.raises(OSError):
        verify(tmp_path, files, terminal)
