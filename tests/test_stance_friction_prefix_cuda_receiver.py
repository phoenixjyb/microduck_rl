"""Mocked receiver contracts only; no CUDA artifact or execution qualification."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from mjlab_microduck import stance_friction_prefix_cuda_receiver as r


def anchor(raw):
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def test_authenticates_whole_files_before_decode(tmp_path):
    raw = b"not JSON but authenticated whole bytes"
    (tmp_path / "raw.json").write_bytes(raw)
    assert r.authenticate(tmp_path, {"raw.json": anchor(raw)}) == {"raw.json": raw}
    with pytest.raises(json.JSONDecodeError):
        r.decode(raw)


@pytest.mark.parametrize(
    "mutation",
    ["length", "digest", "symlink", "traversal", "cap", "directory", "bool", "extra"],
)
def test_authentication_rejects_corrupt_anchor_or_path(tmp_path, mutation):
    raw = b'{"x":1}'
    path = tmp_path / "x.json"
    path.write_bytes(raw)
    info, name, cap = anchor(raw), "x.json", 100
    if mutation == "length":
        info["bytes"] -= 1
    elif mutation == "digest":
        info["sha256"] = "f" * 64
    elif mutation == "symlink":
        (tmp_path / "link").symlink_to(path)
        name = "link"
    elif mutation == "traversal":
        name = "../x.json"
    elif mutation == "cap":
        cap = 1
    elif mutation == "directory":
        name = "sub"
        (tmp_path / name).mkdir()
    elif mutation == "bool":
        info["bytes"] = True
    else:
        info["ignored"] = 1
    with pytest.raises(ValueError):
        r.authenticate(tmp_path, {name: info}, cap=cap)


@pytest.mark.parametrize(
    "raw", [b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}']
)
def test_packet_decode_rejects_ambiguous_json(raw):
    with pytest.raises(ValueError):
        r.decode(raw)


@pytest.mark.parametrize("value", [True, False, 1.0, "1", -1, 2**64])
def test_plain_integer_rejects_coercion(value):
    with pytest.raises(ValueError):
        r.plain(value)


def test_source_binding_requires_external_whole_anchor():
    source = {
        "source": "a" * 40,
        "tree": "b" * 40,
        "branch": r.BRANCH,
        "leaves": {"source.py": {"bytes": 3, "sha256": "c" * 64, "git_blob": "d" * 40}},
    }
    r.source(source, deepcopy(source))
    changed = deepcopy(source)
    changed["leaves"]["source.py"]["sha256"] = "e" * 64
    with pytest.raises(ValueError, match="externally pinned"):
        r.source(changed, source)


@pytest.mark.parametrize("key", list(r.FLAGS))
def test_no_admission_flag_can_be_promoted(key):
    flags = dict(r.FLAGS)
    flags[key] = True
    with pytest.raises(ValueError, match="literally false"):
        r.flags(flags)


def device():
    return {
        "alias": "cuda:0",
        "object_id": 10,
        "context": 11,
        "arch": 120,
        "name": "mock Blackwell",
        "stream": 12,
        "stream_object_id": 13,
        "toolkit_version": [12, 9],
        "driver_version": [13, 0],
    }


def allocation(shape, kind="float32"):
    strides, size = r.contiguous(shape)
    vector = kind in ("solref", "solimp")
    dtype = (
        "<class 'warp._src.types.vec2f'>"
        if kind == "solref"
        else "<class 'mujoco_warp._src.types.vec5f'>"
        if kind == "solimp"
        else "<class 'warp._src.types.int32'>"
        if kind == "int32"
        else "<class 'warp._src.types.float32'>"
    )
    return {
        "object_id": 101,
        "pointer": 4096,
        "device": "cuda:0",
        "context": 11,
        "warp_dtype": dtype,
        "shape": shape[:-1] if vector else shape,
        "strides": strides[:-1] if vector else strides,
        "bytes": size,
        "host_shape": shape,
        "host_dtype": "int32" if kind == "int32" else "float32",
    }


@pytest.mark.parametrize(
    "shape,kind",
    [
        ([3, 20], "float32"),
        ([4, 20, 2], "solref"),
        ([1, 20, 5], "solimp"),
        ([2, 1, 640], "int32"),
    ],
)
def test_scalar_vector_buffers_have_exact_contiguous_bytes(shape, kind):
    row = allocation(shape, kind)
    assert r.allocation(row, shape, kind, device()) == (4096, 4096 + row["bytes"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("object_id", True),
        ("pointer", 0),
        ("context", 12),
        ("device", "cpu"),
        ("warp_dtype", "float32"),
        ("shape", [2, 20, 2]),
        ("strides", [1, 1]),
        ("bytes", 319),
        ("host_shape", [True, 20, 2]),
        ("host_dtype", "float64"),
    ],
)
def test_buffer_metadata_mutation_fails_closed(key, value):
    row = allocation([2, 20, 2], "solref")
    row[key] = value
    with pytest.raises(ValueError):
        r.allocation(row, [2, 20, 2], "solref", device())


def compiled_fixture():
    """Construct mock file-load records, not a retained native execution."""
    directory = r.ROOT + "/artifacts/evaluations/friction-prefix-cuda-run-" + "a" * 12
    raw, inventory, result = {}, {}, {}
    for index, role in enumerate(("original", "candidate")):
        root = Path(directory) / ("compiled-" + role)
        symbol = role + "_cuda_kernel_forward"
        bodies = {
            "generated_source": (role + ".cu", b"mock generated source"),
            "binary": (role + ".cubin", b"\x7fELFmock"),
            "metadata": (
                role + ".meta",
                json.dumps({symbol + "_smem_bytes": 0}).encode(),
            ),
        }
        records = {}
        for kind, (name, body) in bodies.items():
            relative = "compiled-" + role + "/" + name
            raw[relative], inventory[relative] = body, anchor(body)
            records[kind] = {"path": str(root / name), **anchor(body)}
        ids = {
            "kernel": 20 + index * 10,
            "module": 21 + index * 10,
            "device": 10,
            "executable": 22 + index * 10,
            "hooks": 23 + index * 10,
        }
        binding = {
            "protocol": "microduck-retained-cuda-artifact-binding-oct7-v1",
            "artifact_format": "cubin",
            "device": "cuda:0",
            "device_arch": 120,
            "context": 11,
            "block_dim": 256,
            "module_hash": "e" * 64,
            "module_handle": 100 + index,
            "forward_handle": 200 + index,
            "forward_smem_bytes": 0,
            "symbol": symbol,
            "observed_object_ids": ids,
        }
        binding.update(
            {
                key: False
                for key in (
                    "driver_jit_machine_code_observed",
                    "loaded_binary_bytes_observed",
                    "native_execution_qualified",
                    "training_authorized",
                    "physical_acceptance",
                )
            }
        )
        for kind in ("binary", "metadata"):
            for key in ("path", "bytes", "sha256"):
                binding[kind + "_" + key] = records[kind][key]
        result[role] = {
            "compiler_config": dict(r.COMPILER_CONFIG),
            "binding": binding,
            "module_options": {
                "block_dim": 256,
                "enable_backward": False,
                "strip_hash": False,
            },
            "output_arch": 120,
            **records,
            "explicit_load": {
                "module_object_id": ids["module"],
                "returned_executable_id": ids["executable"],
                "device_object_id": 10,
                "block_dim": 256,
                "binary_path": records["binary"]["path"],
                "meta_path": records["metadata"]["path"],
                "output_arch": 120,
                "fresh_cache_before": True,
            },
            "runtime_entries": {
                name: {"object_id": 300 + i, "code_id": 400 + i, "qualname": name}
                for i, name in enumerate(
                    (
                        "launch",
                        "load",
                        "compile",
                        "hooks",
                        "hash",
                        "load_cuda",
                        "build_cuda",
                        "synchronize",
                    )
                )
            },
        }
    return result, raw, inventory, directory


def test_mock_compiled_files_match_explicit_load_and_hooks():
    value, raw, inventory, directory = compiled_fixture()
    r.compiled(value, raw, inventory, directory, device())


@pytest.mark.parametrize(
    "mutation",
    [
        "load-cache",
        "load-result",
        "binding-digest",
        "backward",
        "arch",
        "symbol",
        "metadata",
        "runtime-code",
        "same-module",
        "foreign-path",
        "jit-claim",
        "elf",
        "pch",
    ],
)
def test_mock_compilation_witness_rejects_mismatch(mutation):
    value, raw, inventory, directory = compiled_fixture()
    row = value["candidate"]
    if mutation == "load-cache":
        row["explicit_load"]["fresh_cache_before"] = False
    elif mutation == "load-result":
        row["explicit_load"]["returned_executable_id"] += 1
    elif mutation == "binding-digest":
        row["binding"]["binary_sha256"] = "d" * 64
    elif mutation == "backward":
        row["module_options"]["enable_backward"] = True
    elif mutation == "arch":
        row["output_arch"] = 100
    elif mutation == "symbol":
        row["binding"]["symbol"] = "wrong_cuda_kernel_forward"
    elif mutation == "metadata":
        raw["compiled-candidate/candidate.meta"] = (
            b'{"candidate_cuda_kernel_forward_smem_bytes":true}'
        )
    elif mutation == "runtime-code":
        row["runtime_entries"]["launch"]["code_id"] += 1
    elif mutation == "same-module":
        row["binding"]["observed_object_ids"]["module"] = value["original"]["binding"][
            "observed_object_ids"
        ]["module"]
    elif mutation == "foreign-path":
        row["binary"]["path"] = "/tmp/candidate.cubin"
    elif mutation == "jit-claim":
        row["binding"]["driver_jit_machine_code_observed"] = True
    elif mutation == "pch":
        row["compiler_config"]["use_precompiled_headers"] = True
    else:
        raw["compiled-candidate/candidate.cubin"] = b"notELF"
    with pytest.raises(ValueError):
        r.compiled(value, raw, inventory, directory, device())


def test_independent_receiver_never_imports_cuda_runtime():
    import ast

    tree = ast.parse(Path(r.__file__).read_text())
    names = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    names += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any(name and name.startswith(("warp", "mujoco_warp")) for name in names)
    assert not any(name and name.endswith("cuda_probe") for name in names)


def mock_component_matrix():
    """Plain mocked decisions only; no producer, artifacts, or CUDA runtime."""
    overflow_cases = {"overflow", "maximum"}
    return {
        name: {
            **{key: True for key in r.STRUCTURAL_KEYS},
            "component_exact_without_overflow": name not in overflow_cases,
            "overflow_negative": name in overflow_cases,
            "overflow_cases_match_pinned_matrix": True,
        }
        for name in r.CASES
    }


def test_mock_component_matrix_matches_predeclaration():
    assert r.matrix_matches(mock_component_matrix()) is True


@pytest.mark.parametrize("case", ["overflow", "maximum"])
@pytest.mark.parametrize("key", list(r.STRUCTURAL_KEYS))
def test_overflow_case_structural_failure_is_not_hidden(case, key):
    cases = mock_component_matrix()
    cases[case][key] = False
    assert r.matrix_matches(cases) is False


def test_mock_component_matrix_rejects_wrong_overflow_classification():
    cases = mock_component_matrix()
    cases["overflow"]["component_exact_without_overflow"] = True
    assert r.matrix_matches(cases) is False


def test_mock_component_matrix_requires_literal_boolean_structural_flags():
    cases = mock_component_matrix()
    cases["all-dofs"][r.STRUCTURAL_KEYS[0]] = 1
    assert r.matrix_matches(cases) is False


def mock_environment_alias():
    return {
        "path": r.ROOT + "/.venv",
        "target": r.VENV_TARGET,
        "device": 1,
        "inode": 1794097,
        "bytes": len(r.VENV_TARGET.encode()),
        "mtime_ns": 2,
        "ctime_ns": 3,
    }


def test_receiver_accepts_only_declared_existing_environment_alias():
    record = mock_environment_alias()
    assert r.environment(record) is record


@pytest.mark.parametrize(
    "field", ["path", "target", "device", "inode", "bytes", "mtime_ns", "ctime_ns"]
)
def test_receiver_rejects_environment_alias_substitution(field):
    record = mock_environment_alias()
    record[field] = "other" if field in ("path", "target") else True
    with pytest.raises(ValueError):
        r.environment(record)
