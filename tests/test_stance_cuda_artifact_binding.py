"""Pure-file/mock guards only: no CUDA loading, compilation or launches."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mjlab_microduck import stance_cuda_artifact_binding as binding


def _anchor(raw):
    return len(raw), sha256(raw).hexdigest()


@pytest.fixture
def artifacts(tmp_path):
    binary = tmp_path / "original.ptx"
    meta = tmp_path / "original.meta"
    binary.write_bytes(b"// synthetic test bytes, never executable\n")
    meta.write_bytes(b'{"fixture_cuda_kernel_forward_smem_bytes":32}\n')
    size, digest = _anchor(binary.read_bytes())
    meta_size, meta_digest = _anchor(meta.read_bytes())
    options = dict(
        binary_size=size,
        binary_sha256=digest,
        metadata_size=meta_size,
        metadata_sha256=meta_digest,
    )
    return binary, meta, options


class Device:
    is_cuda = True
    is_capturing = False
    context = 9876
    arch = 120
    max_shared_memory_per_block = 1024

    def __str__(self):
        return "cuda:0"


def _objects():
    device = Device()
    module_hash = bytes(range(32))
    module = SimpleNamespace(
        execs={},
        failed_builds=set(),
        options={"block_dim": 256, "strip_hash": False, "enable_backward": False},
        get_module_hash=lambda block: module_hash,
    )
    kernel = SimpleNamespace(
        module=module,
        adj=object(),
        get_mangled_name=lambda: "fixture",
        options={"enable_backward": False},
    )
    hooks = SimpleNamespace(
        forward=12345, backward=None, forward_smem_bytes=32, backward_smem_bytes=0
    )
    executable = SimpleNamespace(
        handle=67890,
        device=device,
        module_hash=module_hash,
        kernel_hooks={kernel.adj: hooks},
        meta={"fixture_cuda_kernel_forward_smem_bytes": 32},
    )
    return device, module, kernel, executable, hooks, module_hash


def _loaded(artifacts):
    binary, meta, options = artifacts
    artifact = binding.retain_artifact(binary, meta, **options)
    device, module, kernel, executable, hooks, module_hash = _objects()
    binding.assert_fresh_module(module, device, 256)
    # A fake cache entry exercises guards; no loader call or native evidence.
    module.execs[(device.context, 256)] = executable
    held = binding.bind_loaded_module(
        artifact,
        kernel,
        device,
        executable,
        hooks,
        block_dim=256,
        expected_module_hash=module_hash,
    )
    return held, device, module, kernel, executable, hooks


def test_record_is_non_admitting_and_complete_for_artifact_identity(artifacts):
    held, *_ = _loaded(artifacts)
    record = held.record()
    assert record["artifact_format"] == "ptx"
    assert record["binary_path"] == str(artifacts[0])
    assert record["metadata_bytes"] == artifacts[2]["metadata_size"]
    assert record["symbol"] == "fixture_cuda_kernel_forward"
    assert record["device_arch"] == 120
    for key in (
        "driver_jit_machine_code_observed",
        "loaded_binary_bytes_observed",
        "native_execution_qualified",
        "training_authorized",
        "physical_acceptance",
    ):
        assert record[key] is False
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize(
    "field", ["binary_size", "binary_sha256", "metadata_size", "metadata_sha256"]
)
def test_wrong_external_anchor_rejected_before_metadata_decode(artifacts, field):
    binary, meta, options = artifacts
    options[field] = options[field] + 1 if field.endswith("size") else "0" * 64
    with pytest.raises(ValueError):
        binding.retain_artifact(binary, meta, **options)


@pytest.mark.parametrize(
    "raw",
    [
        b"[]",
        b"null",
        b'{"a":1,"a":2}',
        b"not json",
        b'{"a":NaN}',
        b'{"a":1}',
        b'{"fixture_cuda_kernel_forward_smem_bytes":true}',
        b'{"fixture_cuda_kernel_forward_smem_bytes":-1}',
    ],
)
def test_invalid_metadata_rejected_after_whole_authentication(artifacts, raw):
    binary, meta, options = artifacts
    meta.write_bytes(raw)
    options["metadata_size"], options["metadata_sha256"] = _anchor(raw)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        binding.retain_artifact(binary, meta, **options)


def test_symlink_file_and_parent_are_rejected(artifacts, tmp_path):
    binary, meta, options = artifacts
    alias = tmp_path / "alias.ptx"
    alias.symlink_to(binary)
    with pytest.raises(ValueError, match="symlink"):
        binding.retain_artifact(alias, meta, **options)
    parent = tmp_path / "linked"
    parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        binding.retain_artifact(parent / binary.name, meta, **options)


def test_replaced_inode_with_same_bytes_is_not_same_retained_artifact(artifacts):
    binary, meta, options = artifacts
    artifact = binding.retain_artifact(binary, meta, **options)
    replacement = binary.with_suffix(".replacement")
    replacement.write_bytes(binary.read_bytes())
    replacement.replace(binary)
    with pytest.raises(ValueError, match="same retained artifact"):
        artifact.assert_unchanged()


@pytest.mark.parametrize("field", ["binary", "metadata"])
def test_changed_bytes_rejected_at_every_record_boundary(artifacts, field):
    held, *_ = _loaded(artifacts)
    path = getattr(held.artifact, field)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        held.record()


@pytest.mark.parametrize("block", [256, 128])
def test_any_preloaded_executable_for_context_rejected_without_clearing(block):
    device, module, *_ = _objects()
    cached = object()
    module.execs[(device.context, block)] = cached
    with pytest.raises(ValueError, match="preloaded"):
        binding.assert_fresh_module(module, device, 256)
    assert module.execs[(device.context, block)] is cached


@pytest.mark.parametrize(
    "mutation",
    ["cpu", "capturing", "context", "bool-context", "block", "strip", "failed"],
)
def test_bad_preload_state_rejected(mutation):
    device, module, *_ = _objects()
    if mutation == "cpu":
        device.is_cuda = False
    elif mutation == "capturing":
        device.is_capturing = True
    elif mutation == "context":
        device.context = 0
    elif mutation == "bool-context":
        device.context = True
    elif mutation == "block":
        module.options["block_dim"] = 128
    elif mutation == "strip":
        module.options["strip_hash"] = True
    else:
        module.failed_builds.add(device.context)
    with pytest.raises(ValueError):
        binding.assert_fresh_module(module, device, 256)


@pytest.mark.parametrize(
    "mutation",
    [
        "cache",
        "kernel-module",
        "adjoint",
        "device",
        "context",
        "arch",
        "capture",
        "block",
        "hash",
        "handle",
        "forward",
        "backward",
        "smem",
        "metadata",
        "symbol",
        "backward-option",
    ],
)
def test_loaded_binding_detects_mutation(artifacts, mutation):
    held, device, module, kernel, executable, hooks = _loaded(artifacts)
    if mutation == "cache":
        module.execs[(device.context, 256)] = object()
    elif mutation == "kernel-module":
        kernel.module = object()
    elif mutation == "adjoint":
        kernel.adj = object()
    elif mutation == "device":
        executable.device = Device()
    elif mutation == "context":
        device.context += 1
    elif mutation == "arch":
        device.arch = 89
    elif mutation == "capture":
        device.is_capturing = True
    elif mutation == "block":
        module.options["block_dim"] = 128
    elif mutation == "hash":
        executable.module_hash = b"X" * 32
    elif mutation == "handle":
        executable.handle += 1
    elif mutation == "forward":
        hooks.forward += 1
    elif mutation == "backward":
        hooks.backward = 111
    elif mutation == "smem":
        hooks.forward_smem_bytes += 1
    elif mutation == "metadata":
        executable.meta["extra"] = 1
    elif mutation == "symbol":
        kernel.get_mangled_name = lambda: "other"
    else:
        kernel.options["enable_backward"] = True
    with pytest.raises(ValueError, match="same loaded module"):
        held.assert_unchanged()


@pytest.mark.parametrize("value", [-1, True, 1.5, 2048])
def test_invalid_shared_memory_is_rejected(artifacts, value):
    binary, meta, options = artifacts
    artifact = binding.retain_artifact(binary, meta, **options)
    device, module, kernel, executable, hooks, module_hash = _objects()
    module.execs[(device.context, 256)] = executable
    hooks.forward_smem_bytes = value
    with pytest.raises(ValueError, match="shared memory"):
        binding.bind_loaded_module(
            artifact,
            kernel,
            device,
            executable,
            hooks,
            block_dim=256,
            expected_module_hash=module_hash,
        )


def test_module_has_no_gpu_runtime_import_or_load_entry():
    # Scope check, not a sandbox: this component never invokes a loader.
    source = Path(binding.__file__).read_text()
    assert "import warp" not in source
    assert "wp.launch(" not in source
    assert "._compile(" not in source
    assert ".load(" not in source


def test_original_kernel_can_inherit_disabled_backward_from_module(artifacts):
    held, _, _, kernel, _, _ = _loaded(artifacts)
    kernel.options.clear()  # Frozen MuJoCo-Warp original kernel has no override.
    held.assert_unchanged()


def test_hook_object_replacement_even_with_same_handle_is_rejected(artifacts):
    held, _, _, kernel, executable, hooks = _loaded(artifacts)
    executable.kernel_hooks[kernel.adj] = SimpleNamespace(**vars(hooks))
    with pytest.raises(ValueError, match="same loaded module"):
        held.assert_unchanged()


@pytest.mark.parametrize("block", [True, 0, -1, 1025, 256.0])
def test_invalid_block_dimensions_rejected(block):
    device, module, *_ = _objects()
    with pytest.raises(ValueError):
        binding.assert_fresh_module(module, device, block)


def test_forged_format_cannot_mislabel_retained_binary(artifacts):
    held, *_ = _loaded(artifacts)
    forged = replace(held.artifact, format="cubin")
    with pytest.raises(ValueError, match="format matches binary suffix"):
        forged.assert_unchanged()


def test_loaded_metadata_float_cannot_alias_an_integer_value(artifacts):
    held, _, _, _, executable, _ = _loaded(artifacts)
    executable.meta["fixture_cuda_kernel_forward_smem_bytes"] = 32.0
    with pytest.raises(ValueError, match="same loaded module"):
        held.assert_unchanged()
