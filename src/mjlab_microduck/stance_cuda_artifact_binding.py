"""Retained CUDA artifact/cache guards; not a GPU supervisor or admission gate.

No Warp import, compilation, device creation or launch occurs in this module.
The caller owns fresh-process/source authentication, lease, caps, explicit load,
stream/buffer/launch evidence and independent numerical replay. These helpers
bind observed objects, not a security sandbox against arbitrary Python code.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat

PROTOCOL = "microduck-retained-cuda-artifact-binding-oct7-v1"
MAX_BINARY_BYTES = 8 * 1024 * 1024
MAX_META_BYTES = 256 * 1024
_PATH_TYPE = type(Path())


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive_int(value, name):
    _require(type(value) is int and 0 < value < 2**64, name)
    return value


def _stat_identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_anchored(path, size, digest, cap):
    """Authenticate whole regular bytes against an external anchor before decode."""
    _require(type(path) is _PATH_TYPE, "plain absolute artifact Path")
    _require(
        path.is_absolute() and path.resolve(strict=True) == path,
        "no symlink artifact path",
    )
    _require(type(size) is int and 0 < size <= cap, "bounded external artifact length")
    _require(
        type(digest) is str
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest),
        "plain external SHA256",
    )
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _require(
            stat.S_ISREG(before.st_mode) and before.st_size == size,
            "regular whole artifact length",
        )
        chunks = []
        remaining = size + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        current = path.stat(follow_symlinks=False)
        _require(
            _stat_identity(before) == _stat_identity(after) == _stat_identity(current)
            and stat.S_ISREG(current.st_mode)
            and path.resolve(strict=True) == path,
            "stable retained artifact identity",
        )
    finally:
        os.close(fd)
    _require(
        len(raw) == size and sha256(raw).hexdigest() == digest,
        "whole authenticated artifact bytes",
    )
    return raw, _stat_identity(before)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "unique metadata JSON keys")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("finite metadata JSON: " + value)


def _metadata(raw):
    value = json.loads(
        raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
    )
    _require(type(value) is dict, "plain metadata object")
    _require(
        all(
            type(key) is str
            and key.endswith(
                ("_cuda_kernel_forward_smem_bytes", "_cuda_kernel_backward_smem_bytes")
            )
            and type(size) is int
            and 0 <= size <= 1024 * 1024
            for key, size in value.items()
        ),
        "integer shared-memory metadata fields",
    )
    return value


@dataclass(frozen=True)
class RetainedArtifact:
    """Caller-anchored bytes; PTX identity is not driver-JIT machine-code identity."""

    binary: Path
    metadata: Path
    binary_size: int
    binary_sha256: str
    metadata_size: int
    metadata_sha256: str
    format: str
    binary_identity: tuple
    metadata_identity: tuple
    metadata_raw: bytes

    def assert_unchanged(self):
        _require(
            type(self.format) is str
            and self.format in ("ptx", "cubin")
            and self.binary.suffix == "." + self.format,
            "retained artifact format matches binary suffix",
        )
        _, binary_identity = _read_anchored(
            self.binary, self.binary_size, self.binary_sha256, MAX_BINARY_BYTES
        )
        raw, metadata_identity = _read_anchored(
            self.metadata, self.metadata_size, self.metadata_sha256, MAX_META_BYTES
        )
        _require(
            binary_identity == self.binary_identity
            and metadata_identity == self.metadata_identity
            and raw == self.metadata_raw,
            "same retained artifact inodes and bytes",
        )

    def decoded_metadata(self):
        return _metadata(self.metadata_raw)


def retain_artifact(
    binary, metadata, *, binary_size, binary_sha256, metadata_size, metadata_sha256
):
    """Inspect existing files only; never create, rewrite or compile artifacts."""
    _require(
        type(binary) is _PATH_TYPE and type(metadata) is _PATH_TYPE,
        "plain artifact paths",
    )
    _require(
        binary != metadata and binary.suffix in (".ptx", ".cubin"),
        "separate CUDA binary and metadata",
    )
    _, binary_identity = _read_anchored(
        binary, binary_size, binary_sha256, MAX_BINARY_BYTES
    )
    raw, metadata_identity = _read_anchored(
        metadata, metadata_size, metadata_sha256, MAX_META_BYTES
    )
    _require(binary_identity[:2] != metadata_identity[:2], "nonaliased artifact files")
    _metadata(raw)
    artifact = RetainedArtifact(
        binary,
        metadata,
        binary_size,
        binary_sha256,
        metadata_size,
        metadata_sha256,
        binary.suffix[1:],
        binary_identity,
        metadata_identity,
        raw,
    )
    artifact.assert_unchanged()
    return artifact


def assert_fresh_module(module, device, block_dim):
    """Reject Warp1.12's early executable-cache return; clear nothing."""
    _positive_int(block_dim, "plain positive block dimension")
    _require(block_dim <= 1024, "bounded CUDA block dimension")
    _require(
        device.is_cuda is True and device.is_capturing is False, "eager CUDA device"
    )
    _positive_int(device.context, "actual nonzero CUDA context")
    _require(
        type(module.execs) is dict and type(module.failed_builds) is set,
        "plain module caches",
    )
    _require(
        module.options["block_dim"] == block_dim
        and type(module.options["block_dim"]) is int
        and module.options["strip_hash"] is False,
        "declared block dimension and source-hashed module",
    )
    _require(
        not any(key[0] == device.context for key in module.execs),
        "no preloaded executable at any block dimension in actual CUDA context",
    )
    _require(
        device.context not in module.failed_builds,
        "no prior failed build in CUDA context",
    )


@dataclass(frozen=True)
class LoadedModuleBinding:
    artifact: RetainedArtifact
    kernel: object
    module: object
    device: object
    executable: object
    hooks: object
    block_dim: int
    module_hash: bytes
    context: int
    module_handle: int
    forward_handle: int
    forward_smem_bytes: int
    symbol: str
    kernel_adjoint: object
    device_alias: str
    device_arch: int

    def assert_unchanged(self):
        self.artifact.assert_unchanged()
        _require(
            self.kernel.module is self.module
            and self.device.context == self.context
            and type(self.device.context) is int
            and str(self.device) == self.device_alias
            and self.device.arch == self.device_arch
            and type(self.device.arch) is int
            and self.device.is_cuda is True
            and self.device.is_capturing is False
            and self.module.options["block_dim"] == self.block_dim
            and type(self.module.options["block_dim"]) is int
            and self.module.options["strip_hash"] is False
            and self.module.get_module_hash(self.block_dim) == self.module_hash
            and self.module.execs.get((self.context, self.block_dim)) is self.executable
            and self.executable.device is self.device
            and self.executable.module_hash == self.module_hash
            and type(self.executable.module_hash) is bytes
            and self.executable.handle == self.module_handle
            and type(self.executable.handle) is int
            and self.kernel.adj is self.kernel_adjoint
            and self.executable.kernel_hooks.get(self.kernel_adjoint) is self.hooks
            and self.hooks.forward == self.forward_handle
            and type(self.hooks.forward) is int
            and self.hooks.backward is None
            and self.hooks.forward_smem_bytes == self.forward_smem_bytes
            and type(self.hooks.forward_smem_bytes) is int
            and self.hooks.backward_smem_bytes == 0
            and type(self.hooks.backward_smem_bytes) is int
            and self.kernel.get_mangled_name() + "_cuda_kernel_forward" == self.symbol
            and (self.module.options | self.kernel.options).get("enable_backward")
            is False
            and type(self.executable.meta) is dict
            and all(
                type(key) is str and type(value) is int
                for key, value in self.executable.meta.items()
            )
            and self.executable.meta == self.artifact.decoded_metadata(),
            "same loaded module, device, metadata and forward hook",
        )

    def record(self):
        self.assert_unchanged()
        return {
            "protocol": PROTOCOL,
            "artifact_format": self.artifact.format,
            "binary_sha256": self.artifact.binary_sha256,
            "binary_path": str(self.artifact.binary),
            "binary_bytes": self.artifact.binary_size,
            "metadata_sha256": self.artifact.metadata_sha256,
            "metadata_path": str(self.artifact.metadata),
            "metadata_bytes": self.artifact.metadata_size,
            "module_hash": self.module_hash.hex(),
            "block_dim": self.block_dim,
            "context": self.context,
            "module_handle": self.module_handle,
            "forward_handle": self.forward_handle,
            "forward_smem_bytes": self.forward_smem_bytes,
            "symbol": self.symbol,
            "device": str(self.device),
            "device_arch": self.device_arch,
            "observed_object_ids": {
                name: id(getattr(self, name))
                for name in ("kernel", "module", "device", "executable", "hooks")
            },
            "driver_jit_machine_code_observed": False,
            "loaded_binary_bytes_observed": False,
            "native_execution_qualified": False,
            "training_authorized": False,
            "physical_acceptance": False,
        }


def bind_loaded_module(
    artifact, kernel, device, executable, hooks, *, block_dim, expected_module_hash
):
    """Check a caller's explicit-load result, without loading or looking up hooks.

    Caller must call ``assert_fresh_module`` immediately before its held frozen
    ``Module.load`` entry, using the retained paths, arch and block dimension.
    This post-load check alone cannot prove which file the loader consumed.
    """
    _require(type(artifact) is RetainedArtifact, "retained artifact binding")
    _positive_int(block_dim, "plain positive block dimension")
    _require(block_dim <= 1024, "bounded CUDA block dimension")
    _require(
        type(expected_module_hash) is bytes and len(expected_module_hash) == 32,
        "exact module source/options hash",
    )
    context = _positive_int(device.context, "actual nonzero CUDA context")
    module_handle = _positive_int(executable.handle, "nonzero loaded module handle")
    forward_handle = _positive_int(hooks.forward, "nonzero forward hook handle")
    device_arch = _positive_int(device.arch, "plain CUDA device architecture")
    _require(
        type(hooks.forward_smem_bytes) is int
        and 0 <= hooks.forward_smem_bytes <= device.max_shared_memory_per_block,
        "bounded forward shared memory",
    )
    symbol = kernel.get_mangled_name() + "_cuda_kernel_forward"
    metadata = artifact.decoded_metadata()
    _require(
        type(metadata.get(symbol + "_smem_bytes")) is int
        and metadata[symbol + "_smem_bytes"] == hooks.forward_smem_bytes,
        "forward hook shared memory matches authenticated metadata",
    )
    binding = LoadedModuleBinding(
        artifact,
        kernel,
        kernel.module,
        device,
        executable,
        hooks,
        block_dim,
        expected_module_hash,
        context,
        module_handle,
        forward_handle,
        hooks.forward_smem_bytes,
        symbol,
        kernel.adj,
        str(device),
        device_arch,
    )
    binding.assert_unchanged()
    return binding
