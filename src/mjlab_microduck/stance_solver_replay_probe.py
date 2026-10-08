"""One bounded, shared-capacity dense-solver replay; never training admission.

The owner authenticates source, frozen runtime, historical input and a capped
user unit before spawning. Only the fresh inherited-lease child imports CUDA.
The unchanged caller is invoked once, with exact before/after target readback.
The caller must independently close the unit/cgroup and receive its artifacts.
"""

import argparse
import fcntl
from hashlib import sha1, sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from mjlab_microduck import shared_cuda_smoke as shared
from mjlab_microduck import stance_friction_prefix_cuda_probe as frozen
from mjlab_microduck import stance_solver_scratch as scratch
from mjlab_microduck import stance_solver_executable as executable
from mjlab_microduck import stance_solver_supervisor as supervisor

PROTOCOL = "microduck-dense-solver-replay-probe-oct8-v1"
ROOT, BRANCH = shared.ROOT, shared.BRANCH
BASE = "e6cf84d9615ee626390e2b691fbf08889efb0137"
OWN = frozenset({
    "src/mjlab_microduck/stance_solver_replay_probe.py",
    "src/mjlab_microduck/stance_solver_replay_receiver.py",
    "tests/test_stance_solver_replay_probe.py",
    "tests/test_stance_solver_replay_receiver.py",
    "docs/experiments/2026-10-08-dense-solver-replay-diagnostic.md",
})
TESTS = (
    "tests/test_stance_solver_scratch.py", "tests/test_stance_solver_supervisor.py",
    "tests/test_stance_solver_packets.py", "tests/test_stance_solver_executable.py",
    "tests/test_stance_solver_dispatch_guard.py", "tests/test_stance_solver_disassembly_scope.py",
    "tests/test_stance_solver_target_binding.py", "tests/test_stance_cuda_artifact_binding.py",
    "tests/test_stance_solver_binary_scope.py", "tests/test_stance_solver_replay_probe.py",
    "tests/test_stance_solver_replay_receiver.py",
)
FLAGS = dict(scratch.FLAGS)
BOUNDS = dict(service_seconds=480, child_seconds=240, probe_seconds=25,
              cleanup_seconds=5, closeout_seconds=120, margin_seconds=60,
              system_ram_bytes=6 * 1024**3, tasks=64, file_bytes=16 * 1024**2,
              total_used_mib=12288, free_reserve_mib=10240, aggregate_growth_mib=2048,
              temperature_c=65, utilization_percent=85)
INPUT = dict(inventory_sha256="8dcb44a422530c4169e4133a0dfb858d24909dc9d59ac90d163f5037a014b625",
             metadata_sha256="d19e16bcbf8152629b18c514ff627ff8417de1c4b4ed760d9ab0add30f867d91",
             raw_sha256="b47122d67e3568bf4e413c3905b0ca329e727ee900b168f93dd4c5f27bce4441",
             raw_bytes=5375152)
HISTORY = "artifacts/evaluations/solver-init-tick-run-e45a59c412bf"
INVENTORY = "artifacts/tools/solver-init-tick-closeout-e45a59c412bf/inventory.json"
RAW_NAME = "solver-init/original/forward-04.initialized.bin"
MW_TREE = "188d58bfac6ab54e51adfc80a6c504d782aec43856cbb15e54b4109f0ce30a1d"
TOOL_REL = "lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump"
TOOL_SHA = "ad1d0f0699f46603416eb58e7b9fdf9a293e95f9312c9df7fd6fda56a6c30d41"
UNIT_CAPS = dict(Type="exec", RuntimeMaxUSec="8min", TimeoutStopUSec="10s",
                 MemoryMax=str(6 * 1024**3), CPUQuotaPerSecUSec="2s", Nice="10",
                 TasksMax="64", LimitFSIZE=str(16 * 1024**2),
                 KillMode="control-group", Restart="no", NRestarts="0")
CGROUP_PARENT = "/user.slice/user-1000.slice/user@1000.service/app.slice/"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return frozen.canonical(value)


def read_plain(path, cap=16 * 1024**2):
    need(type(path) is type(Path()) and path.is_absolute()
         and path.resolve(strict=True) == path, "canonical plain replay file")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= cap,
             "bounded regular replay file")
        with os.fdopen(os.dup(fd), "rb") as stream:
            raw = stream.read(cap + 1)
        after, current = os.fstat(fd), path.stat(follow_symlinks=False)
        need(executable._identity(before) == executable._identity(after) == executable._identity(current)
             and path.resolve(strict=True) == path and len(raw) == before.st_size,
             "stable whole replay file")
        return raw
    finally:
        os.close(fd)


def read_json(path, cap=8 * 1024**2):
    from mjlab_microduck.stance_solver_replay_receiver import _json
    return _json(read_plain(path, cap), str(path))


def check_deadline(deadline, reserve, *, now=None):
    now = time.time() if now is None else now
    need(type(deadline) in (int, float) and math.isfinite(deadline)
         and type(reserve) is int and reserve >= 0
         and now + reserve < deadline <= now + 1200,
         "fresh bounded wall-clock deadline and full reserve")


def unit(source):
    need(type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source), "exact source SHA")
    return "microduck-dense-solver-replay-" + source[:12] + ".service"


def output(source):
    unit(source)
    return ROOT / "artifacts/evaluations" / ("dense-solver-replay-" + source[:12])


def command(*args, timeout=15):
    return subprocess.run(args, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=timeout).stdout


def source_binding(source):
    unit(source)
    need(Path.cwd().resolve() == ROOT and Path(__file__).resolve().parents[2] == ROOT,
         "exact native worktree")
    need(command("git", "rev-parse", "HEAD").decode().strip() == source
         and command("git", "branch", "--show-current").decode().strip() == BRANCH
         and not command("git", "status", "--porcelain").strip(), "clean exact source branch")
    command("git", "merge-base", "--is-ancestor", BASE, source)
    changed = set(command("git", "diff", "--name-only", BASE, source).decode().splitlines())
    need(changed == OWN, "exact five-path reviewed source fence")
    leaves = []
    for row in command("git", "ls-tree", "-rz", "--full-tree", source).split(b"\0"):
        if not row:
            continue
        header, name_raw = row.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        name = name_raw.decode()
        need(mode in ("100644", "100755") and kind == "blob", "plain committed file")
        raw = read_plain(ROOT / name)
        need(sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == oid,
             "whole committed source blob " + name)
        leaves.append(dict(path=name, bytes=len(raw), git_blob=oid, sha256=sha256(raw).hexdigest()))
    return dict(source=source, branch=BRANCH,
                tree=command("git", "rev-parse", source + "^{tree}").decode().strip(), leaves=leaves)


def historical_packet():
    # Authenticate all three whole anchors BEFORE parsing either JSON. The
    # historical inventory is an external anchor, not a newly regenerated one.
    inv_raw = read_plain(ROOT / INVENTORY)
    meta_raw = read_plain(ROOT / HISTORY / "child.json")
    raw = read_plain(ROOT / HISTORY / RAW_NAME, scratch.MAX_PACKET_BYTES)
    need(sha256(inv_raw).hexdigest() == INPUT["inventory_sha256"]
         and sha256(meta_raw).hexdigest() == INPUT["metadata_sha256"]
         and len(raw) == INPUT["raw_bytes"] and sha256(raw).hexdigest() == INPUT["raw_sha256"],
         "whole historical inventory, metadata and raw replay anchors")
    # Decode only the already authenticated retained bytes.
    from mjlab_microduck.stance_solver_replay_receiver import _json
    inventory = _json(inv_raw, "historical inventory")
    metadata = _json(meta_raw, "historical metadata")
    need(inventory["child.json"] == dict(bytes=len(meta_raw), sha256=INPUT["metadata_sha256"])
         and inventory[RAW_NAME] == dict(bytes=len(raw), sha256=INPUT["raw_sha256"]),
         "historical named leaves match retained whole inventory")
    record = metadata["arms"]["original"]["solver_init"]
    packet = scratch.decode_packet(raw, record["snapshot"], expected_sha256=INPUT["raw_sha256"],
                                   recipe=record["recipe"], elliptic_type=7)
    # Inventory shape is checked separately below; its whole published hash is
    # already the authority for these named historical files.
    need(type(inventory) is dict and inventory, "authenticated nonempty historical inventory")
    return packet


def mujoco_warp_source():
    root = frozen.VENV_TARGET / "lib/python3.12/site-packages/mujoco_warp"
    need(root.resolve(strict=True) == root, "canonical frozen MuJoCo Warp root")
    leaves = {str(path.relative_to(root)): sha256(read_plain(path)).hexdigest()
              for path in sorted(root.rglob("*.py"))}
    need(len(leaves) == 69 and sha256(canonical(leaves)).hexdigest() == MW_TREE,
         "whole frozen MuJoCo Warp source tree")
    return dict(root=str(root), leaves=leaves, sha256=MW_TREE)


def tool_binding():
    path = frozen.VENV_TARGET / TOOL_REL
    raw = read_plain(path)
    need(len(raw) == 569008 and sha256(raw).hexdigest() == TOOL_SHA,
         "whole frozen disassembler executable")
    version = command(str(path), "--version", timeout=5).decode()
    need("V12.8.55" in version and len(version) < 4096, "pinned disassembler version")
    return dict(path=str(path), bytes=len(raw), sha256=TOOL_SHA, version=version)


def runtime_binding():
    need(Path("/etc/machine-id").read_text().strip() == shared.MACHINE,
         "exact WSL machine identity")
    need(Path(sys.prefix).resolve() == frozen.VENV_TARGET
         and Path(sys.executable).resolve() == (frozen.VENV_TARGET / "bin/python").resolve(),
         "actual frozen interpreter")
    need(importlib.metadata.version("triton") == "3.5.1", "frozen Triton version")
    return dict(environment_alias=frozen.environment_binding(), packages=frozen.packages(),
                libraries=frozen.library_binding(), warp_sources=frozen.toolchain_source(),
                mujoco_warp_sources=mujoco_warp_source(), tool=tool_binding())


def unit_record(source, expected_pid):
    props = command("systemctl", "--user", "show", unit(source),
                    *[arg for key in (*UNIT_CAPS, "Id", "MainPID", "ActiveState", "ControlGroup", "InvocationID")
                      for arg in ("-p", key)]).decode()
    record = dict(line.split("=", 1) for line in props.splitlines())
    need({key: record.get(key) for key in UNIT_CAPS} == UNIT_CAPS
         and record["Id"] == unit(source)
         and record["MainPID"] == str(expected_pid) and record["ActiveState"] == "active"
         and record["InvocationID"] == os.environ.get("INVOCATION_ID")
         and re.fullmatch(r"[0-9a-f]{32}", record["InvocationID"])
         and record["ControlGroup"] == CGROUP_PARENT + unit(source),
         "active exact capped exec unit and owner invocation")
    return dict(name=unit(source), **record)


def assert_cgroup_owner_only(record, owner_pid):
    relative = Path(record["ControlGroup"])
    need(relative.is_absolute() and ".." not in relative.parts, "literal owned cgroup path")
    directory = Path("/sys/fs/cgroup") / str(relative).lstrip("/")
    paths = [directory / "cgroup.procs", *directory.glob("**/cgroup.procs")]
    pids = set()
    for path in set(paths):
        need(path.resolve(strict=True) == path and not path.is_symlink(), "plain owned cgroup listing")
        with path.open("rb") as stream:
            raw = stream.read(16385)
        need(len(raw) <= 16384, "bounded owned cgroup process listing")
        pids.update(int(line) for line in raw.splitlines())
    need(pids == {owner_pid}, "kernel owned cgroup contains only reviewed owner after child retirement")


def tests_record(path, source):
    value = read_json(path)
    need(set(value) == {"source", "test_files", "mac", "native", "flags"}
         and value["source"] == source and value["test_files"] == list(TESTS)
         and value["flags"] == FLAGS, "source-bound paired CPU test evidence")
    for platform in ("mac", "native"):
        anchor = value[platform]
        need(type(anchor) is dict and set(anchor) == {"path", "bytes", "sha256", "tests"}
             and type(anchor["tests"]) is int and anchor["tests"] > 335,
             "focused CPU test XML anchor")
        leaf = Path(anchor["path"])
        need(leaf.is_absolute() and leaf.is_relative_to(ROOT / "artifacts/tools")
             and leaf.suffix == ".xml", "retained CPU XML evidence path")
        raw = read_plain(leaf)
        need(len(raw) == anchor["bytes"] and sha256(raw).hexdigest() == anchor["sha256"],
             "whole CPU XML bytes")
        suites = ET.fromstring(raw)
        rows = [suites] if suites.tag == "testsuite" else list(suites)
        need(rows and all(row.tag == "testsuite" and all(int(row.get(key, "-1")) == 0
             for key in ("errors", "failures", "skipped")) for row in rows)
             and sum(int(row.get("tests", "-1")) for row in rows) == anchor["tests"],
             "complete CPU tests with no failures errors or skips")
    need(value["mac"]["tests"] == value["native"]["tests"], "matched focused CPU collection")
    return dict(path=str(path), bytes=len(read_plain(path)), sha256=sha256(read_plain(path)).hexdigest(),
                evidence=value)


def child_env(root):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="0", PYTHONPATH=str(ROOT / "src"),
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    for key, name in (("WARP_CACHE_PATH", "warp-cache"), ("CUDA_CACHE_PATH", "cuda-cache"),
                      ("XDG_CACHE_HOME", "xdg-cache"), ("TORCH_EXTENSIONS_DIR", "torch-cache")):
        path = root / name
        need(not path.exists() and not path.is_symlink(), "fresh absent private child cache")
        env[key] = str(path)
    return env


def recipe_signature(model, data):
    from mjlab_microduck.stance_solver_init_control import SOLVER_RECIPE
    result = {}
    for key in SOLVER_RECIPE:
        owner = data if key in ("nworld", "njmax", "njmax_nnz") else (
            model if key in ("nv", "nv_pad", "is_sparse") else model.opt)
        result[key] = dict(vars(model.block_dim)) if key == "block_dim" else getattr(owner, key)
    need(result == SOLVER_RECIPE and all(type(result[k]) is type(v)
         for k, v in SOLVER_RECIPE.items()), "literal fresh dense solver recipe")
    return result


def scratch_arrays(model, data, context):
    roots = dict(model=model, data=data, contact=data.contact, efc=data.efc, context=context)
    values = {}
    for name in scratch.RESTORE_ORDER:
        if name == "contact.nacon":
            # Capture schema labels the aggregate contact counter with the
            # contact bank; the frozen ABI stores it directly on Data.
            values[name] = data.nacon
            continue
        parts = name.split(".")
        value = roots[parts[0]]
        for part in parts[1:]:
            value = getattr(value, part)
        values[name] = value
    return values


def child(args):
    # No GPU import is reachable before declaration, source/runtime, lease,
    # unit, input and private-cache authentication completes.
    root = output(args.source)
    need(args.child and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
         and os.getppid() == args.owner_pid, "fresh owned CUDA child")
    declaration_raw = read_plain(root / "declaration.json")
    need(sha256(declaration_raw).hexdigest() == args.declaration_sha256,
         "whole owner declaration bytes")
    declaration = read_json(root / "declaration.json")
    need(declaration["protocol"] == PROTOCOL and declaration["source"] == args.source
         and declaration["owner_pid"] == args.owner_pid and declaration["native_root"] == str(root)
         and declaration["flags"] == FLAGS and declaration["bounds"] == BOUNDS
         and declaration["replay_input"] == INPUT, "exact owner replay declaration")
    check_deadline(args.deadline, BOUNDS["child_seconds"] + BOUNDS["margin_seconds"])
    need(declaration["deadline_unix"] == args.deadline
         and source_binding(args.source) == declaration["source_binding"]
         and runtime_binding() == declaration["runtime"]
         and unit_record(args.source, args.owner_pid) == declaration["service"],
         "unchanged source runtime and capped owner service")
    need(shared.lease_identity(args.lease_fd) == declaration["lease"], "same inherited lease")
    fcntl.flock(args.lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    packet = historical_packet()
    for key, name in (("WARP_CACHE_PATH", "warp-cache"), ("CUDA_CACHE_PATH", "cuda-cache"),
                      ("XDG_CACHE_HOME", "xdg-cache"), ("TORCH_EXTENSIONS_DIR", "torch-cache")):
        path = root / name
        need(os.environ.get(key) == str(path) and not path.exists() and not path.is_symlink(),
             "private cache absent before CUDA initialization")
    import numpy as np
    import warp as wp
    from warp._src import context as warp_context
    frozen.configure_compiler(wp)
    wp.init()
    device = wp.get_device("cuda:0")
    need(device.uuid == shared.GPU and device.arch == 120 and device.is_cuda
         and device.context > 0 and not device.is_capturing, "actual mapped sm120 GPU identity")
    for attr, name in (("core", "warp.so"), ("llvm", "warp-clang.so")):
        need(str(getattr(warp_context.runtime, attr)._name)
             == declaration["runtime"]["libraries"][name]["path"], "actual loaded frozen runtime library")
    import mujoco
    import mujoco_warp as mw
    from mujoco_warp._src import solver, types
    from mjlab_microduck.stance_warp_runtime import build_entity
    from mjlab_microduck.stance_plant_evidence import describe
    from mjlab_microduck.stance_com_coupled_receiver import DESCRIPTOR_SHA256
    from mjlab_microduck.stance_solver_dispatch_guard import DenseSolverDispatchGuard
    from mjlab_microduck.stance_solver_packets import DenseSolverPacketCapture
    need(int(types.ConstraintType.CONTACT_ELLIPTIC) == 7, "frozen elliptic enum")
    stream = wp.Stream(device)
    need(type(stream.cuda_stream) is int and stream.cuda_stream > 0, "dedicated nondefault CUDA stream")
    with wp.ScopedDevice(device), wp.ScopedStream(stream):
        native = build_entity().compile()
        need(sha256(canonical(describe(native))).hexdigest() == DESCRIPTOR_SHA256,
             "unchanged compiled plant descriptor")
        model = mw.put_model(native)
        data = mw.put_data(native, mujoco.MjData(native), nworld=64, nconmax=128, njmax=512)
        context = solver.create_solver_context(model, data)
        recipe_signature(model, data)
        prepared = executable.DenseSolverExecutable(wp, warp_context, solver, device,
                                                     root / "compiled-dense-solver")
        prepared.compile()
        tool = declaration["runtime"]["tool"]
        need(tool_binding() == tool, "same disassembler immediately before use")
        with (root / "target.sass").open("xb") as stdout, (root / "disassembler.stderr").open("xb") as stderr:
            subprocess.run((tool["path"], "--dump-sass", "--function", prepared.symbol,
                            str(prepared.binary_path)), stdout=stdout, stderr=stderr,
                           stdin=subprocess.DEVNULL, timeout=20, check=True)
            stdout.flush(); os.fsync(stdout.fileno())
            stderr.flush(); os.fsync(stderr.fileno())
        need((root / "disassembler.stderr").stat().st_size == 0 and tool_binding() == tool,
             "joined successful unchanged disassembler with empty stderr")
        binding = prepared.load(read_plain(root / "target.sass"))
        held_copy, held_sync, held_array = wp.copy, wp.synchronize_stream, wp.array
        entries = tuple((fn, fn.__code__) for fn in (held_copy, held_sync))

        def guard():
            prepared._guard()
            need(wp.copy is held_copy and wp.synchronize_stream is held_sync and wp.array is held_array
                 and all(fn.__code__ is code for fn, code in entries)
                 and wp.get_stream(device) is stream and warp_context.runtime.tape is None
                 and device.uuid == shared.GPU, "same scratch runtime entries stream and eager context")

        def stage(name, raw, logical, host_shape, dtype, warp_dtype):
            source = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(host_shape).copy()
            return held_array(source, dtype=arrays[name].dtype, device="cpu", pinned=True, requires_grad=False)

        arrays = scratch_arrays(model, data, context)
        restored = scratch.DenseSolverScratchRestorer(packet, arrays, device=device, stream=stream,
            stage=stage, copy=held_copy, synchronize=held_sync, guard=guard)
        restored.restore()
        observer = DenseSolverDispatchGuard(solver, wp, runtime=warp_context.runtime, binding=binding,
                                            model=model, data=data, context=context, stream=stream)
        capture = DenseSolverPacketCapture(wp, data, context, stream)
        observer.run(capture=capture)
        frozen.write(root / "packet-before.bin", capture.raw("before"))
        frozen.write(root / "packet-after.bin", capture.raw("after"))
        receipt = dict(protocol=PROTOCOL, source=args.source, native_root=str(root),
            declaration_sha256=args.declaration_sha256, owner_pid=args.owner_pid, child_pid=os.getpid(),
            device=dict(alias=str(device), arch=device.arch, context=device.context,
                        object_id=id(device), gpu_uuid=device.uuid),
            executable=prepared.record(), guard=observer.record(), scratch=restored.receipt(),
            files=dict(cubin=str(prepared.binary_path.relative_to(root)),
                       metadata=str(prepared.meta_path.relative_to(root)),
                       generated_source=str(prepared.source_path.relative_to(root)), sass="target.sass",
                       packet_before="packet-before.bin", packet_after="packet-after.bin"), flags=FLAGS)
    need(runtime_binding() == declaration["runtime"] and source_binding(args.source) == declaration["source_binding"]
         and shared.lease_identity(args.lease_fd) == declaration["lease"], "same complete runtime source and lease after replay")
    frozen.write(root / "receipt.json", receipt)


def owner(args):
    need(not args.child and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only owner")
    check_deadline(args.deadline, BOUNDS["service_seconds"] + BOUNDS["closeout_seconds"] + BOUNDS["margin_seconds"])
    source = source_binding(args.source)
    runtime = runtime_binding()
    service = unit_record(args.source, os.getpid())
    tests = tests_record(Path(args.cpu_evidence), args.source)
    historical_packet()
    root = output(args.source)
    need(root.parent.resolve(strict=True) == root.parent and not root.exists(), "unique absent replay output")
    fd = os.open(shared.LOCK, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        lease = shared.lease_identity(fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        services = shared.service_snapshot()
        baseline = shared.capacity(shared.telemetry())
        root.mkdir(mode=0o700)
        declaration = dict(protocol=PROTOCOL, source=args.source, source_binding=source,
                           native_root=str(root), owner_pid=os.getpid(), runtime=runtime, service=service,
                           cpu_tests=tests, lease=lease, services=services, baseline=baseline,
                           bounds=BOUNDS, deadline_unix=args.deadline, replay_input=INPUT,
                           exclusive_gpu_claimed=False, flags=FLAGS)
        frozen.write(root / "declaration.json", declaration)
        env = child_env(root)
        argv = (sys.executable, "-m", "mjlab_microduck.stance_solver_replay_probe", "--child",
                "--source", args.source, "--deadline", str(args.deadline), "--owner-pid", str(os.getpid()),
                "--lease-fd", str(fd), "--declaration-sha256", sha256(canonical(declaration)).hexdigest())

        def probe():
            shared.READ_DEADLINE = time.monotonic() + BOUNDS["probe_seconds"] - 1
            check_deadline(args.deadline, BOUNDS["margin_seconds"])
            need(shared.lease_identity(fd) == lease, "same held admission lease")
            shared.service_snapshot()
            shared.capacity(shared.telemetry(), baseline)

        try:
            with (root / "child.log").open("xb") as log:
                result = supervisor.supervise(argv, cwd=ROOT, env=env, log=log, probe=probe,
                    timeout=BOUNDS["child_seconds"], probe_timeout=BOUNDS["probe_seconds"],
                    grace=BOUNDS["cleanup_seconds"], interval=0.25, pass_fds=(fd,))
                log.flush(); os.fsync(log.fileno())
            assert_cgroup_owner_only(service, os.getpid())
            need(source_binding(args.source) == source and runtime_binding() == runtime,
                 "same whole source and runtime after joined child")
            probe()
            frozen.write(root / "supervision.json", dict(result=result, kernel_cgroup_only_owner=True,
                unit_retirement_independently_required=True, flags=FLAGS))
        except BaseException as error:
            frozen.write(root / "failure.json", dict(error_type=type(error).__name__, error=str(error)[:4096],
                unit_retirement_independently_required=True, flags=FLAGS))
            raise
        # Owner releases only its FD; every reviewed child inherits the same
        # lease and joins its one tool. External closeout must verify unit/cgroup
        # retirement before interpreting artifacts. No native acceptance here.
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--deadline", required=True, type=float)
    parser.add_argument("--cpu-evidence")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--lease-fd", type=int)
    parser.add_argument("--declaration-sha256")
    args = parser.parse_args()
    if args.child:
        child(args)
    else:
        need(args.cpu_evidence is not None, "paired committed-source CPU evidence required")
        owner(args)


if __name__ == "__main__":
    main()
