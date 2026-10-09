"""Capped shared-capacity two-arm caller-cost diagnostic; never training.

The CPU owner authenticates before spawning an inherited-lease CUDA child.
Only external closeout may authenticate complete output and unit retirement.
"""
import argparse
import fcntl
from hashlib import sha1, sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from mjlab_microduck import stance_solver_replay_probe as retained
from mjlab_microduck import stance_solver_supervisor as supervisor
from mjlab_microduck import stance_solver_cost_stages as stages
from mjlab_microduck import stance_solver_cost_executable as executable
from mjlab_microduck import stance_solver_target_binding as static

PROTOCOL = "microduck-caller-cost-probe-oct9-v1"
BASE = "98aa8a1d1a413f76cadc397492eddb8cb953135e"
OWN = frozenset({
    "src/mjlab_microduck/stance_solver_cost_binary.py",
    "src/mjlab_microduck/stance_solver_cost_executable.py",
    "src/mjlab_microduck/stance_solver_cost_probe.py",
    "src/mjlab_microduck/stance_solver_cost_receiver.py",
    "tests/test_stance_solver_cost_binary.py",
    "tests/test_stance_solver_cost_executable.py",
    "tests/test_stance_solver_cost_probe.py",
    "docs/experiments/2026-10-09-caller-cost-native-diagnostic.md",
})
TESTS = retained.TESTS + ("tests/test_stance_solver_gauss_audit.py", "tests/test_stance_solver_cost_stages.py",
    "tests/test_stance_solver_cost_dispatch.py", "tests/test_stance_solver_cost_binary.py",
    "tests/test_stance_solver_cost_executable.py", "tests/test_stance_solver_cost_probe.py")
ROOT, BRANCH, BOUNDS, UNIT_CAPS = retained.ROOT, retained.BRANCH, retained.BOUNDS, retained.UNIT_CAPS
shared, frozen, need = retained.shared, retained.frozen, retained.need
read_plain, read_json, canonical = retained.read_plain, retained.read_json, retained.canonical


def unit(source):
    need(type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source), "literal source SHA")
    return "microduck-caller-cost-" + source[:12] + ".service"


def output(source):
    unit(source)
    return ROOT / "artifacts/evaluations" / ("caller-cost-" + source[:12])


def source_binding(source):
    unit(source)
    cmd = retained.command
    need(Path.cwd().resolve() == ROOT and Path(__file__).resolve().parents[2] == ROOT
         and cmd("git", "rev-parse", "HEAD").decode().strip() == source
         and cmd("git", "branch", "--show-current").decode().strip() == BRANCH
         and not cmd("git", "status", "--porcelain").strip(), "clean exact native source branch")
    cmd("git", "merge-base", "--is-ancestor", BASE, source)
    need(set(cmd("git", "diff", "--name-only", BASE, source).decode().splitlines()) == OWN,
         "new exact eight-path source fence; historical fences unchanged")
    leaves = []
    for row in cmd("git", "ls-tree", "-rz", "--full-tree", source).split(b"\0"):
        if not row: continue
        header, name = row.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        name = name.decode()
        need(mode in ("100644", "100755") and kind == "blob", "plain committed file")
        raw = read_plain(ROOT / name)
        need(sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == oid, "whole committed blob " + name)
        leaves.append(dict(path=name, bytes=len(raw), git_blob=oid, sha256=sha256(raw).hexdigest()))
    return dict(source=source, branch=BRANCH, tree=cmd("git", "rev-parse", source + "^{tree}").decode().strip(), leaves=leaves)


def unit_record(source, pid):
    keys = (*UNIT_CAPS, "Id", "MainPID", "ActiveState", "ControlGroup", "InvocationID")
    raw = retained.command("systemctl", "--user", "show", unit(source), *[x for k in keys for x in ("-p", k)]).decode()
    row = dict(line.split("=", 1) for line in raw.splitlines())
    need({k: row.get(k) for k in UNIT_CAPS} == UNIT_CAPS and row["Id"] == unit(source)
         and row["MainPID"] == str(pid) and row["ActiveState"] == "active"
         and row["InvocationID"] == os.environ.get("INVOCATION_ID")
         and re.fullmatch(r"[0-9a-f]{32}", row["InvocationID"])
         and row["ControlGroup"] == retained.CGROUP_PARENT + unit(source), "same active capped exec owner unit")
    return dict(name=unit(source), **row)


def tests_record(path, source):
    raw = read_plain(path)
    value = read_json(path)
    need(set(value) == {"source", "test_files", "mac", "native", "flags"}
         and value["source"] == source and value["test_files"] == list(TESTS)
         and value["flags"] == stages.FLAGS, "source-bound paired focused CPU evidence")
    for platform in ("mac", "native"):
        a = value[platform]
        need(type(a) is dict and set(a) == {"path", "bytes", "sha256", "tests"}
             and type(a["tests"]) is int and a["tests"] > 633, "new focused collection anchor")
        p = Path(a["path"])
        need(p.is_absolute() and p.is_relative_to(ROOT / "artifacts/tools") and p.suffix == ".xml", "retained CPU XML path")
        xml = read_plain(p)
        need(len(xml) == a["bytes"] and sha256(xml).hexdigest() == a["sha256"], "whole CPU XML hash")
        tree = ET.fromstring(xml)
        rows = [tree] if tree.tag == "testsuite" else list(tree)
        need(rows and all(r.tag == "testsuite" and all(int(r.get(k, "-1")) == 0 for k in ("errors", "failures", "skipped")) for r in rows)
             and sum(int(r.get("tests", "-1")) for r in rows) == a["tests"], "all focused CPU tests passed without skips")
    need(value["mac"]["tests"] == value["native"]["tests"], "matched focused collection")
    return dict(path=str(path), bytes=len(raw), sha256=sha256(raw).hexdigest(), evidence=value)


def copy_control(wp, arrays, stream, guard):
    """Copy exactly seven literal overrides after full original restoration."""
    import numpy as np
    held_array, held_copy, held_sync = wp.array, wp.copy, wp.synchronize_stream
    held_numpy = held_array.numpy
    codes = tuple(f.__code__ for f in (held_numpy, held_copy, held_sync))
    shapes = {name: host for name, _, host, _ in stages.SPECS}
    for name, raw in stages.control_fields().items():
        guard()
        a = arrays[name]
        host = np.frombuffer(raw, dtype="<f4").reshape(shapes[name]).copy()
        src = held_array(host, dtype=a.dtype, device="cpu", pinned=True, requires_grad=False)
        ptrs = (a.ptr, src.ptr)
        def check():
            guard()
            need(wp.array is held_array and held_array.numpy is held_numpy and wp.copy is held_copy
                 and wp.synchronize_stream is held_sync
                 and tuple(f.__code__ for f in (held_numpy, held_copy, held_sync)) == codes
                 and arrays[name] is a and type(a) is type(src) is held_array
                 and a.dtype is src.dtype is wp.float32 and tuple(a.shape) == tuple(src.shape) == shapes[name]
                 and a.device is stream.device and a.device.is_cuda is True
                 and src.device.is_cpu is True and src.device.is_cuda is False
                 and src.pinned is True and src.requires_grad is a.requires_grad is False
                 and src.is_contiguous is a.is_contiguous is True
                 and (a.ptr, src.ptr) == ptrs and all(type(p) is int and p > 0 for p in ptrs)
                 and abs(a.ptr - src.ptr) >= len(raw), "literal distinct pinned control staging and held copy path")
            view = held_numpy(src)  # deliberately never read a CUDA array through numpy
            need(tuple(view.shape) == shapes[name] and view.dtype.str == "<f4"
                 and view.flags.c_contiguous and view.nbytes == len(raw)
                 and view.tobytes(order="C") == raw, "literal whole control staging bytes")
        check()
        held_copy(a, src, stream=stream)
        check()
        held_sync(stream)
        check()


def child(args):
    root = output(args.source)
    need(args.child and os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and os.getppid() == args.owner_pid,
         "fresh inherited-lease CUDA child")
    raw = read_plain(root / "declaration.json")
    need(sha256(raw).hexdigest() == args.declaration_sha256, "whole declaration hash before decode")
    dcl = read_json(root / "declaration.json")
    need(dcl["protocol"] == PROTOCOL and dcl["source"] == args.source and dcl["owner_pid"] == args.owner_pid
         and dcl["native_root"] == str(root) and dcl["source_binding"] == source_binding(args.source)
         and dcl["runtime"] == retained.runtime_binding() and dcl["service"] == unit_record(args.source, args.owner_pid)
         and dcl["control"] == stages.control_manifest() and dcl["bounds"] == BOUNDS
         and dcl["flags"] == stages.FLAGS and dcl["deadline_unix"] == args.deadline
         and shared.lease_identity(args.lease_fd) == dcl["lease"], "complete child admission before CUDA import")
    retained.check_deadline(args.deadline, BOUNDS["child_seconds"] + BOUNDS["closeout_seconds"] + BOUNDS["margin_seconds"])
    fcntl.flock(args.lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    shared.service_snapshot()
    shared.capacity(shared.telemetry(), dcl["baseline"])
    packet = retained.historical_packet()
    for key, name in (("WARP_CACHE_PATH", "warp-cache"), ("CUDA_CACHE_PATH", "cuda-cache"),
                      ("XDG_CACHE_HOME", "xdg-cache"), ("TORCH_EXTENSIONS_DIR", "torch-cache")):
        p = root / name
        need(os.environ.get(key) == str(p) and not p.exists() and not p.is_symlink(), "absent private cache before CUDA init")
    import numpy as np
    import warp as wp
    from warp._src import context as wc
    frozen.configure_compiler(wp)
    wp.init()
    device = wp.get_device("cuda:0")
    need(device.uuid == shared.GPU and device.arch == 120 and device.is_cuda is True
         and device.context > 0 and not device.is_capturing, "actual mapped sm120 GPU")
    for attr, name in (("core", "warp.so"), ("llvm", "warp-clang.so")):
        need(str(getattr(wc.runtime, attr)._name) == dcl["runtime"]["libraries"][name]["path"], "actual pinned loaded runtime library")
    import mujoco
    import mujoco_warp as mw
    from mujoco_warp._src import solver, types, warp_util
    from mjlab_microduck.stance_warp_runtime import build_entity
    from mjlab_microduck.stance_plant_evidence import describe
    from mjlab_microduck.stance_com_coupled_receiver import DESCRIPTOR_SHA256
    from mjlab_microduck.stance_solver_cost_dispatch import CostDispatchObserver
    need(int(types.ConstraintType.CONTACT_ELLIPTIC) == 7, "frozen domain enum")
    stream = wp.Stream(device)
    with wp.ScopedDevice(device), wp.ScopedStream(stream):
        plant = build_entity().compile()
        need(sha256(canonical(describe(plant))).hexdigest() == DESCRIPTOR_SHA256, "same compiled plant descriptor")
        model = mw.put_model(plant)
        data = mw.put_data(plant, mujoco.MjData(plant), nworld=64, nconmax=128, njmax=512)
        ctx = solver.create_solver_context(model, data)
        retained.recipe_signature(model, data)
        kernels = executable.materialize_kernels(solver, warp_util)
        def graph_guard(): executable.verify_kernel_graph(solver, warp_util, kernels)
        modules, bindings = {}, {}
        for role, names in executable.GROUPS.items():
            prep = executable.CostModuleExecutable(wp=wp, context=wc, device=device, directory=root / ("compiled-" + role),
                role=role, kernels={n: kernels[n] for n in names}, graph_guard=graph_guard)
            prep.compile()
            sass = {}
            tool = dcl["runtime"]["tool"]
            for n, symbol in prep.symbols.items():
                need(retained.tool_binding() == tool, "same held disassembler before use")
                with (root / (n + ".sass")).open("xb") as out, (root / (n + ".stderr")).open("xb") as err:
                    subprocess.run((tool["path"], "--dump-sass", "--function", symbol, str(prep.binary_path)),
                        stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=20, check=True)
                    out.flush(); os.fsync(out.fileno()); err.flush(); os.fsync(err.fileno())
                need((root / (n + ".stderr")).stat().st_size == 0 and retained.tool_binding() == tool,
                     "joined unchanged successful offline disassembler")
                sass[n] = read_plain(root / (n + ".sass"))
            bindings.update(prep.load(sass))
            modules[role] = prep
        arrays = retained.scratch_arrays(model, data, ctx)
        held_array, held_copy, held_sync = wp.array, wp.copy, wp.synchronize_stream
        codes = tuple(f.__code__ for f in (held_copy, held_sync))
        def guard():
            graph_guard()
            need(wp.array is held_array and wp.copy is held_copy and wp.synchronize_stream is held_sync
                 and (held_copy.__code__, held_sync.__code__) == codes
                 and wp.get_stream(device) is stream and wc.runtime.tape is None and device.uuid == shared.GPU,
                 "same scratch copy/sync/runtime/stream")
            for b in bindings.values(): b.assert_unchanged()
        def stage(name, raw, logical, host_shape, dtype, warp_dtype):
            host = np.frombuffer(raw, dtype=dtype).reshape(host_shape).copy()
            return held_array(host, dtype=arrays[name].dtype, device="cpu", pinned=True, requires_grad=False)
        arms = {}
        for arm in ("reference", "control"):
            restored = retained.scratch.DenseSolverScratchRestorer(packet, arrays, device=device, stream=stream,
                stage=stage, copy=held_copy, synchronize=held_sync, guard=guard)
            restored.restore()
            if arm == "control": copy_control(wp, arrays, stream, guard)
            observer = CostDispatchObserver(solver=solver, util=warp_util, wp=wp, runtime=wc.runtime,
                bindings=bindings, model=model, data=data, context=ctx, stream=stream)
            try:
                observer.run()
            finally:
                # Preserve complete failure boundaries too; no successful receipt
                # is written if observer.run/record fails.
                for phase in stages.PHASES:
                    if phase in observer.capture.packets:
                        frozen.write(root / (arm + "-" + phase + ".bin"), observer.capture.raw(phase))
            arms[arm] = dict(observer=observer.record(), scratch=restored.receipt())
        receipt = dict(protocol=PROTOCOL, source=args.source, declaration_sha256=args.declaration_sha256,
            owner_pid=args.owner_pid, child_pid=os.getpid(), native_root=str(root),
            device=dict(alias=str(device), arch=device.arch, context=device.context, object_id=id(device), gpu_uuid=device.uuid),
            modules={role: prep.record() for role, prep in modules.items()}, arms=arms, flags=dict(stages.FLAGS))
    need(retained.runtime_binding() == dcl["runtime"] and source_binding(args.source) == dcl["source_binding"]
         and shared.lease_identity(args.lease_fd) == dcl["lease"], "whole post-run runtime/source/lease pins")
    frozen.write(root / "receipt.json", receipt)


def owner(args):
    need(not args.child and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only capped owner")
    need(type(args.cpu_evidence) is str and bool(args.cpu_evidence), "explicit paired CPU evidence path")
    retained.check_deadline(args.deadline, BOUNDS["service_seconds"] + BOUNDS["closeout_seconds"] + BOUNDS["margin_seconds"])
    source, runtime = source_binding(args.source), retained.runtime_binding()
    service, tests = unit_record(args.source, os.getpid()), tests_record(Path(args.cpu_evidence), args.source)
    retained.historical_packet()
    root = output(args.source)
    need(root.parent.resolve(strict=True) == root.parent and not root.exists(), "unique absent diagnostic output")
    fd = os.open(shared.LOCK, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        lease = shared.lease_identity(fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        services = shared.service_snapshot()
        baseline = shared.capacity(shared.telemetry())
        root.mkdir(mode=0o700)
        dcl = dict(protocol=PROTOCOL, source=args.source, source_binding=source, native_root=str(root), owner_pid=os.getpid(),
            runtime=runtime, service=service, cpu_tests=tests, lease=lease, services=services, baseline=baseline,
            bounds=BOUNDS, deadline_unix=args.deadline, replay_input=retained.INPUT, control=stages.control_manifest(),
            exclusive_gpu_claimed=False, flags=dict(stages.FLAGS))
        frozen.write(root / "declaration.json", dcl)
        env = retained.child_env(root)
        argv = (sys.executable, "-m", "mjlab_microduck.stance_solver_cost_probe", "--child", "--source", args.source,
                "--deadline", str(args.deadline), "--owner-pid", str(os.getpid()), "--lease-fd", str(fd),
                "--declaration-sha256", sha256(canonical(dcl)).hexdigest())
        samples = []
        def probe():
            shared.READ_DEADLINE = time.monotonic() + BOUNDS["probe_seconds"] - 1
            retained.check_deadline(args.deadline, BOUNDS["margin_seconds"])
            need(shared.lease_identity(fd) == lease and shared.service_snapshot() == services, "same held lease/protected services")
            need(len(samples) < 1024, "bounded telemetry collection")
            samples.append(shared.capacity(shared.telemetry(), baseline))
        try:
            with (root / "child.log").open("xb") as log:
                result = supervisor.supervise(argv, cwd=ROOT, env=env, log=log, probe=probe,
                    timeout=BOUNDS["child_seconds"], probe_timeout=BOUNDS["probe_seconds"],
                    grace=BOUNDS["cleanup_seconds"], interval=0.25, pass_fds=(fd,))
                log.flush(); os.fsync(log.fileno())
            retained.assert_cgroup_owner_only(service, os.getpid())
            need(source_binding(args.source) == source and retained.runtime_binding() == runtime, "whole source/runtime pins after child")
            probe()
            frozen.write(root / "telemetry.json", dict(samples=samples, flags=dict(stages.FLAGS)))
            frozen.write(root / "supervision.json", dict(result=result, kernel_cgroup_only_owner=True,
                unit_retirement_independently_required=True, flags=dict(static.FLAGS)))
        except BaseException as error:
            frozen.write(root / "failure.json", dict(error_type=type(error).__name__, error=str(error)[:4096],
                unit_retirement_independently_required=True, flags=dict(stages.FLAGS)))
            raise
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
    child(args) if args.child else owner(args)


if __name__ == "__main__": main()
