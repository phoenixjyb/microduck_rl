"""Fresh one-tick moving-state serial-control supervisor, never admission."""

import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from mjlab_microduck import stance_com_coupled_probe as coupled
from mjlab_microduck import stance_bam_load_receiver as receiver
from mjlab_microduck import stance_serial_step_probe as step_prior
from mjlab_microduck import stance_serial_step_divergence as divergence
from mjlab_microduck import stance_rne_coupled_probe as prior
from mjlab_microduck import stance_com_entry_probe as old
from mjlab_microduck.stance_com_entry_receiver import FLAGS, SERVICE_CAPS, canonical

PROTOCOL = "microduck-bam-load-probe-oct7-v1"
SERVICE_CAPS = {**SERVICE_CAPS, "LimitFSIZE": "2097152"}
MODULE = "mjlab_microduck.stance_bam_load_probe"
BASE = "4458450658ab3f0e12e17d393371003774298138"
ROOT = old.ROOT
SERVICE_SECONDS, CHILD_SECONDS, CLOSEOUT_SECONDS, MARGIN = 300, 240, 300, 60
LEAF_COUNT = 673
OWN = {
    f"src/mjlab_microduck/stance_bam_load_{name}.py"
    for name in ("observer", "probe", "receiver")
}
OWN |= {
    f"tests/test_stance_bam_load_{name}.py"
    for name in ("observer", "probe", "receiver")
}
OWN.add("docs/experiments/2026-10-07-bam-load-boundary.md")
PRIOR_DOC = "docs/experiments/2026-10-07-serial-tick-first-divergence.md"
COM_SOURCE = "6cd03291581684680e6ab3875a96f88cf8dd83ca"
COM_PROOF = {
    "inventory.json": "1048cd38bd8c68c34bba18ea2af65a499d91e07c92ce60fc090593a313036809",
    "terminal.json": "d1cd1c224625fb6171b0da71b24eeec2ec9396ebbb2c51223783e6837efa993f",
    "receiver.json": "51ec8165d49ed398c2704bbe2ce65ce0b9f3e0778dfcd6056c7375d139e113d0",
}
TEMPORAL_SHA256 = "2a871d75faff2d1da3d78ca644fb24332c92cf1ca5830301d5bd77659c4c3e08"


def check_window(reserve, now=None):
    coupled.check_window(reserve, now)


def unit(source, mode):
    old.need(
        type(source) is str
        and re.fullmatch(r"[0-9a-f]{40}", source)
        and type(mode) is str
        and mode in ("tests", "run"),
        "literal fresh serial-step unit identity",
    )
    return f"microduck-bam-load-{mode}-{source[:12]}.service"


def output(source, mode):
    unit(source, mode)
    return (
        ROOT
        / "artifacts"
        / ("tools" if mode == "tests" else "evaluations")
        / f"bam-load-{mode}-{source[:12]}"
    )


def source_binding(source):
    unit(source, "run")
    old.need(
        Path.cwd().resolve() == ROOT
        and Path(__file__).resolve().parents[2] == ROOT
        and old.read("git", "rev-parse", "HEAD") == source
        and old.read("git", "branch", "--show-current") == old.BRANCH
        and old.read("git", "remote", "get-url", "origin") == old.ORIGIN
        and not old.read("git", "status", "--porcelain"),
        "exact clean fresh serial-step source",
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE, source], check=True, timeout=15
    )
    old.need(
        set(old.read("git", "diff", "--name-only", BASE, source).splitlines()) <= OWN,
        "seven-path fresh serial-step source fence",
    )
    leaves = {}
    for path in old.read(
        "git",
        "ls-files",
        "src",
        "tests",
        "pyproject.toml",
        "uv.lock",
        *sorted(
            coupled.OWN
            | prior.OWN
            | prior.prior.OWN
            | step_prior.OWN
            | OWN
            | {PRIOR_DOC}
        ),
    ).splitlines():
        raw = old.bounded(ROOT / path, 4 * 1024**2, allow_empty=True)
        old.need(
            raw
            == subprocess.check_output(["git", "show", f"{source}:{path}"], timeout=15),
            "whole committed RNE leaf " + path,
        )
        leaves[path] = sha256(raw).hexdigest()
    old.need(len(leaves) == LEAF_COUNT, "exact whole RNE leaf inventory")
    return dict(
        source=source,
        branch=old.BRANCH,
        tree=old.read("git", "rev-parse", "HEAD^{tree}"),
        leaf_count=len(leaves),
        leaves_sha256=sha256(canonical(leaves)).hexdigest(),
    )


def properties(source, mode):
    return dict(
        row.split("=", 1)
        for row in old.read(
            "systemctl", "--user", "show", unit(source, mode)
        ).splitlines()
        if "=" in row
    )


def service(source, mode, pid):
    values = properties(source, mode)
    caps = old.TEST_SERVICE_CAPS if mode == "tests" else SERVICE_CAPS
    old.need(
        type(pid) is int
        and pid > 0
        and all(values.get(k) == v for k, v in caps.items())
        and values["MainPID"] == str(pid)
        and values["ActiveState"] == "active"
        and values["SubState"] == "running"
        and re.fullmatch(r"[0-9a-f]{32}", values["InvocationID"]),
        "actual retained capped RNE invocation",
    )
    return {
        k: values[k]
        for k in (*caps, "MainPID", "InvocationID", "ActiveState", "SubState")
    }


def retained_com():
    root = ROOT / "artifacts/tools/com-coupled-run-closeout-6cd032915816"
    raw = {name: old.bounded(root / name, 128 * 1024) for name in COM_PROOF}
    old.need(
        all(
            sha256(raw[name]).hexdigest() == digest
            for name, digest in COM_PROOF.items()
        ),
        "whole immutable paired CoM closeout before JSON",
    )
    old.need(
        json.loads(raw["receiver.json"])["decision"] == "fresh-coupled-control-exact",
        "retained positive paired decision, not temporal admission",
    )
    temporal = old.bounded(
        ROOT / "artifacts/tools/com-coupled-temporal-6cd032915816/native-analysis.json",
        128 * 1024,
    )
    old.need(
        len(temporal) == 53898 and sha256(temporal).hexdigest() == TEMPORAL_SHA256,
        "whole native independent temporal negative before JSON",
    )
    record = json.loads(temporal)
    old.need(
        record["decision"] == "temporal-serial-negative"
        and record["negative_coordinate_count"] == 100
        and record["all_cases_fixed_inputs_temporal_exact"] is True
        and record["serial_temporal_exact"] is False
        and receiver.old.false_flags(record["flags"]),
        "retained separate temporal negative",
    )
    return dict(
        source=COM_SOURCE,
        paired_hashes=dict(COM_PROOF),
        temporal_sha256=TEMPORAL_SHA256,
        paired_decision="fresh-coupled-control-exact",
        temporal_decision="temporal-serial-negative",
    )


def test_files():
    return (
        step_prior.test_files()
        + ["tests/test_stance_serial_step_divergence.py"]
        + [
            f"tests/test_stance_bam_load_{name}.py"
            for name in ("observer", "probe", "receiver")
        ]
    )


def retained_rne():
    """Authenticate the closed paired/temporal proof, never admit a window."""
    root = ROOT / "artifacts/tools/rne-coupled-run-closeout-50864a995f33"
    raw = {
        name: old.bounded(root / name, 4 * 1024**2)
        for name in receiver.PREDECESSOR_FILES
    }
    old.need(
        all(
            len(raw[name]) == anchor["bytes"]
            and sha256(raw[name]).hexdigest() == anchor["sha256"]
            for name, anchor in receiver.PREDECESSOR_FILES.items()
        ),
        "whole native and Mac coupled proof before JSON",
    )
    value = json.loads(raw["receiver.json"])
    old.need(
        value["decision"] == "fresh-coupled-rne-control-exact"
        and value["paired_control_decision"] == "serial-paired-frames-exact"
        and value["temporal_control_decision"] == "serial-temporal-frames-exact"
        and value["all_serial_rne_arithmetic_exact"] is True
        and receiver.old.false_flags(value["flags"]),
        "closed narrow coupled proof with no full-window admission",
    )
    return dict(receiver.PREDECESSOR)


def retained_tick():
    """Reproduce and authenticate the closed negative without weakening it."""
    result = divergence.analyze_closeout(
        ROOT / "artifacts/evaluations/serial-step-run-26e2ee8244b7",
        ROOT / "artifacts/tools/serial-step-run-closeout-26e2ee8244b7",
    )
    payload = canonical(result)
    old.need(
        len(payload) == 252029
        and sha256(payload).hexdigest() == receiver.TICK_ANALYSIS_SHA256,
        "whole retained negative and first observed motor divergence",
    )
    directory = ROOT / "artifacts/tools/serial-step-divergence-26e2ee8244b7"
    for name in ("native-analysis.json", "mac-analysis.json"):
        old.need(
            old.bounded(directory / name, 2 * 1024**2) == payload,
            "both independent retained first-divergence analyses",
        )
    return dict(receiver.TICK_PREDECESSOR)


def checked_junit(raw):
    old.need(
        type(raw) is bytes and 0 < len(raw) <= 1024**2, "bounded complete RNE JUnit"
    )
    suites = list(ET.fromstring(raw).iter("testsuite"))
    old.need(
        receiver.EXPECTED_TESTS > 0
        and suites
        and sum(int(s.get("tests", "-1")) for s in suites) == receiver.EXPECTED_TESTS
        and all(
            s.get(k) == "0" for s in suites for k in ("skipped", "errors", "failures")
        ),
        "exact predeclared serial-step CPU suite without omissions",
    )


def run_tests(source, directory, binding, live, packages):
    environment = old.test_environment()
    fixtures = old.test_fixture_binding(ROOT)
    descriptor = coupled.descriptor_fixture()
    old.need(
        fixtures == old.test_fixture_binding(old.OLD_ROOT),
        "unchanged copied CPU fixtures",
    )
    old.need(
        sha256(canonical(test_files())).hexdigest() == receiver.TEST_FILES_SHA256,
        "exact ordered current test list",
    )
    with (directory / "pytest.log").open("xb") as stream:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--junitxml=" + str(directory / "junit.xml"),
                *test_files(),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=CHILD_SECONDS,
        )
        stream.flush()
        os.fsync(stream.fileno())
    old.need(result.returncode == 0, "fresh full serial-step CPU suite passed")
    xml = old.bounded(directory / "junit.xml", 1024**2)
    checked_junit(xml)
    fd = os.open(directory / "junit.xml", os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    old.need(
        source_binding(source) == binding
        and old.packages() == packages
        and service(source, "tests", os.getpid()) == live
        and coupled.descriptor_fixture() == descriptor
        and fixtures
        == old.test_fixture_binding(ROOT)
        == old.test_fixture_binding(old.OLD_ROOT),
        "closed unchanged serial-step CPU source/packages/fixtures/caps",
    )
    coupled.phase_a()
    old.predecessors()
    retained_com()
    retained_rne()
    retained_tick()
    sample = old.host()
    old.need(
        not sample["processes"] and sample["used_mib"] <= 1024,
        "idle host at CPU closure",
    )
    check_window(CLOSEOUT_SECONDS + MARGIN)
    receipt = dict(
        protocol=PROTOCOL + ":tests",
        source_binding=binding,
        packages=packages,
        tests=receiver.EXPECTED_TESTS,
        test_files=test_files(),
        test_environment=environment,
        retained_test_fixtures=fixtures,
        retained_descriptor_fixture=descriptor,
        service_properties=live,
        junit_sha256=sha256(xml).hexdigest(),
        log_sha256=sha256(old.bounded(directory / "pytest.log", 1024**2)).hexdigest(),
        flags=dict(FLAGS),
    )
    old.write(directory / "receipt.json", canonical(receipt), 128 * 1024)


def test_prerequisite(source, binding, packages):
    directory = output(source, "tests")
    raw = old.bounded(directory / "receipt.json", 128 * 1024)
    record = json.loads(raw)
    old.need(
        record["source_binding"] == binding
        and record["packages"] == packages
        and type(record["tests"]) is int
        and record["tests"] == receiver.EXPECTED_TESTS
        and record["test_files"] == test_files()
        and record["test_environment"] == old.TEST_ENV
        and record["protocol"] == PROTOCOL + ":tests"
        and receiver.old.false_flags(record["flags"])
        and record["retained_descriptor_fixture"] == coupled.descriptor_fixture()
        and record["retained_test_fixtures"]
        == old.test_fixture_binding(ROOT)
        == old.test_fixture_binding(old.OLD_ROOT),
        "fresh exact-source current CPU prerequisite",
    )
    xml = old.bounded(directory / "junit.xml", 1024**2)
    checked_junit(xml)
    old.need(
        sha256(xml).hexdigest() == record["junit_sha256"]
        and sha256(old.bounded(directory / "pytest.log", 1024**2)).hexdigest()
        == record["log_sha256"],
        "whole current CPU log and JUnit hashes",
    )
    terminal = dict(
        old.TEST_SERVICE_CAPS,
        MainPID="0",
        Result="success",
        ExecMainStatus="0",
        ActiveState="active",
        SubState="exited",
        InvocationID=record["service_properties"]["InvocationID"],
    )
    values = properties(source, "tests")
    old.need(
        all(values.get(k) == v for k, v in terminal.items()),
        "same completed current CPU invocation",
    )
    return (
        raw,
        terminal,
        dict(
            count=receiver.EXPECTED_TESTS,
            files=len(test_files()),
            source_binding=binding,
            receipt_sha256=sha256(raw).hexdigest(),
            terminal=terminal,
        ),
    )


def child(source, lease_fd, owner_pid, declaration_sha):
    old.need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
        and type(owner_pid) is int
        and os.getppid() == owner_pid,
        "owned visible native child and actual parent",
    )
    old.inherited_lease(lease_fd)
    check_window(CHILD_SECONDS + CLOSEOUT_SECONDS + MARGIN)
    binding, packages, live = (
        source_binding(source),
        old.packages(),
        service(source, "run", owner_pid),
    )
    directory = output(source, "run")
    raw = old.bounded(directory / "declaration.json", 128 * 1024)
    old.need(
        sha256(raw).hexdigest() == declaration_sha,
        "whole declaration before native child JSON",
    )
    declaration = json.loads(raw)
    old.need(
        declaration["source_binding"] == binding
        and declaration["packages"] == packages
        and declaration["service_properties"] == live
        and declaration["protocol"] == PROTOCOL + ":declaration"
        and declaration["source"] == source
        and receiver.old.false_flags(declaration["flags"]),
        "exact native serial-step declaration",
    )
    sample = old.host()
    old.need(
        not sample["processes"] and sample["used_mib"] <= 1024, "idle before Torch CUDA"
    )
    import torch
    import warp as wp

    old.need(not torch.cuda.is_initialized(), "no early Torch CUDA initialization")
    torch.manual_seed(673)
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(677)
    old.need(
        torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0,
        "one visible initialized GPU",
    )
    cases = {}
    for case in receiver.CASE_ORDER:
        record, packets = _case(torch, wp)
        cases[case] = record
        for name, packet in packets.items():
            name = case + "." + name
            old.write(directory / name, packet, receiver.CAPS[name])
    old.need(
        source_binding(source) == binding
        and old.packages() == packages
        and service(source, "run", owner_pid) == live,
        "closed unchanged serial-step child source/packages/caps",
    )
    for name, module in tuple(sys.modules.items()):
        if name.startswith("mjlab_microduck") and getattr(module, "__file__", None):
            old.need(
                Path(module.__file__)
                .resolve()
                .is_relative_to(ROOT / "src/mjlab_microduck"),
                "bound fresh serial-step imports",
            )
    record = dict(
        protocol=PROTOCOL + ":child",
        source=source,
        source_binding=binding,
        packages=packages,
        declaration_sha256=declaration_sha,
        owner_pid=owner_pid,
        child_pid=os.getpid(),
        child_ppid=os.getppid(),
        case_order=list(receiver.CASE_ORDER),
        cases=cases,
        flags=dict(FLAGS),
    )
    old.write(directory / "child.json", canonical(record), 1024**2)


def _f32(value, shape, torch):
    old.need(
        isinstance(value, torch.Tensor)
        and value.shape == shape
        and value.dtype == torch.float32
        and torch.isfinite(value).all(),
        "complete finite float32 producer tensor",
    )
    return value.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()


def _case(torch, wp):
    from mjlab_microduck.stance_serial_step_control import SerialStepControl
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
    from mjlab_microduck import stance_plant_evidence as plant

    states = {
        f"caller_{dev}_before": coupled._rng_state(torch, dev)
        for dev in ("cpu", "cuda")
    }
    from mjlab_microduck.stance_bam_load_observer import BamLoadObserver

    control = SerialStepControl()
    load = BamLoadObserver()
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(977)
        torch.cuda.manual_seed(983)
        states.update(
            {
                f"private_{dev}_start": coupled._rng_state(torch, dev)
                for dev in ("cpu", "cuda")
            }
        )
        with wp.ScopedDevice("cuda:0"), control, load:
            env = WarpStanceRuntime(
                nworld=64, device="cuda:0", solved_field_check="packed"
            )
            control.bind_runtime(env)
            load.bind_runtime(control, env)
            old.need(
                type(env) is WarpStanceRuntime
                and env._forward.__func__ is WarpStanceRuntime._forward
                and env.step.__func__ is WarpStanceRuntime.step
                and not {"_forward", "step", "reset"} & set(env.__dict__),
                "original plain eager nominal runtime",
            )
            states.update(
                {
                    f"private_{dev}_ctor_end": coupled._rng_state(torch, dev)
                    for dev in ("cpu", "cuda")
                }
            )
            first = coupled._frame(env)
            descriptor = plant.describe(env.native)
            old.need(
                sha256(canonical(descriptor)).hexdigest() == receiver.DESCRIPTOR_SHA256,
                "exact whole native plant descriptor",
            )
            for name, value in dict(
                vin_tensor=7.5,
                vin_drop_gain=0.1,
                kp_scale=1.0,
                kd_scale=1.0,
                friction_scale=1.0,
            ).items():
                tensor = env.motor.nominal_parameters[name]
                old.need(
                    tensor.shape == (64, 1)
                    and torch.isfinite(tensor).all()
                    and (tensor == value).all(),
                    "fixed nominal motor " + name,
                )
            action = torch.zeros((64, 10), dtype=torch.float32, device="cuda:0")
            result = control.step(env, action, capture_control=True)
            old.need(
                len(result["control_evidence"]["proposals"]) == 10
                and len(result["boundaries"]) == 11
                and (env.steps == 10).all()
                and env.live.all()
                and not result["terminated"].any()
                and not result["timed_out"].any()
                and env.terminal == [None] * 64
                and env.forward_graph is None
                and env.model.ntendon == 0
                and not env._view("xfrc_applied").any()
                and not env._view("qfrc_applied").any()
                and result["optimizer_launched"] is False,
                "one complete nominal ten-substep tick without terminal/reset/graph",
            )
            second = coupled._frame(env)
            states.update(
                {
                    f"private_{dev}_forward_end": coupled._rng_state(torch, dev)
                    for dev in ("cpu", "cuda")
                }
            )
            motor, masks = [], []
            for index, row in enumerate(result["control_evidence"]["proposals"]):
                old.need(
                    torch.equal(
                        row["before_steps"],
                        torch.full((64,), index, dtype=torch.long, device=env.device),
                    )
                    and row["live"].all()
                    and row["accepted"].all()
                    and not row["rejected"].any(),
                    "all accepted exact substep counters",
                )
                values = {
                    **{"command." + k: v for k, v in row["command"].items()},
                    "torque": row["torque_nm"],
                    **{"committed." + k: v for k, v in row["committed"].items()},
                }
                old.need(
                    set(values) == set(receiver.MOTOR_FIELDS),
                    "complete actual motor/control field set",
                )
                motor.extend(
                    _f32(values[name], (64, *shape), torch)
                    for name, shape in receiver.MOTOR_FIELDS.items()
                )
                masks.append(
                    row["before_steps"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype("<i8", copy=False)
                    .tobytes()
                )
                masks.extend(
                    row[name].detach().cpu().numpy().astype("u1", copy=False).tobytes()
                    for name in ("live", "accepted", "rejected")
                )
            ledger = {
                name: result[name].detach().cpu().tolist()
                for name in (
                    "reward",
                    "terminated",
                    "timed_out",
                    "episode_steps",
                    "executed_steps",
                    "live",
                )
            }
            ledger["term_sums"] = {
                name: value.detach().cpu().tolist()
                for name, value in result["term_sums"].items()
            }
    states.update(
        {
            f"caller_{dev}_after": coupled._rng_state(torch, dev)
            for dev in ("cpu", "cuda")
        }
    )
    rng = b"".join(states[key] for key in receiver.RNG_KEYS)
    metadata = {
        key: dict(bytes=len(states[key]), sha256=sha256(states[key]).hexdigest())
        for key in receiver.RNG_KEYS
    }
    coupled.receiver.checked_rng(rng, metadata)
    packets = {
        "frames.bin": first + second,
        "rne.entries.bin": b"".join(control.rne_inputs),
        "rne.outputs.bin": b"".join(control.rne_outputs),
        "com.entries.bin": b"".join(control.com_initialized),
        "com.weighted.bin": b"".join(control.com_weighted),
        "rng.bin": rng,
        "motor.bin": b"".join(motor),
        "masks.bin": b"".join(masks),
    }
    old.need(
        set(load.packets) == set(receiver.LOAD_FIELDS),
        "complete actual BAM load observer packet set",
    )
    packets.update(
        {"load." + name + ".bin": value for name, value in load.packets.items()}
    )
    record = dict(
        load_observer=load.receipt,
        compiled_descriptor=descriptor,
        ntendon=env.model.ntendon,
        nominal_parameters={
            "voltage": 7.5,
            "drop_gain": 0.1,
            "kp_scale": 1.0,
            "kd_scale": 1.0,
            "friction_scale": 1.0,
        },
        physics_steps=10,
        graph_created=False,
        actor_model_created=False,
        optimizer_created=False,
        storage_created=False,
        explicit_reset_calls=0,
        control_scope=control.receipt,
        rng_metadata=metadata,
        ledger=ledger,
        rng_end_boundary="after-one-nominal-step",
        control_capture=True,
        control_ids=env.ctrl_ids.detach().cpu().tolist(),
    )
    return record, packets


def run(source, directory, binding, live, packages):
    test_raw, test_terminal, test_summary = test_prerequisite(source, binding, packages)
    before = os.lstat(old.LOCK)
    old.need(
        stat.S_ISREG(before.st_mode) and before.st_size == 0, "existing shared GPU lock"
    )
    fd = os.open(old.LOCK, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    process = None
    try:
        old.need(
            (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
            == (before.st_dev, before.st_ino),
            "stable shared lease inode",
        )
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sample = old.host()
        old.need(
            not sample["processes"] and sample["used_mib"] <= 1024, "idle leased GPU"
        )
        check_window(SERVICE_SECONDS + CLOSEOUT_SECONDS + MARGIN)
        old.write(directory / "tests.receipt.json", test_raw, 128 * 1024)
        old.write(directory / "tests.terminal.json", canonical(test_terminal), 4096)
        declaration = dict(
            protocol=PROTOCOL + ":declaration",
            source=source,
            source_binding=binding,
            packages=packages,
            service_properties=live,
            host=sample,
            phase_a=coupled.phase_a(),
            predecessors=old.predecessors(),
            com_predecessor=retained_com(),
            rne_predecessor=retained_rne(),
            tick_predecessor=retained_tick(),
            case_order=list(receiver.CASE_ORDER),
            expected_tests=receiver.EXPECTED_TESTS,
            tests=test_summary,
            flags=dict(FLAGS),
        )
        raw = canonical(declaration)
        old.write(directory / "declaration.json", raw, 128 * 1024)
        with (directory / "child.log").open("xb") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    MODULE,
                    "--mode",
                    "child",
                    "--source",
                    source,
                    "--lease-fd",
                    str(fd),
                    "--owner-pid",
                    str(os.getpid()),
                    "--declaration-sha",
                    sha256(raw).hexdigest(),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
                pass_fds=(fd,),
            )
            start = time.monotonic()
            observations, seen, observed_ppid = [], False, None
            while process.poll() is None:
                old.need(
                    time.monotonic() - start < CHILD_SECONDS,
                    "bounded native child runtime",
                )
                path = Path(f"/proc/{process.pid}/stat")
                if path.exists():
                    observed_ppid = int(path.read_text().rsplit(") ", 1)[1].split()[1])
                    old.need(observed_ppid == os.getpid(), "actual owned child PPID")
                sample = old.host()
                old.need(
                    all(row["pid"] == process.pid for row in sample["processes"])
                    and sample["used_mib"] <= 12288
                    and sample["free_mib"] >= 10240,
                    "one bounded GPU child, no foreign occupancy",
                )
                seen |= any(row["pid"] == process.pid for row in sample["processes"])
                observations.append(
                    dict(
                        elapsed=time.monotonic() - start,
                        child_pid=process.pid,
                        host=sample,
                    )
                )
                old.need(
                    len(observations) <= 480 and log.tell() <= 1024**2,
                    "bounded monitor/log",
                )
                time.sleep(0.5)
            log.flush()
            os.fsync(log.fileno())
        old.need(
            process.returncode == 0 and seen and observed_ppid == os.getpid(),
            "actually observed successful native child",
        )
        filtered = "\n".join(
            line
            for line in old.bounded(directory / "child.log", 1024**2)
            .decode()
            .splitlines()
            if line != "[mdp] Patches 1-2 active: NaN-safe reward/advantage"
        )
        old.need(
            not old.BAD_LOG.search(filtered), "finite warning-free owned child log"
        )
        old.need(
            source_binding(source) == binding
            and old.packages() == packages
            and service(source, "run", os.getpid()) == live,
            "closed unchanged native owner source/packages/caps",
        )
        coupled.phase_a()
        old.predecessors()
        retained_com()
        retained_rne()
        retained_tick()
        closure_host = old.host()
        old.need(
            not closure_host["processes"] and closure_host["used_mib"] <= 1024,
            "fresh idle GPU and preserved host after child exit",
        )
        old.need(
            {path.name for path in directory.iterdir()}
            == set(receiver.CAPS) - {"report.json"}
            and sum(path.stat().st_size for path in directory.iterdir()) < 32 * 1024**2,
            "bounded exact forty-eight-file paired evidence inventory before report",
        )
        report = dict(
            protocol=PROTOCOL + ":report",
            source=source,
            source_binding=binding,
            child_sha256=sha256(
                old.bounded(directory / "child.json", 1024**2)
            ).hexdigest(),
            returncode=0,
            observed_child_pid=process.pid,
            observed_child_ppid=observed_ppid,
            gpu_child_observed=True,
            monitor=observations,
            closure_host=closure_host,
            flags=dict(FLAGS),
        )
        old.write(directory / "report.json", canonical(report), 256 * 1024)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--mode", required=True, choices=("tests", "run", "child"))
    parser.add_argument("--lease-fd", type=int)
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--declaration-sha")
    args = parser.parse_args()
    if args.mode == "child":
        child(args.source, args.lease_fd, args.owner_pid, args.declaration_sha)
        return
    old.test_environment()
    check_window(
        (2 if args.mode == "tests" else 1) * SERVICE_SECONDS + CLOSEOUT_SECONDS + MARGIN
    )
    binding, packages, live = (
        source_binding(args.source),
        old.packages(),
        service(args.source, args.mode, os.getpid()),
    )
    coupled.phase_a()
    old.predecessors()
    retained_com()
    retained_rne()
    retained_tick()
    sample = old.host()
    old.need(
        not sample["processes"] and sample["used_mib"] <= 1024,
        "idle before fresh service",
    )
    directory = output(args.source, args.mode)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        (run_tests if args.mode == "tests" else run)(
            args.source, directory, binding, live, packages
        )
    except BaseException as error:
        old.write(
            directory / "failure.json",
            canonical(
                dict(
                    protocol=PROTOCOL + ":failure",
                    source=args.source,
                    exception_type=type(error).__name__,
                    flags=dict(FLAGS),
                )
            ),
            4096,
        )
        raise


if __name__ == "__main__":
    main()
