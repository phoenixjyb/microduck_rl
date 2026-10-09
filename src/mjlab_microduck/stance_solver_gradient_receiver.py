"""Independent CPU receiver for a stopped gradient-prefix diagnostic.

Every retained output byte is authenticated before JSON or bank decoding. The
receiver checks the caller's claims and packet contract; it does not establish
capture origin, native execution, runtime cause, or qualification.
"""

from collections import Counter
import ast
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from mjlab_microduck import stance_solver_replay_receiver as old
from mjlab_microduck import stance_solver_efc_receiver as efc_receiver
from mjlab_microduck import stance_solver_gradient_probe as probe
from mjlab_microduck import stance_solver_gradient_prefix as prefix
from mjlab_microduck import stance_solver_gradient_dispatch as dispatch
from mjlab_microduck import stance_solver_gradient_runtime as api
from mjlab_microduck import stance_solver_gradient_executable as executable
from mjlab_microduck import stance_solver_gradient_binary as binary

PROTOCOL = "microduck-stopped-gradient-receiver-oct9-v1"
need = old.need


def _json(raw, label):
    value = old._json(raw, label)
    need(old._canonical(value) == raw, "canonical JSON record " + label)
    return value


def _strict_equal(actual, expected):
    """JSON equality with exact scalar types, especially for false flags."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(_strict_equal(actual[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(_strict_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


_API_LINES = (
    ("warp._src.context", "launch", 7354), ("warp._src.context", "copy", 8708),
    ("warp._src.context", "empty", 6708), ("warp._src.context", "get_device", 5854),
    ("warp._src.context", "get_stream", 6245), ("warp._src.context", "synchronize_stream", 7700),
    ("warp._src.context", "Module.__init__", 2395), ("warp._src.context", "Module.get_module_hash", 2611),
    ("warp._src.context", "Module._compile", 2713), ("warp._src.context", "Module.load", 2935),
    ("warp._src.context", "Module._get_compile_arch", 2653),
    ("warp._src.context", "Module._get_compile_output_name", 2659),
    ("warp._src.context", "Module._get_meta_name", 2706),
    ("warp._src.context", "Module.get_module_identifier", 2639),
    ("warp._src.context", "ModuleExec.__init__", 2279),
    ("warp._src.context", "ModuleExec.get_kernel_hooks", 2301),
    ("warp._src.context", "ModuleBuilder.__init__", 2092),
    ("warp._src.context", "ModuleBuilder.codegen", 2201),
    ("warp._src.context", "Kernel.__init__", 771), ("warp._src.context", "Kernel.get_mangled_name", 882),
    ("warp._src.types", "array.__init__", 2839), ("warp._src.types", "array.numpy", 3827),
    ("warp._src.build", "build_cuda", 40), ("warp._src.build", "load_cuda", 110),
)
_RUNTIME_METHODS = ["init", *["Runtime." + name for name in (
    "__init__", "get_error_string", "get_warp_version", "get_warp_clang_version", "get_llvm_version",
    "get_nanovdb_version", "get_host_compiler_version", "get_libmathdx_version", "get_nvrtc_version",
    "load_dll", "get_device", "set_default_device", "get_current_cuda_device", "rename_device",
    "map_cuda_device", "unmap_cuda_device", "verify_cuda_device",
)]]


def _source_entries(_source_map):
    """Pinned entries from the exact source hashes, independent of host paths."""
    return [dict(module=module, qualified=qualified, line=line) for module, qualified, line in _API_LINES]


def _capture_protocol():
    """Bind the capture wire ID to the checked-in original record implementation."""
    path = Path(dispatch.__file__).resolve(strict=True)
    method = dispatch.GradientStageCapture.record
    need(Path(method.__code__.co_filename).resolve(strict=True) == path,
         "capture recorder comes from checked-in dispatch source")
    tree = ast.parse(path.read_bytes(), filename=str(path))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GradientStageCapture"]
    need(len(classes) == 1, "unique original gradient capture class")
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == "record"]
    need(len(methods) == 1, "unique original gradient capture record method")
    returns = [n.value for n in ast.walk(methods[0]) if isinstance(n, ast.Return)]
    calls = [n for n in returns if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "dict"]
    need(len(calls) == 1 and any(k.arg == "protocol" and isinstance(k.value, ast.Name)
         and k.value.id == "PROTOCOL" for k in calls[0].keywords),
         "original capture record binds protocol to its module constant")
    constants = [n.value.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "PROTOCOL" for t in n.targets)
                 and isinstance(n.value, ast.Constant) and type(n.value.value) is str]
    need(constants == [dispatch.PROTOCOL], "original capture protocol literal matches imported source")
    return dispatch.PROTOCOL


def _cpu_tests(record, source, raw, inventory):
    need(type(record) is dict and set(record) == {"path", "bytes", "sha256", "evidence"}
         and type(record["path"]) is str and str(Path(record["path"])) == record["path"]
         and Path(record["path"]).is_absolute() and ".." not in Path(record["path"]).parts
         and Path(record["path"]).is_relative_to(probe.ROOT / "artifacts/tools")
         and type(record["bytes"]) is int and record["bytes"] > 0
         and type(record["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]),
         "whole paired CPU evidence envelope")
    evidence = record["evidence"]
    evidence_raw = old._canonical(evidence)
    need(len(evidence_raw) == record["bytes"] and sha256(evidence_raw).hexdigest() == record["sha256"],
         "canonical whole CPU evidence record")
    need(type(evidence) is dict and set(evidence) == {"source", "test_files", "mac", "native", "flags"}
         and evidence["source"] == source and evidence["test_files"] == list(probe.TESTS)
         and _strict_equal(evidence["flags"], probe.FLAGS), "source-bound paired focused CPU test evidence")
    cases = []
    for platform in ("mac", "native"):
        row = evidence[platform]
        need(type(row) is dict and set(row) == {"path", "bytes", "sha256", "tests"}
             and type(row["tests"]) is int and row["tests"] > 1199,
             "focused " + platform + " test collection")
        need(type(row["path"]) is str and str(Path(row["path"])) == row["path"],
             "canonical original CPU XML path")
        original_path = Path(row["path"])
        need(original_path.is_absolute() and ".." not in original_path.parts and original_path.suffix == ".xml"
             and original_path.is_relative_to(probe.ROOT / "artifacts/tools"),
             "retained original CPU XML path")
        name = platform + "-cpu.xml"
        need(name in raw and inventory[name] == {k: row[k] for k in ("bytes", "sha256")},
             "copied CPU XML matches original whole-file anchor")
        content = raw[name]
        tree = ET.fromstring(content)
        suites = [tree] if tree.tag == "testsuite" else list(tree)
        need(suites and all(s.tag == "testsuite" and all(int(s.get(k, "-1")) == 0
             for k in ("errors", "failures", "skipped")) for s in suites)
             and sum(int(s.get("tests", "-1")) for s in suites) == row["tests"],
             "CPU XML reports no failures/errors/skips")
        actual = tree.findall(".//testcase")
        need(len(actual) == row["tests"] and not any(tree.findall(".//" + n)
             for n in ("failure", "error", "skipped")), "complete CPU testcase rows")
        cases.append(Counter((c.get("classname"), c.get("name")) for c in actual))
    need(evidence["mac"]["tests"] == evidence["native"]["tests"] and cases[0] == cases[1],
         "paired CPU collection and case multiset")


def _artifact(raw, inventory, row, root):
    need(type(row) is dict and set(row) == {"path", "bytes", "sha256", "identity"},
         "whole generated gradient file anchor")
    need(type(row["path"]) is str and type(row["bytes"]) is int
         and type(row["sha256"]) is str, "plain generated file path and digest types")
    path = Path(row["path"])
    need(path.is_absolute() and str(path) == row["path"] and path.is_relative_to(root),
         "contained canonical generated gradient path")
    name = str(path.relative_to(root))
    need(name in raw and inventory[name] == {k: row[k] for k in ("bytes", "sha256")},
         "generated gradient bytes match inventory")
    identity = row["identity"]
    need(type(identity) is list and len(identity) == 5
         and all(type(x) is int and x > 0 for x in identity[:2])
         and identity[2] == len(raw[name]) and all(type(x) is int and x >= 0 for x in identity[2:]),
         "observed generated file identity schema")
    return raw[name]


def _check_source(source):
    need(type(source) is dict and set(source) == {"source", "branch", "tree", "leaves"}
         and type(source["source"]) is str and re.fullmatch(r"[0-9a-f]{40}", source["source"])
         and source["branch"] == probe.BRANCH and type(source["tree"]) is str
         and re.fullmatch(r"[0-9a-f]{40}", source["tree"]) and type(source["leaves"]) is list,
         "closed committed source inventory")
    seen = set()
    for leaf in source["leaves"]:
        need(type(leaf) is dict and set(leaf) == {"path", "bytes", "git_blob", "sha256"}
             and type(leaf["path"]) is str and leaf["path"] == str(Path(leaf["path"]))
             and leaf["path"] not in ("", ".") and all(ord(c) >= 32 for c in leaf["path"])
             and not Path(leaf["path"]).is_absolute() and ".." not in Path(leaf["path"]).parts
             and "\\" not in leaf["path"] and leaf["path"] not in seen
             and type(leaf["bytes"]) is int and leaf["bytes"] >= 0
             and type(leaf["git_blob"]) is str and re.fullmatch(r"[0-9a-f]{40}", leaf["git_blob"])
             and type(leaf["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", leaf["sha256"]),
             "unique canonical committed source leaf")
        seen.add(leaf["path"])
    need(probe.OWN <= seen, "new exact five-path gradient fence")


def _check_arm_records(arms, observed_bindings, raw, inventory, parent_initial, arm):
    need(type(arms) is dict and set(arms) == {"observer", "restoration"}, "closed arm observer/restoration record")
    need(_strict_equal(arms["restoration"], probe.restoration_record(parent_initial)),
         "exact independently reauthenticated parent restoration record")
    observer = arms["observer"]
    need(type(observer) is dict and set(observer) == {
        "protocol", "base", "caller", "path", "call_site_lines", "excluded_next_branch_line", "stop",
        "call_sites_observed", "intentional_stop_observed", "normal_caller_completion", "next_branch_observed",
        "trace_hooks_installed", "incremental_iteration_observed", "bindings", "capture",
        "parent_capture_authenticated", "explicit_load_provenance_authenticated",
        "native_gpu_execution_authenticated", "external_unit_retirement_authenticated", "qualification",
    } and observer["protocol"] == dispatch.PROTOCOL and observer["base"] == dispatch.BASE
         and observer["caller"] == "_update_gradient" and observer["path"] == "initialization-prefix"
         and observer["call_site_lines"] == [2925, 2927] and observer["excluded_next_branch_line"] == 2934
         and observer["stop"] == "private-sentinel-after-second-launch-and-readback-before-caller-resumption"
         and observer["call_sites_observed"] is True and observer["intentional_stop_observed"] is True
         and observer["normal_caller_completion"] is False and observer["next_branch_observed"] is False
         and observer["trace_hooks_installed"] is False and observer["incremental_iteration_observed"] is False
         and observer["bindings"] == observed_bindings and _strict_equal(observer["qualification"], probe.FLAGS)
         and all(observer[k] is False for k in ("parent_capture_authenticated", "explicit_load_provenance_authenticated",
             "native_gpu_execution_authenticated", "external_unit_retirement_authenticated")),
         "exact two-site intentionally stopped gradient caller observer")
    capture = observer["capture"]
    need(type(capture) is dict and set(capture) == {
        "protocol", "stages", "fields", "packets", "timing_changed_by_readback", "copy_stream_handle",
        "capture_origin_authenticated", "gpu_dispatch_authenticated", "qualification",
    } and capture["protocol"] == _capture_protocol() and capture["stages"] == list(prefix.STAGES)
         and capture["fields"] == list(prefix.ORDER) and capture["timing_changed_by_readback"] is True
         and type(capture["copy_stream_handle"]) is int and capture["copy_stream_handle"] > 0
         and capture["capture_origin_authenticated"] is False and capture["gpu_dispatch_authenticated"] is False
         and _strict_equal(capture["qualification"], probe.FLAGS) and type(capture["packets"]) is dict
         and set(capture["packets"]) == set(prefix.PHASES), "complete four phase, 26 field capture receipt")
    anchors, packets = {}, {}
    for phase in prefix.PHASES:
        name = arm + "-" + phase + ".bin"
        anchor = capture["packets"][phase]
        need(name in raw and anchor == inventory[name], "phase packet matches authenticated inventory")
        anchors[phase], packets[phase] = anchor, raw[name]
    return packets, anchors


def receive(root, inventory, closeout, repository_root, solver_path):
    """Authenticate a complete output inventory, then independently check it."""
    raw = old.authenticate(root, inventory)  # must precede every parse and bank decode
    need({"receipt.json", "declaration.json", "supervision.json", "telemetry.json", "child.log"} <= set(raw)
         and "failure.json" not in raw, "complete successful gradient diagnostic envelope")
    receipt = _json(raw["receipt.json"], "gradient receipt")
    dcl = _json(raw["declaration.json"], "gradient declaration")
    need(type(receipt) is dict and set(receipt) == {
        "protocol", "source", "declaration_sha256", "owner_pid", "child_pid", "native_root", "device",
        "module", "arms", "api", "bootstrap", "caches", "flags",
    } and receipt["protocol"] == dcl.get("protocol") == probe.PROTOCOL
         and _strict_equal(receipt["flags"], probe.FLAGS) and _strict_equal(dcl.get("flags"), probe.FLAGS),
         "closed stopped-gradient protocol and six false qualification flags")
    need(type(dcl) is dict and set(dcl) == {
        "protocol", "source", "source_binding", "native_root", "owner_pid", "runtime", "service", "cpu_tests",
        "lease", "services", "baseline", "bounds", "deadline_unix", "replay_input", "parent",
        "exclusive_gpu_claimed", "flags",
    }, "closed gradient owner declaration")
    source = receipt["source"]
    need(type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source)
         and dcl.get("source") == source and receipt["declaration_sha256"] == sha256(raw["declaration.json"]).hexdigest(),
         "same literal source and whole declaration digest")
    native_root = probe.output(source)
    need(receipt["native_root"] == dcl.get("native_root") == str(native_root)
         and type(receipt["owner_pid"]) is int and type(receipt["child_pid"]) is int
         and 0 < receipt["owner_pid"] != receipt["child_pid"] > 0
         and receipt["owner_pid"] == dcl.get("owner_pid"), "same native output and distinct owner/child")
    _check_source(dcl.get("source_binding"))
    need(dcl["source_binding"]["source"] == source, "source inventory commit matches receipt")
    need(dcl["bounds"] == probe.BOUNDS and dcl["replay_input"] == probe.retained.INPUT
         and dcl["exclusive_gpu_claimed"] is False
         and type(dcl["deadline_unix"]) in (int, float) and math.isfinite(dcl["deadline_unix"])
         and dcl["deadline_unix"] > 0,
         "closed bounded declaration and shared-GPU claim")
    old._check_runtime(dcl["runtime"])
    service = dcl["service"]
    expected_unit = probe.unit(source)
    need(type(service) is dict and set(service) == {
        "name", *probe.UNIT_CAPS, "Id", "MainPID", "ActiveState", "ControlGroup", "InvocationID",
    } and service["name"] == service["Id"] == expected_unit
         and all(service.get(k) == v for k, v in probe.UNIT_CAPS.items())
         and service["MainPID"] == str(receipt["owner_pid"]) and service["ActiveState"] == "active"
         and service["ControlGroup"] == probe.retained.CGROUP_PARENT + expected_unit
         and type(service["InvocationID"]) is str and re.fullmatch(r"[0-9a-f]{32}", service["InvocationID"]),
         "same exact capped active owner unit and invocation")
    need(type(dcl["lease"]) is dict and set(dcl["lease"]) == {"device", "inode", "bytes"}
         and dcl["lease"] == {"device": 2096, "inode": 35886, "bytes": 0}
         and dcl["runtime"]["environment_alias"]["device"] == dcl["lease"]["device"],
         "same pre-existing shared GPU lease")
    baseline = dcl["baseline"]
    need(type(baseline) is dict and set(baseline) == {
        "wall_time_unix", "wsl", "windows", "windows_sampled_at", "windows_active_engines", "counters_simultaneous",
    } and type(baseline["wall_time_unix"]) in (int, float) and math.isfinite(baseline["wall_time_unix"])
         and type(baseline["windows_sampled_at"]) is str and type(baseline["windows_active_engines"]) is list
         and baseline["counters_simultaneous"] is False, "complete shared host capacity baseline")
    for name in ("wsl", "windows"):
        sample = baseline[name]
        need(type(sample) is dict and set(sample) == {
            "uuid", "driver", "total_mib", "used_mib", "free_mib", "utilization_percent", "temperature_c",
        } and sample["uuid"] == probe.shared.GPU and sample["driver"] == probe.shared.DRIVER
             and sample["total_mib"] == 24467 and type(sample["used_mib"]) is int
             and 0 <= sample["used_mib"] <= probe.BOUNDS["total_used_mib"]
             and type(sample["free_mib"]) is int and sample["free_mib"] >= probe.BOUNDS["free_reserve_mib"]
             and type(sample["utilization_percent"]) is int
             and sample["utilization_percent"] <= probe.BOUNDS["utilization_percent"]
             and type(sample["temperature_c"]) is int and sample["temperature_c"] < probe.BOUNDS["temperature_c"]
             and sample["used_mib"] + sample["free_mib"] <= sample["total_mib"],
             "same bounded shared capacity sample " + name)
    probe.shared.capacity(baseline)
    protected = {scope + name for name in probe.shared.PROTECTED for scope in ("system:", "user:")}
    filmbrain = set(probe.shared.FILMBRAIN)
    services = dcl["services"]
    need(type(services) is dict and set(services) == protected | filmbrain
         and all(services[name] == "inactive" for name in protected), "closed protected-service snapshot")
    for name in filmbrain:
        row = services[name]
        need(type(row) is dict and set(row) == {"ActiveState", "MainPID", "NRestarts"}
             and row["ActiveState"] == "active" and type(row["MainPID"]) is str
             and re.fullmatch(r"[1-9][0-9]*", row["MainPID"]) and row["NRestarts"] == "0",
             "retained active FilmBrain service PID/state " + name)
    _cpu_tests(dcl["cpu_tests"], source, raw, inventory)
    supervision = _json(raw["supervision.json"], "gradient supervision")
    need(type(supervision) is dict and set(supervision) == {
        "result", "kernel_cgroup_only_owner", "unit_retirement_independently_required", "flags",
    } and supervision["kernel_cgroup_only_owner"] is True
         and supervision["unit_retirement_independently_required"] is True
         and _strict_equal(supervision["flags"], probe.FLAGS), "successful child supervision and separate retirement gate")
    result = supervision["result"]
    need(type(result) is dict and set(result) == {
        "decision", "returncode", "root", "elapsed", "observed_session_members",
        "process_tree_retirement_proven", "native_qualified", "training_authorized",
    } and result["decision"] == "reviewed-owned-root-exited-no-native-qualification"
         and result["returncode"] == 0 and type(result["root"]) is list and len(result["root"]) == 2
         and result["root"][0] == receipt["child_pid"] and type(result["root"][1]) is int
         and result["root"][1] > 0 and type(result["elapsed"]) in (int, float)
         and math.isfinite(result["elapsed"]) and 0 <= result["elapsed"] <= probe.BOUNDS["child_seconds"]
         and result["observed_session_members"] == []
         and result["process_tree_retirement_proven"] is False and result["native_qualified"] is False
         and result["training_authorized"] is False, "successful supervised child identity and no qualification")
    telemetry = _json(raw["telemetry.json"], "gradient telemetry")
    need(type(telemetry) is dict and set(telemetry) == {"samples", "flags"}
         and _strict_equal(telemetry["flags"], probe.FLAGS) and type(telemetry["samples"]) is list
         and 0 < len(telemetry["samples"]) <= 1024, "bounded retained shared telemetry")
    for sample in [dcl["baseline"], *telemetry["samples"]]:
        need(type(sample) is dict and set(sample) == {
            "wall_time_unix", "wsl", "windows", "windows_sampled_at", "windows_active_engines", "counters_simultaneous",
        } and type(sample["wall_time_unix"]) in (int, float) and math.isfinite(sample["wall_time_unix"])
             and type(sample["windows_sampled_at"]) is str and type(sample["windows_active_engines"]) is list
             and sample["counters_simultaneous"] is False, "closed shared telemetry sample")
        for view in ("wsl", "windows"):
            current = sample[view]
            need(type(current) is dict and set(current) == {
                "uuid", "driver", "total_mib", "used_mib", "free_mib", "utilization_percent", "temperature_c",
            } and current["uuid"] == probe.shared.GPU and current["driver"] == probe.shared.DRIVER
                 and current["total_mib"] == 24467 and type(current["used_mib"]) is int
                 and type(current["free_mib"]) is int and type(current["utilization_percent"]) is int
                 and type(current["temperature_c"]) is int,
                 "closed bounded telemetry device " + view)
        probe.shared.capacity(sample, dcl["baseline"])

    device = receipt["device"]
    need(type(device) is dict and set(device) == {"alias", "arch", "context", "object_id", "gpu_uuid"}
         and device["alias"] == "cuda:0" and type(device["arch"]) is int and device["arch"] == 120
         and probe.shared.GPU == device["gpu_uuid"]
         and all(type(device[k]) is int and device[k] > 0 for k in ("context", "object_id")),
         "literal mapped GPU and context identity")
    module = receipt["module"]
    need(type(module) is dict and set(module) == {
        "protocol", "role", "generated", "module_hash", "offline_disassembly", "explicit_load", "bindings",
        "flags", "loaded_binary_bytes_observed", "driver_jit_machine_code_observed", "actual_dispatch_observed",
    } and module["protocol"] == executable.PROTOCOL and module["role"] == "gradient"
         and _strict_equal(module["flags"], probe.FLAGS)
         and all(module[k] is False for k in ("loaded_binary_bytes_observed", "driver_jit_machine_code_observed", "actual_dispatch_observed"))
         and set(module["generated"]) == {"binary", "metadata", "source"}
         and set(module["bindings"]) == set(module["offline_disassembly"]) == set(prefix.STAGES),
         "closed one-module two-binding executable record without driver claims")
    content = {n: _artifact(raw, inventory, row, native_root) for n, row in module["generated"].items()}
    need(all(Path(row["path"]).parent == native_root / "compiled-gradient" for row in module["generated"].values()),
         "all three generated files in exact compiled-gradient directory")
    load = module["explicit_load"]
    need(type(load) is dict and set(load) == {
        "binary_path", "metadata_path", "output_arch", "block_dim", "returned_executable_is_cache_entry",
        "module_object_id", "executable_object_id", "device_object_id",
    } and load["binary_path"] == module["generated"]["binary"]["path"]
         and load["metadata_path"] == module["generated"]["metadata"]["path"]
         and load["output_arch"] == 120 and load["block_dim"] == 256
         and load["returned_executable_is_cache_entry"] is True
         and load["device_object_id"] == device["object_id"]
         and all(type(load[k]) is int and load[k] > 0 for k in ("module_object_id", "executable_object_id")),
         "exact explicit loader paths and observed module/executable/device identities")
    metadata = old._json(content["metadata"], "gradient metadata")
    observed = {}
    need(type(module["module_hash"]) is str and re.fullmatch(r"[0-9a-f]{64}", module["module_hash"]),
         "whole gradient module hash")
    for stage in prefix.STAGES:
        binding = module["bindings"][stage]
        need(type(binding) is dict and set(binding) == {
            "protocol", "artifact_format", "binary_sha256", "binary_path", "binary_bytes", "metadata_sha256",
            "metadata_path", "metadata_bytes", "module_hash", "block_dim", "context", "module_handle",
            "forward_handle", "forward_smem_bytes", "symbol", "device", "device_arch", "observed_object_ids",
            "driver_jit_machine_code_observed", "loaded_binary_bytes_observed", "native_execution_qualified",
            "training_authorized", "physical_acceptance",
        } and binding["protocol"] == executable.artifacts.PROTOCOL and binding["artifact_format"] == "cubin"
             and binding["binary_path"] == load["binary_path"] and binding["metadata_path"] == load["metadata_path"]
             and binding["binary_sha256"] == sha256(content["binary"]).hexdigest()
             and binding["metadata_sha256"] == sha256(content["metadata"]).hexdigest()
             and binding["binary_bytes"] == len(content["binary"]) and binding["metadata_bytes"] == len(content["metadata"])
             and binding["module_hash"] == module["module_hash"] and binding["context"] == device["context"]
             and binding["device"] == device["alias"] and binding["device_arch"] == 120 and binding["block_dim"] == 256
             and type(binding["observed_object_ids"]) is dict
             and set(binding["observed_object_ids"]) == {"kernel", "module", "device", "executable", "hooks"}
             and binding["observed_object_ids"] == {
                 "kernel": binding["observed_object_ids"]["kernel"], "module": load["module_object_id"],
                 "device": device["object_id"], "executable": load["executable_object_id"],
                 "hooks": binding["observed_object_ids"]["hooks"],
             } and all(type(binding["observed_object_ids"].get(k)) is int and binding["observed_object_ids"][k] > 0
                       for k in ("kernel", "module", "device", "executable", "hooks"))
             and all(type(binding[k]) is int and binding[k] > 0 for k in ("module_handle", "forward_handle"))
             and type(binding["forward_smem_bytes"]) is int and binding["forward_smem_bytes"] >= 0
             and metadata.get(binding["symbol"] + "_smem_bytes") == binding["forward_smem_bytes"]
             and all(binding[k] is False for k in ("driver_jit_machine_code_observed", "loaded_binary_bytes_observed",
                                                   "native_execution_qualified", "training_authorized", "physical_acceptance")),
             "same observed entry-point binary/module/object bindings")
        sass_name = stage + ".sass"
        need(sass_name in raw and raw.get(stage + ".stderr") == b"", "retained successful offline disassembler output")
        checked = binary.verify_disassembly(content["binary"], raw[sass_name], stage, binding["symbol"])
        need(checked == module["offline_disassembly"][stage], "independently reproduced whole CUBIN function span")
        observed[stage] = binding

    api_record = receipt["api"]
    need(type(api_record) is dict and set(api_record) == {
        "protocol", "sources", "entries", "selected_python_bodies_authenticated", "held_direct_globals_checked",
        "transitive_dependencies_authenticated", "initialized_runtime_origin_authenticated", "native_execution_observed", "flags",
    } and api_record["protocol"] == api.PROTOCOL and _strict_equal(api_record["flags"], probe.FLAGS)
         and api_record["selected_python_bodies_authenticated"] is True and api_record["held_direct_globals_checked"] is True
         and api_record["transitive_dependencies_authenticated"] is False
         and api_record["initialized_runtime_origin_authenticated"] is False and api_record["native_execution_observed"] is False,
         "closed selected API boundary record with limited claims")
    source_rows = api_record["sources"]
    need(type(source_rows) is dict and set(source_rows) == set(api.PINS), "exact four pinned Warp source files")
    runtime_sources = dcl["runtime"]["warp_sources"]
    for name, pin in api.PINS.items():
        rel = "__init__.py" if name == "warp" else name.removeprefix("warp.").replace(".", "/") + ".py"
        leaf = runtime_sources["leaves"].get(rel)
        row = source_rows[name]
        need(type(row) is dict and set(row) == {"path", "bytes", "sha256"}
             and row["path"] == runtime_sources["root"] + "/" + rel
             and row["bytes"] == pin[0] and row["sha256"] == pin[1]
             and type(leaf) is dict and leaf == {"bytes": pin[0], "sha256": pin[1]},
             "API source agrees with exact frozen runtime tree pin")
    need(api_record["entries"] == _source_entries(source_rows), "exact 24 qualified API entries and pinned source line IDs")

    bootstrap = receipt["bootstrap"]
    need(type(bootstrap) is dict and set(bootstrap) == {
        "protocol", "methods", "fresh_runtime_before_init", "selected_bootstrap_bodies_checked",
        "held_constructor_completed", "transitive_dependencies_authenticated", "flags",
    } and bootstrap["protocol"] == probe.BOOTSTRAP_PROTOCOL and bootstrap["methods"] == _RUNTIME_METHODS
         and bootstrap["fresh_runtime_before_init"] is True
         and bootstrap["selected_bootstrap_bodies_checked"] is True and bootstrap["held_constructor_completed"] is True
         and bootstrap["transitive_dependencies_authenticated"] is False
         and _strict_equal(bootstrap["flags"], probe.FLAGS),
         "exact fresh Runtime bootstrap methods and bounded claims")

    caches = receipt["caches"]
    need(type(caches) is dict and set(caches) == {"roots", "absent_before_init", "leaves"}
         and caches["roots"] == list(probe.CACHE_NAMES) and caches["absent_before_init"] is True
         and type(caches["leaves"]) is dict and set(caches["leaves"]) <= set(inventory),
         "exact private cache root and inventory leaf set")
    for name, row in caches["leaves"].items():
        need(type(name) is str and not Path(name).is_absolute() and ".." not in Path(name).parts
             and len(Path(name).parts) >= 2
             and Path(name).parts[0] in probe.CACHE_NAMES and type(row) is dict
             and set(row) == {"bytes", "sha256"} and type(row["bytes"]) is int
             and type(row["sha256"]) is str and inventory[name] == row,
             "contained whole private cache leaf anchor")
    cache_inventory = {name for name in inventory if Path(name).parts[0] in probe.CACHE_NAMES}
    need(set(caches["leaves"]) == cache_inventory, "complete authenticated private cache leaf subset")

    parent, parent_gauss, initial, packet = probe.parent_inputs(repository_root, solver_path)
    need(_strict_equal(dcl["parent"], parent), "literal parent receipt reauthenticated after new inventory")
    need(type(receipt["arms"]) is dict and set(receipt["arms"]) == {"reference", "control"}, "exact two gradient arms")
    packets, anchors = {}, {}
    for arm in ("reference", "control"):
        packets[arm], anchors[arm] = _check_arm_records(receipt["arms"][arm], observed, raw, inventory, initial[arm], arm)
    generated = {str(Path(row["path"]).relative_to(native_root)) for row in module["generated"].values()}
    expected_files = {
        "receipt.json", "declaration.json", "supervision.json", "telemetry.json", "child.log",
        "mac-cpu.xml", "native-cpu.xml",
        *(arm + "-" + phase + ".bin" for arm in ("reference", "control") for phase in prefix.PHASES),
        *(stage + suffix for stage in prefix.STAGES for suffix in (".sass", ".stderr")),
        *generated,
        *cache_inventory,
    }
    need(set(raw) == expected_files, "complete known gradient outputs with no unclassified inventory leaves")
    result = prefix.receive_banks(packets, anchors, parent_gauss, packet)
    efc_receiver._retirement(closeout, dcl, inventory)
    return dict(protocol=PROTOCOL, source=source, decision="positive-gradient-prefix-diagnostic-received-not-qualification",
                artifact_files=len(inventory), artifact_bytes=sum(row["bytes"] for row in inventory.values()),
                inventory_sha256=sha256(old._canonical(inventory)).hexdigest(), banks=result,
                external_retirement_evidence_checked=True, timing_changed_by_readback=True,
                loaded_driver_bytes_observed=False, runtime_cause_proven=False, qualification=dict(probe.FLAGS))
