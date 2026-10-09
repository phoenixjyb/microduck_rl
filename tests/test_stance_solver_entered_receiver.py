"""Synthetic outer-envelope checks only; never native or GPU qualification."""

from hashlib import sha256
import ast
import json
from pathlib import Path
import sys
import sysconfig
import xml.etree.ElementTree as ET

import pytest

from test_stance_solver_cost_binary import cubin, sass
from mjlab_microduck import stance_solver_entered_receiver as receiver
from mjlab_microduck import stance_solver_entered_probe as probe
from mjlab_microduck import stance_solver_bootstrap_observer as bootstrap_observer
from mjlab_microduck import stance_solver_gradient_prefix as prefix
from mjlab_microduck import stance_solver_gradient_dispatch as dispatch
from mjlab_microduck import stance_solver_gradient_runtime as api
from mjlab_microduck import stance_solver_gradient_executable as executable
from mjlab_microduck import stance_solver_gradient_binary as binary
from mjlab_microduck import stance_solver_replay_receiver as old
import test_stance_solver_gradient_prefix as prefix_tests


def _json(value):
    return old._canonical(value)


def _capacity():
    sample = dict(uuid=probe.shared.GPU, driver=probe.shared.DRIVER, total_mib=24467, used_mib=1000,
                  free_mib=23467, utilization_percent=0, temperature_c=30)
    return dict(wall_time_unix=1.0, wsl=sample, windows=sample, windows_sampled_at="sample",
                windows_active_engines=[], counters_simultaneous=False)


def _module_record(root):
    symbols = {stage: binary.PREFIXES[stage] + "_1234abcd_cuda_kernel_forward" for stage in prefix.STAGES}
    codes = {stage: bytes(range(i * 16, (i + 1) * 16)) for i, stage in enumerate(prefix.STAGES)}
    cubin_bytes = cubin([(symbols[s], codes[s]) for s in prefix.STAGES])
    metadata_bytes = _json({symbols[s] + "_smem_bytes": 0 for s in prefix.STAGES})
    source_bytes = b"// synthetic compile source\n"
    generated_bytes = {"binary": cubin_bytes, "metadata": metadata_bytes, "source": source_bytes}
    paths = {name: root / "compiled-gradient" / (name + {"binary": ".cubin", "metadata": ".json", "source": ".cu"}[name])
             for name in generated_bytes}
    generated = {}
    for name, content in generated_bytes.items():
        path = paths[name]
        generated[name] = dict(path=str(path), bytes=len(content), sha256=sha256(content).hexdigest(),
                               identity=[1, 1, len(content), 0, 0])
    digest = "a" * 64
    device = dict(alias="cuda:0", arch=120, context=7, object_id=17, gpu_uuid=probe.shared.GPU)
    bindings, offline = {}, {}
    for stage in prefix.STAGES:
        checked = binary.verify_disassembly(cubin_bytes, sass(symbols[stage], codes[stage]), stage, symbols[stage])
        offline[stage] = checked
        bindings[stage] = dict(protocol=executable.artifacts.PROTOCOL, artifact_format="cubin",
            binary_sha256=sha256(cubin_bytes).hexdigest(), binary_path=str(paths["binary"]), binary_bytes=len(cubin_bytes),
            metadata_sha256=sha256(metadata_bytes).hexdigest(), metadata_path=str(paths["metadata"]), metadata_bytes=len(metadata_bytes),
            module_hash=digest, block_dim=256, context=device["context"], module_handle=30 + len(bindings),
            forward_handle=40 + len(bindings), forward_smem_bytes=0, symbol=symbols[stage], device="cuda:0",
            device_arch=120, observed_object_ids=dict(kernel=50 + len(bindings), module=21, device=17,
                executable=22, hooks=60 + len(bindings)), driver_jit_machine_code_observed=False,
            loaded_binary_bytes_observed=False, native_execution_qualified=False, training_authorized=False,
            physical_acceptance=False)
    record = dict(protocol=executable.PROTOCOL, role="gradient", generated=generated, module_hash=digest,
        offline_disassembly=offline, explicit_load=dict(binary_path=str(paths["binary"]), metadata_path=str(paths["metadata"]),
            output_arch=120, block_dim=256, returned_executable_is_cache_entry=True, module_object_id=21,
            executable_object_id=22, device_object_id=17), bindings=bindings, flags=dict(probe.FLAGS),
        loaded_binary_bytes_observed=False, driver_jit_machine_code_observed=False, actual_dispatch_observed=False)
    return record, generated_bytes, symbols, codes


def _write_inventory(root, raw):
    for name, content in raw.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _definition_line(content, qualified):
    tree = ast.parse(content)
    parts = qualified.split(".")
    nodes = tree.body
    for part in parts[:-1]:
        node = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == part)
        nodes = node.body
    node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == parts[-1])
    return node.lineno


def test_pinned_api_line_map_matches_installed_source_bytes_without_importing_warp():
    root = Path(sys.prefix) / "lib/python3.12/site-packages/warp"
    trees = {}
    for module, (size, digest) in api.PINS.items():
        rel = "__init__.py" if module == "warp" else module.removeprefix("warp.").replace(".", "/") + ".py"
        content = (root / rel).read_bytes()
        assert (len(content), sha256(content).hexdigest()) == (size, digest)
        trees[module] = ast.parse(content)
    for module, qualified, expected_line in receiver._API_LINES:
        parts = qualified.split(".")
        nodes = trees[module].body
        if len(parts) == 2:
            cls = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == parts[0])
            nodes = cls.body
            name = parts[1]
        else:
            name = parts[0]
        actual = next(node.lineno for node in nodes
                      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and node.name == name)
        assert actual == expected_line


@pytest.fixture
def envelope(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(probe, "ROOT", repo)
    source = "a" * 40
    native_root = probe.output(source)
    output = repo / "received-entered-stopped-gradient"
    output.mkdir(parents=True)
    original_authenticate = old.authenticate
    original_receive_banks = prefix.receive_banks

    owner, child = 101, 202
    service_name = probe.unit(source)
    service = dict(name=service_name, **probe.UNIT_CAPS, Id=service_name, MainPID=str(owner),
                   ActiveState="active", ControlGroup=probe.retained.CGROUP_PARENT + service_name,
                   InvocationID="1" * 32)
    source_binding = dict(source=source, branch=probe.BRANCH, tree="b" * 40, leaves=[
        dict(path=name, bytes=1, git_blob="c" * 40, sha256="d" * 64) for name in sorted(probe.OWN)])
    baseline = _capacity()
    prior_declaration = json.loads((Path(__file__).resolve().parents[1] /
        "artifacts/evaluations/efc-transition-7717b4cbef3a/declaration.json").read_bytes())
    runtime = prior_declaration["runtime"]
    cpu_xml = ("<testsuite tests='1200' failures='0' errors='0' skipped='0'>" +
               "".join(f"<testcase classname='cpu' name='case{i}'/>" for i in range(1200)) + "</testsuite>").encode()
    test_files = list(probe.TESTS)
    evidence = dict(source=source, test_files=test_files, flags=dict(probe.FLAGS))
    for platform in ("mac", "native"):
        evidence[platform] = dict(path=str(repo / "artifacts/tools" / (platform + "-cpu.xml")), bytes=len(cpu_xml),
                                  sha256=sha256(cpu_xml).hexdigest(), tests=1200)
    evidence_record = _json(evidence)
    cpu_record = dict(path=str(repo / "artifacts/tools" / "gradient-cpu-evidence.json"),
        bytes=len(evidence_record), sha256=sha256(evidence_record).hexdigest(), evidence=evidence)
    dcl = dict(protocol=probe.PROTOCOL, source=source, source_binding=source_binding, native_root=str(native_root),
        owner_pid=owner, runtime=runtime, service=service,
        cpu_tests=cpu_record, lease=dict(device=2096, inode=35886, bytes=0),
        services={"system:recomo-ai-mission-vllm.service": "inactive",
          "user:recomo-ai-mission-vllm.service": "inactive",
          "system:recomo-ai-mission-subject-model-worker.service": "inactive",
          "user:recomo-ai-mission-subject-model-worker.service": "inactive",
          "recomo-filmbrain-observatory.service": {"ActiveState": "active", "MainPID": str(owner + 1000), "NRestarts": "0"},
          "recomo-filmbrain-video-playground.service": {"ActiveState": "active", "MainPID": str(owner + 2000), "NRestarts": "0"}},
        baseline=baseline, bounds=probe.BOUNDS, deadline_unix=1000.0, replay_input=probe.retained.INPUT,
        parent={}, exclusive_gpu_claimed=False, flags=dict(probe.FLAGS))

    initial = {arm: bytes(prefix.PACKET_BYTES) for arm in ("reference", "control")}
    parent_gauss = {arm: bytes(1) for arm in initial}
    packet = object()
    parent_record = {"literal": "parent"}
    monkeypatch.setattr(probe, "parent_inputs", lambda root, solver: (parent_record, parent_gauss, initial, packet))
    dcl["parent"] = parent_record
    module, generated, symbols, codes = _module_record(native_root)
    source_rows = {}
    for name, (size, digest) in api.PINS.items():
        rel = "__init__.py" if name == "warp" else name.removeprefix("warp.").replace(".", "/") + ".py"
        source_rows[name] = dict(path=old.FROZEN_WARP_ROOT + "/" + rel, bytes=size, sha256=digest)
    runtime_api = dict(protocol=api.PROTOCOL, sources=source_rows, entries=receiver._source_entries(source_rows),
        selected_python_bodies_authenticated=True, held_direct_globals_checked=True,
        transitive_dependencies_authenticated=False, initialized_runtime_origin_authenticated=False,
        native_execution_observed=False, flags=dict(probe.FLAGS))
    runtime_methods = list(receiver._RUNTIME_METHODS)
    bootstrap = dict(protocol=probe.BOOTSTRAP_PROTOCOL, methods=runtime_methods, fresh_runtime_before_init=True,
        selected_bootstrap_bodies_checked=True, held_constructor_completed=True,
        transitive_dependencies_authenticated=False, flags=dict(probe.FLAGS))
    runtime_warp = runtime["warp_sources"]
    installed_warp = Path(sys.prefix).resolve() / "lib/python3.12/site-packages/warp"
    entered_frames = []
    for warp_module, rel, qualified in (
        ("warp._src.context", "_src/context.py", "init"),
        ("warp._src.context", "_src/context.py", "Runtime.__init__"),
        ("warp._src.context", "_src/context.py", "Device.__init__"),
    ):
        content = (installed_warp / rel).read_bytes()
        anchor = runtime_warp["leaves"][rel]
        assert len(content) == anchor["bytes"] and sha256(content).hexdigest() == anchor["sha256"]
        entered_frames.append(dict(kind="warp", module=warp_module, qualified=qualified,
            line=_definition_line(content, qualified), path=str(Path(runtime_warp["root"]) / rel), calls=1))
    warp_module, rel, qualified = "warp._src.build", "_src/build.py", "init_kernel_cache"
    content = (installed_warp / rel).read_bytes()
    anchor = runtime_warp["leaves"][rel]
    assert len(content) == anchor["bytes"] and sha256(content).hexdigest() == anchor["sha256"]
    entered_frames.append(dict(kind="warp", module=warp_module, qualified=qualified,
        line=_definition_line(content, qualified), path=str(Path(runtime_warp["root"]) / rel), calls=1))
    stdlib_root = Path(sysconfig.get_path("stdlib")).resolve()
    stdlib_path = stdlib_root / "posixpath.py"
    stdlib_bytes = stdlib_path.read_bytes()
    stdlib_anchor = dict(bytes=len(stdlib_bytes), sha256=sha256(stdlib_bytes).hexdigest())
    entered_frames.append(dict(kind="direct-stdlib", module="posixpath", qualified="join",
        line=_definition_line(stdlib_bytes, "join"), path=str(stdlib_path), calls=1))
    entered_frames.sort(key=lambda r: (r["kind"], r["module"], r["qualified"], r["line"], r["path"]))
    entered_bootstrap = dict(observer=dict(protocol=bootstrap_observer.PROTOCOL,
        scope="main-thread-entered-Warp-and-direct-stdlib-Python-during-init",
        warp_source_tree_sha256="4aa3c865b7e523e1c0bef175f80ed51b00543569246d524914e78c333cdf9e6c",
        frames=entered_frames, stdlib_sources={str(stdlib_path): stdlib_anchor}, python_call_events=5,
        profile_removed=True, timing_changed_by_profile=True, entered_python_bodies_checked=True,
        whole_transitive_dependencies_authenticated=False, native_c_bodies_authenticated=False,
        stdlib_origin_authenticated=False, callable_defaults_authenticated=False,
        other_threads_observed=False, gpu_only_paths_observed=False, qualification=dict(probe.FLAGS)),
        stdlib_root=str(stdlib_root))
    stdlib_leaf = "bootstrap-stdlib/" + stdlib_anchor["sha256"] + ".py"
    cache_record = dict(roots=list(probe.CACHE_NAMES), absent_before_init=True, leaves={})

    raw = {"declaration.json": _json(dcl), "mac-cpu.xml": cpu_xml, "native-cpu.xml": cpu_xml,
           stdlib_leaf: stdlib_bytes,
           "child.log": b"synthetic joined child log\n"}
    for arm in ("reference", "control"):
        capture_packets = {}
        for phase in prefix.PHASES:
            name = arm + "-" + phase + ".bin"
            raw[name] = bytes(prefix.PACKET_BYTES)
            capture_packets[phase] = dict(bytes=prefix.PACKET_BYTES, sha256=sha256(raw[name]).hexdigest())
        observer = dict(protocol=dispatch.PROTOCOL, base=dispatch.BASE, caller="_update_gradient",
            path="initialization-prefix", call_site_lines=[2925, 2927], excluded_next_branch_line=2934,
            stop="private-sentinel-after-second-launch-and-readback-before-caller-resumption",
            call_sites_observed=True, intentional_stop_observed=True, normal_caller_completion=False,
            next_branch_observed=False, trace_hooks_installed=False, incremental_iteration_observed=False,
            bindings=module["bindings"], capture=dict(protocol=dispatch.PROTOCOL, stages=list(prefix.STAGES),
                fields=list(prefix.ORDER), packets=capture_packets, timing_changed_by_readback=True,
                copy_stream_handle=9, capture_origin_authenticated=False, gpu_dispatch_authenticated=False,
                qualification=dict(probe.FLAGS)), parent_capture_authenticated=False,
            explicit_load_provenance_authenticated=False, native_gpu_execution_authenticated=False,
            external_unit_retirement_authenticated=False, qualification=dict(probe.FLAGS))
        arms_record = dict(observer=observer, restoration=probe.restoration_record(initial[arm]))
        if arm == "reference":
            reference_arm = arms_record
        else:
            control_arm = arms_record

    for name, content in generated.items():
        raw[str(Path(module["generated"][name]["path"]).relative_to(native_root))] = content
    for stage in prefix.STAGES:
        raw[stage + ".sass"] = sass(symbols[stage], codes[stage])
        raw[stage + ".stderr"] = b""
    supervision = dict(result=dict(decision="reviewed-owned-root-exited-no-native-qualification", returncode=0,
        root=[child, 303], elapsed=2.0, observed_session_members=[], process_tree_retirement_proven=False,
        native_qualified=False, training_authorized=False), kernel_cgroup_only_owner=True,
        unit_retirement_independently_required=True, flags=dict(probe.FLAGS))
    telemetry = dict(samples=[baseline], flags=dict(probe.FLAGS))

    receipt = dict(protocol=probe.PROTOCOL, source=source, declaration_sha256="", owner_pid=owner, child_pid=child,
        native_root=str(native_root), device=dict(alias="cuda:0", arch=120, context=7, object_id=17, gpu_uuid=probe.shared.GPU),
        module=module, arms={"reference": reference_arm, "control": control_arm}, api=runtime_api,
        bootstrap=bootstrap, entered_bootstrap=entered_bootstrap, caches=cache_record, flags=dict(probe.FLAGS))

    def seal():
        raw["declaration.json"] = _json(dcl)
        receipt["declaration_sha256"] = sha256(raw["declaration.json"]).hexdigest()
        raw["receipt.json"] = _json(receipt)
        raw["supervision.json"] = _json(supervision)
        raw["telemetry.json"] = _json(telemetry)
        inventory = {name: dict(bytes=len(value), sha256=sha256(value).hexdigest()) for name, value in raw.items()}
        closeout = dict(source=source, declaration_sha256=sha256(raw["declaration.json"]).hexdigest(),
            inventory_sha256=sha256(old._canonical(inventory)).hexdigest(),
            service={**probe.UNIT_CAPS, "Id": service_name, "InvocationID": service["InvocationID"],
                "MainPID": "0", "ExecMainStatus": "0", "Result": "success", "ControlGroup": "",
                "ActiveState": "inactive", "SubState": "dead"},
            cgroup={"path": service["ControlGroup"], "absent": True, "observed_pids": []})
        return inventory, closeout

    inventory, closeout = seal()
    monkeypatch.setattr(old, "authenticate", lambda root_arg, inv: raw)
    monkeypatch.setattr(prefix, "receive_banks", lambda arms, anchors, parent, pkt: dict(decision="synthetic packet fixture"))
    return dict(root=output, repository=repo, solver=repo / "solver.py", source=source, raw=raw,
                native_root=native_root, initial=initial, parent_gauss=parent_gauss, packet=packet,
                parent_record=parent_record, inventory=inventory, closeout=closeout,
                receipt=receipt, declaration=dcl, supervision=supervision, telemetry=telemetry,
                seal=seal, original_authenticate=original_authenticate,
                original_receive_banks=original_receive_banks)


def test_synthetic_entered_bootstrap_envelope_is_received_without_qualification(envelope):
    result = receiver.receive(envelope["root"], envelope["inventory"], envelope["closeout"],
                              envelope["repository"], envelope["solver"])
    assert result["decision"] == "positive-entered-stopped-gradient-prefix-diagnostic-received-not-qualification"
    assert result["qualification"] == probe.FLAGS
    assert result["external_retirement_evidence_checked"] is True


def test_real_inventory_and_all_eight_prospective_banks(envelope, monkeypatch):
    """Authenticate files and run the actual packet contract using CPU fixtures."""
    parents = prefix_tests.parents.__wrapped__()
    packet = prefix_tests.packet.__wrapped__()
    prospective = prefix_tests.prospective.__wrapped__(parents, packet)
    initial = prefix.initial_banks(parents, packet)
    parent_record = dict(prior_source=prefix.PRIOR_SOURCE, inventory_sha256=prefix.PRIOR_INVENTORY_SHA256,
        retirement_sha256=prefix.PRIOR_RETIREMENT_SHA256, receiver_sha256=prefix.PRIOR_RESULT_SHA256,
        initial_banks={arm: dict(bytes=len(raw), sha256=sha256(raw).hexdigest()) for arm, raw in initial.items()},
        gauss_banks={arm: dict(bytes=len(raw), sha256=sha256(raw).hexdigest()) for arm, raw in parents.items()})
    envelope["declaration"]["parent"] = parent_record
    for arm in ("reference", "control"):
        observer = envelope["receipt"]["arms"][arm]["observer"]
        observer["capture"]["packets"] = prospective.anchors[arm]
        envelope["receipt"]["arms"][arm]["restoration"] = probe.restoration_record(initial[arm])
        for phase, raw in prospective.arms[arm].items():
            envelope["raw"][arm + "-" + phase + ".bin"] = raw
    envelope["inventory"], envelope["closeout"] = envelope["seal"]()
    monkeypatch.setattr(probe, "parent_inputs", lambda root, solver: (parent_record, parents, initial, packet))
    monkeypatch.setattr(old, "authenticate", envelope["original_authenticate"])
    monkeypatch.setattr(prefix, "receive_banks", envelope["original_receive_banks"])
    _write_inventory(envelope["root"], envelope["raw"])

    result = receiver.receive(envelope["root"], envelope["inventory"], envelope["closeout"],
                              envelope["repository"], envelope["solver"])
    assert result["banks"]["total_bytes"] == 8 * prefix.PACKET_BYTES
    assert result["banks"]["arms"]["control"]["gradient_replacements"] == 1280
    assert not any(result["banks"]["qualification"].values())
    assert not any(result["qualification"].values())


@pytest.mark.parametrize("fault", ["missing", "extra", "tampered", "anchor-type"])
def test_real_inventory_rejects_missing_extra_and_tampered_leaf_before_decode(envelope, monkeypatch, fault):
    _write_inventory(envelope["root"], envelope["raw"])
    if fault == "missing":
        (envelope["root"] / "child.log").unlink()
    elif fault == "extra":
        (envelope["root"] / "extra.bin").write_bytes(b"extra")
    elif fault == "tampered":
        (envelope["root"] / "child.log").write_bytes(b"tampered")
    else:
        envelope["inventory"]["child.log"]["bytes"] = True
    monkeypatch.setattr(old, "authenticate", envelope["original_authenticate"])
    decoded = []
    monkeypatch.setattr(receiver, "_json", lambda *args: decoded.append(args))
    with pytest.raises(ValueError):
        receiver.receive(envelope["root"], envelope["inventory"], envelope["closeout"],
                         envelope["repository"], envelope["solver"])
    assert decoded == []


@pytest.mark.parametrize("mutation", ["protocol", "site", "capture-protocol", "flags", "flag-types", "cache", "loader", "api", "bootstrap", "entered-bootstrap", "cpu", "source", "source-path", "generated-path", "artifact-identity"])
def test_reanchored_outer_claim_mutations_are_refused(envelope, mutation):
    receipt = envelope["receipt"]
    if mutation == "protocol":
        receipt["protocol"] = envelope["declaration"]["protocol"] = "invented-protocol"
    elif mutation == "site":
        receipt["arms"]["reference"]["observer"]["call_site_lines"] = [2925, 2928]
    elif mutation == "capture-protocol":
        receipt["arms"]["reference"]["observer"]["capture"]["protocol"] = prefix.PROTOCOL
    elif mutation == "flags":
        changed = dict(probe.FLAGS, native_qualified=True)
        receipt["flags"] = changed
        envelope["declaration"]["flags"] = changed
    elif mutation == "flag-types":
        receipt["flags"]["native_qualified"] = 0
        envelope["declaration"]["flags"]["native_qualified"] = 0
    elif mutation == "cache":
        receipt["caches"]["leaves"]["warp-cache/omitted.bin"] = {"bytes": 0, "sha256": sha256(b"").hexdigest()}
    elif mutation == "loader":
        receipt["module"]["explicit_load"]["binary_path"] += ".changed"
    elif mutation == "api":
        receipt["api"]["entries"][0]["line"] += 1
    elif mutation == "bootstrap":
        receipt["bootstrap"]["methods"].pop()
    elif mutation == "entered-bootstrap":
        receipt["entered_bootstrap"]["observer"]["profile_removed"] = 1
    elif mutation == "cpu":
        envelope["declaration"]["cpu_tests"]["sha256"] = "0" * 64
    elif mutation == "source":
        envelope["declaration"]["source_binding"]["leaves"].pop()
    elif mutation == "source-path":
        envelope["declaration"]["source_binding"]["leaves"][0]["path"] = "."
    elif mutation == "generated-path":
        envelope["receipt"]["module"]["generated"]["binary"]["path"] = str(
            envelope["native_root"] / "compiled-gradient" / "../compiled-gradient/binary.cubin")
    elif mutation == "artifact-identity":
        envelope["receipt"]["module"]["generated"]["binary"]["identity"][0] = True
    envelope["inventory"], envelope["closeout"] = envelope["seal"]()
    with pytest.raises(ValueError):
        receiver.receive(envelope["root"], envelope["inventory"], envelope["closeout"],
                         envelope["repository"], envelope["solver"])


@pytest.mark.parametrize("surface", ["receipt", "declaration", "api", "bootstrap", "entered-bootstrap", "module",
                                      "observer", "capture", "telemetry", "supervision", "cpu"])
def test_every_false_flag_surface_rejects_integer_zero(envelope, surface):
    key = next(iter(probe.FLAGS))
    if surface == "receipt":
        envelope["receipt"]["flags"][key] = 0
    elif surface == "declaration":
        envelope["declaration"]["flags"][key] = 0
    elif surface == "api":
        envelope["receipt"]["api"]["flags"][key] = 0
    elif surface == "bootstrap":
        envelope["receipt"]["bootstrap"]["flags"][key] = 0
    elif surface == "entered-bootstrap":
        envelope["receipt"]["entered_bootstrap"]["observer"]["qualification"][key] = 0
    elif surface == "module":
        envelope["receipt"]["module"]["flags"][key] = 0
    elif surface == "observer":
        envelope["receipt"]["arms"]["reference"]["observer"]["qualification"][key] = 0
    elif surface == "capture":
        envelope["receipt"]["arms"]["reference"]["observer"]["capture"]["qualification"][key] = 0
    elif surface == "telemetry":
        envelope["telemetry"]["flags"][key] = 0
    elif surface == "supervision":
        envelope["supervision"]["flags"][key] = 0
    elif surface == "cpu":
        envelope["declaration"]["cpu_tests"]["evidence"]["flags"][key] = 0
    envelope["inventory"], envelope["closeout"] = envelope["seal"]()
    with pytest.raises(ValueError):
        receiver.receive(envelope["root"], envelope["inventory"], envelope["closeout"],
                         envelope["repository"], envelope["solver"])


def test_corrupt_inventory_is_rejected_before_any_record_decode(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    source = "e" * 40
    root = repo / "artifacts/evaluations" / ("entered-stopped-gradient-" + source[:12])
    root.mkdir(parents=True)
    content = b"not json"
    (root / "receipt.json").write_bytes(content)
    monkeypatch.setattr(probe, "ROOT", repo)
    decoded = []
    monkeypatch.setattr(receiver, "_json", lambda *args: decoded.append(args))
    inventory = {"receipt.json": {"bytes": len(content), "sha256": "0" * 64}}
    with pytest.raises(ValueError):
        receiver.receive(root, inventory, {}, repo, repo / "solver.py")
    assert decoded == []


def test_external_retirement_must_match_same_invocation(envelope):
    envelope["closeout"]["service"]["InvocationID"] = "2" * 32
    with pytest.raises(ValueError):
        receiver.receive(envelope["root"], envelope["inventory"], envelope["closeout"],
                         envelope["repository"], envelope["solver"])


def test_tampered_retained_stdlib_bytes_are_rejected_after_inventory_auth(envelope, monkeypatch):
    _write_inventory(envelope["root"], envelope["raw"])
    leaf = next(name for name in envelope["raw"] if name.startswith("bootstrap-stdlib/"))
    (envelope["root"] / leaf).write_bytes(b"# changed after receipt\n")
    monkeypatch.setattr(old, "authenticate", envelope["original_authenticate"])
    with pytest.raises(ValueError):
        receiver.receive(envelope["root"], envelope["inventory"], envelope["closeout"],
                         envelope["repository"], envelope["solver"])


def test_entered_stdlib_anchor_requires_hash_named_inventory_leaf(envelope):
    observer = envelope["receipt"]["entered_bootstrap"]["observer"]
    path, anchor = next(iter(observer["stdlib_sources"].items()))
    anchor["sha256"] = "0" * 64
    envelope["inventory"], envelope["closeout"] = envelope["seal"]()
    with pytest.raises(ValueError):
        receiver.receive(envelope["root"], envelope["inventory"], envelope["closeout"],
                         envelope["repository"], envelope["solver"])


def test_whole_source_inventory_may_contain_unchanged_predecessor_files(envelope):
    envelope['declaration']['source_binding']['leaves'].append(
        dict(path='README.md',bytes=1,git_blob='c'*40,sha256='d'*64))
    envelope['inventory'],envelope['closeout']=envelope['seal']()
    result=receiver.receive(envelope['root'],envelope['inventory'],envelope['closeout'],
        envelope['repository'],envelope['solver'])
    assert not any(result['qualification'].values())


@pytest.mark.parametrize('key',('whole_transitive_dependencies_authenticated','native_c_bodies_authenticated',
    'stdlib_origin_authenticated','callable_defaults_authenticated','other_threads_observed','gpu_only_paths_observed'))
@pytest.mark.parametrize('value',(True,0))
def test_every_new_nonqualification_claim_rejects_promotions_and_integer_false(envelope,key,value,monkeypatch):
    envelope['receipt']['entered_bootstrap']['observer'][key]=value
    envelope['inventory'],envelope['closeout']=envelope['seal']()
    monkeypatch.setattr(prefix,'receive_banks',lambda *a:pytest.fail('premature packet decode'))
    with pytest.raises(ValueError):receiver.receive(envelope['root'],envelope['inventory'],envelope['closeout'],
        envelope['repository'],envelope['solver'])


@pytest.mark.parametrize('fault',('extra','missing','scope','tree','event-bool','events-small','events-large',
    'empty','duplicate','unsorted','line-bool','line-missing','calls-bool','zero-calls','unknown-module',
    'relative-path','traversal','extra-source','missing-source','root-type','root-not-stdlib','empty-stdlib','extra-leaf'))
def test_entered_rows_and_exact_source_set_refused_before_packet_decode(envelope,fault,monkeypatch):
    record=envelope['receipt']['entered_bootstrap'];o=record['observer'];rows=o['frames']
    if fault=='extra':o['extra']=False
    elif fault=='missing':del o['profile_removed']
    elif fault=='scope':o['scope']='all-python-origin'
    elif fault=='tree':o['warp_source_tree_sha256']='0'*64
    elif fault=='event-bool':o['python_call_events']=True
    elif fault=='events-small':o['python_call_events']=1
    elif fault=='events-large':o['python_call_events']=20001
    elif fault=='empty':o['frames']=[]
    elif fault=='duplicate':rows.append(rows[-1].copy())
    elif fault=='unsorted':rows.reverse()
    elif fault=='line-bool':rows[0]['line']=True
    elif fault=='line-missing':rows[0]['line']=999999
    elif fault=='calls-bool':rows[0]['calls']=True
    elif fault=='zero-calls':rows[0]['calls']=0
    elif fault=='unknown-module':rows[0]['module']='untrusted'
    elif fault=='relative-path':rows[0]['path']='stdlib.py'
    elif fault=='traversal':rows[0]['path']=record['stdlib_root']+'/../posixpath.py'
    elif fault=='extra-source':o['stdlib_sources']['/extra.py']=next(iter(o['stdlib_sources'].values())).copy()
    elif fault=='missing-source':o['stdlib_sources']={}
    elif fault=='root-type':record['stdlib_root']=None
    elif fault=='root-not-stdlib':record['stdlib_root']='/tmp'
    elif fault=='empty-stdlib':
        o['frames']=[r for r in rows if r['kind']=='warp'];o['stdlib_sources']={}
    else:envelope['raw']['bootstrap-stdlib/'+'0'*64+'.py']=b'pass\n'
    envelope['inventory'],envelope['closeout']=envelope['seal']()
    monkeypatch.setattr(prefix,'receive_banks',lambda *a:pytest.fail('premature packet decode'))
    with pytest.raises(ValueError):receiver.receive(envelope['root'],envelope['inventory'],envelope['closeout'],
        envelope['repository'],envelope['solver'])


def test_reanchored_source_mismatch_refused_before_stdlib_compilation(envelope,monkeypatch):
    record=envelope['receipt']['entered_bootstrap']['observer']
    path,anchor=next(iter(record['stdlib_sources'].items()))
    leaf='bootstrap-stdlib/'+anchor['sha256']+'.py'
    envelope['raw'][leaf]=b'def join():\n    pass\n'
    envelope['inventory'],envelope['closeout']=envelope['seal']()
    original=receiver._code_graph
    def graph(raw,filename):
        assert filename!=path,'compiled unanchored stdlib bytes'
        return original(raw,filename)
    monkeypatch.setattr(receiver,'_code_graph',graph)
    with pytest.raises(ValueError):receiver.receive(envelope['root'],envelope['inventory'],envelope['closeout'],
        envelope['repository'],envelope['solver'])


def test_graph_compiles_but_does_not_execute_retained_stdlib_code():
    code=b"raise RuntimeError('must not execute')\n"
    assert receiver._code_graph(code,'/stdlib.py')[0].co_qualname=='<module>'


def test_package_and_frozen_alias_source_path_mapping():
    assert receiver._stdlib_rel('ctypes')=='ctypes/__init__.py'
    assert receiver._stdlib_rel('collections.abc')=='_collections_abc.py'
    assert receiver._stdlib_rel('importlib._bootstrap')=='importlib/_bootstrap.py'
