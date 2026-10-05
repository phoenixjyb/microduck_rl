"""Synthetic CPU supervision/authentication seams, not native qualification."""

import subprocess
from contextlib import contextmanager

import pytest

from mjlab_microduck import stance_constraint_identity_probe as probe
from mjlab_microduck.first_attempt_smoke import canonical

SOURCE = "b" * 40
INVOCATION = "d" * 32


def props():
    return dict(
        MainPID="123",
        ActiveState="active",
        RuntimeMaxUSec="5min",
        MemoryMax="4294967296",
        CPUQuotaPerSecUSec="2s",
        Nice="10",
        KillMode="control-group",
        InvocationID=INVOCATION,
    )


@pytest.fixture
def evidence_files(tmp_path, monkeypatch):
    for name in probe.prior.COMPLETE_FILES:
        (tmp_path / name).write_bytes(b"{}\n")
    inventory = probe.prior._inventory(tmp_path, probe.prior.COMPLETE_FILES)
    diagnosis = dict(
        protocol=probe.prior.PROTOCOL + ":failure-diagnosis",
        source=probe.PARENT,
        status="failed-pair-independently-diagnosed",
        pair_error="paired rollout semantic state exactness",
        pair_accepted=False,
        files_rehashed=inventory,
        launch_sha256=inventory["launch.json"]["sha256"],
        **probe.prior.FLAGS,
    )
    raw = (canonical(diagnosis) + "\n").encode()
    (tmp_path / "independent-failure-diagnosis.json").write_bytes(raw)
    monkeypatch.setattr(probe, "DIAGNOSIS_SHA", probe.base.digest(raw))
    monkeypatch.setattr(probe.prior, "output_path", lambda _: tmp_path)
    monkeypatch.setattr(
        probe.prior,
        "_read_launch",
        lambda *_: dict(service_properties=dict(InvocationID=probe.RUN_INVOCATION)),
    )
    terminals = []
    monkeypatch.setattr(probe.prior, "_failed_terminal", lambda *a: terminals.append(a))
    return tmp_path, diagnosis, terminals


def test_predeclared_cpu_caps_and_separate_namespaces():
    assert probe.CAP == 300 and probe.MEMORY == 4 * 1024**3
    assert len(probe.TEST_FILES) == 35 and len(probe.INPUT_FILES) == 23
    assert probe.unit(SOURCE, "run") != probe.prior.unit(probe.PARENT, "run")
    assert probe.output_path(SOURCE, "run") != probe.prior.output_path(probe.PARENT)
    assert all(value is False for value in probe.identity.FLAGS.values())
    with pytest.raises(ValueError):
        probe.unit(SOURCE, "train")


@pytest.mark.parametrize(
    "damage", [None, "diagnosis_hash", "original_file", "original_invocation"]
)
def test_input_authentication_before_any_tensor_load(
    evidence_files, monkeypatch, damage
):
    root, _diagnosis, terminals = evidence_files
    monkeypatch.setattr(
        probe.prior.torch,
        "load",
        lambda *_a, **_k: pytest.fail("tensor load during byte authentication"),
    )
    if damage == "diagnosis_hash":
        with (root / "independent-failure-diagnosis.json").open("ab") as f:
            f.write(b" ")
    elif damage == "original_file":
        (root / "capture.pt").write_bytes(b"changed")
    elif damage == "original_invocation":
        monkeypatch.setattr(
            probe.prior,
            "_read_launch",
            lambda *_: dict(service_properties=dict(InvocationID="a" * 32)),
        )
    if damage is None:
        assert probe.authenticate_inputs()[0] == root
        assert terminals == [(probe.PARENT, probe.RUN_INVOCATION)]
    else:
        with pytest.raises(ValueError):
            probe.authenticate_inputs()
        assert terminals == []


@pytest.mark.parametrize("outcome", ["same", "different", "success"])
def test_read_compare_never_substitutes_a_different_pair_result(
    evidence_files, monkeypatch, outcome
):
    root, _diagnosis, _terminals = evidence_files
    inputs = [{"archive": {"records": [{}]}}, {"archive": {"records": [{}]}}]
    monkeypatch.setattr(probe.prior, "_score_inputs", lambda *_: (inputs, [], [{}, {}]))
    monkeypatch.setattr(
        probe.prior,
        "_read_launch",
        lambda *_: dict(
            service_properties=dict(InvocationID=probe.RUN_INVOCATION), schedule={}
        ),
    )

    def pair(*_):
        if outcome == "success":
            return {}
        raise ValueError(
            "paired rollout semantic state exactness"
            if outcome == "same"
            else "paired early trace exactness"
        )

    monkeypatch.setattr(probe.prior, "_strict_pair", pair)
    monkeypatch.setattr(probe.identity, "compare", lambda *_: {"all_flags_false": True})
    if outcome == "same":
        result, _inventory = probe.read_compare()
        assert result == {"all_flags_false": True}
    else:
        with pytest.raises(
            ValueError, match="original strict pair failure reproduced again"
        ):
            probe.read_compare()
    assert set(p.name for p in root.iterdir()) == probe.INPUT_FILES


@pytest.mark.parametrize("damage", [None, "pid", "competitor", "memory"])
def test_exact_cpu_user_service_and_no_competing_duck(monkeypatch, damage):
    values = props()
    if damage == "pid":
        values["MainPID"] = "999"
    if damage == "memory":
        values["MemoryMax"] = "8589934592"
    monkeypatch.setattr(probe.os, "getpid", lambda: 123)

    def read(*args):
        if args[2] == "list-units":
            return probe.unit(SOURCE, "run") + (
                "\n microduck-other.service" if damage == "competitor" else ""
            )
        return values[args[-2]]

    monkeypatch.setattr(probe.base.host, "read", read)
    if damage is None:
        assert probe.properties(SOURCE, "run") == values
    else:
        with pytest.raises(ValueError):
            probe.properties(SOURCE, "run")


@pytest.mark.parametrize("unauthorized", [False, True])
def test_real_git_source_binding_preserves_old_reader_and_limits_new_paths(
    tmp_path, monkeypatch, unauthorized
):
    def git(*args):
        return (
            subprocess.run(
                ["git", *args], cwd=tmp_path, check=True, capture_output=True
            )
            .stdout.decode()
            .strip()
        )

    git("init", "-b", "feat/athletics-obstacle-curriculum")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "CPU fixture")
    stable = set(probe.prior.OWN_FILES) - {probe.OWN[-1]}
    for name in stable | {probe.OWN[-1]}:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    git("add", ".")
    git("commit", "-m", "parent fixture")
    parent = git("rev-parse", "HEAD")
    for name in probe.OWN:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("new " + name)
    if unauthorized:
        (tmp_path / "unrelated.txt").write_text("must refuse")
    git("add", ".")
    git("commit", "-m", "new CPU fixture")
    source = git("rev-parse", "HEAD")
    monkeypatch.setattr(probe, "PARENT", parent)
    monkeypatch.setattr(probe.base.execution, "ROOT", tmp_path)
    if unauthorized:
        with pytest.raises(ValueError, match="only newly declared"):
            probe.source_binding(source)
    else:
        assert set(probe.source_binding(source)) == stable | set(probe.OWN)


def test_cpu_hidden_guard_precedes_source_and_service_access(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        probe,
        "properties",
        lambda *_: pytest.fail("service accessed before hidden guard"),
    )
    with pytest.raises(ValueError, match="CUDA-hidden exact WSL new supervisor"):
        probe.execute(SOURCE, "run")


def test_public_reader_hidden_guard_precedes_file_access(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        probe.prior,
        "output_path",
        lambda *_: pytest.fail("file access before hidden guard"),
    )
    with pytest.raises(ValueError, match="CUDA-hidden constraint identity diagnostic"):
        probe.read_compare()


@pytest.mark.parametrize("failure", [None, "reader", "context"])
def test_new_report_retention_never_qualifies_and_never_overwrites(
    tmp_path, monkeypatch, failure
):
    """Synthetic lifecycle seams, not original artifact or native acceptance."""
    root = tmp_path / "new-analysis"
    monkeypatch.setattr(probe.prior, "_hidden", lambda: None)
    monkeypatch.setattr(probe.prior, "check_window", lambda **_: None)
    monkeypatch.setattr(probe, "properties", lambda *_: props())
    calls = []

    def binding(_):
        calls.append(True)
        return {
            "owned": "changed"
            if failure == "context" and len(calls) > 1
            else "original"
        }

    monkeypatch.setattr(probe, "source_binding", binding)
    monkeypatch.setattr(probe.base.host, "identity", lambda _: {})
    monkeypatch.setattr(probe, "output_path", lambda *_: root)
    monkeypatch.setattr(probe, "_read_tests", lambda *_: "a" * 64)

    @contextmanager
    def lease():
        yield 37

    monkeypatch.setattr(probe.base.gap.base.files, "gpu_lease", lease)
    monkeypatch.setattr(probe.base.gpu_idle_gate, "wait_idle", lambda: {"idle": True})
    monkeypatch.setattr(
        probe.base.gap.base.retained.d0, "filmbrain_state", lambda: {"unchanged": True}
    )

    def compare():
        if failure == "reader":
            raise ValueError("reader failed")
        return {"protocol": probe.identity.PROTOCOL, **probe.identity.FLAGS}, {}

    monkeypatch.setattr(probe, "read_compare", compare)
    monkeypatch.setattr(probe, "authenticate_inputs", lambda: (None, None, None, {}))
    if failure is None:
        assert probe.execute(SOURCE, "run")["status"] == "passed"
    else:
        with pytest.raises(ValueError):
            probe.execute(SOURCE, "run")
    report = probe.base.parse_json((root / "report.json").read_bytes())
    assert report["status"] == ("passed" if failure is None else "failed-retained")
    assert all(report[k] is False for k in probe.identity.FLAGS)
    assert (root / "comparison.json").exists() is (failure != "reader")
    original = (root / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        probe.execute(SOURCE, "run")
    assert (root / "report.json").read_bytes() == original
