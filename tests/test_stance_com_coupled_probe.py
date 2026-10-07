"""Fresh supervisor refusal tests, never native CUDA qualification."""

from types import SimpleNamespace

import pytest

from mjlab_microduck import stance_com_coupled_probe as p


def test_fresh_names_counts_bounds_and_original_windows_unchanged():
    assert p.BASE == "ab88b2621edc14214e96922bc40e7f3685312113"
    assert p.CUTOFF == 1791349200 and p.old.CUTOFF == 1791262800
    assert p.cpu.EXPECTED_TESTS == 1923 and p.old.EXPECTED_TESTS == 1879
    assert len(p.OWN) == 16 and len(p.test_files()) == len(set(p.test_files())) == 63
    assert p.unit("a" * 40, "run") == "microduck-com-coupled-run-aaaaaaaaaaaa.service"
    assert p.output("a" * 40, "tests").name == "com-coupled-tests-aaaaaaaaaaaa"
    assert p.SERVICE_SECONDS == 300 and p.CHILD_SECONDS == 240
    assert len(p.PHASE_A) == 3 and len(p.receiver.CAPS) == 22
    assert p.frames.PACKET_BYTES < 1024**2


@pytest.mark.parametrize("mode", [None, True, "child", "cpu", "RUN"])
def test_only_declared_retained_service_modes(mode):
    with pytest.raises(ValueError):
        p.unit("a" * 40, mode)


@pytest.mark.parametrize(
    "now", [float("nan"), float("inf"), -1, True, p.CUTOFF, p.CUTOFF - 660]
)
def test_no_late_or_ambiguous_job_admission(now):
    with pytest.raises(ValueError):
        p.check_window(660, now)


@pytest.mark.parametrize(
    "changed", ["owner", "invocation", "restart", "memory", "file-cap", "inactive"]
)
def test_actual_service_identity_and_caps(changed, monkeypatch):
    values = dict(
        p.SERVICE_CAPS,
        MainPID="123",
        InvocationID="b" * 32,
        ActiveState="active",
        SubState="running",
    )
    monkeypatch.setattr(p, "properties", lambda *_: values)
    assert p.service("a" * 40, "run", 123)["MainPID"] == "123"
    key, value = {
        "owner": ("MainPID", "124"),
        "invocation": ("InvocationID", "x"),
        "restart": ("NRestarts", "1"),
        "memory": ("MemoryMax", "max"),
        "file-cap": ("LimitFSIZE", "67108864"),
        "inactive": ("ActiveState", "inactive"),
    }[changed]
    values[key] = value
    with pytest.raises(ValueError):
        p.service("a" * 40, "run", 123)


@pytest.mark.parametrize("field", ["tests", "errors", "failures", "skipped"])
def test_junit_is_exact_not_exit_zero_or_subset(field):
    values = dict(tests=str(p.EXPECTED_TESTS), errors="0", failures="0", skipped="0")
    good = (
        "<testsuite " + " ".join(f'{k}="{v}"' for k, v in values.items()) + "/>"
    ).encode()
    p.checked_junit(good)
    values[field] = "1"
    bad = (
        "<testsuite " + " ".join(f'{k}="{v}"' for k, v in values.items()) + "/>"
    ).encode()
    with pytest.raises(ValueError):
        p.checked_junit(bad)


def test_child_refuses_missing_inherited_lease_before_any_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(p.os, "getppid", lambda: 123)
    monkeypatch.setattr(
        p.old,
        "inherited_lease",
        lambda *_: (_ for _ in ()).throw(
            ValueError("no authenticated inherited lease")
        ),
    )
    monkeypatch.setattr(
        p,
        "source_binding",
        lambda *_: pytest.fail("child progressed beyond refused lease"),
    )
    with pytest.raises(ValueError, match="inherited lease"):
        p.child("a" * 40, -1, 123, "b" * 64)


def test_frame_requires_complete_finite_float32_array_without_mutation():
    import numpy as np

    class Value:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def contiguous(self):
            return self

        def numpy(self):
            return self.value

    arrays = {
        name: np.zeros((64, *shape), np.float32)
        for name, shape in p.frames.FRAME_FIELDS
    }
    env = SimpleNamespace(_sync=lambda: None, _view=lambda name: Value(arrays[name]))
    raw = p._frame(env)
    assert len(raw) == p.frames.FRAME_BYTES and not any(raw)
    arrays["qfrc_bias"][0, 0] = np.nan
    with pytest.raises(ValueError, match="qfrc_bias"):
        p._frame(env)


def test_root_or_import_mismatch_precedes_source_leaf_reads(monkeypatch):
    monkeypatch.setattr(p.Path, "cwd", lambda: p.Path("/not-authorized"))
    monkeypatch.setattr(
        p.old, "read", lambda *_: pytest.fail("wrong root executed source command")
    )
    with pytest.raises(ValueError, match="clean fresh"):
        p.source_binding("a" * 40)


@pytest.mark.parametrize("raw", [b"{}", b"\0" * 9948])
def test_descriptor_fixture_whole_hash_precedes_decode(raw, monkeypatch):
    monkeypatch.setattr(p.old, "bounded", lambda *_: raw)
    monkeypatch.setattr(
        p.json, "loads", lambda *_: pytest.fail("decoded wrong fixture")
    )
    with pytest.raises(ValueError, match="before decode"):
        p.descriptor_fixture()


def test_exact_test_list_hash():
    from hashlib import sha256

    assert (
        sha256(p.canonical(p.test_files())).hexdigest() == p.receiver.TEST_FILES_SHA256
    )
