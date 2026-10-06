"""Bounded source/supervisor controls; never a native launch."""

import fcntl
from hashlib import sha256
import os

import pytest

from mjlab_microduck import stance_com_entry_probe as p


@pytest.mark.parametrize("reserve,delta", [(1, 2), (300, 301), (660, 661), (960, 961)])
def test_window_accepts_only_full_strict_reserve(reserve, delta):
    p.check_window(reserve, p.CUTOFF - delta)
    with pytest.raises(ValueError):
        p.check_window(reserve, p.CUTOFF - reserve)


@pytest.mark.parametrize(
    "reserve,now",
    [
        (True, 0),
        (0, 0),
        (-1, 0),
        (1, float("nan")),
        (1, float("inf")),
        (1, True),
        (1, -1),
        (1, p.CUTOFF),
    ],
)
def test_window_rejects_malformed_values(reserve, now):
    with pytest.raises(ValueError):
        p.check_window(reserve, now)


@pytest.mark.parametrize(
    "source,mode",
    [
        ("a" * 40, "child"),
        ("A" * 40, "run"),
        ("a" * 39, "run"),
        ("../bad", "tests"),
        (True, "run"),
    ],
)
def test_unit_rejects_foreign_target(source, mode):
    with pytest.raises(ValueError):
        p.unit(source, mode)


def test_exact_unit_paths_and_new_fence():
    source = "a" * 40
    assert p.unit(source, "run") == "microduck-com-entry-run-aaaaaaaaaaaa.service"
    assert str(p.output(source, "run")).endswith(
        "artifacts/evaluations/com-entry-run-aaaaaaaaaaaa"
    )
    assert len(p.OWN) == 7
    assert p.BASE == "538a29fe360ded475ac58e4fa6ed482a5ddf9f50"
    assert p.OLD_ROOT != p.ROOT
    assert p.SERVICE_SECONDS == 300 and p.CHILD_SECONDS == 240
    assert p.CUTOFF == 1791262800
    assert p.EXPECTED_TESTS == 1859
    assert len(p.test_files()) == len(set(p.test_files())) == 59


def test_exclusive_synced_output_and_caps(tmp_path):
    path = tmp_path / "artifact"
    p.write(path, b"abc", 3)
    assert p.bounded(path, 3) == b"abc"
    with pytest.raises(FileExistsError):
        p.write(path, b"abc", 3)
    for data in (b"", b"abcd", bytearray(b"a")):
        with pytest.raises(ValueError):
            p.write(tmp_path / "other", data, 3)
    with pytest.raises(ValueError):
        p.bounded(path, 2)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        p.bounded(link, 3)


def test_empty_committed_source_leaf_remains_whole_hashed(tmp_path):
    path = tmp_path / "__init__.py"
    path.touch()
    assert sha256(p.bounded(path, 4 * 1024**2, allow_empty=True)).hexdigest() == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    with pytest.raises(ValueError, match="bounded regular file"):
        p.bounded(path, 4 * 1024**2)


def test_installed_python_tree_hashes_include_empty_markers(tmp_path, monkeypatch):
    packages = {
        "mjlab": "mjlab",
        "mujoco-warp": "mujoco_warp",
        "better-actuator-models": "bam",
    }
    for package in packages.values():
        root = tmp_path / package
        root.mkdir()
        (root / "__init__.py").touch()
        for index in range(6):
            (root / f"leaf{index}.py").write_bytes(b"# source fixture\n")

    class Distribution:
        def locate_file(self, package):
            return tmp_path / package

    monkeypatch.setattr(p, "distribution", lambda name: Distribution())
    monkeypatch.setattr(p, "version", lambda name: p.VERSIONS[name])
    monkeypatch.setattr(p.platform, "python_version", lambda: "3.12.13")
    monkeypatch.setattr(p.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(p.sys, "prefix", str(p.OLD_ROOT / ".venv"))
    before = p.packages()
    assert all(row["files"] == 7 for row in before["python_trees"].values())
    (tmp_path / "mjlab/__init__.py").write_bytes(b"# changed marker\n")
    after = p.packages()
    assert (
        after["python_trees"]["mjlab"]["sha256"]
        != before["python_trees"]["mjlab"]["sha256"]
    )
    assert after["python_trees"]["mujoco-warp"] == before["python_trees"]["mujoco-warp"]


def test_existing_inherited_lock_not_created_or_unlinked(tmp_path, monkeypatch):
    path = tmp_path / "lease"
    path.touch()
    monkeypatch.setattr(p, "LOCK", path)
    fd = os.open(path, os.O_RDWR)
    try:
        os.set_inheritable(fd, True)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = p.inherited_lease(fd)
        assert record["inode"] == path.stat().st_ino and record["bytes"] == 0
        second = os.open(path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(second)
        os.set_inheritable(fd, False)
        with pytest.raises(ValueError):
            p.inherited_lease(fd)
    finally:
        os.close(fd)
    assert path.is_file() and path.stat().st_size == 0


def test_foreign_lock_inode_refused(tmp_path, monkeypatch):
    left, right = tmp_path / "left", tmp_path / "right"
    left.touch()
    right.touch()
    monkeypatch.setattr(p, "LOCK", left)
    fd = os.open(right, os.O_RDWR)
    try:
        os.set_inheritable(fd, True)
        with pytest.raises(ValueError):
            p.inherited_lease(fd)
    finally:
        os.close(fd)


def test_original_rejection_artifact_hashes_are_fixed():
    assert len(p.PREDECESSORS) == 3
    assert all(len(v) == 64 for v in p.PREDECESSORS.values())
    assert (
        "dc2aa2d344fcad58d2054a2238c15d0d3678724852160834dc5a2a29bef8151a"
        in p.PREDECESSORS.values()
    )
    assert all(value is False for value in p.FLAGS.values())


@pytest.mark.parametrize("key", list(p.SERVICE_CAPS))
def test_service_rejects_each_changed_cap(monkeypatch, key):
    values = {
        **p.SERVICE_CAPS,
        "MainPID": "123",
        "ActiveState": "active",
        "SubState": "running",
        "InvocationID": "b" * 32,
    }
    values[key] = "foreign"
    monkeypatch.setattr(
        p, "read", lambda *args: "\n".join(f"{k}={v}" for k, v in values.items())
    )
    with pytest.raises(ValueError, match="actual retained service caps"):
        p.service("a" * 40, "run", live_pid=123)


def test_wrong_active_service_owner_refused(monkeypatch):
    values = {
        **p.SERVICE_CAPS,
        "MainPID": "124",
        "ActiveState": "active",
        "SubState": "running",
        "InvocationID": "b" * 32,
    }
    monkeypatch.setattr(
        p, "read", lambda *args: "\n".join(f"{k}={v}" for k, v in values.items())
    )
    with pytest.raises(ValueError, match="actual running"):
        p.service("a" * 40, "run", live_pid=123)


def test_cpu_test_file_cap_does_not_widen_gpu_cap(monkeypatch):
    values = {
        **p.TEST_SERVICE_CAPS,
        "MainPID": "123",
        "ActiveState": "active",
        "SubState": "running",
        "InvocationID": "b" * 32,
    }
    monkeypatch.setattr(
        p, "read", lambda *args: "\n".join(f"{k}={v}" for k, v in values.items())
    )
    assert p.service("a" * 40, "tests", live_pid=123)["LimitFSIZE"] == "67108864"
    with pytest.raises(ValueError, match="actual retained"):
        p.service("a" * 40, "run", live_pid=123)
    assert p.SERVICE_CAPS["LimitFSIZE"] == "1048576"


def test_cpu_test_environment_is_explicit_not_a_late_profile_alias(monkeypatch):
    for key, value in p.TEST_ENV.items():
        monkeypatch.setenv(key, value)
    assert p.test_environment() == p.TEST_ENV
    monkeypatch.setenv("MICRODUCK_STANCE_PROFILE", "linux-100100")
    with pytest.raises(ValueError, match="explicit native CPU test"):
        p.test_environment()


def test_retained_fixture_binding_authenticates_all_whole_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "ROOT", tmp_path)
    rows = {}
    for directory in p.TEST_FIXTURE_DIRS:
        root = tmp_path / directory
        root.mkdir(parents=True)
        for index in range(3):
            path = root / f"fixture{index}.bin"
            path.write_bytes(b"x")
            rows[str(path.relative_to(tmp_path))] = {
                "bytes": 1,
                "sha256": sha256(b"x").hexdigest(),
            }
    monkeypatch.setattr(p, "TEST_FIXTURE_BYTES", 12)
    monkeypatch.setattr(p, "TEST_FIXTURE_SHA256", sha256(p.canonical(rows)).hexdigest())
    assert p.test_fixture_binding(tmp_path)["files"] == 12
    path.write_bytes(b"y")
    with pytest.raises(ValueError, match="whole pinned"):
        p.test_fixture_binding(tmp_path)


def test_existing_source_leaf_byte_hash_is_whole(tmp_path):
    path = tmp_path / "bytes"
    p.write(path, b"a\x00b", 3)
    assert sha256(p.bounded(path, 3)).hexdigest() == sha256(b"a\x00b").hexdigest()
