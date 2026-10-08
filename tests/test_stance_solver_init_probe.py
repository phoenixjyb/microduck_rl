"""CPU guards for the independently owned solver initialization capture."""

from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mjlab_microduck import stance_contact_boundary_probe as historical
from mjlab_microduck import stance_solver_init_probe as probe


def test_separate_owner_fence_and_cutoff():
    assert probe.BASE == "9814ceb8bbeaf656630dd28b97f38e190207a307"
    assert probe.CUTOFF == 1791428400
    assert historical.CUTOFF == 1791415800
    assert probe.OWN.isdisjoint(historical.OWN)
    assert len(probe.OWN) == 7
    assert all("solver_init" in p or "solver-initialization" in p for p in probe.OWN)
    assert probe.PROTOCOL != historical.PROTOCOL
    assert all(value is False for value in probe.FLAGS.values())


@pytest.mark.parametrize("reserve", (0, 60, 240, 900))
def test_deadline_requires_entire_reserve(reserve):
    probe.check_window(reserve, now=probe.CUTOFF - reserve - 1)
    with pytest.raises(ValueError, match="full reserve"):
        probe.check_window(reserve, now=probe.CUTOFF - reserve)


@pytest.mark.parametrize("value", (-1, True, 1.5, "60"))
def test_bad_reserve(value):
    with pytest.raises(ValueError, match="plain nonnegative"):
        probe.check_window(value)


@pytest.mark.parametrize("mode", ("tests", "run"))
def test_fresh_unique_service_and_output(mode):
    source = "a" * 40
    assert (
        probe.unit(source, mode)
        == f"microduck-solver-init-tick-{mode}-aaaaaaaaaaaa.service"
    )
    assert probe.output(source, mode) != historical.output(source, mode)


@pytest.mark.parametrize(
    "source,mode", (("a" * 39, "run"), ("z" * 40, "tests"), ("a" * 40, "train"))
)
def test_invalid_service_identity(source, mode):
    with pytest.raises(ValueError):
        probe.unit(source, mode)


def test_plan_only_adds_passive_solver_boundary():
    from mjlab_microduck import stance_solver_init_control as control

    plan = probe.capture_plan()
    solver = plan.pop("solver_init")
    assert plan == historical.capture_plan()
    assert solver == {
        "protocol": control.PROTOCOL,
        "forward": 4,
        "phase": "initialized-before-search",
        "packet_count_per_arm": 1,
        "max_bytes_per_arm": 8 * 1024**2,
        "max_bytes_all_arms": 24 * 1024**2,
        "order": list(control.SOLVER_INIT_ORDER),
        "flags": probe.FLAGS,
    }


def test_complete_scope_retains_every_historical_test(monkeypatch):
    monkeypatch.setattr(probe.prefix, "ROOT", Path(probe.__file__).resolve().parents[2])
    files = probe.test_files()
    assert files[:94] == historical.test_files()
    assert len(files) == len(set(files)) == 97
    assert sha256(probe.canonical(files)).hexdigest() == probe.TEST_FILES_SHA256
    assert probe.EXPECTED_TESTS > historical.EXPECTED_TESTS


def test_owner_import_and_plan_do_not_import_cuda_packages():
    code = (
        "import sys; from mjlab_microduck import stance_solver_init_probe as p; "
        "p.capture_plan(); assert not any(k in sys.modules for k in ('numpy','torch','warp'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr.decode()


def test_cpu_environment_preserves_parent_and_uses_private_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    before = dict(os.environ)
    environment = probe.cpu_test_environment(tmp_path)
    assert before == dict(os.environ)
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["WARP_CACHE_PATH"] == str(tmp_path)
    assert {
        name: environment[name] for name in probe.CPU_TEST_THREADS
    } == probe.CPU_TEST_THREADS


def test_only_seven_new_paths_may_change_source_fence():
    root = Path(probe.__file__).resolve().parents[2]
    assert all((root / path).is_file() for path in probe.OWN)


@pytest.mark.parametrize("value", ("1", "2"))
def test_cpu_environment_refuses_optimized_child(monkeypatch, tmp_path, value):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONOPTIMIZE", value)
    with pytest.raises(ValueError, match="assertions must remain enabled"):
        probe.cpu_test_environment(tmp_path)


def test_cpu_environment_refuses_optimized_owner_in_isolated_process():
    code = (
        "from pathlib import Path; from mjlab_microduck import stance_solver_init_probe as p; "
        "p.cpu_test_environment(Path('/not-used'))"
    )
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    environment.pop("PYTHONOPTIMIZE", None)
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        env=environment,
        capture_output=True,
        timeout=30,
    )
    if result.returncode == 0:
        raise RuntimeError("optimized owner was not refused")
    assert b"assertions must remain enabled" in result.stderr
