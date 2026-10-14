import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / "scripts" / "stock_walking_overnight.py"
spec = importlib.util.spec_from_file_location("stock_overnight", path)
overnight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overnight)


def test_deadline_admission_reserves_exit_and_review():
    assert overnight.admitted(1000, 100, 3500, review_seconds=1800)
    assert not overnight.admitted(1000, 101, 3500, review_seconds=1800)
    assert not overnight.admitted(0, 100, 3500)
    assert not overnight.admitted(1000, 3500, 3500)


def test_continuation_budget_uses_measured_wall_clock_including_cooling():
    assert overnight.continuation_cap(8000, 5500) == 8908
    assert overnight.continuation_cap(25000, 5500) > 12600
    for bad in (float("nan"), float("inf"), 0, -1):
        with pytest.raises(ValueError): overnight.continuation_cap(bad, 5500)
