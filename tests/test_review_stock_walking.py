import importlib.util
from pathlib import Path
import sys
import pytest

path = Path(__file__).resolve().parents[1] / "scripts" / "review_stock_walking.py"
spec = importlib.util.spec_from_file_location("stock_review", path)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_review_cases_and_heldout_protocol():
    assert review.CASES == (("idle", 0., 0.), ("straight", .3, 0.),
                            ("left", .2, .2), ("right", .2, -.2))
    for _, speed, yaw in review.CASES:
        args = review.script_args("evaluate", Path("model.pt"), Path("out"), speed, yaw)
        assert args[args.index("--task")+1] == "Mjlab-Velocity-Flat-Backlash-MicroDuck"
        assert args[args.index("--seeds")+1] == "42,211,223"
        assert f"--yaw-rates={yaw}" in args
        assert args[args.index("--num-envs")+1] == "8"
        assert args[args.index("--steps")+1] == "500"
        record = review.script_args("record", Path("model.pt"), Path("out"), speed, yaw)
        assert f"--yaw-rate={yaw}" in record
        assert record[record.index("--frames")+1] == "500"


def test_argv_restored_after_script_failure():
    before = sys.argv
    with pytest.raises(RuntimeError):
        with review.script_argv(Path("review.py"), ["--speed", ".3"]):
            assert sys.argv == ["review.py", "--speed", ".3"]
            raise RuntimeError("simulation error")
    assert sys.argv is before


def test_unknown_review_kind_rejected():
    with pytest.raises(ValueError):
        review.script_args("train", Path("x"), Path("y"), 0., 0.)


def test_command_response_sweep_extends_without_rewriting_gentle_protocol():
    assert review.review_cases("gentle") == review.CASES
    cases = review.review_cases("command-response")
    assert cases[:4] == review.CASES
    assert len(cases) == 12
    assert len({name for name, _, _ in cases}) == 12
    assert all(0 <= speed <= .4 and abs(yaw) <= 1. for _, speed, yaw in cases)
    assert ("spin-left", 0., .5) in cases and ("spin-right", 0., -.5) in cases
    assert ("strong-left", .3, .8) in cases and ("strong-right", .3, -.8) in cases
    with pytest.raises(ValueError):
        review.review_cases("choose-best-checkpoint")
