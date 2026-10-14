from __future__ import annotations

from dataclasses import asdict
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

DRIVER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_stock_walking.py"
SPEC = importlib.util.spec_from_file_location("train_stock_walking", DRIVER_PATH)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_phase_budgets_and_world_selection_are_fixed():
    assert driver.phase_settings("smoke") == (64, 5)
    assert driver.phase_settings("profile-1024") == (1024, 20)
    assert driver.phase_settings("profile-2048") == (2048, 20)
    assert driver.phase_settings("profile-4096") == (4096, 20)
    assert driver.phase_settings("train", 256) == (256, 6000)
    for phase in ("smoke", "profile-1024"):
        with pytest.raises(ValueError, match="only valid for train"):
            driver.phase_settings(phase, 256)
    with pytest.raises(ValueError, match="approved choices"):
        driver.phase_settings("train", 128)


def _canonical(value):
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return "<nan>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _without_allowed_overrides(config):
    value = _canonical(asdict(config))
    for path in (
        ("env", "seed"), ("env", "scene", "num_envs"),
        ("env", "scene", "terrain", "num_envs"),
        ("agent", "seed"), ("agent", "max_iterations"),
        ("agent", "logger"), ("agent", "upload_model"),
        ("agent", "resume"), ("agent", "save_interval"),
    ):
        node = value
        for component in path[:-1]:
            node = node[component]
        node.pop(path[-1])
    return value


def test_stock_cfg_differs_only_by_allowed_operational_overrides():
    from mjlab.scripts.train import TrainConfig

    baseline = TrainConfig.from_task(driver.TASK)
    cfg = driver.build_config("profile-1024", seed=1019)
    assert _without_allowed_overrides(cfg) == _without_allowed_overrides(baseline)
    assert cfg.env.scene.num_envs == cfg.env.scene.terrain.num_envs == 1024
    assert cfg.env.seed == cfg.agent.seed == 1019
    assert cfg.agent.max_iterations == 20
    assert cfg.agent.logger == "tensorboard"
    assert cfg.agent.upload_model is False
    assert cfg.agent.resume is False
    assert cfg.agent.save_interval == 250
    assert cfg.video is False


@pytest.mark.parametrize("source", ["", "A" * 40, "a" * 39, "a" * 41])
def test_source_must_be_a_lowercase_full_commit(source):
    with pytest.raises(ValueError, match="40-character"):
        driver.validate_args(source, "stock-walk-profiles-s1019-20261014", 1800)


@pytest.mark.parametrize("run_id", ["../stock-walk-x-s1-20261014", "stock-walk-s1019", "other-x-s1-20261014"])
def test_run_id_is_safe_and_dated(run_id):
    with pytest.raises(ValueError, match="dated stock-walk"):
        driver.validate_args("a" * 40, run_id, 1800)


@pytest.mark.parametrize("cap", [0, -1, 10801, True, 2.5])
def test_deadline_cap_is_bounded_integer(cap):
    with pytest.raises(ValueError, match="between 1 and 10800"):
        driver.validate_args("a" * 40, "stock-walk-profiles-s1019-20261014", cap)


def test_stock_actor_contract_accepts_the_backlash_cfg():
    from mjlab.scripts.train import TrainConfig
    from mjlab_microduck.normalized_onnx_reference import validate_actor_contract

    cfg = TrainConfig.from_task(driver.TASK)
    assert validate_actor_contract(cfg.env, expected_actuator="BacklashEncoderBamActuatorCfg") == (
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
        "actions", "command", "head_command", "body_command",
    )


def test_tensorboard_summary_waits_for_complete_perf_flush():
    rows = [
        {"Perf/collection_time": [SimpleNamespace(step=0, value=0.1)],
         "Perf/learning_time": [SimpleNamespace(step=0, value=0.2)]},
        {"Perf/collection_time": [SimpleNamespace(step=i, value=0.1 + i) for i in range(3)],
         "Perf/learning_time": [SimpleNamespace(step=i, value=0.2 + i) for i in range(3)],
         "Episode_Reward/fall": [SimpleNamespace(step=2, value=-1.0)]},
    ]

    class FakeAccumulator:
        calls = 0

        def Reload(self):
            self.calls += 1

        def Tags(self):
            return {"scalars": tuple(rows[min(self.calls - 1, 1)])}

        def Scalars(self, tag):
            return rows[min(self.calls - 1, 1)][tag]

    accumulator = FakeAccumulator()
    result = driver.tensorboard_summary(
        Path("unused"), 3, accumulator_factory=lambda: accumulator,
        sleep_fn=lambda _seconds: None,
    )
    assert result["Perf/collection_time"]["count"] == 3
    assert result["Perf/collection_time"]["last_step"] == 2
    assert result["Episode_Reward/fall"]["last"] == -1.0
