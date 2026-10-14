from __future__ import annotations

from dataclasses import asdict
import importlib.util
import math
from pathlib import Path
import re
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
    assert driver.phase_settings("thermal-1024") == (1024, 120)
    assert driver.phase_settings("train", 256) == (256, 6000)
    for phase in ("smoke", "profile-1024"):
        with pytest.raises(ValueError, match="only valid for train"):
            driver.phase_settings(phase, 256)
    with pytest.raises(ValueError, match="approved choices"):
        driver.phase_settings("train", 128)
    for updates in driver.RESUME_UPDATE_CHOICES:
        assert driver.phase_settings("resume", 1024, additional_updates=updates) == (1024, updates)
    for bad_worlds, bad_updates in ((None, 5), (2048, 5), (1024, 6)):
        with pytest.raises(ValueError, match="resume"):
            driver.phase_settings("resume", bad_worlds, additional_updates=bad_updates)
    with pytest.raises(ValueError, match="only valid for resume"):
        driver.phase_settings("train", 1024, additional_updates=5)


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
        ("agent", "load_run"), ("agent", "load_checkpoint"),
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
    assert cfg.agent.save_interval == 100
    assert cfg.video is False


@pytest.mark.parametrize("updates,checkpoint,final_iteration", [
    (5, "model_500.pt", 504), (5500, "model_500.pt", 5999),
    (4000, "model_5999.pt", 9998),
])
def test_resume_cfg_keeps_stock_recipe_and_uses_exact_parent_checkpoint(
    updates, checkpoint, final_iteration,
):
    from mjlab.scripts.train import TrainConfig

    baseline = TrainConfig.from_task(driver.TASK)
    cfg = driver.build_config("resume", worlds=1024, additional_updates=updates,
                              resume_checkpoint_name=checkpoint)
    assert _without_allowed_overrides(cfg) == _without_allowed_overrides(baseline)
    assert cfg.agent.resume is True
    assert cfg.agent.load_run == "resume_parent"
    assert cfg.agent.load_checkpoint == f"^{re.escape(checkpoint)}$"
    assert cfg.agent.max_iterations == updates
    assert cfg.env.scene.num_envs == cfg.env.scene.terrain.num_envs == 1024
    assert final_iteration == int(checkpoint.removeprefix("model_").removesuffix(".pt")) + updates - 1


def test_resume_options_are_fail_closed():
    source = "a" * 40
    digest = "b" * 64
    checkpoint = Path("/tmp/artifacts/training/run/train/model_500.pt")
    assert driver.validate_phase_options(
        "resume", worlds=1024, additional_updates=5, seed=1019,
        cooldown_seconds=0.5, max_temperature_c=80, parent_checkpoint=checkpoint,
        parent_source=source, parent_sha256=digest,
    ) == (1024, 5)
    for overrides in (
        {"seed": 7}, {"cooldown_seconds": 0}, {"max_temperature_c": 74},
        {"parent_source": "bad"}, {"parent_sha256": "f" * 63},
        {"parent_checkpoint": None},
    ):
        values = dict(seed=1019, cooldown_seconds=0.5, max_temperature_c=75,
                      parent_checkpoint=checkpoint, parent_source=source,
                      parent_sha256=digest)
        values.update(overrides)
        with pytest.raises(ValueError):
            driver.validate_phase_options(
                "resume", worlds=1024, additional_updates=5, **values,
            )
    with pytest.raises(ValueError, match="only valid for resume"):
        driver.validate_phase_options(
            "train", worlds=1024, additional_updates=None, seed=1019,
            cooldown_seconds=0, max_temperature_c=75,
            parent_checkpoint=checkpoint, parent_source=source, parent_sha256=digest,
        )
    with pytest.raises(ValueError, match="only approved"):
        driver.validate_phase_options(
            "train", worlds=1024, additional_updates=None, seed=1019,
            cooldown_seconds=0, max_temperature_c=80,
        )


@pytest.mark.parametrize("iteration,step_counter,launch", [
    (500, 12024, {"updates": 6000, "resume_parent": None}),
    (5999, 144024, {"updates": 5500, "resume_parent": {"common_step_counter": 12024}}),
])
def test_resume_parent_provenance_and_cumulative_counter(iteration, step_counter, launch, tmp_path):
    import json
    import torch

    run_id = "stock-walk-cooled-foundation-s1019-20261014"
    phase = "train" if iteration == 500 else "resume"
    output_phase = "train" if iteration == 500 else "resume-main"
    checkpoint = tmp_path / "artifacts" / "training" / run_id / output_phase / f"model_{iteration}.pt"
    checkpoint.parent.mkdir(parents=True)
    launch_data = {
        "source": "c" * 40, "task": driver.TASK, "run_id": run_id,
        "phase": phase, "seed": 1019, "worlds": 1024, **launch,
    }
    if launch_data["resume_parent"] is not None:
        launch_data["resume_parent"].update({
            "source": "c" * 40, "checkpoint_sha256": "f" * 64,
            "checkpoint_iteration": 500,
        })
    (checkpoint.parent / "launch.json").write_text(json.dumps(launch_data), encoding="utf-8")
    torch.save({
        "actor_state_dict": {"weight": torch.tensor([1.0])},
        "critic_state_dict": {"weight": torch.tensor([2.0])},
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(1)}}},
        "iter": iteration,
        "infos": {"env_state": {"common_step_counter": step_counter}},
    }, checkpoint)

    info = driver.validate_resume_parent(
        tmp_path, checkpoint, "c" * 40, driver.sha256_file(checkpoint), torch,
    )
    assert info["checkpoint_iteration"] == iteration
    assert info["common_step_counter"] == step_counter
    assert info["completed_updates"] == step_counter // 24
    assert info["cumulative_transitions"] == step_counter * 1024

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        driver.validate_resume_parent(tmp_path, checkpoint, "c" * 40, "d" * 64, torch)


def test_resume_parent_rejects_mismatched_cumulative_step_counter(tmp_path):
    import json
    import torch

    checkpoint = tmp_path / "artifacts" / "training" / "stock-walk-parent-s1019-20261014" / "train" / "model_500.pt"
    checkpoint.parent.mkdir(parents=True)
    launch = {"source": "e" * 40, "task": driver.TASK,
              "run_id": checkpoint.parents[1].name, "seed": 1019,
              "worlds": 1024, "phase": "train", "updates": 6000,
              "resume_parent": None}
    (checkpoint.parent / "launch.json").write_text(json.dumps(launch), encoding="utf-8")
    torch.save({
        "actor_state_dict": {"weight": torch.tensor([1.0])},
        "critic_state_dict": {"weight": torch.tensor([1.0])},
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(1)}}},
        "iter": 500, "infos": {"env_state": {"common_step_counter": 12000}},
    }, checkpoint)
    with pytest.raises(RuntimeError, match="iteration/state/finite"):
        driver.validate_resume_parent(
            tmp_path, checkpoint, "e" * 40, driver.sha256_file(checkpoint), torch,
        )
@pytest.mark.parametrize("source", ["", "A" * 40, "a" * 39, "a" * 41])
def test_source_must_be_a_lowercase_full_commit(source):
    with pytest.raises(ValueError, match="40-character"):
        driver.validate_args(source, "stock-walk-profiles-s1019-20261014", 1800)


@pytest.mark.parametrize("run_id", ["../stock-walk-x-s1-20261014", "stock-walk-s1019", "other-x-s1-20261014"])
def test_run_id_is_safe_and_dated(run_id):
    with pytest.raises(ValueError, match="dated stock-walk"):
        driver.validate_args("a" * 40, run_id, 1800)


@pytest.mark.parametrize("cap", [0, -1, 12601, True, 2.5])
def test_deadline_cap_is_bounded_integer(cap):
    with pytest.raises(ValueError, match="between 1 and 12600"):
        driver.validate_args("a" * 40, "stock-walk-profiles-s1019-20261014", cap)


def test_cap_accepts_the_predeclared_three_and_half_hour_limit():
    driver.validate_args("a" * 40, "stock-walk-profiles-s1019-20261014", 12600)


@pytest.mark.parametrize("cooldown", [-0.01, float("inf"), float("-inf"), float("nan"), 1.01, True])
def test_cooldown_must_be_finite_and_bounded(cooldown):
    with pytest.raises(ValueError, match="finite and between 0 and 1"):
        driver.validate_args("a" * 40, "stock-walk-profiles-s1019-20261014", 1800, cooldown)


def test_paced_logger_synchronizes_then_sleeps_once_and_restores():
    calls = []

    class FakeLogger:
        def log(self, *args, **kwargs):
            calls.append(("log", args, kwargs))
            return "logged"

    original = FakeLogger.log
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda: calls.append(("synchronize",)),
    ))
    with driver.paced_update_logging(
        0.25, logger_class=FakeLogger, torch_module=fake_torch,
        sleep_fn=lambda seconds: calls.append(("sleep", seconds)),
    ):
        assert FakeLogger().log(7, collect_time=1.5) == "logged"
        assert FakeLogger.log is not original
    assert FakeLogger.log is original
    assert calls == [
        ("log", (7,), {"collect_time": 1.5}),
        ("synchronize",), ("sleep", 0.25),
    ]


def test_paced_logger_restores_when_training_raises():
    class FakeLogger:
        def log(self, *_args, **_kwargs):
            return None

    original = FakeLogger.log
    with pytest.raises(RuntimeError, match="training failed"):
        with driver.paced_update_logging(
            0.1, logger_class=FakeLogger,
            torch_module=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)),
            sleep_fn=lambda _seconds: None,
        ):
            raise RuntimeError("training failed")
    assert FakeLogger.log is original


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


def test_tensorboard_resume_waits_for_cumulative_final_step():
    events = [SimpleNamespace(step=step, value=0.1) for step in range(500, 505)]

    class FakeAccumulator:
        def Reload(self):
            pass

        def Tags(self):
            return {"scalars": ("Perf/collection_time", "Perf/learning_time")}

        def Scalars(self, tag):
            return events if tag.startswith("Perf/") else []

    result = driver.tensorboard_summary(
        Path("unused"), 5, expected_last_step=504,
        accumulator_factory=FakeAccumulator, sleep_fn=lambda _seconds: None,
    )
    assert result["Perf/collection_time"]["last_step"] == 504
