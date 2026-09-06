"""CPU contracts and mocked one-control lifecycle, not simulation evidence."""

import datetime as dt
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import speed_response_control as control
from mjlab_microduck.recovery_measurement import RecoveryMeasurement


def inputs(body=.3, route=.3, count=400, terminal=False):
    v = torch.zeros(count, 8, 4, dtype=torch.float64)
    v[:, :, 0] = body
    v[:, :, 1] = route
    command = torch.tensor([.3, 0., 0.]).expand(count, 8, 3).clone()
    force, speed = torch.full((count, 8, 14), .1), torch.ones(count, 8, 14)
    observer = RecoveryMeasurement(8, .3, .02)
    for step in range(count):
        observer.begin(step, torch.full((8,), 0 if step < 100 else 2), v[step, :, 1])
        observer.finish(torch.full((8,), terminal and step == count-1))
    return dict(velocities=v, commands=command, legacy_force=force, legacy_speed=speed,
                pre_force=force.clone(), pre_speed=speed.clone(),
                terminal_steps=[count-1] if terminal else [], measurement=observer.report())


def test_frame_projection_and_rotation_invariance_do_not_equate_body_with_route():
    yaw = math.pi / 3
    world = torch.tensor([[.3 * math.cos(yaw), .3 * math.sin(yaw), 0.]], dtype=torch.float64)
    body = torch.tensor([[.3, 0., 0.]], dtype=torch.float64)
    q = torch.tensor([[math.cos(yaw/2), 0., 0., math.sin(yaw/2)]], dtype=torch.float64)
    route = torch.tensor([[1., 0.]], dtype=torch.float64)
    before = world.clone()
    row = control.velocity_rows(world, body, q, route)
    assert row[0, 0] == pytest.approx(.3) and row[0, 1] == pytest.approx(.15)
    assert row[0, 2] == pytest.approx(.3*math.sin(yaw)) and row[0, 3] == pytest.approx(yaw)
    rotate = .7
    rotated_world = torch.tensor([[.3*math.cos(yaw+rotate), .3*math.sin(yaw+rotate), 0.]], dtype=torch.float64)
    rotated_q = torch.tensor([[math.cos((yaw+rotate)/2), 0., 0., math.sin((yaw+rotate)/2)]], dtype=torch.float64)
    rotated_route = torch.tensor([[math.cos(rotate), math.sin(rotate)]], dtype=torch.float64)
    torch.testing.assert_close(row, control.velocity_rows(rotated_world, body, rotated_q, rotated_route))
    assert torch.equal(world, before)


@pytest.mark.parametrize("which", ["nan", "route", "quaternion", "shape"])
def test_invalid_frames_rejected(which):
    world, body = torch.zeros(1, 3), torch.zeros(1, 3)
    q, route = torch.tensor([[1., 0., 0., 0.]]), torch.tensor([[1., 0.]])
    if which == "nan": world.fill_(float("nan"))
    if which == "route": route *= 2
    if which == "quaternion": q *= 2
    if which == "shape": body = body[:, :2]
    with pytest.raises(ValueError): control.velocity_rows(world, body, q, route)


@pytest.mark.parametrize("body,route,expected", [
    (.3, .3, "straight-response-within-both-criteria"),
    (.2, .2, "straight-body-mean-outside-band"),
    (.3, .2, "body-route-response-diverge"),
])
def test_classifications_are_diagnostic_not_training_admission(body, route, expected):
    report = control.summarize(**inputs(body, route))
    assert report["classification"] == expected
    assert not report["training_admitted"] and not report["reopens_recovery_ab"]
    assert report["groups"]["settled"]["steps"] == 300
    assert report["sample_steps"] == 400 and report["startup_steps"] == 100


def test_one_undertracking_environment_is_not_hidden_by_mean():
    values = inputs()
    values["velocities"][:, 0, 0] = .2
    report = control.summarize(**values)
    assert report["classification"] == "straight-body-mean-outside-band"


def test_mean_tracking_does_not_relabel_an_instantaneous_window_failure():
    values = inputs()
    values["velocities"][::2, :, 1] = .2
    values["velocities"][1::2, :, 1] = .4
    observer = RecoveryMeasurement(8, .3, .02)
    for step, v in enumerate(values["velocities"]):
        observer.begin(step, torch.full((8,), 0 if step < 100 else 2), v[:, 1])
        observer.finish(torch.zeros(8, dtype=torch.bool))
    values["measurement"] = observer.report()
    report = control.summarize(**values)
    assert report["classification"] == "mean-tracking-but-instantaneous-window-missed"
    assert not report["stable_route_window_all_envs"] and not report["training_admitted"]


@pytest.mark.parametrize("failure", ["startup_terminal", "incomplete", "command", "legacy_torque", "rated_speed", "pre_speed"])
def test_safety_including_startup_is_not_hidden(failure):
    values = inputs(count=20, terminal=True) if failure == "startup_terminal" else inputs(count=200 if failure == "incomplete" else 400)
    if failure == "command": values["commands"][0, 0, 0] = .4
    if failure == "legacy_torque": values["legacy_force"][:100] = .5
    if failure == "rated_speed": values["legacy_speed"][0] = 100
    if failure == "pre_speed": values["pre_speed"][0] = 100
    result = control.summarize(**values)
    assert result["classification"] == "safety-or-coverage-stop" and result["safety_failures"]


@pytest.mark.parametrize("key", ["velocities", "commands", "legacy_force", "pre_force", "legacy_speed", "pre_speed"])
def test_nonfinite_measurements_refused(key):
    values = inputs()
    values[key][0].fill_(float("nan"))
    with pytest.raises(ValueError): control.summarize(**values)


def test_real_task_config_has_no_obstacle_and_pins_body_commands():
    cfg, agent = control.prepare_config()
    assert "obstacle" not in cfg.scene.entities
    assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == 8
    assert cfg.seed == agent.seed == 383 and not cfg.curriculum
    assert "push_robot" not in cfg.events
    command = cfg.commands["twist"]
    assert command.ranges.lin_vel_x == (.3, .3) and command.ranges.ang_vel_z == (0., 0.)
    assert command.rel_forward_envs == command.rel_world_envs == command.rel_heading_envs == 0.
    assert command.init_velocity_prob == 0. and not command.heading_command
    assert "microduck_motor_step_stream" in cfg.metrics


def test_control_loop_stops_on_startup_terminal_before_reset_state_can_enter(monkeypatch):
    import mjlab.envs
    import mjlab.rl
    import mjlab.tasks.registry
    from mjlab_microduck.motor_step_stream import MotorStepStream

    raw_data = NS(root_link_lin_vel_w=torch.full((8, 3), .1),
                  root_link_lin_vel_b=torch.full((8, 3), .1),
                  root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(8, 1),
                  actuator_force=torch.full((8, 14), .1), joint_vel=torch.ones(8, 14))
    class Env:
        def __init__(self, **kwargs):
            self.step_dt = .02
            self.scene = {"robot": NS(data=raw_data)}
            self.command_manager = NS(get_command=lambda _: torch.tensor([[.3, 0., 0.]]).repeat(8, 1))
            self.steps = 0
            self.closed = False
        def close(self): self.closed = True
    env = Env()
    obs = {"actor": torch.zeros(8, 61)}
    class Wrapper:
        def __init__(self, raw, **kwargs): self.raw = raw
        def get_observations(self): return obs
        def step(self, actions):
            done = torch.zeros(8, dtype=torch.bool)
            done[0] = self.raw.steps == 2
            self.raw._microduck_motor_step_stream.capture(raw_data, done)
            if done.any():
                raw_data.root_link_lin_vel_w.fill_(999.)
                raw_data.root_link_lin_vel_b.fill_(999.)
                raw_data.actuator_force.fill_(0.)
            self.raw.steps += 1
            return obs, torch.zeros(8), done, {}
    class Runner:
        def __init__(self, *args, **kwargs): pass
        def load(self, *args, **kwargs): pass
        def get_inference_policy(self, **kwargs): return lambda _: torch.zeros(8, 14)
    monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv", lambda **kwargs: env)
    monkeypatch.setattr(mjlab.rl, "RslRlVecEnvWrapper", Wrapper)
    monkeypatch.setattr(mjlab.tasks.registry, "load_runner_cls", lambda _: Runner)
    monkeypatch.setattr(control.MotorStepStream, "from_robot", lambda *args, **kwargs:
        MotorStepStream(8, control.JOINTS, tuple(range(14)), device="cpu", cost_cfg=control.MotorStepCostCfg()))
    monkeypatch.setattr(control, "sha256", lambda _: control.ACTOR_SHA256)
    report = control.run_control(device="cpu")
    assert env.steps == 3 and env.closed
    assert report["sample_steps"] == 3 and report["terminal_steps"] == [2]
    assert report["groups"]["all"]["body_forward_mean"] == pytest.approx(.1)
    assert "settled" not in report["groups"] and report["classification"] == "safety-or-coverage-stop"
    assert report["motor_stream"]["trainer_integration_validated"] is False


@pytest.mark.parametrize("failure", [None, "expired", "source", "actor", "runtime", "dependency", "busy", "existing", "simulation", "safety"])
def test_main_one_control_no_retry_retained_no_admission(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(control, "OUTPUT", tmp_path / "control")
    monkeypatch.setattr(control, "DEADLINE", (dt.datetime.min if failure == "expired" else
                        dt.datetime.max).replace(tzinfo=dt.timezone.utc))
    monkeypatch.setattr(sys, "argv", ["control", "--source", "a" * 40])
    def source(value):
        if failure == "source": raise ValueError("source mismatch")
    monkeypatch.setattr(control, "verify_source", source)
    original_hash = control.sha256
    def hashed(path):
        if path == control.ACTOR: return "bad" if failure == "actor" else control.ACTOR_SHA256
        for name, h in control.DEPENDENCIES.items():
            if str(path).endswith(name): return "bad" if failure == "dependency" else h
        return original_hash(path)
    monkeypatch.setattr(control, "sha256", hashed)
    monkeypatch.setattr(control.importlib.metadata, "version", lambda k: "bad" if failure == "runtime" else control.VERSIONS[k])
    def host():
        if failure == "busy": raise ValueError("GPU occupied")
        return dict(utilization_percent=0, temperature_c=45, memory_mib=12)
    monkeypatch.setattr(control, "check_host", host)
    calls = []
    def run():
        calls.append(True)
        if failure == "simulation": raise ValueError("nonfinite sample")
        return control.summarize(**(inputs(count=10, terminal=True) if failure == "safety" else inputs()))
    monkeypatch.setattr(control, "run_control", run)
    if failure == "existing":
        control.OUTPUT.mkdir()
        (control.OUTPUT / "keep").write_text("untouched")
    if failure in ("expired", "source", "actor", "runtime", "dependency", "busy", "existing"):
        with pytest.raises((ValueError, FileExistsError)): control.main()
        assert calls == []
        return
    with pytest.raises(SystemExit) as ended: control.main()
    assert calls == [True] and ended.value.code == (2 if failure else 0)
    decision = json.loads((control.OUTPUT / "decision.json").read_text())
    assert not any(decision[k] for k in ("training_admitted", "reopens_recovery_ab", "physical_motion_authorized"))
    assert (control.OUTPUT / "launch.json").exists() and (control.OUTPUT / "runtime.json").exists()


def test_retained_straight_control_hashes_and_classification_when_available():
    root = Path(__file__).resolve().parents[1]
    options = [root / "artifacts" / kind / control.PROTOCOL for kind in ("diagnostics", "evaluations")]
    evidence = next((p for p in options if (p / "response.json").exists()), None)
    if evidence is None:
        pytest.skip("retained control artifacts not distributed in Git")
    expected = {
        "response.json": "4e4d94d82f90621fe4441df907236de1b76c7a0e5950b948dfdd1354d6fa3c73",
        "decision.json": "e47e8dbea684e7d65e968bcc9a505436576b9173c95ae90f00c95ded37d73959",
        "launch.json": "c3fb377ce4aa8735cc0f69d5cd52be669135577d6db63230f6383d0766fc94fc",
        "runtime.json": "a207b34590137bdc5d8ca908a8e3bd86f16cf0d24dcf610ac3a005bc8424ee8a",
    }
    for name, digest in expected.items(): assert control.sha256(evidence / name) == digest
    report = json.loads((evidence / "response.json").read_text())
    decision = json.loads((evidence / "decision.json").read_text())
    assert decision["report_sha256"] == expected["response.json"]
    assert report["source"] == "0679be398b29ffc79dcf003001869e4d9a146afe"
    assert report["checkpoint_sha256"] == control.ACTOR_SHA256
    assert report["sample_steps"] == 400 and report["terminal_steps"] == []
    assert report["safety_failures"] == []
    assert len(report["groups"]["settled"]["body_forward_per_env_mean"]) == 8
    assert all(v < .27 for v in report["groups"]["settled"]["body_forward_per_env_mean"])
    assert report["classification"] == decision["classification"] == "straight-body-mean-outside-band"
    for key in ("training_admitted", "reopens_recovery_ab", "physical_motion_authorized"):
        assert report[key] is decision[key] is False
