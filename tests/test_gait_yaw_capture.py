from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import gait_yaw_capture as capture


def fixture():
    data = NS(root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(8, 1),
              root_link_ang_vel_b=torch.zeros(8, 3), root_link_ang_vel_w=torch.zeros(8, 3),
              site_lin_vel_w=torch.arange(8*5*3, dtype=torch.float32).reshape(8, 5, 3),
              joint_pos=torch.arange(8*28, dtype=torch.float32).reshape(8, 28),
              default_joint_pos=torch.ones(8, 28), joint_vel=torch.ones(8, 28)*2,
              data=NS(time=torch.arange(8, dtype=torch.float64)*.01))
    robot = NS(data=data, find_sites=lambda *args, **kw: ([3, 1], list(capture.SITES)),
               find_joints=lambda *args, **kw: ([1, 7, 14, 23], list(capture.HEAD_JOINTS)))
    contact = NS(found=torch.ones(8, 2), force=torch.ones(8, 2, 3),
                 current_air_time=torch.zeros(8, 2), current_contact_time=torch.ones(8, 2)*.02)
    cfg = NS(name="feet_ground_contact", num_slots=1, fields=("found", "force"),
             reduce="netforce", track_air_time=True, history_length=0,
             primary=NS(entity="robot", mode="geom"), secondary=NS(entity=None, mode="body", pattern="terrain"))
    sensor = NS(cfg=cfg, data=contact,
                _slots=[NS(primary_name=name, field_name=field)
                        for name in capture.CONTACTS for field in ("found", "force")])
    commands = dict(twist=torch.tensor([[.2, 0., -.2]]).repeat(8, 1), head_pose=torch.zeros(8, 4))
    env = NS(num_envs=8, step_dt=.02, scene=NS(sensors=dict(feet_ground_contact=sensor)),
             episode_length_buf=torch.zeros(8, dtype=torch.long),
             command_manager=NS(get_command=commands.__getitem__))
    obs = torch.zeros(8, 61)
    obs[:, 48:51] = commands["twist"]
    return env, robot, obs


def test_named_interleaved_capture_detached_nonaliasing_no_rng_or_input_mutation():
    env, robot, obs = fixture()
    state = torch.random.get_rng_state().clone()
    expected = robot.data.joint_pos[:, [1, 7, 14, 23]] - 1.
    sample = capture.capture(env, robot, obs.requires_grad_())
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(sample["head_position_offset"], expected)
    assert torch.equal(sample["foot_velocity_w"], robot.data.site_lin_vel_w[:, [3, 1]])
    assert not sample["actor_head_command"].requires_grad
    robot.data.joint_pos.zero_()
    env.scene.sensors["feet_ground_contact"].data.force.zero_()
    env.episode_length_buf.fill_(99)
    assert torch.equal(sample["head_position_offset"], expected)
    assert sample["contact_force_w"].eq(1).all()
    assert sample["episode_length"].eq(0).all()
    assert sample["episode_length"].dtype == torch.long
    raw = capture.trace([sample, sample], "checkpoint")
    assert raw["protocol"] == "microduck-gait-yaw-trace-v1"
    assert raw["force_frame"] == "world" and raw["terrain"] == "stationary-flat"
    assert type(raw["episode_length"][0][0]) is int
    assert raw["head_joints"] == list(capture.HEAD_JOINTS)
    assert raw["policy_acceptance"] is raw["physical_motion_authorized"] is False


@pytest.mark.parametrize("fault", ["contact-order", "contact-field", "force-frame", "missing-air",
    "slots", "ground", "head-names", "site-names", "head-command", "episode-type",
    "negative-clock", "nonfinite-force", "force-shape", "negative-found", "fractional-found", "worlds", "dt"])
def test_capture_rejects_ambiguous_channels_or_timing(fault):
    env, robot, obs = fixture()
    sensor = env.scene.sensors["feet_ground_contact"]
    if fault == "contact-order": sensor._slots.reverse()
    elif fault == "contact-field": sensor.cfg.fields = ("force",)
    elif fault == "force-frame": sensor.cfg.reduce = "maxforce"
    elif fault == "missing-air": sensor.data.current_air_time = None
    elif fault == "slots": sensor.cfg.num_slots = 2
    elif fault == "ground": sensor.cfg.secondary.pattern = "moving_ground"
    elif fault == "head-names": robot.find_joints = lambda *a, **kw: ([1, 7, 14, 23], list(reversed(capture.HEAD_JOINTS)))
    elif fault == "site-names": robot.find_sites = lambda *a, **kw: ([1, 3], list(reversed(capture.SITES)))
    elif fault == "head-command": obs[:, 51] = 1.
    elif fault == "episode-type": env.episode_length_buf = torch.zeros(8)
    elif fault == "negative-clock": sensor.data.current_air_time[0, 0] = -.01
    elif fault == "nonfinite-force": sensor.data.force[0, 0, 0] = float("nan")
    elif fault == "force-shape": sensor.data.force = torch.ones(8, 3)
    elif fault == "negative-found": sensor.data.found[0, 0] = -1.
    elif fault == "fractional-found": sensor.data.found[0, 0] = .5
    elif fault == "worlds": env.num_envs = 7
    else: env.step_dt = .01
    with pytest.raises(ValueError): capture.capture(env, robot, obs)


def test_capture_never_steps_sensor_or_updates_air_timers():
    from mjlab.sensor.contact_sensor import ContactData, ContactSensor, ContactSensorCfg
    env, robot, obs = fixture()
    prototype = env.scene.sensors["feet_ground_contact"]
    sensor = ContactSensor(ContactSensorCfg(name="feet_ground_contact", primary=prototype.cfg.primary))
    sensor.cfg = prototype.cfg
    sensor._slots = prototype._slots
    state = NS(current_air_time=torch.zeros(8, 2), last_air_time=torch.ones(8, 2),
               current_contact_time=torch.ones(8, 2)*.02, last_contact_time=torch.zeros(8, 2),
               last_time=torch.ones(8))
    sensor._air_time_state = state
    sensor._extract_sensor_data = lambda: ContactData(found=torch.ones(8, 2), force=torch.ones(8, 2, 3))
    def forbidden(*args): raise AssertionError("diagnostic must not step/reset the sensor")
    sensor.update = sensor.reset = forbidden
    env.scene.sensors["feet_ground_contact"] = sensor
    before = {key: value.clone() for key, value in vars(state).items()}
    sample = capture.capture(env, robot, obs)
    assert sample["current_contact_time"].eq(.02).all()
    assert all(torch.equal(value, before[key]) for key, value in vars(state).items())


def test_real_warp_clock_proxy_is_read_as_tensor_without_mutation_or_aliasing():
    import warp as wp
    from mjlab.sim.sim_data import TorchArray
    env, robot, obs = fixture()
    clock = TorchArray(wp.array([float(i) * .02 for i in range(8)],
                                dtype=wp.float32, device="cpu"))
    assert not isinstance(clock, torch.Tensor)
    robot.data.data.time = clock
    before = clock[:].clone()
    sample = capture.capture(env, robot, obs)
    assert isinstance(sample["physics_time"], torch.Tensor)
    assert torch.equal(sample["physics_time"], before)
    assert torch.equal(clock[:], before)
    clock[:].fill_(99.)
    assert torch.equal(sample["physics_time"], before)
