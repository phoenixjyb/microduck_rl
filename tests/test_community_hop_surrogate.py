"""CPU compile/motor/contact fixtures, never published-policy inference."""

import copy
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mjlab_microduck import community_hop_surrogate as s


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def stock_buffers():
    plan, _ = s.load_declaration(ROOT)
    scene, assets, _ = s.verified_assets(ROOT, plan)
    return plan, scene, assets


@pytest.fixture
def plant(stock_buffers):
    plan, scene, assets = stock_buffers
    bam, _ = s.verified_bam(plan)
    return s.NativePlant(scene, assets, plan, bam)


def test_effective_torque_motors_and_passive_hinges_are_preserved(plant, stock_buffers):
    plan, scene, assets = stock_buffers
    raw = mujoco.MjSpec.from_string(scene.decode(),
        include={name: value for name, value in assets.items() if name.endswith(".xml")},
        assets={name: value for name, value in assets.items() if not name.endswith(".xml")}).compile()
    model, ctrl = plant.model, plant.controller
    assert model.nq == 35 and model.nv == 34 and model.nu == 14
    assert not np.any(model.actuator_forcelimited) and not np.any(model.actuator_ctrllimited)
    assert np.all(model.actuator_biastype == 0) and np.all(model.actuator_biasprm == 0)
    assert np.all(model.actuator_gainprm[:, 0] == 1)
    assert model.dof_armature[ctrl.dof] == pytest.approx([ctrl.bam.armature.value] * 14)
    for field in ("geom_contype", "geom_conaffinity", "geom_condim", "geom_friction", "geom_solref", "geom_solimp",
                  "jnt_range", "jnt_solref", "jnt_solimp", "eq_data", "eq_solref", "eq_solimp"):
        assert np.array_equal(getattr(raw, field), getattr(model, field)), field
    for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
        assert np.array_equal(getattr(raw, field)[ctrl.bdof], getattr(model, field)[ctrl.bdof]), field
    assert plant.data.qpos[ctrl.q] == pytest.approx(plan["interface"]["home_rad"])
    assert np.all(plant.data.qpos[ctrl.bq] == 0) and np.all(plant.data.qvel == 0)
    assert len(plant.ground_geoms) == 10


def test_bam_feedback_includes_backlash_but_rotor_math_does_not(plant):
    d, ctrl, bam = plant.data, plant.controller, plant.controller.bam
    d.qpos[ctrl.bq] = 0.012
    d.qvel[ctrl.dof] = 8
    d.qvel[ctrl.bdof] = 100  # Must NOT enter back-EMF/friction math.
    mujoco.mj_forward(plant.model, d)
    ctrl.target = plant.home + 0.02
    q_encoder = d.qpos[ctrl.q] + d.qpos[ctrl.bq]
    voltage = bam.actuator.compute_control(ctrl.target, q_encoder, d.qvel[ctrl.dof], 0.005)
    torque = bam.actuator.compute_torque(voltage, True, d.qpos[ctrl.q], d.qvel[ctrl.dof])
    external = s.friction_without_self(d, ctrl.dof)
    friction, damping = bam.compute_frictions(torque, external, d.qvel[ctrl.dof])
    passive_damping = plant.model.dof_damping[ctrl.bdof].copy()
    ctrl.update()
    assert ctrl.voltage == pytest.approx(voltage)
    assert ctrl.current == pytest.approx((voltage - bam.kt.value * 8) / bam.R.value)
    assert d.ctrl[ctrl.act] == pytest.approx(torque)
    assert plant.model.dof_frictionloss[ctrl.dof] == pytest.approx(friction)
    assert plant.model.dof_damping[ctrl.dof] == pytest.approx(damping)
    assert np.array_equal(plant.model.dof_damping[ctrl.bdof], passive_damping)
    wrong_voltage = bam.actuator.compute_control(ctrl.target, d.qpos[ctrl.q], d.qvel[ctrl.dof], 0.005)
    assert not np.allclose(voltage, wrong_voltage)


def test_unreachable_high_speed_current_limit_is_reported_not_hard_clipped(plant):
    ctrl, d = plant.controller, plant.data
    d.qvel[ctrl.dof] = 100
    ctrl.update()
    assert np.max(np.abs(ctrl.current)) > 1.75
    assert np.max(np.abs(ctrl.voltage)) <= 7.4
    assert ctrl.torque == pytest.approx(ctrl.current * ctrl.bam.kt.value)


def test_friction_selector_uses_dof_ids_not_joint_ids():
    data = SimpleNamespace(efc_type=np.array([mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF,
        mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT, mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF]),
        efc_id=np.array([6, 6, 8]), efc_force=np.array([3., 100., 5.]),
        qfrc_bias=np.arange(10.) * 0.2, qfrc_constraint=np.arange(10.) * 2)
    assert s.friction_without_self(data, np.array([6, 8])) == pytest.approx([7.8, 9.4])


@pytest.mark.parametrize("height,pitch", [(0.12, 0.), (0.17, 0.3), (0.119, -0.12)])
def test_mesh_bottom_matches_native_signed_plane_distance(plant, height, pitch):
    d = plant.data
    d.qpos[plant.root_q + 2] = height
    d.qpos[plant.root_q + 3:plant.root_q + 7] = [np.cos(pitch / 2), 0, np.sin(pitch / 2), 0]
    mujoco.mj_forward(plant.model, d)
    for foot, actual in zip(plant.feet, plant.kinematics()["foot_clearance_m"]):
        expected = mujoco.mj_geomDistance(plant.model, d, foot, plant.floor, 1., None)
        assert actual == pytest.approx(expected, abs=1e-7)
        assert abs(actual - d.geom_xpos[foot, 2]) > 0.001  # Not a foot-origin proxy.


def test_world_velocity_and_local_observation_frames(plant):
    d = plant.data
    yaw = np.pi / 2
    d.qpos[plant.root_q + 3:plant.root_q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    d.qvel[plant.root_dof:plant.root_dof + 6] = [0.2, 0.03, 0.04, 1., 2., 3.]
    mujoco.mj_forward(plant.model, d)
    assert plant.kinematics()["base_velocity_world_m_s"] == pytest.approx([0.2, 0.03, 0.04])
    assert plant.kinematics()["base_angular_velocity_rad_s"] == pytest.approx([-2., 1., 3.])
    assert plant.observation_state()["angular_velocity"] == pytest.approx([1., 2., 3.])
    assert plant.observation_state()["projected_gravity"] == pytest.approx([0, 0, -1])


def test_contact_force_sum_and_nonsole_floor_contact(plant):
    d, m = plant.data, plant.model
    d.qpos[plant.root_q + 2] -= 0.0035
    mujoco.mj_forward(m, d)
    measured = plant.contacts()
    assert measured["foot_contact"] == [True, True]
    expected = [0., 0.]
    for i, contact in enumerate(d.contact):
        if plant.floor not in (contact.geom1, contact.geom2) or contact.efc_address < 0:
            continue
        other = contact.geom2 if contact.geom1 == plant.floor else contact.geom1
        if other in plant.feet:
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)
            expected[plant.feet.index(other)] += abs(force[0])
    assert measured["foot_normal_force_n"] == pytest.approx(expected)
    assert sum(expected) > 0
    d.qpos[plant.root_q + 2] = 0.005
    mujoco.mj_forward(m, d)
    assert plant.contacts()["body_contact"]
    assert plant.unsafe_for_inference()


def test_single_fixture_step_measures_preintegration_state_and_terminal_does_not_step(plant):
    plant.prepare()
    before = copy.deepcopy(plant.kinematics())
    sample = plant.advance_and_measure([0.] * 14, terminal=False)
    assert plant.data.time == pytest.approx(0.005)
    for field, value in before.items():
        assert sample[field] == value
    assert not sample["reset_event"]
    plant.prepare()
    plant.advance_and_measure([0.] * 14, terminal=True)
    assert plant.data.time == pytest.approx(0.005)


def test_new_bam_friction_budget_enters_same_substep_solver(plant):
    plant.prepare()
    plant.advance_and_measure([0.2] * 14, terminal=False)
    model, data, ctrl = plant.model, plant.data, plant.controller
    dry = data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
    for dof in ctrl.dof:
        budget = data.efc_frictionloss[dry & (data.efc_id == dof)]
        assert len(budget) == 1
        assert budget[0] == pytest.approx(model.dof_frictionloss[dof])


def test_source_and_dependency_pin_fail_closed(stock_buffers, monkeypatch):
    plan, _, _ = stock_buffers
    with pytest.raises(ValueError, match="predeclaration"):
        s.load_declaration(Path("/nonexistent"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="explicitly empty"):
        s.verified_versions(plan)
    bad = copy.deepcopy(plan)
    bad["plant"]["bam_parameters_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parameter SHA256"):
        s.verified_bam(bad)


def test_static_preflight_cannot_infer_public_weights_or_step_physics(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    import onnxruntime
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **k: pytest.fail("session forbidden"))
    monkeypatch.setattr(mujoco, "mj_step", lambda *a: pytest.fail("step forbidden"))
    monkeypatch.setattr(mujoco, "mj_step1", lambda *a: pytest.fail("step1 forbidden"))
    monkeypatch.setattr(mujoco, "mj_step2", lambda *a: pytest.fail("step2 forbidden"))
    plan, _ = s.load_declaration(ROOT)
    if not all((ROOT / p["local_path"]).exists() for p in plan["policies"].values()):
        pytest.skip("local retained public artifacts are optional for CI")
    _, _, payloads, plant, receipt = s.preflight(ROOT)
    assert len(payloads) == 2
    assert receipt["runtime"]["physics_steps"] == receipt["runtime"]["policy_inferences"] == 0
    assert not receipt["runtime"]["sessions_created"]
    assert receipt["predeclaration_sha256"] == s.DECLARATION_SHA256
