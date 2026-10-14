"""Unchanged-recipe exposure lesson, no mutation of historical protocols."""
import copy
import json
from types import SimpleNamespace

import pytest

from mjlab_microduck import v8_steering_budget as lesson
from mjlab_microduck import v8_gait_scale_contrast as driver
from mjlab_microduck import v8_motor_headroom as headroom
from test_v8_gait_scale_contrast import _payload, _runner, _FakeFrozenActor


@pytest.mark.parametrize("mode",lesson.MODES)
def test_complete_hashes_equal_for_short_stages_only_budget_differs_in_foundation(mode):
    proof = lesson.recipe_match(mode)
    assert proof["control_recipe_sha256"] == proof["treatment_recipe_sha256"]
    assert proof["full_recipe_hashes_distinct"] is (mode == "foundation")
    assert proof["comparison_mask"] == ["agent.max_iterations"]
    assert json.loads(json.dumps(proof)) == proof
    for arm in lesson.ARMS:
        cfg, agent = lesson.prepare_config(mode, arm)
        parent, pa = headroom.prepare_config(mode,"control")
        assert driver.stage_spec(lesson,mode,arm) == (cfg.scene.num_envs,agent.max_iterations,agent.seed)
        agent.experiment_name, agent.max_iterations = pa.experiment_name, pa.max_iterations
        assert driver._full_canonical(cfg,agent) == driver._full_canonical(parent,pa)
        assert cfg.rewards["selected_motor_load"].weight == 0
        assert cfg.rewards["yaw_bias"].params["tau_s"] == .5
        observer = lesson.Observer(mode,arm)
        assert observer.selected_arm == "control"
        assert observer.report_arm == arm and observer.protocol == lesson.PROTOCOL


def test_initializer_preserves_parent_adam_and_critic_counts_without_importing_clocks(monkeypatch,tmp_path):
    from mjlab_microduck import v8_actor_seed
    payload = _payload()
    for state in payload["optimizer_state_dict"]["state"].values(): state["step"].fill_(60000)
    payload["critic_state_dict"]["obs_normalizer.count"].fill_(18432000)
    payload["optimizer_state_dict"]["param_groups"][0]["lr"] = .00011390625000000005
    monkeypatch.setattr(lesson,"_read_parent",lambda:payload)
    monkeypatch.setattr(v8_actor_seed,"FrozenV8Actor",_FakeFrozenActor)
    before=(driver.PARENT_ADAM_STEP,headroom.PARENT_ADAM_STEP)
    runner,env=_runner()
    report=lesson.initialize(runner,env,tmp_path)
    assert report["adam_step_min"]==report["adam_step_max"]==60000
    assert report["critic_normalizer_count"]==18432000
    assert report["local_iteration"]==report["local_common_step_counter"]==0
    assert report["parent_sha256"]==lesson.PARENT_SHA
    assert runner.alg.learning_rate==runner.alg.optimizer.param_groups[0]["lr"]==.00011390625000000005
    assert before==(driver.PARENT_ADAM_STEP,headroom.PARENT_ADAM_STEP)


def test_final_checkpoints_are_the_declared_endpoints_not_cherry_picked(tmp_path):
    assert driver.foundation_checkpoint(tmp_path,lesson,"control").name=="model_999.pt"
    assert driver.foundation_checkpoint(tmp_path,lesson,"extended").name=="model_1999.pt"
    assert driver.projection_seconds(8,2000)<lesson.CAPS["foundation"]-30
    assert driver.projection_seconds(13,2000)>lesson.CAPS["foundation"]-30


def test_predeclared_diagnostic_requires_complete_world_gain_and_rejects_wrong_matrix():
    from test_gait_path_qualification import make_case_trace, quaternion
    rows=[]
    for seed in base_seeds():
        for speed,yaw in driver.base.CASES:
            row,trace=make_case_trace(speed=speed,yaw=yaw,seed=seed)
            row["gait_path_trace"]=trace
            rows.append(row)
    worse=copy.deepcopy(rows)
    for row in worse:
        if row["speed"]==0:continue
        row["yaw_mae"]=[.2,.2,0.,0.,0.,0.,0.,0.]
        trace=row["gait_path_trace"]
        for step in range(60,240):
            t=(step-60)*.02
            for world in (0,1):
                trace["quat_wxyz"][step][world]=quaternion((row["yaw"]+.2)*t)
                trace["angular_b"][step][world][2]=row["yaw"]+.2
                trace["angular_w"][step][world][2]=row["yaw"]+.2
    report=lesson.diagnostic_comparison(worse,rows)
    assert report["decision"]=="budget-lesson-useful-not-qualified"
    assert report["groups"]["extended"]["complete_moving_route_worlds"]==72
    assert report["groups"]["control"]["complete_moving_route_worlds"]==54
    assert report["policy_acceptance"] is report["physical_motion_authorized"] is False
    assert lesson.diagnostic_comparison(rows,rows)["decision"]=="budget-lesson-rejected"
    with pytest.raises(ValueError,match="exact ordered"):
        lesson.diagnostic_comparison(rows[::-1],rows)


def base_seeds():
    return driver.base.SEEDS
