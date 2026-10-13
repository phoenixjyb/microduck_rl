"""CPU-only gates for v8-seeded bounded learning and fixed evaluation."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import v8_actor_checkpoint as checkpoint_module
from mjlab_microduck import v8_actor_learning as learning


SOURCE = "a" * 40


def _without_learning_overrides(value, *, cfg=False):
    result = copy.deepcopy(value)
    if cfg:
        result.seed = 821
    else:
        result.seed = 821
        result.max_iterations = 5
        result.experiment_name = base.PROTOCOL
        result.run_name = "smoke"
    return result


@pytest.mark.parametrize("mode,worlds,updates,seed", [
    ("benchmark", 256, 10, 953),
    ("foundation", 256, 1000, 967),
])
def test_learning_recipe_changes_only_budget_seed_and_run_identity(mode, worlds, updates, seed):
    smoke_cfg, smoke_agent = learning.smoke.prepare_config()
    cfg, agent = learning.prepare_config(mode)

    expected_cfg = _without_learning_overrides(smoke_cfg, cfg=True)
    expected_cfg.seed = seed
    expected_cfg.scene.num_envs = expected_cfg.scene.terrain.num_envs = worlds
    expected_agent = _without_learning_overrides(smoke_agent)
    expected_agent.seed = seed
    expected_agent.max_iterations = updates
    expected_agent.experiment_name = learning.PROTOCOL
    expected_agent.run_name = mode

    assert cfg == expected_cfg
    assert agent == expected_agent
    assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == 256
    assert cfg.seed == agent.seed == seed
    assert agent.max_iterations == updates
    assert agent.actor.class_name == "mjlab_microduck.v8_actor_seed:FrozenV8Actor"
    assert agent.actor.obs_normalization is True
    assert agent.actor.distribution_cfg["init_std"] == .3
    assert agent.clip_actions is None
    assert cfg.rewards == smoke_cfg.rewards and cfg.curriculum == smoke_cfg.curriculum


def test_benchmark_projection_formula_and_strict_foundation_cap():
    assert base.SEEDS == (839, 853, 857)
    assert base.CASES == ((0., 0.), (.30, 0.), (.20, -.20), (.20, .20))
    assert learning.CAPS == {"benchmark": 300, "initial-eval": 600,
                             "foundation": 2100, "final-eval": 600}
    assert learning.benchmark_projection_seconds(10) == 1560
    just_under = (learning.CAPS["foundation"] - 30 - 60) / 150
    assert learning.benchmark_projection_seconds(just_under - 1e-6) < learning.CAPS["foundation"] - 30
    assert learning.benchmark_projection_seconds(just_under) == learning.CAPS["foundation"] - 30
    assert not learning.benchmark_projection_seconds(just_under) < learning.CAPS["foundation"] - 30


@pytest.mark.parametrize("elapsed", [0, -1, float("nan"), float("inf"), True, "12"])
def test_benchmark_projection_rejects_invalid_elapsed_values(elapsed):
    with pytest.raises(ValueError, match="finite positive measured benchmark time"):
        learning.benchmark_projection_seconds(elapsed)


def _baseline_rows():
    columns = [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "neck_pitch", "head_pitch", "head_yaw", "head_roll",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ]
    rows = []
    for seed in base.SEEDS:
        for speed, yaw in base.CASES:
            rows.append({
                "seed": seed, "speed": speed, "yaw": yaw,
                "complete": True, "steps": 240, "worlds": 8,
                "terminal_worlds": [], "rated_speed_exceed_fraction": 0., "torque_p99": .6,
                # Tracking can miss badly here: this is a catastrophic-baseline gate only.
                "speed_mae": [9.] * 8, "yaw_mae": [8.] * 8,
                "response_diagnostics": {
                    "joint_columns": columns,
                    "min_distance_to_hard_stop_rad_by_joint": [0.] * 9 + [-.005] + [0.] * 4,
                    "hard_stop_proximity_fraction_by_joint": [0.] * 9 + [.01] + [0.] * 4,
                },
            })
    return rows


def test_baseline_requires_exact_ordered_matrix_and_only_catastrophic_gates():
    rows = _baseline_rows()

    assert learning.validate_baseline(rows) is None


@pytest.mark.parametrize("damage", ["count", "seed", "order", "motor", "rated_speed",
                                     "terminal", "margin", "right_yaw"])
def test_baseline_refuses_missing_reordered_or_catastrophic_rows(damage):
    rows = _baseline_rows()
    if damage == "count":
        rows.pop()
    elif damage == "seed":
        rows[0]["seed"] = 1
    elif damage == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif damage == "motor":
        rows[0]["torque_p99"] = .600001
    elif damage == "rated_speed":
        rows[0]["rated_speed_exceed_fraction"] = 1e-9
    elif damage == "terminal":
        rows[0]["terminal_worlds"] = [0]
    elif damage == "margin":
        rows[0]["response_diagnostics"]["min_distance_to_hard_stop_rad_by_joint"][0] = -.005001
    elif damage == "right_yaw":
        rows[0]["response_diagnostics"]["hard_stop_proximity_fraction_by_joint"][9] = .010001

    with pytest.raises(ValueError):
        learning.validate_baseline(rows)


def _comparison_row(seed, speed, yaw, *, complete=True, value=0.):
    return {
        "seed": seed, "speed": speed, "yaw": yaw, "complete": complete,
        "speed_mae": [value] * 8, "yaw_mae": [value + 1.] * 8,
        "torque_p99": value + 2., "abs_force_p99_nm": value + 3.,
        "soft_limit_fraction": value + 4., "rated_speed_exceed_fraction": value + 5.,
        "mean_abs_mechanical_power_w": value + 6., "thermal_load_proxy": value + 7.,
    }


def test_paired_comparison_keeps_fixed_order_and_only_computes_means_for_complete_pairs():
    identities = [(s, v, w) for s in base.SEEDS for v, w in base.CASES]
    left = [_comparison_row(*identity, value=1.) for identity in identities]
    right = [_comparison_row(*identity, complete=(i != 1), value=3.)
             for i, identity in enumerate(identities)]

    rows = learning.paired_comparison(left, right)

    assert [(r["seed"], r["speed"], r["yaw"]) for r in rows] == identities
    assert rows[0]["speed_mae_mean_delta"] == 2.
    assert rows[0]["yaw_mae_mean_delta"] == 2.
    assert rows[1]["complete_pair"] is False
    assert rows[1]["speed_mae_mean_delta"] is None
    assert rows[1]["yaw_mae_mean_delta"] is None
    assert rows[1]["torque_p99_delta"] == 2.
    assert all(isinstance(value, (int, float)) and value == value
               for row in rows for key, value in row.items()
               if key.endswith("_delta") and value is not None)


def test_paired_comparison_rejects_wrong_pairing_and_different_matrix_lengths():
    identities = [(s, v, w) for s in base.SEEDS for v, w in base.CASES]
    left = [_comparison_row(*identity) for identity in identities]
    right = [_comparison_row(*identity) for identity in identities]
    right[0]["yaw"] = .01
    with pytest.raises(ValueError, match="paired held-out identity"):
        learning.paired_comparison(left, right)
    right[0]["yaw"] = identities[0][2]
    with pytest.raises(ValueError):
        learning.paired_comparison(left[:-1], right)


def _training_receipt(tmp_path, mode="benchmark"):
    root = tmp_path
    worlds, updates, seed = learning.MODES[mode]
    output = root / mode
    output.mkdir()
    (output / "initial.pt").write_bytes(b"initial checkpoint fixture")
    checkpoint = output / f"model_{updates - 1}.pt"
    checkpoint.write_bytes(b"final checkpoint fixture")
    actor_import = {"ppo_resume": False, "adam_state_entries": 0,
                    "artifact_sha256": learning.pinned.V8.sha256,
                    "maximum_actor_error": 1e-6,
                    "maximum_export_forward_error": 1e-6}
    (output / "actor-import.json").write_text(json.dumps(actor_import))
    result = {
        "protocol": learning.PROTOCOL, "source": SOURCE,
        "status": "training-complete-not-accepted",
        "worlds": worlds, "updates": updates, "seed": seed,
        "steps": updates * 24, "transitions": worlds * updates * 24,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "initial_checkpoint_sha256": hashlib.sha256((output / "initial.pt").read_bytes()).hexdigest(),
        "actor_import": actor_import,
        "elapsed_s": 60. if mode == "benchmark" else 1200.,
        **dict.fromkeys(base.FULL_COMMAND_COVERAGE, 1),
    }
    (output / "result.json").write_text(json.dumps(result))
    return root, output, result, actor_import


@pytest.mark.parametrize("mode", ["benchmark", "foundation"])
def test_training_receipt_binds_exact_budget_and_hashes_both_checkpoints(tmp_path, monkeypatch, mode):
    root, output, expected, _ = _training_receipt(tmp_path, mode)
    calls = []
    sha256 = base.sha256

    def observed_sha(path):
        calls.append(path)
        return sha256(path)

    monkeypatch.setattr(learning.base, "sha256", observed_sha)
    result = learning.validate_training_receipt(root, mode, SOURCE)

    assert result == expected
    assert calls == [output / f"model_{learning.MODES[mode][1] - 1}.pt",
                     output / "initial.pt"]


@pytest.mark.parametrize("damage", ["source", "worlds", "updates", "checkpoint-sha",
                                     "initial-sha", "import-binding", "resume", "adam",
                                     "coverage", "elapsed", "nonfinite"])
def test_training_receipt_rejects_counterfeit_or_incomplete_evidence(tmp_path, damage):
    root, output, result, actor_import = _training_receipt(tmp_path)
    changed = copy.deepcopy(result)
    if damage == "source":
        changed["source"] = "b" * 40
    elif damage == "worlds":
        changed["worlds"] = 64
    elif damage == "updates":
        changed["updates"] = 9
    elif damage == "checkpoint-sha":
        changed["checkpoint_sha256"] = "0" * 64
    elif damage == "initial-sha":
        changed["initial_checkpoint_sha256"] = "0" * 64
    elif damage == "import-binding":
        changed["actor_import"]["artifact_sha256"] = "0" * 64
    elif damage == "resume":
        changed["actor_import"]["ppo_resume"] = True
        (output / "actor-import.json").write_text(json.dumps(changed["actor_import"]))
    elif damage == "adam":
        changed["actor_import"]["adam_state_entries"] = 1
        (output / "actor-import.json").write_text(json.dumps(changed["actor_import"]))
    elif damage == "coverage":
        changed[base.FULL_COMMAND_COVERAGE[0]] = 0
    elif damage == "elapsed":
        changed["elapsed_s"] = learning.CAPS["benchmark"] + 1
    elif damage == "nonfinite":
        changed["sample"] = float("nan")
    (output / "result.json").write_text(json.dumps(changed))

    with pytest.raises(ValueError):
        learning.validate_training_receipt(root, "benchmark", SOURCE)


def _matrix_row(checkpoint_sha, seed, speed, yaw):
    return {"checkpoint_sha256": checkpoint_sha, "seed": seed, "speed": speed, "yaw": yaw,
            "complete": True, "steps": 240, "worlds": 8, "terminal_worlds": [],
            "speed_mae": [.02] * 8, "yaw_mae": [.05] * 8,
            "torque_p99": .5, "rated_speed_exceed_fraction": 0.}


@pytest.mark.parametrize("checkpoint_name", ["initial.pt", "model_999.pt"])
def test_evaluate_matrix_uses_bound_typed_checkpoint_ordered_diagnostics_and_exclusive_outputs(
        tmp_path, monkeypatch, checkpoint_name):
    checkpoint = tmp_path / checkpoint_name
    checkpoint.write_bytes(b"frozen final checkpoint fixture")
    output = tmp_path / "final-eval"
    output.mkdir()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    seen = {"construct": [], "evaluate": [], "health": 0, "decision": None}

    class FakeFrozenCheckpoint:
        def __init__(self, path, expected_sha256):
            seen["construct"].append((path, expected_sha256))
            self.path, self.sha256 = path, expected_sha256

        def provenance(self):
            return {"backend": "frozen-v8-ppo-actor-only", "artifact_sha256": self.sha256}

    def fake_evaluate(path, seed, speed, yaw, health, **kwargs):
        seen["evaluate"].append((path, seed, speed, yaw, health, kwargs))
        return _matrix_row(digest, seed, speed, yaw)

    gate = {"decision": "foundation-ready-for-next-experiment", "failures": []}

    def fake_decision(rows):
        seen["decision"] = rows
        return gate

    def health():
        seen["health"] += 1

    monkeypatch.setattr(checkpoint_module, "FrozenV8Checkpoint", FakeFrozenCheckpoint)
    monkeypatch.setattr(learning.base, "evaluate_case", fake_evaluate)
    monkeypatch.setattr(learning.limits, "decision", fake_decision)

    result = learning.evaluate_matrix(checkpoint, output, health)
    identities = [(seed, speed, yaw) for seed in base.SEEDS for speed, yaw in base.CASES]

    assert seen["construct"] == [(checkpoint, digest)]
    assert [(row[1], row[2], row[3]) for row in seen["evaluate"]] == identities
    assert seen["health"] == len(identities)
    assert all(row[0] == checkpoint and row[4] is health for row in seen["evaluate"])
    typed = seen["evaluate"][0][5]["frozen_v8_checkpoint"]
    assert type(typed) is FakeFrozenCheckpoint
    assert all(row[5] == {"diagnostics": True, "yaw_trace": True,
                          "gait_trace": True, "frozen_v8_checkpoint": typed}
               for row in seen["evaluate"])
    assert result["checkpoint_sha256"] == digest
    assert result["numerical_gate"] is gate and seen["decision"] == result["cases"]
    assert result["inference_backend"]["backend"] == "frozen-v8-ppo-actor-only"
    for index, row in enumerate(result["cases"]):
        path = output / f"s{identities[index][0]}-c{index % len(base.CASES)}.json"
        assert json.loads(path.read_text()) == row

    prior = (output / "s839-c0.json").read_bytes()
    with pytest.raises(FileExistsError):
        learning.evaluate_matrix(checkpoint, output, health)
    assert (output / "s839-c0.json").read_bytes() == prior


def test_initial_matrix_refuses_nonoriginal_seed_before_any_rollout(tmp_path, monkeypatch):
    path = tmp_path / "initial.pt"
    path.write_bytes(b"rejected initial baseline")
    class ChangedSeed:
        def __init__(self, *args):
            self.sha256 = "b" * 64

        def require_initial_seed(self):
            raise ValueError("initial baseline must equal original v8")
    monkeypatch.setattr(checkpoint_module, "FrozenV8Checkpoint", ChangedSeed)
    calls = []
    monkeypatch.setattr(learning.base, "evaluate_case", lambda *a, **k: calls.append(True))
    with pytest.raises(ValueError, match="initial baseline"):
        learning.evaluate_matrix(path, tmp_path, lambda: calls.append(True), initial=True)
    assert calls == []
