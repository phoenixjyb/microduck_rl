"""Inert YAML, exact artifact identity and incomplete-binding refusal."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from mjlab_microduck import retained_skill_binding as binding


def test_python_tags_are_preserved_but_never_executed(tmp_path):
    marker = tmp_path / "must-not-exist"
    tree = binding.tagged_yaml(
        "spec_fn: !!python/name:example.model_factory ''\n"
        f"malicious: !!python/object/apply:os.system ['touch {marker}']\n"
        "joints: !!python/tuple [left, right]\n"
    )
    assert binding.select(tree, "spec_fn")["tag"].endswith("python/name:example.model_factory")
    assert binding.select(tree, "malicious")["tag"].endswith("python/object/apply:os.system")
    assert binding.select(tree, "joints")["tag"].endswith("python/tuple")
    assert not marker.exists()


def test_mapping_order_and_scalar_spelling_are_preserved():
    tree = binding.tagged_yaml("z: 0.005\na: false\nq: null\n")
    assert [k for k, _ in tree["value"]] == ["z", "a", "q"]
    assert [v["value"] for _, v in tree["value"]] == ["0.005", "false", "null"]


def test_changed_callable_tag_or_term_order_changes_snapshot():
    assert binding.tagged_yaml("f: !!python/name:a.b ''") != binding.tagged_yaml("f: !!python/name:a.c ''")
    assert binding.tagged_yaml("a: 1\nb: 2") != binding.tagged_yaml("b: 2\na: 1")


def test_aliases_are_expanded_as_independent_inert_values():
    tree = binding.tagged_yaml("a: &x [1, 2]\nb: *x\n")
    a, b = binding.select(tree, "a"), binding.select(tree, "b")
    assert a == b and a is not b


@pytest.mark.parametrize("text", ["", "a: 1\na: 2", "a: &x [*x]",
                                  "? [a, b]\n: c", "a: [", "---\na: 1\n---\na: 2"])
def test_ambiguous_or_invalid_yaml_rejected(text):
    with pytest.raises(ValueError):
        binding.tagged_yaml(text)


def test_resource_bounds():
    for text in ("a" * 2_000_001, "[" * 82 + "0" + "]" * 82,
                 "a: &a [0,0,0,0,0,0,0,0,0,0]\nb: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a,*a]\n"
                 "c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b,*b]\n"
                 "d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c,*c]\ne: [*d,*d,*d,*d,*d,*d,*d,*d,*d,*d]\n"):
        with pytest.raises(ValueError):
            binding.tagged_yaml(text)


def config_text():
    env = dict(decimation=4, sim=dict(mujoco=dict(timestep=.005)),
               scene=dict(entities=dict(robot={"name": "synthetic-not-effective"})),
               observations=dict(actor=dict(terms=dict(z={}, a={}))),
               actions=dict(joint_pos={}), commands=dict(twist={}))
    # JSON is valid YAML and preserves insertion ordering for these fixtures.
    return json.dumps(env), json.dumps(dict(actor={"class_name": "MLPModel"}, clip_actions=None))


def test_declared_extraction_keeps_order_and_partial_scope():
    result = binding.declared_config(*config_text())
    assert result["actor_term_order"] == ["z", "a"]
    assert result["provenance"] == "saved-declarations-only-not-effective-runtime"
    assert result["declared_control_period_s"] == .02


@pytest.mark.parametrize("field", ["decimation", "sim", "actions", "commands", "observations", "scene"])
def test_missing_config_never_defaults_to_compatible(field):
    env, agent = config_text()
    value = json.loads(env)
    del value[field]
    with pytest.raises(ValueError):
        binding.declared_config(json.dumps(value), agent)


@pytest.mark.parametrize("value", [True, 4.0, 5, None])
def test_wrong_declared_timing_rejected(value):
    env, agent = config_text()
    cfg = json.loads(env)
    cfg["decimation"] = value
    with pytest.raises(ValueError, match="timing"):
        binding.declared_config(json.dumps(cfg), agent)


def synthetic_checkpoint():
    actor = {}
    for i, inputs, outputs in ((0, 61, 512), (2, 512, 256), (4, 256, 128), (6, 128, 14)):
        actor[f"mlp.{i}.weight"] = torch.zeros(outputs, inputs)
        actor[f"mlp.{i}.bias"] = torch.zeros(outputs)
    for key in ("_mean", "_var", "_std"):
        actor[f"obs_normalizer.{key}"] = torch.ones(1, 61)
    actor["obs_normalizer.count"] = torch.tensor(1, dtype=torch.int64)
    actor["distribution.std_param"] = torch.ones(14)
    return dict(iter=8998, infos=dict(env_state=dict(common_step_counter=216000)), actor_state_dict=actor)


def test_checkpoint_structure_is_read_only_not_inference_or_joint_binding():
    checkpoint = synthetic_checkpoint()
    original = deepcopy(checkpoint)
    report = binding.checkpoint_structure(checkpoint)
    assert (report["actor_dimension"], report["motor_output_dimension"]) == (61, 14)
    for field in ("inference_executed", "observation_order_bound_to_tensor_columns", "motor_output_order_bound_to_joints"):
        assert report[field] is False
    for key, tensor in checkpoint["actor_state_dict"].items():
        assert torch.equal(tensor, original["actor_state_dict"][key])


@pytest.mark.parametrize("damage", ["nan", "dtype", "dim", "count", "std", "extra", "iter", "counter"])
def test_invalid_checkpoint_structure_rejected(damage):
    c = synthetic_checkpoint()
    actor = c["actor_state_dict"]
    if damage == "nan":
        actor["mlp.0.weight"][0, 0] = float("nan")
    elif damage == "dtype":
        actor["mlp.0.weight"] = actor["mlp.0.weight"].double()
    elif damage == "dim":
        actor["mlp.6.bias"] = torch.zeros(13)
    elif damage == "count":
        actor["obs_normalizer.count"] = torch.tensor(0)
    elif damage == "std":
        actor["obs_normalizer._std"][0, 0] = 0
    elif damage == "extra":
        actor["mystery"] = torch.ones(1)
    elif damage == "iter":
        c["iter"] = 8999
    else:
        c["infos"]["env_state"]["common_step_counter"] = 216000.0
    with pytest.raises(ValueError):
        binding.checkpoint_structure(c)


def fake_inventory(root, monkeypatch):
    files = {}
    for i in range(105):
        p = root / f"file-{i}"
        p.write_bytes(str(i).encode())
        files[p.name] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
    data = json.dumps(dict(protocol=binding.CAMPAIGN, source=binding.SOURCE, files=files)).encode()
    (root / "manifest.json").write_bytes(data)
    monkeypatch.setattr(binding, "MANIFEST_SHA", hashlib.sha256(data).hexdigest())


def test_exact_inventory_verified(tmp_path, monkeypatch):
    fake_inventory(tmp_path, monkeypatch)
    assert len(binding.verified_f1y_inventory(tmp_path)) == 105


@pytest.mark.parametrize("damage", ["manifest", "payload", "missing", "extra", "symlink-dir", "symlink-manifest"])
def test_inventory_damage_rejected(tmp_path, monkeypatch, damage):
    fake_inventory(tmp_path, monkeypatch)
    if damage == "manifest":
        (tmp_path / "manifest.json").write_bytes(b"{}")
    elif damage == "payload":
        (tmp_path / "file-0").write_bytes(b"x")
    elif damage == "missing":
        (tmp_path / "file-0").unlink()
    elif damage == "extra":
        (tmp_path / "extra").write_bytes(b"x")
    elif damage == "symlink-dir":
        (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    else:
        (tmp_path / "manifest.json").rename(tmp_path / "original")
        (tmp_path / "manifest.json").symlink_to(tmp_path / "original")
    with pytest.raises(ValueError):
        binding.verified_f1y_inventory(tmp_path)


def test_binding_requires_cuda_hidden_before_reading(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="hide CUDA"):
        binding.audit_retained(tmp_path)


@pytest.mark.parametrize("field,value", [("source", "b" * 40), ("protocol", "other"), ("files", {})])
def test_manifest_identity_and_coverage_are_not_inferred(tmp_path, monkeypatch, field, value):
    fake_inventory(tmp_path, monkeypatch)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest[field] = value
    data = json.dumps(manifest).encode()
    (tmp_path / "manifest.json").write_bytes(data)
    monkeypatch.setattr(binding, "MANIFEST_SHA", hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError):
        binding.verified_f1y_inventory(tmp_path)


def test_cli_writes_only_new_separate_report(tmp_path, monkeypatch, capsys):
    root = tmp_path / "closed"
    root.mkdir()
    output = tmp_path / "derived" / "binding.json"
    report = dict(decision="partial-artifact-binding-only", unresolved=["synthetic"], descriptor=None)
    monkeypatch.setattr(binding, "audit_retained", lambda _: report)
    monkeypatch.setattr("sys.argv", ["binding", str(root), "--output", str(output)])
    binding.main()
    assert json.loads(output.read_text()) == report
    assert json.loads(capsys.readouterr().out)["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    before = output.read_bytes()
    with pytest.raises(ValueError, match="new output"):
        binding.main()
    assert output.read_bytes() == before
    assert list(root.iterdir()) == []


def test_cli_refuses_to_write_inside_closed_evidence_before_audit(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["binding", str(tmp_path), "--output", str(tmp_path / "result.json")])
    monkeypatch.setattr(binding, "audit_retained", lambda _: pytest.fail("must refuse before audit"))
    with pytest.raises(ValueError, match="closed evidence"):
        binding.main()


def test_real_retained_artifact_when_available(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    roots = [repo / "artifacts" / prefix / binding.CAMPAIGN for prefix in ("diagnostics", "experiments")]
    root = next((r for r in roots if (r / "manifest.json").is_file()), None)
    if root is None:
        pytest.skip("optional retained artifact is not in source checkout")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    report = binding.audit_retained(root)
    assert report["descriptor"] is None
    assert report["decision"] == "partial-artifact-binding-only"
    assert len(report["unresolved"]) == 5
    assert len(report["recorded_evaluation"]["failed_gates"]) == 16
    assert binding.select(report["declared_config"]["robot"], "spec_fn")["tag"].endswith("microduck_constants.get_walk_spec")
    assert report["checkpoint_structure"]["actor_dimension"] == 61
    assert report["policy_acceptance"] is report["transition_authorized"] is False
