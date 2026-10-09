"""Frozen installed-source inspection only; no solver or runtime library imports."""
import ast
from importlib.metadata import distribution
from pathlib import Path
import subprocess
import sys

import pytest
from mjlab_microduck import ada_solver_stage_audit as p


def sources():
    dist = distribution("mujoco-warp")
    return {k: Path(dist.locate_file("mujoco_warp/_src/" + k + ".py")).read_bytes() for k in p.HASHES}


def test_import_and_audit_load_no_runtime():
    code = "import sys; from mjlab_microduck.ada_solver_stage_audit import audit; r=audit(); assert r['new_solver_calls']==0; assert not {'numpy','torch','warp','mujoco','mujoco_warp'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)


def test_installed_stages_and_false_qualification():
    r = p.audit()
    assert len(r["library_sources"]) == 72
    assert len(r["stages"]) == 10
    assert [x["call"] for x in r["stages"]["forward.fwd_position"]["lexical_calls"]] == list(p.POSITION)
    assert r["ordinary_forward_recomputes_kinematics"]
    assert not r["lexical_inventory_is_transitive_execution_proof"]
    assert not r["generated_manifold_is_same_manifold_solver_control"]
    assert not r["measured_historical_gpu_collision_pose_complete"]
    for k in ("new_collision_calls", "new_constraint_calls", "new_solver_calls", "new_integration_steps"):
        assert r[k] == 0
    for k in ("runtime_imported", "solver_qualified", "training_authorized", "physical_motion_authorized"):
        assert r[k] is False
    assert "d.qacc_warmstart" in r["stages"]["solver._solve"]["lexical_data_attributes"]
    copies = [x["arguments"] for x in r["stages"]["solver._solve"]["lexical_calls"] if x["call"] == "wp.copy"]
    assert copies == [["d.qacc", "d.qacc_warmstart"], ["d.qacc", "d.qacc_smooth"]]


@pytest.mark.parametrize("name", p.HASHES)
def test_any_changed_source_byte_refuses(name):
    raw = sources(); raw[name] += b"\n"
    with pytest.raises(ValueError, match="unchanged stage source"): p.inspect_sources(raw)


def test_inventory_and_ambiguous_function_refuse():
    raw = sources(); raw.pop("solver")
    with pytest.raises(ValueError, match="complete declared"): p.inspect_sources(raw)
    for source in ("pass", "def x(): pass\ndef x(): pass"):
        with pytest.raises(ValueError, match="one exact"): p.function(ast.parse(source), "x")


def test_lexical_order_preserves_branch_calls_without_claiming_execution():
    node = ast.parse("def f():\n if False: a.b()\n c.d(factorize=False)\n").body[0]
    assert [x["call"] for x in p.calls(node)] == ["a.b", "c.d"]
    assert p.calls(node)[1]["keywords"] == {"factorize": "False"}
    assert p.attributes(node, "a") == ["a.b"]


def test_no_physics_runtime_import_or_execution_calls():
    tree = ast.parse(Path(p.__file__).read_text())
    calls = {p.dotted(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not any(c.split(".")[-1] in {"forward", "kinematics", "collision", "make_constraint", "solve", "step"} for c in calls)
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imports |= {x.name for n in ast.walk(tree) if isinstance(n, ast.Import) for x in n.names}
    assert not imports & {"numpy", "torch", "warp", "mujoco", "mujoco_warp"}
