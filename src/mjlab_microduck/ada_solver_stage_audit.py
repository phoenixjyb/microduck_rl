"""Read-only frozen solver-stage source audit; no runtime or solver execution."""
import argparse
import ast
from hashlib import sha256
from importlib.metadata import distribution
import os
from pathlib import Path
import re

from mjlab_microduck import ada_saved_pose_collision as saved

need = saved.need
PROTOCOL = "microduck-saved-pose-solver-stage-source-audit-v1"
HASHES = {
    "forward": "c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3",
    "constraint": "b69f15e5c7206b30bfe1af12b5ca6c0bdf3e37398116846643df73a2e8f8ef53",
    "solver": "bba0c67182ade84f5375d6a066048e111edd3371b33d46a6f1246349f22bb30a",
    "sensor": "2ba411f182ba5fb7d88e85050c55ef4055fafefe2d26555f429ecd4e0d80c19e",
    "passive": "4928d125d85deb57607d738fd40e2653ed4e4a9e690b88665d649adee8936ffb",
    "smooth": saved.SOURCE_FILES["mujoco_warp/_src/smooth.py"],
}
# Lexical call order includes conditional/unreachable branches, NOT execution.
POSITION = ("smooth.kinematics", "smooth.com_pos", "smooth.camlight", "smooth.flex",
            "smooth.tendon", "smooth.crb", "smooth.tendon_armature", "smooth.factor_m",
            "collision_driver.collision", "constraint.make_constraint", "island.island",
            "smooth.transmission")
FORWARD = ("fwd_position", "d.sensordata.zero_", "sensor.sensor_pos", "sensor.energy_pos",
           "d.energy.zero_", "fwd_velocity", "sensor.sensor_vel", "sensor.energy_vel",
           "m.callback.control", "fwd_actuation", "fwd_acceleration", "solver.solve",
           "sensor.sensor_acc")


def dotted(node):
    if isinstance(node, ast.Name): return node.id
    if isinstance(node, ast.Attribute):
        root = dotted(node.value)
        return root + "." + node.attr if root else ""
    return ""


def function(tree, name):
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    need(len(matches) == 1, "one exact top-level stage: " + name)
    return matches[0]


def calls(node):
    return [dict(call=dotted(n.func), line=n.lineno, column=n.col_offset,
                 arguments=[ast.unparse(a) for a in n.args],
                 keywords={k.arg: ast.unparse(k.value) for k in n.keywords})
            for n in sorted((n for n in ast.walk(node) if isinstance(n, ast.Call)),
                            key=lambda n: (n.lineno, n.col_offset))]


def attributes(node, root):
    """Conservative lexical references, not a transitive read/write proof."""
    return sorted({dotted(n) for n in ast.walk(node) if isinstance(n, ast.Attribute)
                   and dotted(n).startswith(root + ".")})


def inspect_sources(raw):
    need(set(raw) == set(HASHES), "complete declared stage source inventory")
    for name, value in raw.items():
        need(type(value) is bytes and len(value) < 2 * 1024**2
             and sha256(value).hexdigest() == HASHES[name], "unchanged stage source: " + name)
    trees = {k: ast.parse(v) for k, v in raw.items()}
    position = calls(function(trees["forward"], "fwd_position"))
    forward = calls(function(trees["forward"], "forward"))
    need(tuple(x["call"] for x in position) == POSITION
         and tuple(x["call"] for x in forward) == FORWARD, "exact lexical forward stage ordering")
    need(forward[0]["keywords"] == {"factorize": "False"}
         and next(x for x in forward if x["call"] == "fwd_acceleration")["keywords"] == {"factorize": "True"},
         "deferred forward mass factorization")
    com = function(trees["smooth"], "com_pos")
    # For this frozen function, outputs are complete supplied-body-pose dependents.
    outputs = [ast.unparse(k.value) for n in ast.walk(com) if isinstance(n, ast.Call)
               for k in n.keywords if k.arg == "outputs"]
    need(set(outputs) == {"[d.subtree_com]", "[d.cinert]", "[d.cdof]"},
         "com_pos does not overwrite supplied pose arrays")
    solver_copies = [x["arguments"] for x in calls(function(trees["solver"], "_solve"))
                     if x["call"] == "wp.copy"]
    need(solver_copies == [["d.qacc", "d.qacc_warmstart"], ["d.qacc", "d.qacc_smooth"]],
         "solver copies source warmstart/smooth into destination qacc")
    constraint_outputs = [ast.unparse(k.value) for n in ast.walk(function(trees["constraint"], "make_constraint"))
                          if isinstance(n, ast.Call) for k in n.keywords if k.arg == "outputs"]
    need(any("d.contact.efc_address" in value for value in constraint_outputs),
         "constructed contact EFC address is an output, not a frozen input")
    stages = {}
    for module, names in {
        "forward": ("fwd_position", "fwd_velocity", "fwd_actuation", "fwd_acceleration", "forward"),
        "smooth": ("com_pos", "transmission"), "constraint": ("make_constraint",),
        "solver": ("solve", "_solve"),
    }.items():
        for name in names:
            node = function(trees[module], name)
            stages[module + "." + name] = dict(line=node.lineno, end_line=node.end_lineno,
                lexical_calls=calls(node), lexical_data_attributes=attributes(node, "d"),
                lexical_model_attributes=attributes(node, "m"))
    return dict(protocol=PROTOCOL, source_sha256=dict(HASHES), stages=stages,
                lexical_inventory_is_transitive_execution_proof=False,
                ordinary_forward_recomputes_kinematics=True,
                ordinary_forward_factorizes_at_acceleration=True,
                com_pos_outputs=["subtree_com", "cinert", "cdof"],
                solver_uses_supplied_qacc_warmstart=True,
                constraint_construction_mutates_contact_efc_address=True,
                measured_historical_gpu_collision_pose_complete=False,
                generated_manifold_is_same_manifold_solver_control=False,
                new_collision_calls=0, new_constraint_calls=0, new_solver_calls=0,
                new_integration_steps=0, runtime_imported=False,
                solver_qualified=False, training_authorized=False, physical_motion_authorized=False)


def audit():
    sources = saved.library_sources()  # All69 wheel Python files + three Warp files.
    dist = distribution("mujoco-warp")
    raw = {name: Path(dist.locate_file("mujoco_warp/_src/" + name + ".py")).read_bytes()
           for name in HASHES}
    result = inspect_sources(raw)
    need(all(sources["mujoco_warp/_src/" + k + ".py"]["sha256"] == v for k, v in HASHES.items())
         and saved.library_sources() == sources, "stable complete wheel-bound installed stage sources")
    result["library_sources"] = sources
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args(); host = saved.p.p.base.host
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "explicit CPU-hidden source audit")
    need(re.fullmatch(r"[0-9a-f]{40}", args.source) and Path.cwd().resolve() == Path(__file__).resolve().parents[2]
         and host.read("git", "rev-parse", "HEAD") == args.source
         and host.read("git", "branch", "--show-current") == host.BRANCH
         and not host.read("git", "status", "--porcelain"), "clean exact stage audit source")
    raw = Path(__file__).read_bytes()
    need(raw == host.read("git", "show", args.source + ":src/mjlab_microduck/ada_solver_stage_audit.py", binary=True),
         "committed stage audit bytes")
    need(args.output.is_absolute() and not args.output.exists()
         and args.output.parent.resolve(strict=True) == args.output.parent, "fresh canonical stage audit output")
    result = audit()
    result.update(source=args.source, module_sha256=sha256(raw).hexdigest(),
                  decision="frozen-stage-source-audit-not-execution-or-admission")
    host.write_json(args.output, result)
    print(result["decision"])


if __name__ == "__main__": main()
