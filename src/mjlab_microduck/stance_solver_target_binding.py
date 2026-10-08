"""Static pinned solver-source contract, not loaded binary or launch evidence.

No solver import, CUDA initialization, compilation, subprocess or file mutation.
The eventual native collector must separately bind the actual kernel dispatch.
"""

import ast
from hashlib import sha256

PROTOCOL = "microduck-dense-solver-static-target-oct8-v1"
SOLVER_SHA256 = "bba0c67182ade84f5375d6a066048e111edd3371b33d46a6f1246349f22bb30a"
TARGET = "update_constraint_init_qfrc_constraint_dense"
MODULE = "mujoco_warp._src.solver"
MAX_SOURCE_BYTES = 256 * 1024
FLAGS = dict(native_qualified=False, full_window_qualified=False,
             runtime_cause_proven=False, training_authorized=False,
             physical_acceptance=False)

KERNEL = """
@wp.kernel
def update_constraint_init_qfrc_constraint_dense(
    nefc_in: wp.array[int], efc_J_in: wp.array3d[float],
    efc_force_in: wp.array2d[float], njmax_in: int,
    ctx_done_in: wp.array[bool], qfrc_constraint_out: wp.array2d[float],
):
    worldid, dofid = wp.tid()
    if ctx_done_in[worldid]:
        return
    sum_qfrc = float(0.0)
    for efcid in range(min(njmax_in, nefc_in[worldid])):
        efc_J = efc_J_in[worldid, efcid, dofid]
        force = efc_force_in[worldid, efcid]
        sum_qfrc += efc_J * force
    qfrc_constraint_out[worldid, dofid] = sum_qfrc
"""
BRANCH = """
if m.is_sparse:
    d.qfrc_constraint.zero_()
    wp.launch(
        update_constraint_init_qfrc_constraint_sparse,
        dim=(d.nworld, d.njmax),
        inputs=[d.nefc, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind,
                d.efc.J, d.efc.force, ctx.done],
        outputs=[d.qfrc_constraint],
    )
else:
    wp.launch(
        update_constraint_init_qfrc_constraint_dense,
        dim=(d.nworld, m.nv),
        inputs=[d.nefc, d.efc.J, d.efc.force, d.njmax, ctx.done],
        outputs=[d.qfrc_constraint],
    )
"""


def need(value, message):
    if not value:
        raise ValueError(message)


def shape(node):
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def dispatch_profile(raw):
    """Low-level structural check only; does not authenticate installed bytes."""
    need(type(raw) is bytes and 0 < len(raw) <= MAX_SOURCE_BYTES,
         "bounded literal solver source bytes")
    tree = ast.parse(raw.decode("utf-8"))
    definitions = [node for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name == TARGET]
    need(len(definitions) == 1 and definitions[0] in tree.body,
         "one top-level literal dense target definition")
    kernel = definitions[0]
    need(shape(kernel) == shape(ast.parse(KERNEL).body[0]),
         "unchanged dense kernel decorator, signature and arithmetic")
    callers = [node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == "_update_constraint"]
    need(len(callers) == 1, "one top-level frozen dispatch function")
    expected = ast.parse(BRANCH).body[0]
    branches = [node for node in callers[0].body
                if isinstance(node, ast.If) and shape(node.test) == shape(expected.test)]
    need(len(branches) == 1 and shape(branches[0]) == shape(expected),
         "exact sparse/dense branch, dimensions, inputs and outputs")
    refs = [node for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id == TARGET]
    launch = branches[0].orelse[0].value
    need(len(refs) == 1 and refs[0] is launch.args[0],
         "one literal dense target reference at the declared dispatch")
    return dict(module=MODULE, target=TARGET, caller="_update_constraint",
                kernel_line=kernel.lineno, dispatch_line=launch.lineno,
                kernel_ast_sha256=sha256(shape(kernel).encode()).hexdigest(),
                branch_ast_sha256=sha256(shape(branches[0]).encode()).hexdigest(),
                branch="else-of-m.is_sparse", dimensions=["d.nworld", "m.nv"],
                inputs=["d.nefc", "d.efc.J", "d.efc.force", "d.njmax", "ctx.done"],
                outputs=["d.qfrc_constraint"])


def verify_source(raw):
    """Require the whole frozen 3.8.1 solver bytes before returning a contract."""
    need(type(raw) is bytes and 0 < len(raw) <= MAX_SOURCE_BYTES,
         "bounded literal solver source bytes")
    need(sha256(raw).hexdigest() == SOLVER_SHA256, "whole frozen solver source hash")
    profile = dispatch_profile(raw)
    return dict(protocol=PROTOCOL, source=dict(bytes=len(raw), sha256=SOLVER_SHA256),
                dispatch=profile, flags=dict(FLAGS),
                decision="static-dense-solver-contract-only-not-native",
                actual_dispatch_observed=False, binary_binding_observed=False,
                instructions_observed=False)
