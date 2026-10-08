"""Synthetic static-contract refusal tests, never a CUDA or runtime proof."""
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import pytest
from mjlab_microduck import stance_solver_target_binding as contract


def fixture():
    # Readable independent dispatcher carrier; arithmetic uses the pinned template.
    caller = '''
def _update_constraint(m, d, ctx):
    if m.is_sparse:
        d.qfrc_constraint.zero_()
        wp.launch(update_constraint_init_qfrc_constraint_sparse,
                  dim=(d.nworld, d.njmax),
                  inputs=[d.nefc, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind,
                          d.efc.J, d.efc.force, ctx.done], outputs=[d.qfrc_constraint])
    else:
        wp.launch(update_constraint_init_qfrc_constraint_dense, dim=(d.nworld, m.nv),
                  inputs=[d.nefc, d.efc.J, d.efc.force, d.njmax, ctx.done],
                  outputs=[d.qfrc_constraint])
'''
    return (contract.KERNEL + caller).encode()


def test_static_profile():
    profile = contract.dispatch_profile(fixture())
    assert profile["target"] == contract.TARGET
    assert profile["dimensions"] == ["d.nworld", "m.nv"]
    assert profile["inputs"] == ["d.nefc", "d.efc.J", "d.efc.force", "d.njmax", "ctx.done"]
    assert profile["branch"] == "else-of-m.is_sparse"


@pytest.mark.parametrize("old,new", [
    (b"@wp.kernel", b"@wp.kernel(enable_backward=False)"),
    (b"njmax_in: int", b"njmax_in: float"),
    (b"sum_qfrc += efc_J * force", b"sum_qfrc += force * efc_J"),
    (b"range(min(njmax_in, nefc_in[worldid]))", b"range(njmax_in)"),
    (b"if ctx_done_in[worldid]", b"if not ctx_done_in[worldid]"),
    (b"dim=(d.nworld, m.nv)", b"dim=(d.nworld, d.njmax)"),
    (b"d.nefc, d.efc.J, d.efc.force, d.njmax, ctx.done",
     b"d.nefc, d.efc.force, d.efc.J, d.njmax, ctx.done"),
    (b"outputs=[d.qfrc_constraint])", b"outputs=[d.qfrc_smooth])"),
    (b"if m.is_sparse:", b"if not m.is_sparse:"),
    (b"update_constraint_init_qfrc_constraint_dense, dim",
     b"update_constraint_init_qfrc_constraint_sparse, dim"),
    (b"d.qfrc_constraint.zero_()", b"d.qfrc_constraint.fill_(0)"),
])
def test_changed_contract_refused(old, new):
    raw = fixture()
    assert old in raw
    with pytest.raises(ValueError):
        contract.dispatch_profile(raw.replace(old, new))


@pytest.mark.parametrize("extra", [
    contract.KERNEL.encode(),
    b"\nother = update_constraint_init_qfrc_constraint_dense\n",
    b"\ndef _update_constraint(m,d,ctx): pass\n",
])
def test_duplicate_or_extra_reference_refused(extra):
    with pytest.raises(ValueError):
        contract.dispatch_profile(fixture() + extra)


def test_dense_launch_moved_to_sparse_arm_refused():
    raw = fixture().replace(b"    else:\n", b"    if m.is_sparse:\n")
    with pytest.raises(ValueError):
        contract.dispatch_profile(raw)


@pytest.mark.parametrize("raw", [None, "source", b"", b"x" * (contract.MAX_SOURCE_BYTES + 1)])
def test_bad_source_type_size(raw):
    with pytest.raises(ValueError):
        contract.verify_source(raw)


def test_whole_source_required():
    with pytest.raises(ValueError, match="whole frozen solver source hash"):
        contract.verify_source(fixture())


def test_pinned_installed_source_without_import():
    # Frozen environments on both CPU hosts supply these bytes; do not import MuJoCo/Warp.
    root = Path(sys.prefix) / "lib/python3.12/site-packages/mujoco_warp/_src/solver.py"
    raw = root.read_bytes()
    assert sha256(raw).hexdigest() == contract.SOLVER_SHA256
    report = contract.verify_source(raw)
    assert report["dispatch"]["kernel_line"] == 1988
    assert report["dispatch"]["dispatch_line"] == 2197
    assert report["flags"] == contract.FLAGS
    assert not any(report["flags"].values())
    assert report["decision"] == "static-dense-solver-contract-only-not-native"
    assert report["actual_dispatch_observed"] is False
    assert report["binary_binding_observed"] is False
    assert report["instructions_observed"] is False


def test_import_does_not_load_gpu_packages():
    code = "from mjlab_microduck import stance_solver_target_binding; import sys; assert not any(n.split('.')[0] in {'warp','torch','mujoco','mujoco_warp'} for n in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)
