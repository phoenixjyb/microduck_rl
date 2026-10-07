"""Ascending dense DOF-friction row kernel for source-only CPU comparison.

This is a standalone kernel component, not a runtime hook or simulator
admission. Callers must reject sparse models before launching it.
"""

import warp as wp

from mujoco_warp._src import constraint as frozen_constraint
from mujoco_warp._src.types import ConstraintType, vec5

wp.set_module_options({"enable_backward": False})


@wp.kernel(enable_backward=False)
def ascending_friction_dof(
    # Model:
    nv: int,
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    dof_solref: wp.array2d[wp.vec2],
    dof_solimp: wp.array2d[vec5],
    dof_frictionloss: wp.array2d[float],
    dof_invweight0: wp.array2d[float],
    is_sparse: bool,
    # Data in:
    qvel_in: wp.array2d[float],
    njmax_in: int,
    njmax_nnz_in: int,
    # Data out:
    nf_out: wp.array[int],
    nefc_out: wp.array[int],
    efc_type_out: wp.array2d[int],
    efc_id_out: wp.array2d[int],
    efc_J_rownnz_out: wp.array2d[int],
    efc_J_rowadr_out: wp.array2d[int],
    efc_J_colind_out: wp.array3d[int],
    efc_J_out: wp.array3d[float],
    efc_pos_out: wp.array2d[float],
    efc_margin_out: wp.array2d[float],
    efc_D_out: wp.array2d[float],
    efc_vel_out: wp.array2d[float],
    efc_aref_out: wp.array2d[float],
    efc_frictionloss_out: wp.array2d[float],
    # Out:
    efc_nnz_out: wp.array[int],
):
    worldid, slot = wp.tid()
    if slot != 0 or is_sparse:
        return

    dof_frictionloss_id = worldid % dof_frictionloss.shape[0]
    for dofid in range(nv):
        if dof_frictionloss[dof_frictionloss_id, dofid] <= 0.0:
            continue

        wp.atomic_add(nf_out, worldid, 1)
        efcid = wp.atomic_add(nefc_out, worldid, 1)
        if efcid >= njmax_in:
            continue

        for index in range(nv):
            efc_J_out[worldid, efcid, index] = 0.0
        efc_J_out[worldid, efcid, dofid] = 1.0

        Jqvel = qvel_in[worldid, dofid]
        dof_invweight0_id = worldid % dof_invweight0.shape[0]
        dof_solref_id = worldid % dof_solref.shape[0]
        dof_solimp_id = worldid % dof_solimp.shape[0]
        frozen_constraint._efc_row(
            opt_disableflags,
            worldid,
            opt_timestep[worldid % opt_timestep.shape[0]],
            efcid,
            0.0,
            0.0,
            dof_invweight0[dof_invweight0_id, dofid],
            dof_solref[dof_solref_id, dofid],
            dof_solimp[dof_solimp_id, dofid],
            0.0,
            Jqvel,
            dof_frictionloss[dof_frictionloss_id, dofid],
            ConstraintType.FRICTION_DOF,
            dofid,
            efc_type_out,
            efc_id_out,
            efc_pos_out,
            efc_margin_out,
            efc_D_out,
            efc_vel_out,
            efc_aref_out,
            efc_frictionloss_out,
        )
