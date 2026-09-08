"""Necessary point-force equilibrium test, not a contact or motor solver.

Relax unilateral/friction/motor constraints. Failure rules out this frozen pose
under the declared point-contact model; success cannot establish feasible stance.
"""

import numpy as np

from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = 'football-b0-relaxed-point-wrench-v1'
LENGTH_SCALE_M = .11
RESIDUAL_TOLERANCE_N = 1e-6


def wrench_matrix(point, center):
    """World force -> world force and moment about center."""
    x, y, z = np.asarray(point, dtype=float) - np.asarray(center, dtype=float)
    return np.vstack((np.eye(3), [[0, -z, y], [z, 0, -x], [-y, x, 0]]))


def relaxed_equilibrium(foot_points, robot_com, robot_mass, *, ball_center=None,
                        ball_mass=None, floor_point=None):
    """Fixed ball: robot balance only. Free ball: coupled equal/opposite loads.

    Unbounded signed point forces, zero contact couples. Moments divided by the
    fixed 0.11 m length scale for the least-squares objective, then reported in
    physical units. This is intentionally more permissive than real contacts.
    """
    points = np.asarray(foot_points, dtype=float)
    center = np.asarray(robot_com, dtype=float)
    require(points.ndim == 2 and points.shape[1] == 3 and len(points) > 0,
            'one or more 3D foot contact points')
    require(center.shape == (3,) and np.isfinite(center).all()
            and np.isfinite(points).all(), 'finite points and robot COM')
    require(np.isfinite(robot_mass) and robot_mass > 0, 'positive robot mass')
    free = ball_center is not None
    require(free == (ball_mass is not None) == (floor_point is not None),
            'all or no free-ball fields')
    count = len(points)
    matrix = np.zeros((12 if free else 6, 3 * (count + int(free))))
    for i, point in enumerate(points):
        matrix[:6, i*3:i*3+3] = wrench_matrix(point, center)
    target = np.zeros(len(matrix)); target[2] = robot_mass * 9.81
    if free:
        ball_center = np.asarray(ball_center, dtype=float)
        floor_point = np.asarray(floor_point, dtype=float)
        require(ball_center.shape == floor_point.shape == (3,)
                and np.isfinite(ball_center).all() and np.isfinite(floor_point).all()
                and np.isfinite(ball_mass) and ball_mass > 0, 'finite free-ball plant')
        for i, point in enumerate(points):
            matrix[6:, i*3:i*3+3] = -wrench_matrix(point, ball_center)
        matrix[6:, -3:] = wrench_matrix(floor_point, ball_center)
        target[8] = ball_mass * 9.81
    scale = np.tile([1, 1, 1, 1/LENGTH_SCALE_M, 1/LENGTH_SCALE_M,
                     1/LENGTH_SCALE_M], 2 if free else 1)
    forces, _, rank, singular = np.linalg.lstsq(matrix * scale[:, None], target * scale, rcond=1e-12)
    residual = matrix @ forces - target
    maximum = float(np.max(np.abs(residual * scale)))
    result = dict(protocol=PROTOCOL, support='free' if free else 'fixed',
        force_convention='world force on robot at each foot; equal/opposite on free ball',
        least_squares_foot_forces_world_n=forces[:3*count].reshape(count, 3).tolist(),
        least_squares_floor_force_on_ball_world_n=forces[-3:].tolist() if free else None,
        robot_force_residual_n=residual[:3].tolist(), robot_moment_residual_nm=residual[3:6].tolist(),
        ball_force_residual_n=residual[6:9].tolist() if free else None,
        ball_moment_residual_nm=residual[9:12].tolist() if free else None,
        scaled_residual_max_n=maximum, residual_tolerance_n=RESIDUAL_TOLERANCE_N,
        moment_length_scale_m=LENGTH_SCALE_M, matrix_rank=int(rank),
        singular_values=singular.tolist(),
        relaxed_equilibrium_consistent=maximum <= RESIDUAL_TOLERANCE_N,
        unilateral_constraints_enforced=False, friction_constraints_enforced=False,
        contact_couples_allowed=False, motor_load_verified=False,
        stance_feasibility_verified=False, policy_acceptance=False,
        physical_motion_authorized=False, physics_steps=0)
    canonical(result); return result


def probe_fixture(*, support='free'):
    from mjlab_microduck.football_contact_fixture import vertical_placement_probe, FEET
    model, data, geometry = vertical_placement_probe(support=support)
    require(geometry['geometry_within_joint_ranges'] and not geometry['forbidden_contact_candidates'],
            'valid geometry candidate')
    ball = model.geom('b0_ball_surface').id; floor = model.geom('b0_floor').id
    foot_ids = {model.geom(name).id for name in FEET}
    foot_points = []; floor_points = []; contacts = []; seen_feet = set()
    for c in data.contact:
        pair = set(map(int, c.geom))
        require(c.dim == 3, 'point-force-only condim 3')
        require(c.dist < c.includemargin, 'contact within force generation margin')
        contacts.append(dict(names=[model.geom(int(g)).name for g in c.geom],
            position_world_m=c.pos.tolist(), dim=int(c.dim), friction=c.friction.tolist(),
            distance_m=float(c.dist), include_margin_m=float(c.includemargin)))
        if ball in pair and pair & foot_ids:
            foot_points.append(c.pos.copy()); seen_feet.update(pair & foot_ids)
        elif pair == {ball, floor}: floor_points.append(c.pos.copy())
        else: raise ValueError('unexpected contact in wrench probe')
    # Fixed ball and floor belong to the world and have no mutual contact.
    require(len(foot_points) == 2 and seen_feet == foot_ids
            and len(floor_points) == int(support == 'free'),
            'declared fixed/free contact topology')
    kwargs = dict(ball_center=geometry['ball_center_world_m'], ball_mass=geometry['ball_mass_kg'],
                  floor_point=floor_points[0]) if support == 'free' else {}
    report = relaxed_equilibrium(foot_points, geometry['robot_com_world_m'], geometry['robot_mass_kg'], **kwargs)
    report.update(contact_candidates=contacts, geometry=geometry,
                  mujoco_force_solver_executed=False,
                  interpretation='necessary condition only for this frozen point-contact pose; not learned balance')
    canonical(report); return report
