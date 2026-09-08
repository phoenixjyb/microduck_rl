"""B1-N observation/action reference only; no registered env or training launch."""

import numpy as np

from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = 'football-b1n-nominal-stance-v1'
JOINTS = ('left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
          'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll', 'right_hip_yaw',
          'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle')
LEG_IDS = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)
ACTOR_DIM = 44
CRITIC_DIM = 50
CONTROL_DT = .02
MAX_CORRECTION = .2
MAX_CORRECTION_RATE = 1.  # rad/s; at most .02 rad per policy tick


def vector(value, size, label):
    value = np.asarray(value, dtype=float)
    require(value.shape == (size,) and np.isfinite(value).all(), 'finite '+label)
    return value


def actor_observation(gravity_body, angular_velocity_body, joint_offset, joint_velocity,
                      previous_limited_correction):
    """Fixed scales, no running normalizer or privileged translation/contact data."""
    gravity = vector(gravity_body, 3, 'unit projected gravity')
    require(abs(np.linalg.norm(gravity)-1.) < 1e-4, 'unit projected gravity')
    previous = vector(previous_limited_correction, 10, 'previous limited correction')
    require(np.max(np.abs(previous)) <= MAX_CORRECTION, 'bounded previous correction')
    return np.concatenate((gravity,
        vector(angular_velocity_body, 3, 'angular velocity')/5.,
        vector(joint_offset, 14, 'joint offset')/.5,
        vector(joint_velocity, 14, 'joint velocity')/10., previous/MAX_CORRECTION))


def critic_observation(actor, root_linear_velocity, root_height, foot_normal_forces):
    actor = vector(actor, ACTOR_DIM, 'actor observation')
    height = vector([root_height], 1, 'root height')
    support = vector(foot_normal_forces, 2, 'foot normal forces')
    require(np.min(support) >= 0, 'nonnegative support forces')
    return np.concatenate((actor, vector(root_linear_velocity, 3, 'root velocity'),
                           (height-.12)/.1, support/10.))


def limited_targets(action, previous_correction, nominal, joint_ranges):
    """Rate- and position-limited leg targets; four head/neck targets stay nominal.

    The next actor receives returned realized corrections, NOT the raw action.
    Apply the fixed physics-delay queue afterward. This reference has no queue
    state and performs no physics; actual manager/normalizer wiring needs tests.
    """
    action = vector(action, 10, 'action')
    previous = vector(previous_correction, 10, 'previous correction')
    nominal = vector(nominal, 14, 'nominal pose')
    ranges = np.asarray(joint_ranges, dtype=float)
    require(ranges.shape == (14, 2) and np.isfinite(ranges).all()
            and (ranges[:, 1] > ranges[:, 0]).all(), 'finite ordered joint ranges')
    center = ranges.mean(1); half = .9*(ranges[:, 1]-ranges[:, 0])/2
    lo, hi = center-half, center+half
    require(((nominal >= lo) & (nominal <= hi)).all(), 'nominal inside soft range')
    ids = np.asarray(LEG_IDS)
    lower = np.maximum(-MAX_CORRECTION, lo[ids]-nominal[ids])
    upper = np.minimum(MAX_CORRECTION, hi[ids]-nominal[ids])
    require(((previous >= lower) & (previous <= upper)).all(), 'valid previous correction')
    desired = np.clip(MAX_CORRECTION*np.clip(action, -1., 1.), lower, upper)
    increment = np.clip(desired-previous, -MAX_CORRECTION_RATE*CONTROL_DT,
                        MAX_CORRECTION_RATE*CONTROL_DT)
    realized = np.clip(previous+increment, lower, upper)
    target = nominal.copy(); target[ids] += realized
    return target, realized
