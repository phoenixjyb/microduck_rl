"""Pure protocol identity for centered, exact-geometry O1 evidence."""

from copy import deepcopy


O1_PROTOCOL_NAME = "O1-centered-exact-v1"
O1_OBSTACLE_FORWARD_M = 1.15
O1_OBSTACLE_LATERAL_M = 0.0
O1_MIN_COMMAND_SPEED_MPS = 0.25
O1_MAX_COMMAND_SPEED_MPS = 0.50
O1_SENSOR_PARAMS = {
    "range_noise_m": 0.0,
    "bearing_noise_rad": 0.0,
    "width_noise_m": 0.0,
    "height_noise_m": 0.0,
    "closing_rate_noise_mps": 0.0,
    "dropout_probability": 0.0,
}


def o1_evaluation_protocol() -> dict:
    """Return a JSON-compatible copy of the exact O1 environment contract."""
    return {
        "name": O1_PROTOCOL_NAME,
        "obstacle_forward_range_m": [O1_OBSTACLE_FORWARD_M, O1_OBSTACLE_FORWARD_M],
        "obstacle_lateral_range_m": [O1_OBSTACLE_LATERAL_M, O1_OBSTACLE_LATERAL_M],
        "actor_delay_lag_steps": [0, 0],
        "sensor": deepcopy(O1_SENSOR_PARAMS),
        "commanded_speed_range_mps": [
            O1_MIN_COMMAND_SPEED_MPS,
            O1_MAX_COMMAND_SPEED_MPS,
        ],
    }
