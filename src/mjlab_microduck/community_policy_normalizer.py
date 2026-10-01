"""Static affine-prefix inspection, never actor inference or parity admission.

Recognizes only an explicit obs -> Sub(mean) -> Div(denominator) -> Gemm
prefix. Constant values describe the exported graph, not its training source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .community_policy_inspection import _read_payload, inspect_policy_payload


TERMS = (("base_ang_vel", 0, 3), ("projected_gravity", 3, 6),
         ("joint_pos", 6, 20), ("joint_vel", 20, 34), ("actions", 34, 48),
         ("command", 48, 51), ("head_command", 51, 55), ("body_command", 55, 61))
OBSERVATION_NAMES = ",".join(name for name, _, _ in TERMS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_affine_prefix(payload: bytes, expected_sha256: str) -> dict:
    """Read hash-checked bytes once; report narrowly recognized graph structure.

    No model/session execution. In particular, nonzero first-layer columns do
    not prove effective downstream command sensitivity or correct locomotion.
    """
    inspected = inspect_policy_payload(payload, expected_sha256)
    require(inspected["interface_classification"] == "feedforward-microduck-api1",
            "exact API1 interface required")
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load_model_from_string(payload)
    graph = model.graph
    metadata = {}
    for pair in model.metadata_props:
        require(pair.key not in metadata, "duplicate metadata key")
        metadata[pair.key] = pair.value
    require(metadata.get("observation_names") == OBSERVATION_NAMES,
            "declared API1 observation-term order required")
    require(not graph.sparse_initializer, "dense prefix initializers required")
    constants = {value.name: value for value in graph.initializer}
    require(not set(constants).intersection(value.name for value in graph.input),
            "overridable initializer input is not a fixed constant")
    require(len(graph.node) >= 3, "affine prefix not recognized")
    sub, div, linear = graph.node[:3]
    require((sub.op_type, div.op_type, linear.op_type) == ("Sub", "Div", "Gemm")
            and all(node.domain in ("", "ai.onnx") for node in (sub, div, linear)),
            "affine prefix not recognized")
    require(len(sub.input) == 2 and len(sub.output) == 1 and sub.input[0] == "obs"
            and len(div.input) == 2 and len(div.output) == 1
            and div.input[0] == sub.output[0]
            and len(linear.input) in (2, 3) and len(linear.output) == 1
            and linear.input[0] == div.output[0]
            and not sub.attribute and not div.attribute,
            "affine prefix wiring not recognized")
    for name, expected_node in (("obs", sub), (sub.output[0], div), (div.output[0], linear)):
        require([node for node in graph.node if name in node.input] == [expected_node]
                and name not in {value.name for value in graph.output},
                "normalizer prefix has a bypass or additional consumer")

    def array(name, dimensions=None):
        require(name in constants, "prefix input is not a fixed dense initializer")
        tensor = constants[name]
        require(tensor.data_type == onnx.TensorProto.FLOAT,
                "float32 prefix constants required")
        values = numpy_helper.to_array(tensor)
        require(np.isfinite(values).all(), "nonfinite prefix constant")
        require(dimensions is None or values.shape == dimensions,
                "prefix constant shape mismatch")
        return values

    mean = array(sub.input[1], (1, 61)).reshape(61)
    denominator = array(div.input[1], (1, 61)).reshape(61)
    require((denominator > 0).all(), "strictly positive affine denominator required")
    attributes = {attr.name: onnx.helper.get_attribute_value(attr) for attr in linear.attribute}
    require(len(attributes) == len(linear.attribute)
            and set(attributes) <= {"alpha", "beta", "transA", "transB"}
            and attributes.get("alpha", 1.0) == 1.0
            and attributes.get("beta", 1.0) == 1.0
            and attributes.get("transA", 0) == 0
            and attributes.get("transB", 0) in (0, 1), "first Gemm convention not recognized")
    weights = array(linear.input[1])
    require(weights.ndim == 2, "first Gemm must have a matrix weight")
    weights = weights.T if attributes.get("transB", 0) == 1 else weights
    require(weights.shape[0] == 61 and weights.shape[1] > 0,
            "first Gemm must consume all 61 input features")
    if len(linear.input) == 3:
        bias = array(linear.input[2])
        require(bias.shape in ((weights.shape[1],), (1, weights.shape[1])),
                "first Gemm bias shape not recognized")
    descriptions = []
    for name, first, last in TERMS:
        descriptions.append(dict(term=name, start=first, stop=last,
            mean_min=float(mean[first:last].min()), mean_max=float(mean[first:last].max()),
            denominator_min=float(denominator[first:last].min()),
            denominator_max=float(denominator[first:last].max()),
            first_layer_nonzero_counts=np.count_nonzero(weights[first:last], axis=1).tolist()))
    commands = [dict(index=index, mean=float(mean[index]), denominator=float(denominator[index]),
                     first_layer_nonzero_count=int(np.count_nonzero(weights[index])))
                for index in range(48, 61)]
    return dict(schema="community-policy-static-affine-prefix-v1", artifact=inspected["artifact"],
        decision="recognized-static-affine-prefix-only", formula="(obs - mean) / denominator",
        constants=dict(mean=sub.input[1], denominator=div.input[1], first_weight=linear.input[1]),
        affine_values=dict(mean=mean.tolist(), denominator=denominator.tolist()),
        hidden_width=weights.shape[1], terms=descriptions, command_features=commands,
        actor_forward_executed=False, policy_inferences=0, simulation_executed=False,
        effective_command_sensitivity_verified=False, normalizer_verified=False,
        training_source_binding_verified=False, behavioral_acceptance=False,
        training_authorized=False, transition_authorized=False, physical_motion_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    try:
        result = inspect_affine_prefix(_read_payload(args.path), args.sha256)
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
