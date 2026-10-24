import argparse
import onnx
from onnx import shape_inference
from collections import defaultdict


def get_tensor_shape(value_info):
    """
    Extract tensor shape from ValueInfoProto.
    Returns a tuple, e.g. (1, 3, 224, 224),
    or None if shape information is unavailable.
    """
    if not value_info.type.HasField("tensor_type"):
        return None

    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)  # dynamic dimension
        else:
            shape.append("?")
    return tuple(shape)


def get_initializer_shape(initializer):
    """
    Extract shape from TensorProto (initializer).
    Initializers always have static shapes.
    """
    return tuple(initializer.dims)


def build_shape_dict(graph):
    """
    Build a mapping:
        tensor_name -> tensor_shape

    Shape sources:
        graph.initializer (weights / constants)
        graph.input / graph.output / graph.value_info
        Constant node value attributes (override exported value_info)
    """
    shape_dict = {}

    # 1. Initializers
    for init in graph.initializer:
        shape_dict[init.name] = get_initializer_shape(init)

    # 2. Inputs / outputs / intermediate tensors
    for v in list(graph.input) + list(graph.output) + list(graph.value_info):
        shape = get_tensor_shape(v)
        if shape is not None:
            # Do not override initializer shapes
            shape_dict.setdefault(v.name, shape)

    # 3. Constant nodes: exported value_info may report an empty shape
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                shape_dict[node.output[0]] = tuple(attr.t.dims)
                break

    return shape_dict


def analyze_onnx_model(onnx_path):
    """
    Analyze an ONNX model:
    - Print basic model information
    - Group nodes by op_type
    - Count input/output shape patterns per op_type
    """
    # Weights are not needed for shape analysis; skip external data loading.
    model = onnx.load(onnx_path, load_external_data=False)

    # Run shape inference to fill missing / incorrect value_info.
    model = shape_inference.infer_shapes(model)
    graph = model.graph

    opset_versions = [
        (opset.domain or "ai.onnx", opset.version)
        for opset in model.opset_import
    ]

    print("=== Basic Model Information ===")
    print(f"IR version       : {model.ir_version}")
    print(f"Opset versions   : {opset_versions}")
    print(f"Number of nodes  : {len(graph.node)}")
    print(f"Graph inputs     : {len(graph.input)}")
    print(f"Graph outputs    : {len(graph.output)}")
    print(f"Initializers     : {len(graph.initializer)}")
    print()

    shape_dict = build_shape_dict(graph)

    # Statistics:
    # op_type -> {(input_shapes, output_shapes): count}
    stats = defaultdict(lambda: defaultdict(int))

    for node in graph.node:
        op_type = node.op_type

        input_shapes = tuple(
            shape_dict.get(name, None) for name in node.input
        )
        output_shapes = tuple(
            shape_dict.get(name, None) for name in node.output
        )

        stats[op_type][(input_shapes, output_shapes)] += 1

    # Print statistics
    for op_type, shape_map in stats.items():
        print(f"=== OpType: {op_type} ===")
        total = sum(shape_map.values())
        print(f"Total nodes: {total}")

        for (in_shapes, out_shapes), count in shape_map.items():
            print(f"  Count: {count}")
            print(f"    Inputs : {in_shapes}")
            print(f"    Outputs: {out_shapes}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ONNX model operators and input/output shape patterns"
    )
    parser.add_argument(
        "onnx_path",
        type=str,
        help="Path to the ONNX model file"
    )

    args = parser.parse_args()
    analyze_onnx_model(args.onnx_path)


if __name__ == "__main__":
    main()
