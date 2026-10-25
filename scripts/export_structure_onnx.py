#!/usr/bin/env python3
"""Export a weight-free ONNX graph for Netron.

Initializers are removed and exposed as graph inputs with their original
dtype/shape, so the file keeps the graph structure but no weight data.
Use --layer N to keep decoder layer N plus the shared input preparation it
depends on; tensors crossing the layer boundary become graph inputs/outputs,
so the layer can be read on its own.
Add --runtime to keep only the operator types the C++ runtime executes; the
shared input preparation is then dropped and exposed as graph inputs.
Use --json for ONNX JSON.
"""

import argparse
import os
import re

import onnx
from google.protobuf import json_format
from onnx import TensorProto, helper

LAYER = re.compile(r"/layers\.(\d+)/")

RUNTIME_ALIASES = {
    "/model/pos_ids_reformat/Reshape/output_0": "position_ids",
    "/model/attn_mask_reformat/attn_mask_subgraph/Sub/Cast/output_0": "seqlens_k",
    "/model/attn_mask_reformat/attn_mask_subgraph/Gather/Cast/output_0": "total_sequence_length",
}

RUNTIME_INPUTS = {
    "seqlens_k": (TensorProto.INT32, ["batch_size", 1]),
    "total_sequence_length": (TensorProto.INT32, []),
}


def last_decoder_layer(graph):
    layers = set()
    for node in graph.node:
        match = LAYER.search(node.name)
        if match and ("/attn/" in node.name or "/mlp/" in node.name):
            layers.add(int(match.group(1)))
    return max(layers) if layers else -1


def clone_value_info(value_info):
    result = onnx.ValueInfoProto()
    result.CopyFrom(value_info)
    return result


def weight_input(initializer):
    return helper.make_tensor_value_info(
        initializer.name, initializer.data_type, list(initializer.dims)
    )


def clear_graph(graph, nodes, inputs, outputs, value_infos, initializers=()):
    graph.ClearField("node")
    graph.node.extend(nodes)
    graph.ClearField("initializer")
    graph.initializer.extend(initializers)
    graph.ClearField("sparse_initializer")
    graph.ClearField("input")
    graph.input.extend(inputs)
    graph.ClearField("output")
    graph.output.extend(outputs)
    graph.ClearField("value_info")
    graph.value_info.extend(value_infos)


def export_weightless(model):
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    infos = {
        value_info.name: value_info
        for value_info in (*graph.input, *graph.output, *graph.value_info)
    }
    input_names = {value_info.name for value_info in graph.input}

    produced = {name for node in graph.node for name in node.output if name}
    consumed = {name for node in graph.node for name in node.input if name}
    referenced = produced | consumed

    inputs = [
        clone_value_info(value_info)
        for value_info in graph.input
        if value_info.name in consumed
    ]
    inputs += [
        weight_input(init) for name, init in initializers.items() if name in consumed
    ]
    for name in sorted(consumed - produced - set(initializers) - input_names):
        if name in infos:
            inputs.append(clone_value_info(infos[name]))
        else:
            inputs.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            )

    outputs = [
        clone_value_info(value_info)
        for value_info in graph.output
        if value_info.name in referenced
    ]
    value_infos = [
        clone_value_info(value_info)
        for value_info in graph.value_info
        if value_info.name in referenced
    ]

    clear_graph(graph, list(graph.node), inputs, outputs, value_infos)
    return model


def export_layer(model, layer, runtime=False):
    graph = model.graph
    prefix = f"/model/layers.{layer}/"
    if not any(prefix in node.name for node in graph.node):
        raise SystemExit(f"Layer {layer} not found")

    nodes = [node for node in graph.node if prefix in node.name]
    if not runtime:
        producer = {}
        for index, node in enumerate(graph.node):
            for name in node.output:
                if name:
                    producer[name] = index

        selected = {
            index for index, node in enumerate(graph.node) if prefix in node.name
        }
        pending = list(selected)
        while pending:
            node = graph.node[pending.pop()]
            for name in node.input:
                source = producer.get(name)
                if source is None or source in selected:
                    continue
                if LAYER.search(graph.node[source].name):
                    continue
                selected.add(source)
                pending.append(source)

        nodes = [node for index, node in enumerate(graph.node) if index in selected]

    shape_initializers = []
    shape_names = set()
    if runtime:
        constants = {}
        for node in graph.node:
            if node.op_type != "Constant" or not node.output:
                continue
            value = next(
                (
                    attribute.t
                    for attribute in node.attribute
                    if attribute.name == "value"
                    and attribute.type == onnx.AttributeProto.TENSOR
                ),
                None,
            )
            if value is not None:
                constants[node.output[0]] = value
        for node in nodes:
            for name in node.input:
                if name in constants and name not in shape_names:
                    tensor = onnx.TensorProto()
                    tensor.CopyFrom(constants[name])
                    tensor.name = name
                    shape_initializers.append(tensor)
                    shape_names.add(name)
        for node in nodes:
            for index, name in enumerate(node.input):
                if name in RUNTIME_ALIASES:
                    node.input[index] = RUNTIME_ALIASES[name]

    initializers = {init.name: init for init in graph.initializer}
    input_infos = {value_info.name: value_info for value_info in graph.input}
    output_infos = {value_info.name: value_info for value_info in graph.output}
    value_infos = {value_info.name: value_info for value_info in graph.value_info}

    produced = {name for node in nodes for name in node.output if name}
    consumed = {name for node in nodes for name in node.input if name}

    def boundary_input(name):
        if name in initializers:
            return weight_input(initializers[name])
        if name in input_infos:
            return clone_value_info(input_infos[name])
        if name in RUNTIME_INPUTS:
            data_type, shape = RUNTIME_INPUTS[name]
            return helper.make_tensor_value_info(name, data_type, shape)
        if name in value_infos:
            return clone_value_info(value_infos[name])
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, None)

    def boundary_output(name):
        if name in output_infos:
            return clone_value_info(output_infos[name])
        if name in value_infos:
            return clone_value_info(value_infos[name])
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, None)

    inputs = [
        boundary_input(name)
        for name in sorted(consumed - produced)
        if name not in shape_names
    ]
    outputs = [boundary_output(name) for name in sorted(produced - consumed)]
    declared = {value_info.name for value_info in inputs + outputs}
    kept_value_infos = [
        clone_value_info(value_info)
        for name, value_info in value_infos.items()
        if name in produced | consumed and name not in declared
    ]

    clear_graph(graph, nodes, inputs, outputs, kept_value_infos, shape_initializers)
    if runtime:
        graph.name = f"decoder_layer_{layer}_runtime"
        graph.doc_string = (
            f"Decoder layer {layer} as executed by the runtime: only the "
            "layer's own operator types are kept; the shared input "
            "preparation is exposed as graph inputs."
        )
    else:
        graph.name = f"decoder_layer_{layer}"
        graph.doc_string = (
            f"Decoder layer {layer} with the shared input preparation it uses. "
            "Weights are exposed as graph inputs; tensors crossing the layer "
            "boundary are graph inputs/outputs."
        )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Export a weight-free ONNX graph for Netron"
    )
    parser.add_argument("model_path")
    parser.add_argument("-o", "--output")
    parser.add_argument(
        "--layer",
        type=int,
        help="Export decoder layer N, with boundary tensors as graph inputs/outputs",
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Keep only the operator types the C++ runtime executes (requires --layer)",
    )
    parser.add_argument("--json", action="store_true", help="Write ONNX JSON")
    args = parser.parse_args()

    if args.runtime and args.layer is None:
        raise SystemExit("--runtime requires --layer")

    if args.output is None:
        stem, _ = os.path.splitext(args.model_path)
        suffix = "" if args.layer is None else f".layer{args.layer}"
        suffix += ".runtime" if args.runtime else ""
        args.output = stem + suffix + (".structure.onnx.json" if args.json else ".structure.onnx")

    model = onnx.load(args.model_path, load_external_data=False)
    last_layer = last_decoder_layer(model.graph)
    if args.layer is None:
        model = export_weightless(model)
    else:
        if not 0 <= args.layer <= last_layer:
            raise SystemExit(f"Layer {args.layer} not found; model has layers 0..{last_layer}")
        model = export_layer(model, args.layer, runtime=args.runtime)

    if args.json:
        content = json_format.MessageToJson(model, indent=2, ensure_ascii=False)
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(content + "\n")
    else:
        onnx.save(model, args.output)

    print(f"Wrote {args.output}")
    print(f"  nodes   : {len(model.graph.node)}")
    print(f"  inputs  : {len(model.graph.input)}")
    print(f"  outputs : {len(model.graph.output)}")
    print(f"  size    : {os.path.getsize(args.output)} bytes")


if __name__ == "__main__":
    main()
