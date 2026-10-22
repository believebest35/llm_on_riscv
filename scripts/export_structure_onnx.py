#!/usr/bin/env python3
"""Export a weight-free ONNX graph for Netron.

Initializers are removed and exposed as graph inputs with their original
dtype/shape, so the file keeps the graph structure but no weight data.
Use --layer N for a single decoder layer and --json for ONNX JSON.
"""

import argparse
import os
import re

import onnx
from google.protobuf import json_format
from onnx import TensorProto, helper

LAYER = re.compile(r"/layers\.(\d+)/")


def layer_of(node):
    for text in (node.name, *node.output):
        match = LAYER.search(text)
        if match:
            return int(match.group(1))
    return None


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


def export_weightless(model, layer=None):
    graph = model.graph
    last_layer = last_decoder_layer(graph)
    initializers = {init.name: init for init in graph.initializer}
    infos = {
        value_info.name: value_info
        for value_info in (*graph.input, *graph.output, *graph.value_info)
    }

    nodes = []
    for node in graph.node:
        index = layer_of(node)
        if layer is None or index is None or index > last_layer or index == layer:
            nodes.append(node)

    produced = {name for node in nodes for name in node.output if name}
    consumed = {name for node in nodes for name in node.input if name}
    referenced = produced | consumed

    inputs = [
        clone_value_info(value_info)
        for value_info in graph.input
        if value_info.name in consumed
    ]
    inputs += [
        helper.make_tensor_value_info(init.name, init.data_type, list(init.dims))
        for name, init in initializers.items()
        if name in consumed
    ]
    for name in sorted(consumed - produced - set(initializers) - {v.name for v in graph.input}):
        if name in infos:
            value_info = clone_value_info(infos[name])
        else:
            value_info = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        value_info.name = name
        inputs.append(value_info)

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

    graph.ClearField("node")
    graph.node.extend(nodes)
    graph.ClearField("initializer")
    graph.ClearField("sparse_initializer")
    graph.ClearField("input")
    graph.input.extend(inputs)
    graph.ClearField("output")
    graph.output.extend(outputs)
    graph.ClearField("value_info")
    graph.value_info.extend(value_infos)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Export a weight-free ONNX graph for Netron"
    )
    parser.add_argument("model_path")
    parser.add_argument("-o", "--output")
    parser.add_argument("--layer", type=int, help="Keep only decoder layer N")
    parser.add_argument("--json", action="store_true", help="Write ONNX JSON")
    args = parser.parse_args()

    if args.output is None:
        stem, _ = os.path.splitext(args.model_path)
        suffix = "" if args.layer is None else f".layer{args.layer}"
        args.output = stem + suffix + (".structure.onnx.json" if args.json else ".structure.onnx")

    model = onnx.load(args.model_path, load_external_data=False)
    last_layer = last_decoder_layer(model.graph)
    if args.layer is not None and not 0 <= args.layer <= last_layer:
        raise SystemExit(f"Layer {args.layer} not found; model has layers 0..{last_layer}")

    model = export_weightless(model, args.layer)

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
