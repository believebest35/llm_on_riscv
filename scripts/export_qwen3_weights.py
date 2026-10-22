#!/usr/bin/env python3
"""Export Qwen3-0.6B ONNX weights to raw float32 files for the C++ runtime.

Weights are written in column-major order so the C++ loader can read them
directly into Eigen matrices. Each decoder layer is one file with a fixed
tensor order; see manifest.txt.
"""

import argparse
import os

import numpy as np
import onnx
from onnx import numpy_helper

NUM_LAYERS = 28
HIDDEN = 1024
QUERY_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
QUERY_DIM = QUERY_HEADS * HEAD_DIM
KV_DIM = KV_HEADS * HEAD_DIM
INTERMEDIATE = 3072
VOCAB = 151936
MAX_POS = 40960

LAYER_TENSORS = [
    ("input_norm", (HIDDEN,), "input_layernorm.weight"),
    ("q_proj", (HIDDEN, QUERY_DIM), "attn.q_proj.MatMul.weight"),
    ("k_proj", (HIDDEN, KV_DIM), "attn.k_proj.MatMul.weight"),
    ("v_proj", (HIDDEN, KV_DIM), "attn.v_proj.MatMul.weight"),
    ("q_norm", (HEAD_DIM,), "attn.q_norm.layernorm.weight"),
    ("k_norm", (HEAD_DIM,), "attn.k_norm.layernorm.weight"),
    ("o_proj", (QUERY_DIM, HIDDEN), "attn.o_proj.MatMul.weight"),
    ("post_norm", (HIDDEN,), "post_attention_layernorm.weight"),
    ("gate_proj", (HIDDEN, INTERMEDIATE), "mlp.gate_proj.MatMul.weight"),
    ("up_proj", (HIDDEN, INTERMEDIATE), "mlp.up_proj.MatMul.weight"),
    ("down_proj", (INTERMEDIATE, HIDDEN), "mlp.down_proj.MatMul.weight"),
]


def read_weight(initializers, name):
    if name not in initializers:
        raise KeyError(f"missing initializer: {name}")
    return numpy_helper.to_array(initializers[name]).astype(np.float32)


def write_tensor(output_file, tensor, expected_shape):
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"shape mismatch: got {tuple(tensor.shape)}, expected {expected_shape}"
        )
    output_file.write(np.asfortranarray(tensor).tobytes(order="F"))


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen3-0.6B ONNX weights to raw float32 files"
    )
    parser.add_argument(
        "model_path", help="Path to the FP32 ONNX model"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="models/qwen3_0.6b",
        help="Output directory (default: models/qwen3_0.6b)",
    )
    args = parser.parse_args()

    model = onnx.load(args.model_path)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    os.makedirs(args.output_dir, exist_ok=True)

    manifest = [
        "qwen3-0.6b float32 column-major",
        f"vocab {VOCAB} hidden {HIDDEN} layers {NUM_LAYERS}",
        "",
    ]

    def write_file(filename, entries):
        path = os.path.join(args.output_dir, filename)
        with open(path, "wb") as output_file:
            for name, shape, onnx_name in entries:
                tensor = read_weight(initializers, onnx_name)
                write_tensor(output_file, tensor, shape)
                manifest.append(f"{filename}: {name} {shape}")
        return path

    files = [
        write_file("embedding.bin", [("embedding", (VOCAB, HIDDEN), "model.embed_tokens.weight")]),
        write_file("cos_cache.bin", [("cos_cache", (MAX_POS, HEAD_DIM // 2), "cos_cache")]),
        write_file("sin_cache.bin", [("sin_cache", (MAX_POS, HEAD_DIM // 2), "sin_cache")]),
        write_file("final_norm.bin", [("final_norm", (HIDDEN,), "model.layers.28.final_norm_layernorm.weight")]),
    ]
    manifest.append("")

    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}."
        entries = [
            (name, shape, prefix + suffix)
            for name, shape, suffix in LAYER_TENSORS
        ]
        files.append(write_file(f"layer_{layer:02d}.bin", entries))
        manifest.append("")

    manifest_path = os.path.join(args.output_dir, "manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        manifest_file.write("\n".join(manifest) + "\n")

    total = sum(os.path.getsize(path) for path in files)
    print(f"Wrote {len(files)} weight files to {args.output_dir}")
    print(f"  manifest : {manifest_path}")
    print(f"  total    : {total / (1024 ** 3):.2f} GiB")


if __name__ == "__main__":
    main()
