#!/usr/bin/env python3
"""Export Qwen3 prefill/decode golden data with ONNX Runtime.

Saves the prompt token ids, prefill logits, per-step logits and the greedy
token sequence. Logits are float32 row-major [sequence, vocab].
"""

import argparse
import json
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

NUM_LAYERS = 28
KV_HEADS = 8
HEAD_DIM = 128


def run(session, feed, output_names):
    return dict(zip(output_names, session.run(None, feed)))


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen3 ORT golden logits and greedy tokens"
    )
    parser.add_argument(
        "model_path", help="Path to the FP32 ONNX model"
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen3-0.6B-ONNX/tokenizer.json",
        help="Path to tokenizer.json",
    )
    parser.add_argument(
        "--prompt", default="The capital of France is", help="Prompt text"
    )
    parser.add_argument("--max-new", type=int, default=8)
    parser.add_argument(
        "-o",
        "--output-dir",
        default="models/qwen3_0.6b/golden",
        help="Output directory",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    prompt_ids = tokenizer.encode(args.prompt).ids
    input_ids = np.array([prompt_ids], dtype=np.int64)
    sequence = input_ids.shape[1]

    session = ort.InferenceSession(
        args.model_path, providers=["CPUExecutionProvider"]
    )
    output_names = [output.name for output in session.get_outputs()]

    feed = {
        "input_ids": input_ids,
        "attention_mask": np.ones((1, sequence), dtype=np.int64),
        "position_ids": np.arange(sequence, dtype=np.int64).reshape(1, sequence),
    }
    for layer in range(NUM_LAYERS):
        feed[f"past_key_values.{layer}.key"] = np.zeros(
            (1, KV_HEADS, 0, HEAD_DIM), dtype=np.float32
        )
        feed[f"past_key_values.{layer}.value"] = np.zeros(
            (1, KV_HEADS, 0, HEAD_DIM), dtype=np.float32
        )

    result = run(session, feed, output_names)
    logits = result["logits"].astype(np.float32)

    logits_files = []
    prefill_path = os.path.join(args.output_dir, "logits_step_00.bin")
    logits.tofile(prefill_path)
    logits_files.append(os.path.basename(prefill_path))

    generated_ids = []
    total = sequence
    for step in range(args.max_new):
        next_id = int(np.argmax(logits[0, -1]))
        generated_ids.append(next_id)
        if step + 1 == args.max_new:
            break

        feed = {
            "input_ids": np.array([[next_id]], dtype=np.int64),
            "attention_mask": np.ones((1, total + 1), dtype=np.int64),
            "position_ids": np.array([[total]], dtype=np.int64),
        }
        for layer in range(NUM_LAYERS):
            feed[f"past_key_values.{layer}.key"] = result[f"present.{layer}.key"]
            feed[f"past_key_values.{layer}.value"] = result[f"present.{layer}.value"]

        result = run(session, feed, output_names)
        logits = result["logits"].astype(np.float32)
        total += 1

        path = os.path.join(args.output_dir, f"logits_step_{step + 1:02d}.bin")
        logits.tofile(path)
        logits_files.append(os.path.basename(path))

    meta = {
        "prompt": args.prompt,
        "prompt_ids": prompt_ids,
        "generated_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_ids),
        "prefill_sequence": sequence,
        "vocab": 151936,
        "logits_order": "row-major",
        "logits_files": logits_files,
        "max_new": args.max_new,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as meta_file:
        json.dump(meta, meta_file, indent=2, ensure_ascii=False)

    print(f"Wrote golden data to {args.output_dir}")
    print(f"  prompt ids     : {prompt_ids}")
    print(f"  generated ids  : {generated_ids}")
    print(f"  generated text : {meta['generated_text']!r}")
    print(f"  logits files   : {len(logits_files)}")


if __name__ == "__main__":
    main()
