#!/usr/bin/env python3
"""Run the minimal C++ Qwen3 runtime with the official Python tokenizer."""

import argparse
import os
import subprocess
import tempfile

from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Prompt -> Qwen3 C++ runtime")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new", type=int, default=8)
    parser.add_argument("--weights", default="models/qwen3_0.6b")
    parser.add_argument("--tokenizer", default="Qwen3-0.6B-ONNX/tokenizer.json")
    parser.add_argument(
        "--binary", default="build/src/llm/qwen3_run", help="C++ runner path"
    )
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    prompt_ids = tokenizer.encode(args.prompt).ids

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
        tokens_path = temp.name

    try:
        subprocess.run(
            [
                args.binary,
                args.weights,
                "--tokens",
                ",".join(str(token) for token in prompt_ids),
                "--max-new",
                str(args.max_new),
                "--tokens-out",
                tokens_path,
            ],
            check=True,
        )
        with open(tokens_path, encoding="utf-8") as tokens_file:
            text = tokens_file.read().strip()
            generated_ids = [int(token) for token in text.split(",") if token]
    finally:
        os.unlink(tokens_path)

    print(f"prompt ids    : {prompt_ids}")
    print(f"generated ids : {generated_ids}")
    print(f"output        : {tokenizer.decode(generated_ids, skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
