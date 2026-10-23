# llm_on_riscv

Run Qwen3-0.6B (ONNX) on RISC-V by implementing its operators with Eigen,
validating them against unit tests, and optimizing for RVV later.

## Build

```sh
./build.sh          # Release
./build.sh -d       # Debug
```

## Test

```sh
ctest --test-dir build --output-on-failure
```

Each operator has `tests/test_<op>.cpp` with `ValidInputs` (compared against a
reference implementation) and, where the kernel validates inputs,
`InvalidInputs` (expected to throw).

## Layout

```
src/core/            operator kernels (header-only, Eigen)
src/core/operators/  Tensor-based operator + registry layer
tests/               gtest suites, auto-discovered by CMake
docs/                design notes and ONNX operator profile
scripts/             model inspection utilities
```

See `docs/01_dev.md` for the operator statistics of the target model.
