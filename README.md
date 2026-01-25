# llm_on_riscv

## Project goal

Run Qwen-3-0.6B (ONNX) on RISC-V simulators and targets by implementing and optimizing the required operators, validating inference correctness, and measuring performance.

## Current status

- [x] Model operator export and analysis completed (see `docs/01_dev.md`).
- [x] Basic unit-test framework and several example tests exist in `tests/`.
- [x] Implementations present in `src/core/`.

## Quickstart

1. Build the project:

```sh
./build.sh
```

2. Run tests:

```sh
ctest --test-dir build --output-on-failure
```

3. Run on a RISC-V simulator:

- Cross-compile or deploy the runtime to your chosen RISC-V simulator (e.g. Spike, QEMU-riscv). Steps depend on your cross-toolchain and target.

## How to add operators and unit tests

- Add operator implementations and headers in `src/core/`.
- Add unit tests in `tests/` following the existing `tests/test_*.cpp` pattern.
- Update `CMakeLists.txt` to include new sources and run tests in `build/`.

## Roadmap

What works today

- [x] Model operator export and analysis (`docs/01_dev.md`).
- [x] Basic unit test harness and example tests (`tests/`).
- [x] Implemented operators: `MatMul`, `Mul`, `SkipSimplifiedLayerNormalization`.

### Short term — v0.2 (Now)

Goal: implement a minimal set of operators required for model inference and validate single-layer inference on host.

Core checklist

- [ ] Implement and test core ops for single-layer inference: `MatMul`, `Reshape`, `Gather`, `Concat`.
- [ ] Implement attention/embedding ops: `RotaryEmbedding`, `GroupQueryAttention`.
- [ ] Implement normalization and residual ops: `SimplifiedLayerNormalization`, `SkipSimplifiedLayerNormalization`, `ReduceSum`.
- [ ] Implement primitive and shape ops: `Mul`, `Sub`, `Sigmoid`, `Cast`, `Unsqueeze`, `Transpose`, `Constant`, `Shape`.
- [ ] Add unit tests and small correctness examples for each operator (`tests/test_<op>.cpp`).

Operator status

| Operator                         | Area          | Priority | Status | ETA |
| -------------------------------- | ------------- | :------: | ------ | --- |
| MatMul                           | Core          |   High   | ✓      | -   |
| Reshape                          | Core          |   High   | ✗      | -   |
| Gather                           | Embedding     |   High   | ✗      | -   |
| Concat                           | Tensor ops    |  Medium  | ✗      | -   |
| RotaryEmbedding                  | Embedding     |   High   | ✗      | -   |
| GroupQueryAttention              | Attention     |   High   | ✗      | -   |
| SimplifiedLayerNormalization     | Normalization |   High   | ✗      | -   |
| SkipSimplifiedLayerNormalization | Normalization |   High   | ✓      | -   |
| ReduceSum                        | Reduction     |  Medium  | ✗      | -   |
| Transpose                        | Tensor ops    |  Medium  | ✗      | -   |
| Unsqueeze                        | Tensor ops    |   Low    | ✗      | -   |
| Cast                             | Tensor ops    |   Low    | ✗      | -   |
| Shape / Constant                 | Helpers       |   Low    | ✗      | -   |
| Sigmoid                          | Primitives    |   Low    | ✗      | -   |
| Mul                              | Primitives    |   Low    | ✓      | -   |
| Sub                              | Primitives    |   Low    | ✗      | -   |

### Medium term — v0.3

- [ ] Cross-compile the runtime to RISC-V and run on QEMU/Spike.
- [ ] End-to-end correctness tests on RISC-V simulator for a single layer and small batch.

### Performance & optimization — v0.4

- [ ] Profile hotspots on RISC-V (cycle-accurate where possible).
- [ ] Optimize MatMul and attention kernels (vector intrinsics, blocking, memory layout).
- [ ] Add optional multithreading and accelerator hooks.

### Stability & ecosystem — v0.5+

- [ ] CI for host and RISC-V simulator (unit tests + regression).
- [ ] Benchmark suite and example inference scripts (throughput & latency).
- [ ] Packaging & docs: `examples/` and complete tutorials in `docs/`.

## Notes

- This roadmap follows an iterative approach: correctness first, then performance.
- See `docs/01_dev.md` for detailed operator analysis and `scripts/summary_op.py` for profiling.
