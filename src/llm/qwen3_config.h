#pragma once

namespace llm_on_riscv {
namespace llm {

// Fixed Qwen3-0.6B configuration (batch size 1).
struct Qwen3Config {
  static constexpr int kLayers = 28;
  static constexpr int kHidden = 1024;
  static constexpr int kQueryHeads = 16;
  static constexpr int kKvHeads = 8;
  static constexpr int kHeadDim = 128;
  static constexpr int kQueryDim = kQueryHeads * kHeadDim;
  static constexpr int kKvDim = kKvHeads * kHeadDim;
  static constexpr int kIntermediate = 3072;
  static constexpr int kVocab = 151936;
  static constexpr int kMaxPositions = 40960;
  static constexpr int kRopeDim = kHeadDim / 2;
  static constexpr float kEpsilon = 1e-6f;
};

}  // namespace llm
}  // namespace llm_on_riscv
