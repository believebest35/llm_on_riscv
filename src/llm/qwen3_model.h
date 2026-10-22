#pragma once

#include <Eigen/Dense>
#include <vector>

#include "llm/weights.h"

namespace llm_on_riscv {
namespace llm {

// Minimal batch-size-one Qwen3 prefill path.
class Qwen3Model {
 public:
  explicit Qwen3Model(const Qwen3Weights& weights);

  // Process a prompt from position 0 and return logits (S x vocab).
  Eigen::MatrixXf prefill(const std::vector<int>& tokens);

 private:
  Eigen::MatrixXf forward(const std::vector<int>& tokens);
  Eigen::MatrixXf embed(const std::vector<int>& tokens) const;

  const Qwen3Weights& weights_;
};

}  // namespace llm
}  // namespace llm_on_riscv
