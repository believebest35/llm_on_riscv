#pragma once

#include <Eigen/Dense>
#include <vector>

#include "llm/weights.h"

namespace llm_on_riscv {
namespace llm {

// Minimal batch-size-one Qwen3 runner. It keeps one KV cache per layer and
// exposes prefill/decode separately so the forward path stays easy to follow.
class Qwen3Model {
 public:
  explicit Qwen3Model(const Qwen3Weights& weights);

  // Process a full prompt from position 0 and return logits (S x vocab).
  Eigen::MatrixXf prefill(const std::vector<int>& tokens);

  // Process one token after the current cache and return logits (1 x vocab).
  Eigen::MatrixXf decode(int token);

  void reset();
  int past_length() const { return past_length_; }

 private:
  struct LayerCache {
    Eigen::MatrixXf key;
    Eigen::MatrixXf value;
  };

  Eigen::MatrixXf forward(const std::vector<int>& tokens, bool reset_cache);
  Eigen::MatrixXf embed(const std::vector<int>& tokens) const;

  const Qwen3Weights& weights_;
  std::vector<LayerCache> caches_;
  int past_length_ = 0;
};

}  // namespace llm
}  // namespace llm_on_riscv
