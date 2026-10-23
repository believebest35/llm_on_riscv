#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace llm_on_riscv {
namespace llm {

struct LayerWeights {
  Eigen::VectorXf input_norm;
  Eigen::MatrixXf q_proj;
  Eigen::MatrixXf k_proj;
  Eigen::MatrixXf v_proj;
  Eigen::VectorXf q_norm;
  Eigen::VectorXf k_norm;
  Eigen::MatrixXf o_proj;
  Eigen::VectorXf post_norm;
  Eigen::MatrixXf gate_proj;
  Eigen::MatrixXf up_proj;
  Eigen::MatrixXf down_proj;
};

struct Qwen3Weights {
  Eigen::MatrixXf embedding;
  Eigen::MatrixXf cos_cache;
  Eigen::MatrixXf sin_cache;
  std::vector<LayerWeights> layers;
  Eigen::VectorXf final_norm;
};

// Load the raw float32 files written by scripts/export_qwen3_weights.py.
Qwen3Weights load_qwen3_weights(const std::string& directory);

}  // namespace llm
}  // namespace llm_on_riscv
