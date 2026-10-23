#include "llm/qwen3_model.h"

#include <tuple>
#include <vector>

#include "llm/qwen3_config.h"

#include "core/group_query_attention.h"
#include "core/matmul.h"
#include "core/mul.h"
#include "core/reshape.h"
#include "core/rotary_embedding.h"
#include "core/sigmoid.h"
#include "core/simplified_layernorm.h"
#include "core/skip_simplified_layernorm.h"

namespace llm_on_riscv {
namespace llm {

Qwen3Model::Qwen3Model(const Qwen3Weights& weights)
    : weights_(weights), caches_(Qwen3Config::kLayers) {
  for (auto& cache : caches_) {
    cache.key.resize(0, Qwen3Config::kHeadDim);
    cache.value.resize(0, Qwen3Config::kHeadDim);
  }
}

void Qwen3Model::reset() {
  for (auto& cache : caches_) {
    cache.key.resize(0, Qwen3Config::kHeadDim);
    cache.value.resize(0, Qwen3Config::kHeadDim);
  }
  past_length_ = 0;
}

Eigen::MatrixXf Qwen3Model::embed(const std::vector<int>& tokens) const {
  Eigen::MatrixXf result(tokens.size(), Qwen3Config::kHidden);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    result.row(i) = weights_.embedding.row(tokens[i]);
  }
  return result;
}

Eigen::MatrixXf Qwen3Model::prefill(const std::vector<int>& tokens) {
  reset();
  return forward(tokens, true);
}

Eigen::MatrixXf Qwen3Model::decode(int token) {
  return forward({token}, false);
}

Eigen::MatrixXf Qwen3Model::forward(
    const std::vector<int>& tokens, bool reset_cache) {
  const int sequence = static_cast<int>(tokens.size());
  const int position_start = reset_cache ? 0 : past_length_;
  const int total_sequence = position_start + sequence;

  Eigen::MatrixXf residual = embed(tokens);
  Eigen::MatrixXf mlp_down;
  Eigen::MatrixXf attn_input;
  Eigen::MatrixXf mlp_input;

  for (int layer_index = 0; layer_index < Qwen3Config::kLayers; ++layer_index) {
    const auto& layer = weights_.layers[layer_index];

    // Layer 0 normalizes the embedding directly; later layers add the previous
    // MLP output to the residual first, matching the ONNX SkipLN chain.
    if (layer_index == 0) {
      attn_input = simplified_layer_normalization(
          residual, layer.input_norm, Qwen3Config::kEpsilon);
    } else {
      std::tie(attn_input, residual) = skip_simplified_layer_normalization(
          residual, mlp_down, layer.input_norm, Qwen3Config::kEpsilon);
    }

    Eigen::MatrixXf query = matrix_multiply(attn_input, layer.q_proj);
    Eigen::MatrixXf key = matrix_multiply(attn_input, layer.k_proj);
    Eigen::MatrixXf value = matrix_multiply(attn_input, layer.v_proj);

    // Per-head RMSNorm on q/k, then back to the flat hidden layout.
    query = reshape(query, sequence * Qwen3Config::kQueryHeads, Qwen3Config::kHeadDim);
    query = simplified_layer_normalization(
        query, layer.q_norm, Qwen3Config::kEpsilon);
    query = reshape(query, sequence, Qwen3Config::kQueryDim);

    key = reshape(key, sequence * Qwen3Config::kKvHeads, Qwen3Config::kHeadDim);
    key = simplified_layer_normalization(
        key, layer.k_norm, Qwen3Config::kEpsilon);
    key = reshape(key, sequence, Qwen3Config::kKvDim);

    query = rotary_embedding(
        query, position_start, weights_.cos_cache, weights_.sin_cache,
        Qwen3Config::kQueryHeads);
    key = rotary_embedding(
        key, position_start, weights_.cos_cache, weights_.sin_cache,
        Qwen3Config::kKvHeads);

    Eigen::MatrixXf attention;
    Eigen::MatrixXf present_key;
    Eigen::MatrixXf present_value;
    std::tie(attention, present_key, present_value) = group_query_attention(
        query, key, value, caches_[layer_index].key, caches_[layer_index].value,
        total_sequence - 1, total_sequence);
    caches_[layer_index].key = present_key;
    caches_[layer_index].value = present_value;

    Eigen::MatrixXf attention_output =
        matrix_multiply(attention, layer.o_proj);
    std::tie(mlp_input, residual) = skip_simplified_layer_normalization(
        residual, attention_output, layer.post_norm, Qwen3Config::kEpsilon);

    Eigen::MatrixXf gate = matrix_multiply(mlp_input, layer.gate_proj);
    Eigen::MatrixXf up = matrix_multiply(mlp_input, layer.up_proj);
    Eigen::MatrixXf silu = matrix_multiply_elementwise(gate, sigmoid(gate));
    mlp_down = matrix_multiply(matrix_multiply_elementwise(silu, up), layer.down_proj);
  }

  Eigen::MatrixXf final_output;
  std::tie(final_output, std::ignore) = skip_simplified_layer_normalization(
      residual, mlp_down, weights_.final_norm, Qwen3Config::kEpsilon);

  past_length_ = total_sequence;
  return final_output * weights_.embedding.transpose();
}

}  // namespace llm
}  // namespace llm_on_riscv
