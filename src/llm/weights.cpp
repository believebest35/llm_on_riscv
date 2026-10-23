#include "llm/weights.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "llm/qwen3_config.h"

namespace llm_on_riscv {
namespace llm {
namespace {

std::string layer_path(const std::string& directory, int layer) {
  char name[32];
  std::snprintf(name, sizeof(name), "/layer_%02d.bin", layer);
  return directory + name;
}

std::ifstream open_weight_file(const std::string& path, std::size_t expected_bytes) {
  const auto actual_bytes = std::filesystem::file_size(path);
  if (actual_bytes != expected_bytes) {
    throw std::runtime_error(
        "weight file size mismatch: " + path + " expected " +
        std::to_string(expected_bytes) + " bytes, got " +
        std::to_string(actual_bytes));
  }
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("cannot open weight file: " + path);
  }
  return file;
}

void read_floats(std::ifstream& file, float* data, std::size_t count) {
  file.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
  if (!file) {
    throw std::runtime_error("unexpected end of weight file");
  }
}

Eigen::MatrixXf read_matrix(
    const std::string& path, int rows, int cols) {
  Eigen::MatrixXf matrix(rows, cols);
  auto file = open_weight_file(path, static_cast<std::size_t>(rows) * cols * sizeof(float));
  read_floats(file, matrix.data(), static_cast<std::size_t>(rows) * cols);
  return matrix;
}

Eigen::VectorXf read_vector(const std::string& path, int size) {
  Eigen::VectorXf vector(size);
  auto file = open_weight_file(path, static_cast<std::size_t>(size) * sizeof(float));
  read_floats(file, vector.data(), static_cast<std::size_t>(size));
  return vector;
}

std::size_t layer_file_bytes() {
  return (static_cast<std::size_t>(Qwen3Config::kHidden) +
          static_cast<std::size_t>(Qwen3Config::kHidden) * Qwen3Config::kQueryDim +
          static_cast<std::size_t>(Qwen3Config::kHidden) * Qwen3Config::kKvDim * 2 +
          Qwen3Config::kHeadDim * 2 +
          static_cast<std::size_t>(Qwen3Config::kQueryDim) * Qwen3Config::kHidden +
          static_cast<std::size_t>(Qwen3Config::kHidden) +
          static_cast<std::size_t>(Qwen3Config::kHidden) * Qwen3Config::kIntermediate * 2 +
          static_cast<std::size_t>(Qwen3Config::kIntermediate) * Qwen3Config::kHidden) *
         sizeof(float);
}

LayerWeights read_layer(const std::string& directory, int index) {
  auto file = open_weight_file(layer_path(directory, index), layer_file_bytes());
  LayerWeights layer;

  layer.input_norm.resize(Qwen3Config::kHidden);
  read_floats(file, layer.input_norm.data(), Qwen3Config::kHidden);

  layer.q_proj.resize(Qwen3Config::kHidden, Qwen3Config::kQueryDim);
  read_floats(file, layer.q_proj.data(), layer.q_proj.size());

  layer.k_proj.resize(Qwen3Config::kHidden, Qwen3Config::kKvDim);
  read_floats(file, layer.k_proj.data(), layer.k_proj.size());

  layer.v_proj.resize(Qwen3Config::kHidden, Qwen3Config::kKvDim);
  read_floats(file, layer.v_proj.data(), layer.v_proj.size());

  layer.q_norm.resize(Qwen3Config::kHeadDim);
  read_floats(file, layer.q_norm.data(), Qwen3Config::kHeadDim);

  layer.k_norm.resize(Qwen3Config::kHeadDim);
  read_floats(file, layer.k_norm.data(), Qwen3Config::kHeadDim);

  layer.o_proj.resize(Qwen3Config::kQueryDim, Qwen3Config::kHidden);
  read_floats(file, layer.o_proj.data(), layer.o_proj.size());

  layer.post_norm.resize(Qwen3Config::kHidden);
  read_floats(file, layer.post_norm.data(), Qwen3Config::kHidden);

  layer.gate_proj.resize(Qwen3Config::kHidden, Qwen3Config::kIntermediate);
  read_floats(file, layer.gate_proj.data(), layer.gate_proj.size());

  layer.up_proj.resize(Qwen3Config::kHidden, Qwen3Config::kIntermediate);
  read_floats(file, layer.up_proj.data(), layer.up_proj.size());

  layer.down_proj.resize(Qwen3Config::kIntermediate, Qwen3Config::kHidden);
  read_floats(file, layer.down_proj.data(), layer.down_proj.size());

  return layer;
}

}  // namespace

Qwen3Weights load_qwen3_weights(const std::string& directory) {
  Qwen3Weights weights;
  weights.embedding = read_matrix(
      directory + "/embedding.bin", Qwen3Config::kVocab, Qwen3Config::kHidden);
  weights.cos_cache = read_matrix(
      directory + "/cos_cache.bin", Qwen3Config::kMaxPositions, Qwen3Config::kRopeDim);
  weights.sin_cache = read_matrix(
      directory + "/sin_cache.bin", Qwen3Config::kMaxPositions, Qwen3Config::kRopeDim);
  weights.final_norm = read_vector(directory + "/final_norm.bin", Qwen3Config::kHidden);

  weights.layers.reserve(Qwen3Config::kLayers);
  for (int layer = 0; layer < Qwen3Config::kLayers; ++layer) {
    weights.layers.push_back(read_layer(directory, layer));
  }
  return weights;
}

}  // namespace llm
}  // namespace llm_on_riscv
