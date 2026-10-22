#include <exception>
#include <iostream>
#include <string>

#include "llm/weights.h"

int main(int argc, char** argv) {
  const std::string directory =
      argc > 1 ? argv[1] : "models/qwen3_0.6b";
  try {
    const auto weights = llm_on_riscv::llm::load_qwen3_weights(directory);
    std::cout << "weights loaded from " << directory << "\n";
    std::cout << "  embedding   : " << weights.embedding.rows() << " x "
              << weights.embedding.cols() << "\n";
    std::cout << "  layers      : " << weights.layers.size() << "\n";
    std::cout << "  q_proj      : " << weights.layers.front().q_proj.rows()
              << " x " << weights.layers.front().q_proj.cols() << "\n";
    std::cout << "  gate_proj   : " << weights.layers.front().gate_proj.rows()
              << " x " << weights.layers.front().gate_proj.cols() << "\n";
    std::cout << "  final_norm  : " << weights.final_norm.size() << "\n";
    std::cout << "  cos_cache   : " << weights.cos_cache.rows() << " x "
              << weights.cos_cache.cols() << "\n";
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
