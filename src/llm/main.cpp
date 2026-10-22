#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "llm/qwen3_model.h"

namespace {

std::vector<int> parse_tokens(const std::string& text) {
  std::vector<int> tokens;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    tokens.push_back(std::stoi(item));
  }
  return tokens;
}

}  // namespace

int main(int argc, char** argv) {
  std::string directory = "models/qwen3_0.6b";
  std::string tokens_text = "785,6722,315,9625,374";
  std::string logits_path;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--tokens" && i + 1 < argc) {
      tokens_text = argv[++i];
    } else if (argument == "--logits" && i + 1 < argc) {
      logits_path = argv[++i];
    } else {
      directory = argument;
    }
  }

  try {
    const auto weights = llm_on_riscv::llm::load_qwen3_weights(directory);
    const auto tokens = parse_tokens(tokens_text);
    llm_on_riscv::llm::Qwen3Model model(weights);

    const Eigen::MatrixXf logits = model.prefill(tokens);
    std::cout << "prefill tokens: " << tokens.size() << "\n";
    std::cout << "logits shape  : " << logits.rows() << " x " << logits.cols()
              << "\n";

    Eigen::Index next = 0;
    logits.row(logits.rows() - 1).maxCoeff(&next);
    std::cout << "next token    : " << next << "\n";

    if (!logits_path.empty()) {
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          row_major = logits;
      std::ofstream output(logits_path, std::ios::binary);
      output.write(
          reinterpret_cast<const char*>(row_major.data()),
          static_cast<std::streamsize>(row_major.size() * sizeof(float)));
      std::cout << "wrote logits  : " << logits_path << "\n";
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
