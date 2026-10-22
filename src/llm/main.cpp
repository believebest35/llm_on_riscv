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
  std::string tokens_out_path;
  int max_new = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--tokens" && i + 1 < argc) {
      tokens_text = argv[++i];
    } else if (argument == "--logits" && i + 1 < argc) {
      logits_path = argv[++i];
    } else if (argument == "--max-new" && i + 1 < argc) {
      max_new = std::stoi(argv[++i]);
    } else if (argument == "--tokens-out" && i + 1 < argc) {
      tokens_out_path = argv[++i];
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

    if (max_new > 0) {
      std::vector<int> generated;
      Eigen::MatrixXf step_logits = logits;
      for (int step = 0; step < max_new; ++step) {
        Eigen::Index token = 0;
        step_logits.row(step_logits.rows() - 1).maxCoeff(&token);
        generated.push_back(static_cast<int>(token));
        if (step + 1 < max_new) {
          step_logits = model.decode(static_cast<int>(token));
        }
      }

      std::cout << "generated     :";
      for (int token : generated) {
        std::cout << " " << token;
      }
      std::cout << "\n";

      if (!tokens_out_path.empty()) {
        std::ofstream output(tokens_out_path);
        for (std::size_t i = 0; i < generated.size(); ++i) {
          if (i > 0) {
            output << ",";
          }
          output << generated[i];
        }
        output << "\n";
      }
    }

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
