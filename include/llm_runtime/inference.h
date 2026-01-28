// Public inference convenience API.
// Thin wrapper around Model + future execution engine.

#pragma once

#include <string>
#include <unordered_map>

#include "llm_runtime/model.h"
#include "llm_runtime/tensor.h"

namespace llm_on_riscv {

// Simple options placeholder; extend as needed.
struct InferenceOptions {
  // e.g. maximum sequence length, number of threads, etc.
  int dummy{0};
};

// A minimal inference session facade. For now it only stores the model
// and echoes inputs back as outputs, so that example code can compile
// and run without the full graph engine being implemented.
class InferenceSession {
 public:
  InferenceSession() = default;

  explicit InferenceSession(Model model) : model_(std::move(model)) {}

  [[nodiscard]] bool HasModel() const { return model_.IsLoaded(); }

  // Configure the session; currently unused but kept for API stability.
  void SetOptions(const InferenceOptions& options) { options_ = options; }

  // Run inference. For now this is a stub that simply returns the inputs
  // as outputs. It is mainly intended to allow wiring example code; the
  // real implementation will route to the execution engine.
  std::unordered_map<std::string, Tensor> Run(
      const std::unordered_map<std::string, Tensor>& inputs) {
    last_error_.clear();
    if (!model_.IsLoaded()) {
      last_error_ = "Model is not loaded";
      return {};
    }
    return inputs;
  }

  [[nodiscard]] const std::string& LastError() const { return last_error_; }

 private:
  Model model_{};
  InferenceOptions options_{};
  std::string last_error_{};
};

}  // namespace llm_on_riscv

