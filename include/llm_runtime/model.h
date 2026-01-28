// Public lightweight model facade.
// This is intentionally minimal for now and will be extended
// when the ONNX graph loader / executor are implemented.

#pragma once

#include <string>

namespace llm_on_riscv {

class Model {
 public:
  Model() = default;

  // Load a model from an ONNX file. For now this is a stub that only
  // records the path; later it can be wired to a real loader.
  bool LoadFromOnnx(const std::string& onnx_path) {
    last_error_.clear();
    onnx_path_ = onnx_path;
    loaded_ = true;
    return true;
  }

  [[nodiscard]] bool IsLoaded() const { return loaded_; }

  [[nodiscard]] const std::string& OnnxPath() const { return onnx_path_; }

  [[nodiscard]] const std::string& LastError() const { return last_error_; }

 private:
  bool loaded_{false};
  std::string onnx_path_;
  std::string last_error_;
};

}  // namespace llm_on_riscv

