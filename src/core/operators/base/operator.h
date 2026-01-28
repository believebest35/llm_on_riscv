// Operator base class for core runtime operators.

#pragma once

#include <string>
#include <vector>

#include "core/tensor/tensor.h"

namespace llm_on_riscv {
namespace core {
namespace operators {

// Base class for all operators. This is intentionally minimal for now
// and can be extended later (e.g., with attributes, shape inference, etc.).
class Operator {
 public:
  virtual ~Operator() = default;

  // A short, stable name for this operator (e.g. "Add", "MatMul").
  [[nodiscard]] virtual const char* Name() const = 0;

  // Execute the operator on the given inputs and return the outputs.
  // For now we assume a small number of inputs/outputs and do not model
  // attributes explicitly.
  [[nodiscard]] virtual std::vector<Tensor> Compute(
      const std::vector<Tensor>& inputs) = 0;
};

}  // namespace operators
}  // namespace core
}  // namespace llm_on_riscv

