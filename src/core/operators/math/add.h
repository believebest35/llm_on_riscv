// Example Add operator implementation using the core Tensor.

#pragma once

#include <stdexcept>
#include <utility>
#include <vector>

#include "core/operators/base/operator.h"
#include "core/operators/base/operator_registry.h"

namespace llm_on_riscv {
namespace core {
namespace operators {

class AddOp : public Operator {
 public:
  [[nodiscard]] const char* Name() const override { return "Add"; }

  [[nodiscard]] std::vector<Tensor> Compute(
      const std::vector<Tensor>& inputs) override {
    if (inputs.size() != 2) {
      throw std::invalid_argument("AddOp expects exactly 2 input tensors");
    }

    const Tensor& a = inputs[0];
    const Tensor& b = inputs[1];

    if (a.Dtype() != b.Dtype()) {
      throw std::invalid_argument("AddOp: input dtypes must match");
    }
    if (a.GetShape() != b.GetShape()) {
      throw std::invalid_argument("AddOp: input shapes must match");
    }

    if (a.Dtype() != DataType::FLOAT32) {
      throw std::invalid_argument("AddOp currently only supports FLOAT32");
    }

    Tensor out(a.Dtype(), a.GetShape());
    out.Allocate();

    const auto numel = a.NumElements();
    float* out_data = out.DataAs<float>();
    const float* a_data = a.DataAs<float>();
    const float* b_data = b.DataAs<float>();

    for (std::int64_t i = 0; i < numel; ++i) {
      out_data[i] = a_data[i] + b_data[i];
    }

    std::vector<Tensor> outputs;
    outputs.emplace_back(std::move(out));
    return outputs;
  }
};

// Register this operator under the name "Add".
REGISTER_OPERATOR("Add", AddOp);

}  // namespace operators
}  // namespace core
}  // namespace llm_on_riscv

