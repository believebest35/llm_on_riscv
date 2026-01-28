#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "core/operators/base/operator_registry.h"
#include "core/operators/math/add.h"
#include "core/tensor/tensor.h"

namespace {

using llm_on_riscv::core::DataType;
using llm_on_riscv::core::Shape;
using llm_on_riscv::core::Tensor;
using llm_on_riscv::core::operators::Operator;
using llm_on_riscv::core::operators::OperatorRegistry;

Tensor MakeTensorF32(const Shape& shape, float value) {
  Tensor t(DataType::FLOAT32, shape);
  t.Allocate();
  float* data = t.DataAs<float>();
  const auto numel = t.NumElements();
  for (std::int64_t i = 0; i < numel; ++i) {
    data[i] = value;
  }
  return t;
}

}  // namespace

TEST(AddOpTest, BasicAddition) {
  const Shape shape({2, 3});

  Tensor a = MakeTensorF32(shape, 1.5f);
  Tensor b = MakeTensorF32(shape, 2.0f);

  auto op = OperatorRegistry::Instance().Create("Add");
  ASSERT_NE(op, nullptr) << "Add operator should be registered";

  std::vector<Tensor> inputs;
  inputs.emplace_back(std::move(a));
  inputs.emplace_back(std::move(b));

  auto outputs = op->Compute(inputs);
  ASSERT_EQ(outputs.size(), 1u);

  const Tensor& out = outputs[0];
  EXPECT_EQ(out.GetShape(), shape);
  EXPECT_EQ(out.Dtype(), DataType::FLOAT32);

  const float* out_data = out.DataAs<float>();
  const auto numel = out.NumElements();
  for (std::int64_t i = 0; i < numel; ++i) {
    EXPECT_NEAR(out_data[i], 3.5f, 1e-6f);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

