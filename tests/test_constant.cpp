#include "core/constant.h"
#include "gtest_base.h"

class ConstantTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_constant(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& value) {
    return value;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& value,
      Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result = constant<Scalar>(value))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_constant<Scalar>(value))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_result, golden_result, tolerance, test_description);
  }
};

TEST_F(ConstantTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1}, {2, 3}, {64, 64}, {128, 256}, {1024, 1024},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {4, 1024},
  };

  const float tolerance = 0.0f;

  for (const auto& [rows, cols] : test_cases) {
    auto value = generate_random_matrix<float>(rows, cols, -1.0, 1.0);

    std::string test_name = "Constant " + std::to_string(rows) + "x" +
                            std::to_string(cols);

    test_with_golden(value, tolerance, test_name);
  }
}

