#include "core/constant.h"
#include "gtest_base.h"

class ConstantTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_constant(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input) {
    const int rows = static_cast<int>(input.rows());
    const int cols = static_cast<int>(input.cols());
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(rows, cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        output(i, j) = input(i, j);
      }
    }
    return output;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
      Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_output;
    ASSERT_NO_THROW(impl_output = constant<Scalar>(input))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_output;
    ASSERT_NO_THROW(golden_output = golden_reference_constant<Scalar>(input))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_output, golden_output, tolerance, test_description);
  }
};

TEST_F(ConstantTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {2, 3},
      {64, 64},
      {128, 256},
      {1024, 1024},
      {4, 1024},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {1, 1},
      {3, 1},
  };

  const float tolerance = 0.0f;

  for (const auto& [rows, cols] : test_cases) {
    auto input = generate_random_matrix<float>(rows, cols, -1.0f, 1.0f);

    std::string test_description = "Constant " + std::to_string(rows) + "x" +
                            std::to_string(cols);

    test_with_golden(input, tolerance, test_description);
  }
}

