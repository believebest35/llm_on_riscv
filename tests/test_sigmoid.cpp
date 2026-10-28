#include "core/sigmoid.h"
#include "gtest_base.h"

class SigmoidTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_sigmoid(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input) {
    const int rows = static_cast<int>(input.rows());
    const int cols = static_cast<int>(input.cols());
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(rows, cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        const Scalar x = input(i, j);
        output(i, j) =
            static_cast<Scalar>(1) / (static_cast<Scalar>(1) + std::exp(-x));
      }
    }
    return output;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
      Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_output;
    ASSERT_NO_THROW(impl_output = sigmoid<Scalar>(input))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_output;
    ASSERT_NO_THROW(golden_output = golden_reference_sigmoid(input))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_output, golden_output, tolerance, test_description);
  }
};

TEST_F(SigmoidTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1},
      {1, 1024},
      {1024, 1},
      {2, 3},
      {64, 64},
      {256, 256},
      {1024, 1024},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {4, 3072},
  };

  const float tolerance = 1e-5f;

  for (const auto& [rows, cols] : test_cases) {
    auto input = generate_random_matrix<float>(rows, cols, -5.0f, 5.0f);

    std::string test_description = "Sigmoid " + std::to_string(rows) + "x" +
                            std::to_string(cols);

    test_with_golden(input, tolerance, test_description);
  }
}
