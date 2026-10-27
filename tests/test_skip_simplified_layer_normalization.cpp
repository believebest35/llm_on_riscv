#include "core/skip_simplified_layer_normalization.h"
#include "gtest_base.h"

class SkipSimplifiedLayerNormalizationTest : public GTestBase {
 protected:
  template <typename Scalar>
  std::tuple<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
             Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
  golden_reference_skip_simplified_layer_normalization(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& skip,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& gamma, Scalar epsilon) {
    const int rows = static_cast<int>(input.rows());
    const int cols = static_cast<int>(input.cols());
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(rows, cols);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
        input_skip_bias_sum(rows, cols);
    for (int i = 0; i < rows; ++i) {
      Scalar sum_sq = static_cast<Scalar>(0);
      for (int j = 0; j < cols; ++j) {
        input_skip_bias_sum(i, j) = input(i, j) + skip(i, j);
        sum_sq += input_skip_bias_sum(i, j) * input_skip_bias_sum(i, j);
      }
      const Scalar mean_sq = sum_sq / static_cast<Scalar>(cols);
      const Scalar rms = std::sqrt(mean_sq + epsilon);

      for (int j = 0; j < cols; ++j) {
        output(i, j) = (input_skip_bias_sum(i, j) / rms) * gamma(j);
      }
    }
    return {output, input_skip_bias_sum};
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& skip,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& gamma, Scalar epsilon,
      Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_output;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
        impl_input_skip_bias_sum;
    ASSERT_NO_THROW(
        std::tie(impl_output, impl_input_skip_bias_sum) =
            skip_simplified_layer_normalization<Scalar>(input, skip, gamma,
                                                        epsilon))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_output;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
        golden_input_skip_bias_sum;
    ASSERT_NO_THROW(
        std::tie(golden_output, golden_input_skip_bias_sum) =
            golden_reference_skip_simplified_layer_normalization<Scalar>(
                input, skip, gamma, epsilon))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_output, golden_output, tolerance,
                       test_description + " [output]");
    expect_matrix_near(impl_input_skip_bias_sum, golden_input_skip_bias_sum,
                       tolerance,
                       test_description + " [input_skip_bias_sum]");
  }
};

TEST_F(SkipSimplifiedLayerNormalizationTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1},
      {2, 3},
      {4, 8},
      {64, 128},
      {128, 256},
      {256, 512},
      {1024, 128},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {4, 1024},
  };

  const float epsilon = 1e-5f;
  const float tolerance = 0.0f;

  for (const auto& [rows, cols] : test_cases) {
    auto input = generate_random_matrix<float>(rows, cols, -1.0f, 1.0f);
    auto skip = generate_random_matrix<float>(rows, cols, -0.5f, 0.5f);
    auto gamma = generate_random_vector<float>(cols, 0.5f, 1.5f);

    std::string test_name =
        "SkipSimplifiedLayerNormalization " + std::to_string(rows) + "x" +
        std::to_string(cols);

    test_with_golden(input, skip, gamma, epsilon, tolerance, test_name);
  }
}

TEST_F(SkipSimplifiedLayerNormalizationTest, InvalidInputs) {
  auto input = generate_random_matrix<float>(4, 8);
  auto gamma = generate_random_vector<float>(8);

  EXPECT_THROW(
      skip_simplified_layer_normalization<float>(
          input, generate_random_matrix<float>(3, 8), gamma, 1e-5f),
      std::invalid_argument);
  EXPECT_THROW(
      skip_simplified_layer_normalization<float>(
          input, generate_random_matrix<float>(4, 8),
          generate_random_vector<float>(7), 1e-5f),
      std::invalid_argument);
}
