#include "core/simplified_layer_normalization.h"
#include "gtest_base.h"

class SimplifiedLayerNormalizationTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_simplified_layer_normalization(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& scale, Scalar epsilon) {
    const int rows = static_cast<int>(input.rows());
    const int cols = static_cast<int>(input.cols());
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(rows, cols);
    for (int i = 0; i < rows; ++i) {
      Scalar sum_sq = static_cast<Scalar>(0);
      for (int j = 0; j < cols; ++j) {
        const Scalar v = input(i, j);
        sum_sq += v * v;
      }
      const Scalar mean_sq = sum_sq / static_cast<Scalar>(cols);
      const Scalar rms = std::sqrt(mean_sq + epsilon);

      for (int j = 0; j < cols; ++j) {
        output(i, j) = (input(i, j) / rms) * scale(j);
      }
    }
    return output;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& scale, Scalar epsilon,
      Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_output;
    ASSERT_NO_THROW(impl_output =
                        simplified_layer_normalization(input, scale, epsilon))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_output;
    ASSERT_NO_THROW(
        golden_output = golden_reference_simplified_layer_normalization(
            input, scale, epsilon))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_output, golden_output, tolerance, test_description);
  }
};

TEST_F(SimplifiedLayerNormalizationTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1},     {2, 3},      {4, 8},      {64, 128},
      {128, 256}, {256, 1024}, {1024, 128}, {1024, 3072},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {4, 1024},
  };

  const float epsilon = 1e-5f;
  const float tolerance = 0.0f;

  for (const auto& [rows, cols] : test_cases) {
    auto input = generate_random_matrix<float>(rows, cols, -1.0f, 1.0f);
    auto scale = generate_random_vector<float>(cols, 0.5f, 1.5f);

    std::string test_name =
        "SimplifiedLayerNormalization " + std::to_string(rows) + "x" +
        std::to_string(cols);

    test_with_golden(input, scale, epsilon, tolerance, test_name);
  }
}

TEST_F(SimplifiedLayerNormalizationTest, InvalidInputs) {
  auto input = generate_random_matrix<float>(4, 8);
  auto bad_scale = generate_random_vector<float>(7);
  EXPECT_THROW(simplified_layer_normalization(input, bad_scale, 1e-5f),
               std::invalid_argument);
}
