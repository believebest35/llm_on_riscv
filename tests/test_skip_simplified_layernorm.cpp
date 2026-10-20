#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "core/skip_simplified_layernorm.h"
#include "gtest_base.h"

class SkipSimplifiedLayerNormTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_skip_simplified_layernorm(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& skip,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& gamma, Scalar epsilon) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y(X.rows(), X.cols());
    for (int i = 0; i < X.rows(); ++i) {
      Scalar sum_sq = static_cast<Scalar>(0);
      for (int j = 0; j < X.cols(); ++j) {
        const Scalar v = X(i, j) + skip(i, j);
        sum_sq += v * v;
      }
      const Scalar mean_sq = sum_sq / static_cast<Scalar>(X.cols());
      const Scalar rms = std::sqrt(mean_sq + epsilon);

      for (int j = 0; j < X.cols(); ++j) {
        Y(i, j) = ((X(i, j) + skip(i, j)) / rms) * gamma(j);
      }
    }
    return Y;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& skip,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& gamma, Scalar epsilon,
      Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result = skip_simplified_layer_normalization(
                        X, skip, gamma, epsilon))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_skip_simplified_layernorm(X, skip, gamma, epsilon))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_result, golden_result, tolerance, test_description);
  }
};

TEST_F(SkipSimplifiedLayerNormTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1}, {2, 3}, {4, 8}, {64, 128}, {128, 256}, {256, 512}, {1024, 128},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {4, 1024},
  };

  const float epsilon = 1e-5f;
  const float tolerance = 1e-4f;

  for (const auto& [rows, cols] : test_cases) {
    auto X = generate_random_matrix<float>(rows, cols, -1.0f, 1.0f);
    auto skip = generate_random_matrix<float>(rows, cols, -0.5f, 0.5f);
    auto gamma = generate_random_vector<float>(cols, 0.5f, 1.5f);

    std::string test_name =
        "SkipSimplifiedLayerNorm " + std::to_string(rows) + "x" +
        std::to_string(cols);

    test_with_golden(X, skip, gamma, epsilon, tolerance, test_name);
  }
}


TEST_F(SkipSimplifiedLayerNormTest, InvalidInputs) {
  auto X = generate_random_matrix<float>(4, 8);
  auto gamma = generate_random_vector<float>(8);

  EXPECT_THROW(
      skip_simplified_layer_normalization(
          X, generate_random_matrix<float>(3, 8), gamma, 1e-5f),
      std::invalid_argument);
  EXPECT_THROW(
      skip_simplified_layer_normalization(
          X, generate_random_matrix<float>(4, 8), generate_random_vector<float>(7),
          1e-5f),
      std::invalid_argument);
}
