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
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_reference(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& skip,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& weight, Scalar epsilon) {
    if (X.rows() != skip.rows() || X.cols() != skip.cols()) {
      throw std::invalid_argument("dimension mismatch");
    }
    if (X.cols() != weight.size()) {
      throw std::invalid_argument("dimension mismatch");
    }

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
        Y(i, j) = ((X(i, j) + skip(i, j)) / rms) * weight(j);
      }
    }
    return Y;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& skip,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& weight, Scalar epsilon,
      Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result = skip_simplified_layer_normalization(
                        X, skip, weight, epsilon))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference(X, skip, weight, epsilon))
        << "Golden reference failed: " << test_description;

    EXPECT_EQ(impl_result.rows(), golden_result.rows());
    EXPECT_EQ(impl_result.cols(), golden_result.cols());

    Scalar l2_diff = calculate_l2_difference(impl_result, golden_result);
    EXPECT_LE(l2_diff, tolerance) << "L2 diff too large: " << test_description;
  }
};

TEST_F(SkipSimplifiedLayerNormTest, RandomMatricesMatchGolden) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1}, {2, 3}, {4, 8}, {64, 128}, {128, 256}, {256, 512}, {1024, 128},
  };

  const float epsilon = 1e-5f;
  const float tolerance = 1e-4f;

  for (const auto& [rows, cols] : test_cases) {
    auto X = generate_random_matrix<float>(rows, cols, -1.0f, 1.0f);
    auto skip = generate_random_matrix<float>(rows, cols, -0.5f, 0.5f);
    auto w = generate_random_vector<float>(cols, 0.5f, 1.5f);

    std::string test_name =
        "Shape " + std::to_string(rows) + "x" + std::to_string(cols);

    test_with_golden(X, skip, w, epsilon, tolerance, test_name);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
