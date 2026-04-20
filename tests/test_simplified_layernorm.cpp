#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "core/simplified_layernorm.h"
#include "gtest_base.h"

class SimplifiedLayerNormTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_simplified_layernorm(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& scale, Scalar epsilon) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y(X.rows(), X.cols());
    for (int i = 0; i < X.rows(); ++i) {
      Scalar sum_sq = static_cast<Scalar>(0);
      for (int j = 0; j < X.cols(); ++j) {
        const Scalar v = X(i, j);
        sum_sq += v * v;
      }
      const Scalar mean_sq = sum_sq / static_cast<Scalar>(X.cols());
      const Scalar rms = std::sqrt(mean_sq + epsilon);

      for (int j = 0; j < X.cols(); ++j) {
        Y(i, j) = (X(i, j) / rms) * scale(j);
      }
    }
    return Y;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& scale, Scalar epsilon,
      Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result =
                        simplified_layer_normalization(X, scale, epsilon))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_simplified_layernorm(X, scale, epsilon))
        << "Golden reference failed: " << test_description;

    EXPECT_EQ(impl_result.rows(), golden_result.rows())
        << "Row count mismatch: " << test_description;
    EXPECT_EQ(impl_result.cols(), golden_result.cols())
        << "Column count mismatch: " << test_description;

    Scalar l2_diff = calculate_l2_difference(impl_result, golden_result);
    EXPECT_LE(l2_diff, tolerance)
        << "L2 difference exceeds tolerance: " << test_description;

    std::cout << test_description << "\nMax absolute difference: "
              << calculate_max_abs_difference(impl_result, golden_result)
              << "\nRMSE: " << calculate_rmse(impl_result, golden_result)
              << "\nL2 difference: " << l2_diff << std::endl;
  }
};

TEST_F(SimplifiedLayerNormTest, RandomMatrices) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1},     {2, 3},      {4, 8},      {64, 128},
      {128, 256}, {256, 1024}, {1024, 128}, {1024, 3072},
  };

  const float epsilon = 1e-5f;
  const float tolerance = 1e-4f;

  for (const auto& [rows, cols] : test_cases) {
    auto X = generate_random_matrix<float>(rows, cols, -1.0f, 1.0f);
    auto w = generate_random_vector<float>(cols, 0.5f, 1.5f);

    std::string test_name =
        "SimplifiedLayerNorm " + std::to_string(rows) + "x" +
        std::to_string(cols);

    test_with_golden(X, w, epsilon, tolerance, test_name);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
