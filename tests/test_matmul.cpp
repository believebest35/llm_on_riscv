#include <gtest/gtest.h>

#include <string>

#include "core/matmul.h"
#include "gtest_base.h"

class MatMulTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_reference_matmul(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
    const int m = A.rows();
    const int n = A.cols();
    const int p = B.cols();

    if (n != B.rows()) {
      throw std::invalid_argument("Matrix dimensions incompatible");
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> C(m, p);
    C.setZero();

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < p; ++j) {
        Scalar sum = static_cast<Scalar>(0);
        for (int k = 0; k < n; ++k) {
          sum += A(i, k) * B(k, j);
        }
        C(i, j) = sum;
      }
    }

    return C;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B,
      Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result = matrix_multiply(A, B))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_matmul(A, B))
        << "Golden reference failed: " << test_description;

    EXPECT_EQ(impl_result.rows(), A.rows())
        << "Result rows incorrect: " << test_description;
    EXPECT_EQ(impl_result.cols(), B.cols())
        << "Result cols incorrect: " << test_description;
    EXPECT_EQ(impl_result.rows(), golden_result.rows())
        << "Row count mismatch with golden: " << test_description;
    EXPECT_EQ(impl_result.cols(), golden_result.cols())
        << "Column count mismatch with golden: " << test_description;

    Scalar l2_diff = calculate_l2_difference(impl_result, golden_result);
    EXPECT_LE(l2_diff, tolerance)
        << "L2 difference exceeds tolerance: " << test_description;

    std::cout << test_description << "\nMax absolute difference: "
              << calculate_max_abs_difference(impl_result, golden_result)
              << "\nRMSE: " << calculate_rmse(impl_result, golden_result)
              << "\nL2 difference: " << l2_diff << std::endl;
  }
};

TEST_F(MatMulTest, RandomMatrices) {
  const std::vector<std::tuple<int, int, int>> test_cases = {
      {1, 1, 1},    {1, 1024, 1},    {1024, 1, 2048},    {2, 3, 4},
      {64, 64, 64}, {256, 256, 256}, {1024, 1024, 2048},
  };

  const float tolerance = 1e-2;

  for (const auto& [m, n, p] : test_cases) {
    auto A = generate_random_matrix<float>(m, n, -1.0, 1.0);
    auto B = generate_random_matrix<float>(n, p, -1.0, 1.0);

    std::string test_name = "Random matrix " + std::to_string(m) + "x" +
                            std::to_string(n) + " * " + std::to_string(n) +
                            "x" + std::to_string(p);

    test_with_golden(A, B, tolerance, test_name);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}