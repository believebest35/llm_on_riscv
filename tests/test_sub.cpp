#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "core/sub.h"
#include "gtest_base.h"

class SubTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_sub(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
    const int m = std::max(A.rows(), B.rows());
    const int n = std::max(A.cols(), B.cols());
    if ((A.rows() != m && A.rows() != 1) ||
        (A.cols() != n && A.cols() != 1) ||
        (B.rows() != m && B.rows() != 1) ||
        (B.cols() != n && B.cols() != 1)) {
      throw std::invalid_argument("Matrix dimensions incompatible");
    }
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> C(m, n);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        const int ai = (A.rows() == 1) ? 0 : i;
        const int aj = (A.cols() == 1) ? 0 : j;
        const int bi = (B.rows() == 1) ? 0 : i;
        const int bj = (B.cols() == 1) ? 0 : j;
        C(i, j) = A(ai, aj) - B(bi, bj);
      }
    }
    return C;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B,
      Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> C;
    ASSERT_NO_THROW(C = matrix_subtract_elementwise(A, B))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_sub(A, B))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(C, golden_result, tolerance, test_description);
  }
};

TEST_F(SubTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1},     {1, 1024},    {1024, 1},    {2, 3},       {64, 64},
      {256, 256}, {1024, 1024}, {1024, 2048}, {1024, 3072}, {2048, 1024},
  };

  const float tolerance = 1e-5;

  for (const auto& [m, n] : test_cases) {
    auto A = generate_random_matrix<float>(m, n, -1.0, 1.0);
    auto B = generate_random_matrix<float>(m, n, -1.0, 1.0);

    std::string test_name = "Random matrix " + std::to_string(m) + "x" +
                            std::to_string(n) + " subtract";

    test_with_golden(A, B, tolerance, test_name);
  }

  // Broadcasting cases used by the Qwen3 graph: (batch, 1) - scalar.
  {
    auto A = generate_random_matrix<float>(4, 1);
    Eigen::MatrixXf scalar(1, 1);
    scalar << 2.5f;
    test_with_golden(A, scalar, tolerance, "Sub (4x1) - scalar");

    auto C = generate_random_matrix<float>(4, 3);
    auto row = generate_random_matrix<float>(1, 3);
    test_with_golden(C, row, tolerance, "Sub (4x3) - (1x3)");

    auto col = generate_random_matrix<float>(4, 1);
    test_with_golden(C, col, tolerance, "Sub (4x3) - (4x1)");

    test_with_golden(scalar, C, tolerance, "Sub scalar - (4x3)");
  }
}


TEST_F(SubTest, InvalidInputs) {
  auto A = generate_random_matrix<float>(2, 3);
  auto B = generate_random_matrix<float>(3, 2);
  EXPECT_THROW(matrix_subtract_elementwise(A, B), std::invalid_argument);

  auto C = generate_random_matrix<float>(2, 2);
  EXPECT_THROW(matrix_subtract_elementwise(A, C), std::invalid_argument);
}
