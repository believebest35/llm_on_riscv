#include <gtest/gtest.h>

#include <string>

#include "core/mul.h"
#include "gtest_base.h"

class MulTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_mul(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
    const int m = A.rows();
    const int n = A.cols();

    if (m != B.rows() || n != B.cols()) {
      throw std::invalid_argument("Matrix dimensions incompatible");
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> C(m, n);

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        C(i, j) = A(i, j) * B(i, j);
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
    ASSERT_NO_THROW(C = matrix_multiply_elementwise(A, B))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_mul(A, B))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(C, golden_result, tolerance, test_description);
  }
};

TEST_F(MulTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1},     {1, 1024},    {1024, 1},    {2, 3},       {64, 64},
      {256, 256}, {1024, 1024}, {1024, 2048}, {1024, 3072}, {2048, 1024},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {4, 3072},
  };

  const float tolerance = 1e-5;

  for (const auto& [rows, cols] : test_cases) {
    auto A = generate_random_matrix<float>(rows, cols, -1.0, 1.0);
    auto B = generate_random_matrix<float>(rows, cols, -1.0, 1.0);

    std::string test_name = "Random matrix " + std::to_string(rows) + "x" +
                            std::to_string(cols) + " element-wise multiply";

    test_with_golden(A, B, tolerance, test_name);
  }
}


TEST_F(MulTest, InvalidInputs) {
  auto A = generate_random_matrix<float>(2, 3);
  auto B = generate_random_matrix<float>(3, 2);
  EXPECT_THROW(matrix_multiply_elementwise(A, B), std::invalid_argument);
}
