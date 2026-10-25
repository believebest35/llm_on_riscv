#include "core/matmul.h"
#include "gtest_base.h"

class MatMulTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_reference_matmul(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int p = static_cast<int>(B.cols());
    const int b_rows = static_cast<int>(B.rows());

    if (n != b_rows) {
      throw std::invalid_argument("Matrix dimensions incompatible");
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(m, p);

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < p; ++j) {
        Scalar sum = static_cast<Scalar>(0);
        for (int k = 0; k < n; ++k) {
          sum += A(i, k) * B(k, j);
        }
        output(i, j) = sum;
      }
    }
    return output;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B,
      Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_output;
    ASSERT_NO_THROW(impl_output = matmul(A, B))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_output;
    ASSERT_NO_THROW(golden_output = golden_reference_matmul(A, B))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_output, golden_output, tolerance, test_description);
  }
};

TEST_F(MatMulTest, ValidInputs) {
  const std::vector<std::tuple<int, int, int>> test_cases = {
      {1, 1, 1},    {1, 1024, 1},    {1024, 1, 2048},    {2, 3, 4},
      {64, 64, 64}, {256, 256, 256}, {1024, 1024, 2048},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {4, 1024, 2048},   {4, 1024, 1024},  {4, 2048, 1024},
      {4, 1024, 3072},   {4, 3072, 1024},  {4, 1024, 151936},
  };

  const float tolerance = 1e-2f;

  for (const auto& [m, n, p] : test_cases) {
    auto A = generate_random_matrix<float>(m, n, -1.0f, 1.0f);
    auto B = generate_random_matrix<float>(n, p, -1.0f, 1.0f);

    std::string test_name = "MatMul " + std::to_string(m) + "x" +
                            std::to_string(n) + " * " + std::to_string(n) +
                            "x" + std::to_string(p);

    test_with_golden(A, B, tolerance, test_name);
  }
}

TEST_F(MatMulTest, InvalidInputs) {
  auto A = generate_random_matrix<float>(2, 3);
  auto B = generate_random_matrix<float>(4, 5);
  EXPECT_THROW(matmul(A, B), std::invalid_argument);
}
