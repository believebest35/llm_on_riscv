#include "core/mul.h"
#include "gtest_base.h"

class MulTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_mul(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int b_rows = static_cast<int>(B.rows());
    const int b_cols = static_cast<int>(B.cols());

    if (m != b_rows || n != b_cols) {
      throw std::invalid_argument("Matrix dimensions incompatible");
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(m, n);

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        output(i, j) = A(i, j) * B(i, j);
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
    ASSERT_NO_THROW(impl_output = mul<Scalar>(A, B))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_output;
    ASSERT_NO_THROW(golden_output = golden_reference_mul<Scalar>(A, B))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_output, golden_output, tolerance, test_description);
  }
};

TEST_F(MulTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1},
      {1, 1024},
      {1024, 1},
      {2, 3},
      {64, 64},
      {256, 256},
      {1024, 1024},
      {1024, 2048},
      {1024, 3072},
      {2048, 1024},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {4, 3072},
  };

  const float tolerance = 0.0f;

  for (const auto& [rows, cols] : test_cases) {
    auto A = generate_random_matrix<float>(rows, cols, -1.0f, 1.0f);
    auto B = generate_random_matrix<float>(rows, cols, -1.0f, 1.0f);

    std::string test_name = "Mul " + std::to_string(rows) + "x" +
                            std::to_string(cols) + " element-wise multiply";

    test_with_golden(A, B, tolerance, test_name);
  }
}

TEST_F(MulTest, InvalidInputs) {
  auto A = generate_random_matrix<float>(2, 3);
  auto B = generate_random_matrix<float>(3, 2);
  EXPECT_THROW(mul<float>(A, B), std::invalid_argument);
}
