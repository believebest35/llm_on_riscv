#include <algorithm>

#include "core/sub.h"
#include "gtest_base.h"

class SubTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_sub(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
    const int a_rows = static_cast<int>(A.rows());
    const int a_cols = static_cast<int>(A.cols());
    const int b_rows = static_cast<int>(B.rows());
    const int b_cols = static_cast<int>(B.cols());
    const int m = std::max(a_rows, b_rows);
    const int n = std::max(a_cols, b_cols);
    if ((a_rows != m && a_rows != 1) || (a_cols != n && a_cols != 1) ||
        (b_rows != m && b_rows != 1) || (b_cols != n && b_cols != 1)) {
      throw std::invalid_argument("Matrix dimensions incompatible");
    }
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(m, n);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        const int ai = (a_rows == 1) ? 0 : i;
        const int aj = (a_cols == 1) ? 0 : j;
        const int bi = (b_rows == 1) ? 0 : i;
        const int bj = (b_cols == 1) ? 0 : j;
        output(i, j) = A(ai, aj) - B(bi, bj);
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
    ASSERT_NO_THROW(impl_output = sub(A, B))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_output;
    ASSERT_NO_THROW(golden_output = golden_reference_sub(A, B))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_output, golden_output, tolerance, test_description);
  }
};

TEST_F(SubTest, ValidInputs) {
  const std::vector<std::tuple<int, int, int, int>> test_cases = {
      {1, 1, 1, 1},             {1, 1024, 1, 1024},
      {1024, 1, 1024, 1},       {2, 3, 2, 3},
      {64, 64, 64, 64},         {256, 256, 256, 256},
      {1024, 1024, 1024, 1024}, {1024, 2048, 1024, 2048},
      {1024, 3072, 1024, 3072}, {2048, 1024, 2048, 1024},
      // docs/01_dev.md (Qwen3-0.6B) shapes: (batch, 1) - scalar.
      {4, 1, 1, 1},
  };

  const float tolerance = 0.0f;

  for (const auto& [rA, cA, rB, cB] : test_cases) {
    auto A = generate_random_matrix<float>(rA, cA, -1.0f, 1.0f);
    auto B = generate_random_matrix<float>(rB, cB, -1.0f, 1.0f);

    std::string test_name = "Sub " + std::to_string(rA) + "x" +
                            std::to_string(cA) + " - " + std::to_string(rB) +
                            "x" + std::to_string(cB);

    test_with_golden(A, B, tolerance, test_name);
  }
}

TEST_F(SubTest, InvalidInputs) {
  auto A = generate_random_matrix<float>(2, 3);
  auto B = generate_random_matrix<float>(3, 2);
  EXPECT_THROW(sub(A, B), std::invalid_argument);

  auto other = generate_random_matrix<float>(2, 2);
  EXPECT_THROW(sub(A, other), std::invalid_argument);
}
