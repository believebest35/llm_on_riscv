#include "core/concat.h"
#include "gtest_base.h"

class ConcatTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_reference_concat(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B,
      int axis) {
    if (axis < 0) {
      axis += 2;  // 2-D data: -1 -> 1, -2 -> 0
    }
    const int a_rows = static_cast<int>(A.rows());
    const int a_cols = static_cast<int>(A.cols());
    const int b_rows = static_cast<int>(B.rows());
    const int b_cols = static_cast<int>(B.cols());

    if (axis == 0) {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(
          a_rows + b_rows, a_cols);
      for (int i = 0; i < a_rows; ++i) {
        for (int j = 0; j < a_cols; ++j) {
          output(i, j) = A(i, j);
        }
      }
      for (int i = 0; i < b_rows; ++i) {
        for (int j = 0; j < b_cols; ++j) {
          output(a_rows + i, j) = B(i, j);
        }
      }
      return output;
    } else {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(
          a_rows, a_cols + b_cols);
      for (int i = 0; i < a_rows; ++i) {
        for (int j = 0; j < a_cols; ++j) {
          output(i, j) = A(i, j);
        }
        for (int j = 0; j < b_cols; ++j) {
          output(i, a_cols + j) = B(i, j);
        }
      }
      return output;
    }
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B,
      int axis, Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_output;
    ASSERT_NO_THROW(impl_output = concat<Scalar>(A, B, axis))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_output;
    ASSERT_NO_THROW(golden_output = golden_reference_concat<Scalar>(A, B, axis))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_output, golden_output, tolerance, test_description);
  }
};

TEST_F(ConcatTest, ValidInputs) {
  const std::vector<std::tuple<int, int, int, int, int>> test_cases = {
      {2, 3, 1, 3, 0},
      {2, 2, 2, 1, 1},
      {3, 4, 2, 4, 0},
      {5, 2, 5, 3, 1},
      {64, 128, 32, 128, 0},
      {128, 64, 128, 64, 1},
      {2, 3, 1, 3, -2},
      {2, 2, 2, 1, -1},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {1, 1, 1, 1, 0},
      {1, 1, 3, 1, 0},
  };

  const float tolerance = 0.0f;

  for (const auto& [a_rows, a_cols, b_rows, b_cols, axis] : test_cases) {
    auto A = generate_random_matrix<float>(a_rows, a_cols, -1.0f, 1.0f);
    auto B = generate_random_matrix<float>(b_rows, b_cols, -1.0f, 1.0f);

    std::string test_description =
        "Concat " + std::to_string(a_rows) + "x" + std::to_string(a_cols) +
        " + " + std::to_string(b_rows) + "x" + std::to_string(b_cols) +
        " axis=" + std::to_string(axis);

    test_with_golden(A, B, axis, tolerance, test_description);
  }
}

TEST_F(ConcatTest, InvalidInputs) {
  auto A = generate_random_matrix<float>(2, 3);
  auto B = generate_random_matrix<float>(2, 3);

  EXPECT_THROW(concat<float>(A, B, 2), std::invalid_argument);
  EXPECT_THROW(concat<float>(A, B, -3), std::invalid_argument);
  // axis 0 needs matching column counts.
  EXPECT_THROW(concat<float>(A, generate_random_matrix<float>(2, 4), 0),
               std::invalid_argument);
  // axis 1 needs matching row counts.
  EXPECT_THROW(concat<float>(A, generate_random_matrix<float>(3, 3), 1),
               std::invalid_argument);
}
