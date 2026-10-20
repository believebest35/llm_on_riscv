#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/concat.h"
#include "gtest_base.h"

class ConcatTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_reference_concat(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B,
      int axis) {
    if (axis == 0) {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(
          A.rows() + B.rows(), A.cols());
      out << A, B;
      return out;
    }
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(A.rows(),
                                                              A.cols() + B.cols());
    out << A, B;
    return out;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B,
      int axis, Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result = concat<Scalar>(A, B, axis))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_concat<Scalar>(A, B, axis))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_result, golden_result, tolerance, test_description);
  }
};

TEST_F(ConcatTest, ValidInputs) {
  const std::vector<std::tuple<int, int, int, int, int>> test_cases = {
      {2, 3, 1, 3, 0}, {2, 2, 2, 1, 1}, {3, 4, 2, 4, 0},
      {5, 2, 5, 3, 1}, {64, 128, 32, 128, 0}, {128, 64, 128, 64, 1},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {1, 1, 1, 1, 0},
  };

  const float tolerance = 1e-5;

  for (const auto& [rA, cA, rB, cB, axis] : test_cases) {
    auto A = generate_random_matrix<float>(rA, cA, -1.0f, 1.0f);
    auto B = generate_random_matrix<float>(rB, cB, -1.0f, 1.0f);

    std::string test_name = "Concat " + std::to_string(rA) + "x" +
                            std::to_string(cA) + " + " + std::to_string(rB) +
                            "x" + std::to_string(cB) +
                            " axis=" + std::to_string(axis);

    test_with_golden(A, B, axis, tolerance, test_name);
  }
}


TEST_F(ConcatTest, InvalidInputs) {
  auto A = generate_random_matrix<float>(2, 3);
  auto B = generate_random_matrix<float>(2, 3);

  EXPECT_THROW(concat<float>(A, B, 2), std::invalid_argument);
  // axis 0 needs matching column counts.
  EXPECT_THROW(concat<float>(A, generate_random_matrix<float>(2, 4), 0),
               std::invalid_argument);
  // axis 1 needs matching row counts.
  EXPECT_THROW(concat<float>(A, generate_random_matrix<float>(3, 3), 1),
               std::invalid_argument);
}
