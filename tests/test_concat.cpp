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

TEST_F(ConcatTest, RandomMatrices) {
  const std::vector<std::tuple<int, int, int, int, int>> test_cases = {
      {2, 3, 1, 3, 0}, {2, 2, 2, 1, 1}, {3, 4, 2, 4, 0},
      {5, 2, 5, 3, 1}, {64, 128, 32, 128, 0}, {128, 64, 128, 64, 1},
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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
