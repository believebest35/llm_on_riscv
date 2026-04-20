#include <gtest/gtest.h>

#include <string>

#include "core/shape.h"
#include "gtest_base.h"

class ShapeTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_reference_shape(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(1, 2);
    out(0, 0) = static_cast<Scalar>(A.rows());
    out(0, 1) = static_cast<Scalar>(A.cols());
    return out;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> impl_vec;
    ASSERT_NO_THROW(impl_vec = shape<Scalar>(A))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result(1, 2);
    impl_result(0, 0) = impl_vec(0);
    impl_result(0, 1) = impl_vec(1);

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_shape(A))
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

TEST_F(ShapeTest, RandomMatrices) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1}, {1, 1024}, {1024, 1}, {2, 3}, {64, 64}, {256, 256}, {1024, 2048},
  };

  const float tolerance = 1e-5;

  for (const auto& [rows, cols] : test_cases) {
    auto A = generate_random_matrix<float>(rows, cols, -1.0f, 1.0f);

    std::string test_name = "Shape " + std::to_string(rows) + "x" +
                            std::to_string(cols);

    test_with_golden(A, tolerance, test_name);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
