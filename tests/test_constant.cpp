#include <gtest/gtest.h>

#include <string>

#include "core/constant.h"
#include "gtest_base.h"

class ConstantTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_constant(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& value) {
    return value;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& value,
      Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result = constant_matrix<Scalar>(value))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_constant<Scalar>(value))
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

TEST_F(ConstantTest, RandomMatrices) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1}, {2, 3}, {64, 64}, {128, 256}, {1024, 1024},
  };

  const float tolerance = 1e-5;

  for (const auto& [rows, cols] : test_cases) {
    auto value = generate_random_matrix<float>(rows, cols, -1.0, 1.0);

    std::string test_name = "Constant " + std::to_string(rows) + "x" +
                            std::to_string(cols);

    test_with_golden(value, tolerance, test_name);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
