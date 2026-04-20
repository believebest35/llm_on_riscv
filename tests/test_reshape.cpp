#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/reshape.h"
#include "gtest_base.h"

class ReshapeTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_reshape(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      int new_rows, int new_cols) {
    const std::int64_t total = A.size();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(new_rows,
                                                              new_cols);
    for (std::int64_t i = 0; i < total; ++i) {
      std::int64_t r = i / A.cols();
      std::int64_t c = i % A.cols();
      std::int64_t r2 = i / new_cols;
      std::int64_t c2 = i % new_cols;
      out(r2, c2) = A(r, c);
    }
    return out;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      int new_rows, int new_cols, Scalar tolerance,
      const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result = reshape<Scalar>(A, new_rows, new_cols))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(
        golden_result = golden_reference_reshape<Scalar>(A, new_rows, new_cols))
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

TEST_F(ReshapeTest, RandomMatrices) {
  const std::vector<std::tuple<int, int, int, int>> test_cases = {
      {2, 3, 3, 2}, {2, 3, 1, 6}, {2, 3, 6, 1},
      {4, 4, 2, 8}, {64, 128, 128, 64}, {256, 256, 1, 65536},
  };

  const float tolerance = 1e-5;

  for (const auto& [rows, cols, new_rows, new_cols] : test_cases) {
    auto A = generate_random_matrix<float>(rows, cols, -5.0f, 5.0f);

    std::string test_name = "Reshape " + std::to_string(rows) + "x" +
                            std::to_string(cols) + " -> " +
                            std::to_string(new_rows) + "x" +
                            std::to_string(new_cols);

    test_with_golden(A, new_rows, new_cols, tolerance, test_name);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
