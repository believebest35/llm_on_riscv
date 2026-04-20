#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/unsqueeze.h"
#include "gtest_base.h"

class UnsqueezeTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_unsqueeze(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& data, int axes) {
    const int N = data.size();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> expanded;
    if (axes == 0) {
      expanded.resize(1, N);
      expanded.row(0) = data.transpose();
    } else {
      expanded.resize(N, 1);
      expanded.col(0) = data;
    }
    return expanded;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& data, int axes,
      Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result = unsqueeze<Scalar>(data, axes))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_unsqueeze<Scalar>(data, axes))
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

TEST_F(UnsqueezeTest, RandomMatrices) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 0}, {1, 1}, {5, 0}, {5, 1}, {10, 0}, {100, 0}, {100, 1},
  };

  const float tolerance = 1e-5;

  for (const auto& [size, axes] : test_cases) {
    auto data = generate_random_vector<float>(size, -2.0f, 2.0f);

    std::string test_name = "Unsqueeze size=" + std::to_string(size) +
                            " axes=" + std::to_string(axes);

    test_with_golden(data, axes, tolerance, test_name);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
