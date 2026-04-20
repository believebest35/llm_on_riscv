#include <gtest/gtest.h>

#include <set>
#include <string>
#include <vector>

#include "core/reduce_sum.h"
#include "gtest_base.h"

class ReduceSumTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_reduce_sum(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& data,
      const std::vector<int>& axes) {
    std::set<int> unique_axes;
    for (int a : axes) {
      unique_axes.insert(a);
    }

    const int rows = data.rows();
    const int cols = data.cols();

    if (unique_axes.size() == 2) {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> reduced(1, 1);
      reduced(0, 0) = data.sum();
      return reduced;
    }

    if (*unique_axes.begin() == 0) {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> reduced(1, cols);
      reduced.setZero();
      for (int i = 0; i < rows; ++i) {
        reduced += data.row(i);
      }
      return reduced;
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> reduced(rows, 1);
    reduced.setZero();
    for (int i = 0; i < rows; ++i) {
      reduced(i, 0) = data.row(i).sum();
    }
    return reduced;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& data,
      const std::vector<int>& axes, Scalar tolerance,
      const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result = reduce_sum<Scalar>(data, axes))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_reduce_sum(data, axes))
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

TEST_F(ReduceSumTest, RandomMatrices) {
  const std::vector<std::tuple<int, int, std::vector<int>>> test_cases = {
      {3, 4, {0}},      {3, 4, {1}},        {3, 4, {0, 1}},
      {5, 5, {0}},      {10, 20, {1}},      {64, 128, {0}},
      {128, 64, {1}},   {256, 256, {0, 1}},
  };

  const float tolerance = 1e-5;

  for (const auto& [rows, cols, axes] : test_cases) {
    auto data = generate_random_matrix<float>(rows, cols, -3.0f, 3.0f);

    std::string axes_str;
    for (size_t i = 0; i < axes.size(); ++i) {
      if (i > 0) axes_str += ",";
      axes_str += std::to_string(axes[i]);
    }

    std::string test_name = "ReduceSum " + std::to_string(rows) + "x" +
                            std::to_string(cols) + " axes={" + axes_str + "}";

    test_with_golden(data, axes, tolerance, test_name);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
