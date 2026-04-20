#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/gather.h"
#include "gtest_base.h"

class GatherTest : public GTestBase {
 protected:
  template <typename Scalar, typename IndexType>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_gather(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& data,
      const Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic>& indices,
      int axis) {
    const int rows = data.rows();
    const int cols = data.cols();
    const int num_indices = static_cast<int>(indices.size());

    if (axis == 0) {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(num_indices,
                                                                   cols);
      int out_i = 0;
      for (int i = 0; i < indices.rows(); ++i) {
        for (int j = 0; j < indices.cols(); ++j) {
          IndexType idx = indices(i, j);
          result.row(out_i++) = data.row(static_cast<int>(idx));
        }
      }
      return result;
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(rows,
                                                                 num_indices);
    int out_j = 0;
    for (int i = 0; i < indices.rows(); ++i) {
      for (int j = 0; j < indices.cols(); ++j) {
        IndexType idx = indices(i, j);
        result.col(out_j++) = data.col(static_cast<int>(idx));
      }
    }
    return result;
  }

  template <typename Scalar, typename IndexType>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& data,
      const Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic>& indices,
      int axis, Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(
        (impl_result = gather<Scalar, IndexType>(data, indices, axis)))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(
        (golden_result = golden_reference_gather<Scalar, IndexType>(
             data, indices, axis)))
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

TEST_F(GatherTest, RandomMatrices) {
  const std::vector<std::tuple<int, int, int, int>> test_cases = {
      {10, 20, 5, 0}, {10, 20, 5, 1}, {5, 5, 10, 0},
      {5, 5, 10, 1},  {64, 128, 32, 0}, {64, 128, 32, 1},
  };

  const float tolerance = 1e-5;

  for (const auto& [rows, cols, num_idx, axis] : test_cases) {
    auto data = generate_random_matrix<float>(rows, cols, -10.0f, 10.0f);
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> idx(1, num_idx);
    int max_idx = (axis == 0) ? rows : cols;
    for (int i = 0; i < num_idx; ++i) {
      idx(0, i) = std::rand() % max_idx;
    }

    std::string test_name = "Gather " + std::to_string(rows) + "x" +
                            std::to_string(cols) + " num_idx=" +
                            std::to_string(num_idx) +
                            " axis=" + std::to_string(axis);

    test_with_golden<float, int>(data, idx, axis, tolerance, test_name);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
