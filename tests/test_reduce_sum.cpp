#include <set>

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
      if (a < 0) {
        a += 2;  // 2-D data: -1 -> 1, -2 -> 0
      }
      unique_axes.insert(a);
    }

    const int rows = data.rows();
    const int cols = data.cols();

    if (unique_axes.size() == 2) {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> reduced(1, 1);
      double total = 0.0;
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          total += static_cast<double>(data(i, j));
        }
      }
      reduced(0, 0) = static_cast<Scalar>(total);
      return reduced;
    } else if (*unique_axes.begin() == 0) {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> reduced(1, cols);
      for (int j = 0; j < cols; ++j) {
        double total = 0.0;
        for (int i = 0; i < rows; ++i) {
          total += static_cast<double>(data(i, j));
        }
        reduced(0, j) = static_cast<Scalar>(total);
      }
      return reduced;
    } else {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> reduced(rows, 1);
      for (int i = 0; i < rows; ++i) {
        double total = 0.0;
        for (int j = 0; j < cols; ++j) {
          total += static_cast<double>(data(i, j));
        }
        reduced(i, 0) = static_cast<Scalar>(total);
      }
      return reduced;
    }
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

    expect_matrix_near(impl_result, golden_result, tolerance, test_description);
  }
};

TEST_F(ReduceSumTest, ValidInputs) {
  const std::vector<std::tuple<int, int, std::vector<int>>> test_cases = {
      {3, 4, {0}},      {3, 4, {1}},        {3, 4, {0, 1}},
      {5, 5, {0}},      {10, 20, {1}},      {64, 128, {0}},
      {128, 64, {1}},   {256, 256, {0, 1}},
      // Negative axes are used by the Qwen3 graph.
      {3, 4, {-1}},      {3, 4, {-2}},      {3, 4, {-1, -2}},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {1, 7, {-1}},
  };

  // Float accumulation over large reductions differs from the exact sum.
  const float tolerance = 1e-3;

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

TEST_F(ReduceSumTest, InvalidInputs) {
  auto data = generate_random_matrix<float>(2, 3);
  EXPECT_THROW(reduce_sum<float>(data, std::vector<int>{}),
               std::invalid_argument);
  EXPECT_THROW(reduce_sum<float>(data, {2}), std::invalid_argument);
}
