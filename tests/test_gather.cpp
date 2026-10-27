#include "core/gather.h"
#include "gtest_base.h"

class GatherTest : public GTestBase {
 protected:
  template <typename Scalar, typename IndexType>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_gather(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
      const Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic>& indices,
      int axis) {
    if (axis < 0) {
      axis += 2;  // 2-D input: -1 -> 1, -2 -> 0
    }
    const int rows = static_cast<int>(input.rows());
    const int cols = static_cast<int>(input.cols());
    const int index_rows = static_cast<int>(indices.rows());
    const int index_cols = static_cast<int>(indices.cols());
    const int num_indices = static_cast<int>(indices.size());

    if (axis == 0) {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(num_indices,
                                                                   cols);
      int out_i = 0;
      for (int i = 0; i < index_rows; ++i) {
        for (int j = 0; j < index_cols; ++j) {
          int idx = static_cast<int>(indices(i, j));
          if (idx < 0) {
            idx += rows;
          }
          for (int k = 0; k < cols; ++k) {
            output(out_i, k) = input(idx, k);
          }
          ++out_i;
        }
      }
      return output;
    } else {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(rows,
                                                                   num_indices);
      int out_j = 0;
      for (int i = 0; i < index_rows; ++i) {
        for (int j = 0; j < index_cols; ++j) {
          int idx = static_cast<int>(indices(i, j));
          if (idx < 0) {
            idx += cols;
          }
          for (int k = 0; k < rows; ++k) {
            output(k, out_j) = input(k, idx);
          }
          ++out_j;
        }
      }
      return output;
    }
  }

  template <typename Scalar, typename IndexType>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
      const Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic>& indices,
      int axis, Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_output;
    ASSERT_NO_THROW(
        (impl_output = gather<Scalar, IndexType>(input, indices, axis)))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_output;
    ASSERT_NO_THROW(
        (golden_output = golden_reference_gather<Scalar, IndexType>(
             input, indices, axis)))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_output, golden_output, tolerance, test_description);
  }
};

TEST_F(GatherTest, ValidInputs) {
  const std::vector<std::tuple<int, int, int, int>> test_cases = {
      {10, 20, 5, 0},
      {10, 20, 5, 1},
      {5, 5, 10, 0},
      {5, 5, 10, 1},
      {64, 128, 32, 0},
      {64, 128, 32, 1},
      {10, 20, 5, -2},
      {10, 20, 5, -1},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {151936, 1024, 4, 0},
      {2, 1, 1, 0},
  };

  const float tolerance = 0.0f;

  for (const auto& [rows, cols, num_idx, axis] : test_cases) {
    auto input = generate_random_matrix<float>(rows, cols, -10.0f, 10.0f);
    const int norm_axis = (axis < 0) ? axis + 2 : axis;
    const int axis_size = (norm_axis == 0) ? rows : cols;
    auto int_indices =
        generate_random_indices<int>(num_idx, -axis_size, axis_size - 1);
    auto int64_indices = generate_random_indices<std::int64_t>(
        num_idx, -axis_size, axis_size - 1);

    std::string test_name = "Gather " + std::to_string(rows) + "x" +
                            std::to_string(cols) + " num_idx=" +
                            std::to_string(num_idx) +
                            " axis=" + std::to_string(axis);

    test_with_golden<float, int>(input, int_indices, axis, tolerance,
                                 test_name);
    test_with_golden<float, std::int64_t>(input, int64_indices, axis, tolerance,
                                          test_name + " int64");
  }
}

TEST_F(GatherTest, InvalidInputs) {
  auto input = generate_random_matrix<float>(2, 3);

  Eigen::MatrixXi idx(1, 1);
  idx(0, 0) = 0;
  EXPECT_THROW((gather<float, int>(input, idx, 2)), std::invalid_argument);

  Eigen::MatrixXi out_of_range(1, 1);
  out_of_range(0, 0) = static_cast<int>(input.size());
  EXPECT_THROW((gather<float, int>(input, out_of_range, 0)),
               std::invalid_argument);
  EXPECT_THROW((gather<float, int>(input, out_of_range, 1)),
               std::invalid_argument);
}
