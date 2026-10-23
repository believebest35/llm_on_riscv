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
    const int kept_size = (axis == 0) ? cols : rows;

    // ONNX output: indices.shape + the non-gathered data dim, with the
    // gathered axis flattened first.
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(num_indices,
                                                                 kept_size);
    int out_i = 0;
    for (int i = 0; i < indices.rows(); ++i) {
      for (int j = 0; j < indices.cols(); ++j) {
        const int idx = static_cast<int>(indices(i, j));
        if (axis == 0) {
          result.row(out_i) = data.row(idx);
        } else {
          result.row(out_i) = data.col(idx).transpose();
        }
        ++out_i;
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

    expect_matrix_near(impl_result, golden_result, tolerance, test_description);
  }
};

TEST_F(GatherTest, ValidInputs) {
  const std::vector<std::tuple<int, int, int, int>> test_cases = {
      {10, 20, 5, 0}, {10, 20, 5, 1}, {5, 5, 10, 0},
      {5, 5, 10, 1},  {64, 128, 32, 0}, {64, 128, 32, 1},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {151936, 1024, 4, 0}, {1, 2, 1, 1},
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

  // Hand-picked cases: non-random values / alternative
  // argument forms, checked against the same reference.
  {
    Eigen::MatrixXf data(2, 3);
    data << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f;

    // Negative index counts from the end: -1 selects the last row.
    Eigen::MatrixXi row_idx(1, 2);
    row_idx << -1, 0;
    auto rows = gather<float, int>(data, row_idx, 0);
    EXPECT_EQ(rows.rows(), 2);
    EXPECT_EQ(rows.cols(), 3);
    EXPECT_FLOAT_EQ(rows(0, 0), 4.0f);
    EXPECT_FLOAT_EQ(rows(1, 0), 1.0f);

    // Negative axis counts from the back: -1 is the columns.
    Eigen::MatrixXi col_idx(1, 2);
    col_idx << 2, 0;
    auto cols = gather<float, int>(data, col_idx, -1);
    EXPECT_EQ(cols.rows(), 2);
    EXPECT_EQ(cols.cols(), 2);
    EXPECT_FLOAT_EQ(cols(0, 0), 3.0f);
    EXPECT_FLOAT_EQ(cols(0, 1), 6.0f);
    EXPECT_FLOAT_EQ(cols(1, 0), 1.0f);
    EXPECT_FLOAT_EQ(cols(1, 1), 4.0f);

    // Gather a scalar from a shape vector: data (2,) and indices () in the
    // Qwen3 graph, represented as 2x1 data and a 1x1 index here.
    Eigen::MatrixXf shape_vec(2, 1);
    shape_vec << 4.0f, 7.0f;
    Eigen::MatrixXi scalar_idx(1, 1);
    scalar_idx << 1;
    auto scalar = gather<float, int>(shape_vec, scalar_idx, 0);
    EXPECT_EQ(scalar.rows(), 1);
    EXPECT_EQ(scalar.cols(), 1);
    EXPECT_FLOAT_EQ(scalar(0, 0), 7.0f);
  }
}



TEST_F(GatherTest, InvalidInputs) {
  Eigen::MatrixXf data(2, 3);
  data << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f;

  Eigen::MatrixXi idx(1, 1);
  idx << 0;
  EXPECT_THROW((gather<float, int>(data, idx, 2)), std::invalid_argument);

  Eigen::MatrixXi out_of_range(1, 1);
  out_of_range << 5;
  EXPECT_THROW((gather<float, int>(data, out_of_range, 0)),
               std::invalid_argument);
  EXPECT_THROW((gather<float, int>(data, out_of_range, 1)),
               std::invalid_argument);
}
