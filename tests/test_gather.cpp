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
          int idx = static_cast<int>(indices(i, j));
          if (idx < 0) {
            idx += rows;
          }
          for (int k = 0; k < cols; ++k) {
            result(out_i, k) = data(idx, k);
          }
          ++out_i;
        }
      }
      return result;
    } else {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(rows,
                                                                   num_indices);
      int out_j = 0;
      for (int i = 0; i < indices.rows(); ++i) {
        for (int j = 0; j < indices.cols(); ++j) {
          int idx = static_cast<int>(indices(i, j));
          if (idx < 0) {
            idx += cols;
          }
          for (int k = 0; k < rows; ++k) {
            result(k, out_j) = data(k, idx);
          }
          ++out_j;
        }
      }
      return result;
    }
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
      {151936, 1024, 4, 0}, {2, 1, 1, 0},
  };

  const float tolerance = 0.0f;

  for (const auto& [rows, cols, num_idx, axis] : test_cases) {
    auto data = generate_random_matrix<float>(rows, cols, -10.0f, 10.0f);
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> idx(1, num_idx);
    int axis_size = (axis == 0) ? rows : cols;
    for (int i = 0; i < num_idx; ++i) {
      // Negative indices count from the end, as allowed by ONNX.
      idx(0, i) = std::rand() % (2 * axis_size) - axis_size;
    }

    std::string test_name = "Gather " + std::to_string(rows) + "x" +
                            std::to_string(cols) + " num_idx=" +
                            std::to_string(num_idx) +
                            " axis=" + std::to_string(axis);

    test_with_golden<float, int>(data, idx, axis, tolerance, test_name);
  }
}

TEST_F(GatherTest, InvalidInputs) {
  auto data = generate_random_matrix<float>(2, 3);

  Eigen::MatrixXi idx(1, 1);
  idx(0, 0) = std::rand() % data.cols();
  EXPECT_THROW((gather<float, int>(data, idx, 2)), std::invalid_argument);

  Eigen::MatrixXi out_of_range(1, 1);
  out_of_range(0, 0) = static_cast<int>(data.size());
  EXPECT_THROW((gather<float, int>(data, out_of_range, 0)),
               std::invalid_argument);
  EXPECT_THROW((gather<float, int>(data, out_of_range, 1)),
               std::invalid_argument);
}
