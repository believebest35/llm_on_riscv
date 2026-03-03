#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/gather.h"
#include "gtest_base.h"

class GatherTest : public GTestBase {
 protected:
  template <typename Scalar, typename IndexType>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_reference_gather(
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
          if (idx < 0 || idx >= rows) {
            throw std::invalid_argument("index out of bounds");
          }
          result.row(out_i++) = data.row(static_cast<int>(idx));
        }
      }
      return result;
    } else if (axis == 1) {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(rows,
                                                                   num_indices);
      int out_j = 0;
      for (int i = 0; i < indices.rows(); ++i) {
        for (int j = 0; j < indices.cols(); ++j) {
          IndexType idx = indices(i, j);
          if (idx < 0 || idx >= cols) {
            throw std::invalid_argument("index out of bounds");
          }
          result.col(out_j++) = data.col(static_cast<int>(idx));
        }
      }
      return result;
    } else {
      throw std::invalid_argument("invalid axis");
    }
  }

  template <typename Scalar, typename IndexType>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& data,
      const Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic>& indices,
      int axis, const std::string& desc = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl;
    ASSERT_NO_THROW((impl = gather<Scalar, IndexType>(data, indices, axis)))
        << "implementation threw: " << desc;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden;
    ASSERT_NO_THROW((golden = golden_reference_gather<Scalar, IndexType>(
                         data, indices, axis)))
        << "golden threw: " << desc;

    EXPECT_EQ(impl.rows(), golden.rows()) << "row count mismatch: " << desc;
    EXPECT_EQ(impl.cols(), golden.cols()) << "col count mismatch: " << desc;

    const auto numel = impl.rows() * impl.cols();
    for (int i = 0; i < numel; ++i) {
      EXPECT_EQ(impl.data()[i], golden.data()[i])
          << "value mismatch at index " << i << " (" << desc << ")";
    }
  }
};

TEST_F(GatherTest, RowAxisBasic) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> data(5, 3);
  data << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15;

  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> idx(2, 2);
  idx << 3, 0, 4, 1;

  std::string desc = "simple gather rows axis=0";
  test_with_golden<float, int>(data, idx, 0, desc);
}

TEST_F(GatherTest, ColAxisBasic) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> data(2, 4);
  data << 1, 2, 3, 4, 5, 6, 7, 8;

  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> idx(1, 3);
  idx << 2, 0, 3;

  std::string desc = "simple gather cols axis=1";
  test_with_golden<float, int>(data, idx, 1, desc);
}

TEST_F(GatherTest, Randomized) {
  std::vector<std::tuple<int, int, int>> dims = {{10, 20, 5}, {5, 5, 10}};

  for (auto [rows, cols, num_idx] : dims) {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> data =
        generate_random_matrix<float>(rows, cols, -10, 10);
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> idx(1, num_idx);
    for (int i = 0; i < num_idx; ++i) {
      idx(0, i) = std::rand() % rows;
    }
    test_with_golden<float, int>(data, idx, 0, "random row gather");

    // also test axis=1 with appropriately sized indices
    idx.resize(1, num_idx);
    for (int i = 0; i < num_idx; ++i) {
      idx(0, i) = std::rand() % cols;
    }
    test_with_golden<float, int>(data, idx, 1, "random col gather");
  }
}

TEST_F(GatherTest, ErrorCases) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> data(3, 3);
  data.setRandom();
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> idx(1, 1);
  idx(0, 0) = -1;
  EXPECT_THROW((gather<float, int>(data, idx, 0)), std::invalid_argument);
  idx(0, 0) = 3;
  EXPECT_THROW((gather<float, int>(data, idx, 0)), std::invalid_argument);
  EXPECT_THROW((gather<float, int>(data, idx, 2)), std::invalid_argument);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
