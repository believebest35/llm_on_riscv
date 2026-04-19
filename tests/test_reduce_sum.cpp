#include <gtest/gtest.h>

#include <set>
#include <string>
#include <vector>

#include "core/reduce_sum.h"
#include "gtest_base.h"

class ReduceSumTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_reference_reduce(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& data,
      const std::vector<int>& axes) {
    if (axes.empty()) {
      throw std::invalid_argument("empty axes");
    }
    std::set<int> unique_axes;
    for (int a : axes) {
      if (a < 0 || a > 1) {
        throw std::invalid_argument("invalid axis");
      }
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
      const std::vector<int>& axes, const std::string& desc = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl;
    ASSERT_NO_THROW(impl = reduce_sum<Scalar>(data, axes))
        << "implementation threw: " << desc;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden;
    ASSERT_NO_THROW(golden = golden_reference_reduce<Scalar>(data, axes))
        << "golden threw: " << desc;

    EXPECT_EQ(impl.rows(), golden.rows()) << "row mismatch: " << desc;
    EXPECT_EQ(impl.cols(), golden.cols()) << "col mismatch: " << desc;

    const auto numel = impl.rows() * impl.cols();
    for (int i = 0; i < numel; ++i) {
      EXPECT_EQ(impl.data()[i], golden.data()[i])
          << "value mismatch at " << i << " (" << desc << ")";
    }
  }
};

TEST_F(ReduceSumTest, Basic) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> data(3, 4);
  data << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

  test_with_golden<float>(data, {0}, "axis0 basic");
  test_with_golden<float>(data, {1}, "axis1 basic");
}

TEST_F(ReduceSumTest, BothAxes) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> data(3, 4);
  data << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

  test_with_golden<float>(data, {0, 1}, "axes {0,1}");
  test_with_golden<float>(data, {1, 0, 0}, "axes with duplicates");
}

TEST_F(ReduceSumTest, Random) {
  std::vector<std::tuple<int, int>> dims = {{5, 5}, {10, 20}, {1, 6}};
  for (auto [r, c] : dims) {
    auto M = generate_random_matrix<float>(r, c, -3.0f, 3.0f);
    test_with_golden<float>(M, {0}, "random axis0");
    test_with_golden<float>(M, {1}, "random axis1");
    test_with_golden<float>(M, {0, 1}, "random both axes");
  }
}

TEST_F(ReduceSumTest, InvalidAxis) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> data(2, 2);
  data.setRandom();
  EXPECT_THROW(reduce_sum<float>(data, {2}), std::invalid_argument);
}

TEST_F(ReduceSumTest, EmptyAxes) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> data(2, 2);
  data.setRandom();
  EXPECT_THROW(reduce_sum<float>(data, {}), std::invalid_argument);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
