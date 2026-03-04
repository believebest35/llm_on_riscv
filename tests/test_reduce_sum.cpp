#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/reduce_sum.h"
#include "gtest_base.h"

class ReduceSumTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_reference_reduce(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      int axis) {
    const int rows = A.rows();
    const int cols = A.cols();
    if (axis == 0) {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> res(1, cols);
      res.setZero();
      for (int i = 0; i < rows; ++i) {
        res += A.row(i);
      }
      return res;
    } else if (axis == 1) {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> res(rows, 1);
      res.setZero();
      for (int i = 0; i < rows; ++i) {
        res(i, 0) = A.row(i).sum();
      }
      return res;
    } else {
      throw std::invalid_argument("invalid axis");
    }
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A, int axis,
      const std::string& desc = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl;
    ASSERT_NO_THROW(impl = reduce_sum<Scalar>(A, axis))
        << "implementation threw: " << desc;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden;
    ASSERT_NO_THROW(golden = golden_reference_reduce<Scalar>(A, axis))
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
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(3, 4);
  A << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

  test_with_golden<float>(A, 0, "axis0 basic");
  test_with_golden<float>(A, 1, "axis1 basic");
}

TEST_F(ReduceSumTest, Random) {
  std::vector<std::tuple<int, int>> dims = {{5, 5}, {10, 20}, {1, 6}};
  for (auto [r, c] : dims) {
    auto M = generate_random_matrix<float>(r, c, -3.0f, 3.0f);
    test_with_golden<float>(M, 0, "random axis0");
    test_with_golden<float>(M, 1, "random axis1");
  }
}

TEST_F(ReduceSumTest, InvalidAxis) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(2, 2);
  A.setRandom();
  EXPECT_THROW(reduce_sum<float>(A, 2), std::invalid_argument);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
