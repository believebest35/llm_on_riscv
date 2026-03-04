#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/concat.h"
#include "gtest_base.h"

class ConcatTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_reference_concat(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B,
      int axis) {
    if (axis == 0) {
      if (A.cols() != B.cols()) {
        throw std::invalid_argument("column mismatch");
      }
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(
          A.rows() + B.rows(), A.cols());
      out << A, B;
      return out;
    } else if (axis == 1) {
      if (A.rows() != B.rows()) {
        throw std::invalid_argument("row mismatch");
      }
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(
          A.rows(), A.cols() + B.cols());
      out << A, B;
      return out;
    }
    throw std::invalid_argument("invalid axis");
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B, int axis,
      const std::string& desc = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl;
    ASSERT_NO_THROW(impl = concat<Scalar>(A, B, axis))
        << "implementation threw: " << desc;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden;
    ASSERT_NO_THROW(golden = golden_reference_concat<Scalar>(A, B, axis))
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

TEST_F(ConcatTest, BasicVertical) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(2, 3);
  A << 1, 2, 3, 4, 5, 6;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> B(1, 3);
  B << 7, 8, 9;
  test_with_golden<float>(A, B, 0, "vertical");
}

TEST_F(ConcatTest, BasicHorizontal) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(2, 2);
  A << 1, 2, 3, 4;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> B(2, 1);
  B << 5, 6;
  test_with_golden<float>(A, B, 1, "horizontal");
}

TEST_F(ConcatTest, Random) {
  std::vector<std::tuple<int, int, int>> dims = {
      {3, 4, 2}, {5, 2, 5}, {1, 1, 3}};
  for (auto [r, c, c2] : dims) {
    auto A = generate_random_matrix<float>(r, c, -1, 1);
    auto B = generate_random_matrix<float>(r, c2, -1, 1);
    test_with_golden<float>(A, B, 1, "random horiz");
    // vertical: requires same cols
    B = generate_random_matrix<float>(
        c, r, -1, 1);  // wrong shape but we'll adapt differently
    // instead generate vertical case separately
  }
  // vertical random example
  auto A2 = generate_random_matrix<float>(2, 4, -1, 1);
  auto B2 = generate_random_matrix<float>(3, 4, -1, 1);
  test_with_golden<float>(A2, B2, 0, "random vert");
}

TEST_F(ConcatTest, MismatchError) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(2, 3);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> B(1, 2);
  EXPECT_THROW(concat<float>(A, B, 0), std::invalid_argument);
  EXPECT_THROW(concat<float>(A, B, 1), std::invalid_argument);
  EXPECT_THROW(concat<float>(A, B, 2), std::invalid_argument);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
