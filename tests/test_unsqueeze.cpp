#include <gtest/gtest.h>

#include <string>

#include "core/unsqueeze.h"
#include "gtest_base.h"

class UnsqueezeTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_unsqueeze(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      int axis) {
    const int total = static_cast<int>(A.size());
    if (axis == 0) {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(1, total);
      for (int i = 0; i < total; ++i) {
        out(0, i) = A.data()[i];
      }
      return out;
    } else if (axis == 1) {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(total, 1);
      for (int i = 0; i < total; ++i) {
        out(i, 0) = A.data()[i];
      }
      return out;
    }
    throw std::invalid_argument("invalid axis");
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A, int axis,
      const std::string& desc = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl;
    ASSERT_NO_THROW(impl = unsqueeze<Scalar>(A, axis))
        << "implementation threw: " << desc;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden;
    ASSERT_NO_THROW(golden = golden_reference_unsqueeze<Scalar>(A, axis))
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

TEST_F(UnsqueezeTest, SmallMatrices) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(2, 2);
  A << 1, 2, 3, 4;
  test_with_golden<float>(A, 0, "axis0 small");
  test_with_golden<float>(A, 1, "axis1 small");
}

TEST_F(UnsqueezeTest, Random) {
  std::vector<std::pair<int, int>> dims = {{1, 1}, {1, 5}, {5, 1}, {3, 4}};
  for (auto [r, c] : dims) {
    auto M = generate_random_matrix<float>(r, c, -2.0f, 2.0f);
    test_with_golden<float>(M, 0, "rand axis0");
    test_with_golden<float>(M, 1, "rand axis1");
  }
}

TEST_F(UnsqueezeTest, InvalidAxis) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(2, 3);
  A.setRandom();
  EXPECT_THROW(unsqueeze<float>(A, 2), std::invalid_argument);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
