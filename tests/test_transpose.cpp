#include <gtest/gtest.h>

#include <string>

#include "core/transpose.h"
#include "gtest_base.h"

class TransposeTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_transpose(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(A.cols(),
                                                                 A.rows());
    for (int i = 0; i < A.rows(); ++i) {
      for (int j = 0; j < A.cols(); ++j) {
        result(j, i) = A(i, j);
      }
    }
    return result;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const std::string& desc = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl;
    ASSERT_NO_THROW(impl = transpose<Scalar>(A))
        << "implementation threw: " << desc;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden;
    ASSERT_NO_THROW(golden = golden_reference_transpose(A))
        << "golden threw: " << desc;

    EXPECT_EQ(impl.rows(), golden.rows()) << "row mismatch: " << desc;
    EXPECT_EQ(impl.cols(), golden.cols()) << "col mismatch: " << desc;

    const auto numel = impl.rows() * impl.cols();
    for (int i = 0; i < numel; ++i) {
      EXPECT_EQ(impl.data()[i], golden.data()[i])
          << "value mismatch at index " << i << " (" << desc << ")";
    }
  }
};

TEST_F(TransposeTest, FixedMatrices) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(2, 3);
  A << 1, 2, 3, 4, 5, 6;
  test_with_golden<float>(A, "2x3 fixed");

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> B(3, 1);
  B << 7, 8, 9;
  test_with_golden<float>(B, "3x1 fixed");
}

TEST_F(TransposeTest, RandomMatrices) {
  const std::vector<std::pair<int, int>> cases = {
      {1, 1}, {1, 5}, {5, 1}, {10, 10}, {64, 128}};
  for (auto [r, c] : cases) {
    auto M = generate_random_matrix<float>(r, c, -5.0f, 5.0f);
    std::string desc = "random " + std::to_string(r) + "x" + std::to_string(c);
    test_with_golden<float>(M, desc);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
