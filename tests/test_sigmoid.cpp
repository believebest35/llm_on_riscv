#include <gtest/gtest.h>

#include <string>

#include "core/sigmoid.h"
#include "gtest_base.h"

class SigmoidTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_sigmoid(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(A.rows(),
                                                              A.cols());
    for (std::int64_t i = 0; i < A.size(); ++i) {
      Scalar x = A.data()[i];
      out.data()[i] =
          static_cast<Scalar>(1) / (static_cast<Scalar>(1) + std::exp(-x));
    }
    return out;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const std::string& desc = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl;
    ASSERT_NO_THROW(impl = sigmoid<Scalar>(A))
        << "implementation threw: " << desc;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden;
    ASSERT_NO_THROW(golden = golden_reference_sigmoid(A))
        << "golden threw: " << desc;

    EXPECT_EQ(impl.rows(), golden.rows()) << "row mismatch: " << desc;
    EXPECT_EQ(impl.cols(), golden.cols()) << "col mismatch: " << desc;

    for (std::int64_t i = 0; i < impl.size(); ++i) {
      EXPECT_NEAR(impl.data()[i], golden.data()[i], 1e-6)
          << "value mismatch at " << i << " (" << desc << ")";
    }
  }
};

TEST_F(SigmoidTest, Fixed) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(2, 2);
  A << -1.0, 0.0, 1.0, 2.0;
  test_with_golden<float>(A, "fixed values");
}

TEST_F(SigmoidTest, Random) {
  auto M = generate_random_matrix<float>(5, 5, -5.0f, 5.0f);
  test_with_golden<float>(M, "random");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
