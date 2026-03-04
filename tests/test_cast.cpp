#include <gtest/gtest.h>

#include <string>

#include "core/cast.h"
#include "gtest_base.h"

class CastTest : public GTestBase {
 protected:
  template <typename Out, typename In>
  Eigen::Matrix<Out, Eigen::Dynamic, Eigen::Dynamic> golden_reference_cast(
      const Eigen::Matrix<In, Eigen::Dynamic, Eigen::Dynamic>& A) {
    Eigen::Matrix<Out, Eigen::Dynamic, Eigen::Dynamic> B(A.rows(), A.cols());
    for (int i = 0; i < A.size(); ++i) {
      B.data()[i] = static_cast<Out>(A.data()[i]);
    }
    return B;
  }

  template <typename Out, typename In>
  void test_with_golden(
      const Eigen::Matrix<In, Eigen::Dynamic, Eigen::Dynamic>& A,
      const std::string& desc = "") {
    Eigen::Matrix<Out, Eigen::Dynamic, Eigen::Dynamic> impl;
    ASSERT_NO_THROW((impl = cast_matrix<Out, In>(A)))
        << "implementation threw: " << desc;

    Eigen::Matrix<Out, Eigen::Dynamic, Eigen::Dynamic> golden;
    ASSERT_NO_THROW((golden = golden_reference_cast<Out, In>(A)))
        << "golden threw: " << desc;

    EXPECT_EQ(impl.rows(), golden.rows()) << "row mismatch: " << desc;
    EXPECT_EQ(impl.cols(), golden.cols()) << "col mismatch: " << desc;

    for (int i = 0; i < impl.size(); ++i) {
      EXPECT_EQ(impl.data()[i], golden.data()[i])
          << "value mismatch at " << i << " (" << desc << ")";
    }
  }
};

TEST_F(CastTest, FloatToDouble) {
  auto A = generate_random_matrix<float>(3, 4, -1.0f, 1.0f);
  test_with_golden<double, float>(A, "float->double");
}

TEST_F(CastTest, DoubleToInt) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A(2, 3);
  A << 1.1, -2.9, 3.5, -4.2, 5.0, 6.7;
  test_with_golden<int, double>(A, "double->int");
}

TEST_F(CastTest, IntToFloatRandom) {
  auto A = generate_random_matrix<int>(5, 5, -10, 10);
  test_with_golden<float, int>(A, "int->float");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
