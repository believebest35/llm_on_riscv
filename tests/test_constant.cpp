#include <gtest/gtest.h>

#include <string>

#include "core/constant.h"
#include "gtest_base.h"

class ConstantTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_constant(int rows, int cols, Scalar value) {
    if (rows < 0 || cols < 0) {
      throw std::invalid_argument("negative dimension");
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(rows, cols);
    result.setConstant(value);
    return result;
  }

  template <typename Scalar>
  void test_with_golden(int rows, int cols, Scalar value,
                        const std::string& desc = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result = constant_matrix<Scalar>(rows, cols, value))
        << "Implementation threw: " << desc;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result =
                        golden_reference_constant<Scalar>(rows, cols, value))
        << "Golden reference threw: " << desc;

    EXPECT_EQ(impl_result.rows(), golden_result.rows())
        << "row mismatch: " << desc;
    EXPECT_EQ(impl_result.cols(), golden_result.cols())
        << "col mismatch: " << desc;

    const auto numel = impl_result.rows() * impl_result.cols();
    for (int i = 0; i < numel; ++i) {
      EXPECT_EQ(impl_result.data()[i], golden_result.data()[i])
          << "value mismatch at index " << i << " (" << desc << ")";
    }
  }
};

TEST_F(ConstantTest, VariousSizes) {
  const std::vector<std::tuple<int, int, float>> cases = {
      {1, 1, 0.0f},  {3, 5, 1.23f},   {10, 10, -4.56f},
      {64, 1, 2.0f}, {1, 64, -3.14f}, {128, 128, 7.77f},
  };

  for (const auto& [r, c, v] : cases) {
    std::ostringstream ss;
    ss << "constant " << r << "x" << c << " value=" << v;
    test_with_golden<float>(r, c, v, ss.str());
  }
}

TEST_F(ConstantTest, NegativeDimensions) {
  EXPECT_THROW(constant_matrix<float>(-1, 5, 0.0f), std::invalid_argument);
  EXPECT_THROW(constant_matrix<float>(5, -1, 0.0f), std::invalid_argument);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
