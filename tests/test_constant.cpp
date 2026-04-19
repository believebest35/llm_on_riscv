#include <gtest/gtest.h>

#include <string>

#include "core/constant.h"
#include "gtest_base.h"

class ConstantTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_constant(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& value) {
    return value;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& value,
                        const std::string& desc = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result = constant_matrix<Scalar>(value))
        << "Implementation threw: " << desc;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(
        golden_result = golden_reference_constant<Scalar>(value))
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
  const std::vector<Eigen::MatrixXf> cases = {
      (Eigen::MatrixXf(1, 1) << 0.0f).finished(),
      (Eigen::MatrixXf(2, 3) << 1.0f, -2.0f, 3.5f, 4.5f, 0.0f, -6.0f)
          .finished(),
      (Eigen::MatrixXf::Random(8, 8) * 10.0f),
      (Eigen::MatrixXf::Constant(64, 1, 2.0f)),
      (Eigen::MatrixXf::Constant(1, 64, -3.14f)),
      (Eigen::MatrixXf::Constant(128, 128, 7.77f)),
  };

  for (const auto& value : cases) {
    std::ostringstream ss;
    ss << "constant " << value.rows() << "x" << value.cols();
    test_with_golden<float>(value, ss.str());
  }
}

TEST_F(ConstantTest, EmptyMatrix) {
  const Eigen::MatrixXf value(0, 0);
  EXPECT_NO_THROW({
    const auto output = constant_matrix<float>(value);
    EXPECT_EQ(output.rows(), 0);
    EXPECT_EQ(output.cols(), 0);
  });
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
