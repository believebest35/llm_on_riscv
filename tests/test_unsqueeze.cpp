#include <gtest/gtest.h>

#include <string>

#include "core/unsqueeze.h"
#include "gtest_base.h"

class UnsqueezeTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_unsqueeze(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& data,
      int axes) {
    const int N = data.size();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> expanded;
    if (axes == 0) {
      expanded.resize(1, N);
      expanded.row(0) = data.transpose();
    } else {
      expanded.resize(N, 1);
      expanded.col(0) = data;
    }
    return expanded;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& data, int axes,
      const std::string& desc = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl;
    ASSERT_NO_THROW(impl = unsqueeze<Scalar>(data, axes))
        << "implementation threw: " << desc;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden;
    ASSERT_NO_THROW(golden = golden_reference_unsqueeze<Scalar>(data, axes))
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

TEST_F(UnsqueezeTest, Axes0) {
  Eigen::Matrix<float, Eigen::Dynamic, 1> data(4);
  data << 1, 2, 3, 4;
  test_with_golden<float>(data, 0, "axes0");
}

TEST_F(UnsqueezeTest, Axes1) {
  Eigen::Matrix<float, Eigen::Dynamic, 1> data(4);
  data << 1, 2, 3, 4;
  test_with_golden<float>(data, 1, "axes1");
}

TEST_F(UnsqueezeTest, Random) {
  std::vector<int> sizes = {1, 5, 10, 100};
  for (int n : sizes) {
    auto v = generate_random_vector<float>(n, -2.0f, 2.0f);
    test_with_golden<float>(v, 0, "rand axes0");
    test_with_golden<float>(v, 1, "rand axes1");
  }
}

TEST_F(UnsqueezeTest, InvalidAxes) {
  Eigen::Matrix<float, Eigen::Dynamic, 1> data(5);
  data.setRandom();
  EXPECT_THROW(unsqueeze<float>(data, 2), std::invalid_argument);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
