#include <gtest/gtest.h>

#include <string>

#include "core/rotary_embedding.h"
#include "gtest_base.h"

class RotaryEmbeddingTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_reference_rope(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& data,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cos,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& sin) {
    // same algorithm as in implementation
    const int rows = data.rows();
    const int cols = data.cols();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(rows, cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; j += 2) {
        Scalar x0 = data(i, j);
        Scalar x1 = data(i, j + 1);
        Scalar c = cos(i, j);
        Scalar s = sin(i, j);
        out(i, j) = x0 * c - x1 * s;
        out(i, j + 1) = x0 * s + x1 * c;
      }
    }
    return out;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& data,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cos,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& sin,
      const std::string& desc = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl;
    ASSERT_NO_THROW(impl = rotary_embedding<Scalar>(data, cos, sin))
        << "implementation threw: " << desc;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden;
    ASSERT_NO_THROW(golden = golden_reference_rope<Scalar>(data, cos, sin))
        << "golden threw: " << desc;

    EXPECT_EQ(impl.rows(), golden.rows()) << "row mismatch: " << desc;
    EXPECT_EQ(impl.cols(), golden.cols()) << "col mismatch: " << desc;

    for (int i = 0; i < impl.size(); ++i) {
      EXPECT_NEAR(impl.data()[i], golden.data()[i], 1e-6)
          << "value mismatch at " << i << " (" << desc << ")";
    }
  }
};

TEST_F(RotaryEmbeddingTest, PairwiseSimple) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> d(1, 4);
  d << 1, 2, 3, 4;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> c(1, 4);
  c << 0, 0, 1, 1;  // with broadcasting, only first element used per pair
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> s(1, 4);
  s << 1, 1, 0, 0;
  test_with_golden<float>(d, c, s, "simple pairs");
}

TEST_F(RotaryEmbeddingTest, RandomSmall) {
  auto data = generate_random_matrix<float>(2, 6, -1, 1);
  auto cos = generate_random_matrix<float>(2, 6, -1, 1);
  auto sin = generate_random_matrix<float>(2, 6, -1, 1);
  // force even dimension
  cos = cos.array().abs();
  sin = sin.array().abs();
  test_with_golden<float>(data, cos, sin, "random");
}

TEST_F(RotaryEmbeddingTest, Broadcasting) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> data(3, 2);
  data.setRandom();
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> cos(1, 2);
  cos << 0.5f, 0.5f;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> sin(1, 2);
  sin << 0.5f, 0.5f;
  test_with_golden<float>(data, cos, sin, "broadcast row");
}

TEST_F(RotaryEmbeddingTest, ErrorCases) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> data(1, 3);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> cos(1, 3);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> sin(1, 3);
  EXPECT_THROW(rotary_embedding<float>(data, cos, sin), std::invalid_argument);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> data2(2, 2);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> cos2(2, 1);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> sin2(3, 2);
  EXPECT_THROW(rotary_embedding<float>(data2, cos2, sin2),
               std::invalid_argument);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
