#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "core/skip_simplified_layernorm.h"

class SkipSimplifiedLayerNormTest : public ::testing::Test {
 protected:
  void SetUp() override { std::srand(42); }

  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> generate_random_matrix(
      int rows, int cols, Scalar min_val = static_cast<Scalar>(-1.0),
      Scalar max_val = static_cast<Scalar>(1.0)) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> mat(rows, cols);
    mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Random(rows,
                                                                        cols);
    mat = (mat.array() + static_cast<Scalar>(1.0)) / static_cast<Scalar>(2.0) *
              (max_val - min_val) +
          min_val;
    return mat;
  }

  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> generate_random_vector(
      int size, Scalar min_val = static_cast<Scalar>(-1.0),
      Scalar max_val = static_cast<Scalar>(1.0)) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> vec(size);
    vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Random(size);
    vec = (vec.array() + static_cast<Scalar>(1.0)) / static_cast<Scalar>(2.0) *
              (max_val - min_val) +
          min_val;
    return vec;
  }

  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_reference(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& skip,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& weight, Scalar epsilon) {
    if (X.rows() != skip.rows() || X.cols() != skip.cols()) {
      throw std::invalid_argument("dimension mismatch");
    }
    if (X.cols() != weight.size()) {
      throw std::invalid_argument("dimension mismatch");
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y(X.rows(), X.cols());
    for (int i = 0; i < X.rows(); ++i) {
      Scalar sum_sq = static_cast<Scalar>(0);
      for (int j = 0; j < X.cols(); ++j) {
        const Scalar v = X(i, j) + skip(i, j);
        sum_sq += v * v;
      }
      const Scalar mean_sq = sum_sq / static_cast<Scalar>(X.cols());
      const Scalar rms = std::sqrt(mean_sq + epsilon);

      for (int j = 0; j < X.cols(); ++j) {
        Y(i, j) = ((X(i, j) + skip(i, j)) / rms) * weight(j);
      }
    }
    return Y;
  }

  template <typename Scalar>
  Scalar l2_norm_diff(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
    return (A - B).norm();
  }
};

TEST_F(SkipSimplifiedLayerNormTest, RandomMatricesMatchGolden) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1}, {2, 3}, {4, 8}, {64, 128}, {128, 256}, {256, 512}, {1024, 128},
  };

  const float epsilon = 1e-5f;
  const float tolerance = 1e-4f;

  for (const auto& [rows, cols] : test_cases) {
    auto X = generate_random_matrix<float>(rows, cols, -1.0f, 1.0f);
    auto skip = generate_random_matrix<float>(rows, cols, -0.5f, 0.5f);
    auto w = generate_random_vector<float>(cols, 0.5f, 1.5f);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> y_impl;
    ASSERT_NO_THROW(
        y_impl = skip_simplified_layer_normalization(X, skip, w, epsilon))
        << "impl threw for shape " << rows << "x" << cols;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> y_gold;
    ASSERT_NO_THROW(y_gold = golden_reference(X, skip, w, epsilon))
        << "golden threw for shape " << rows << "x" << cols;

    EXPECT_EQ(y_impl.rows(), y_gold.rows());
    EXPECT_EQ(y_impl.cols(), y_gold.cols());

    const float l2 = l2_norm_diff(y_impl, y_gold);
    EXPECT_LE(l2, tolerance) << "L2 diff too large for shape " << rows << "x"
                             << cols << ", l2=" << l2;
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
