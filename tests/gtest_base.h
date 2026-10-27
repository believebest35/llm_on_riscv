#ifndef TESTS_GTEST_BASE_H_
#define TESTS_GTEST_BASE_H_

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

class GTestBase : public ::testing::Test {
 protected:
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

  // Generate `count` integer indices in [min_val, max_val]. Negative values are
  // allowed; ONNX operators normalize them from the end of the axis.
  template <typename IndexType>
  Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic>
  generate_random_indices(int count, int min_val, int max_val) {
    const Eigen::VectorXf raw = Eigen::VectorXf::Random(count);
    Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic> indices(1, count);
    for (int i = 0; i < count; ++i) {
      const float scaled = (raw(i) + 1.0f) * 0.5f * (max_val - min_val + 1);
      const int value = min_val + static_cast<int>(scaled);
      indices(0, i) = static_cast<IndexType>(std::min(max_val, value));
    }
    return indices;
  }

  template <typename Scalar>
  Scalar calculate_l2_difference(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
      throw std::invalid_argument("Matrices must have same dimensions");
    }
    return (A - B).norm();
  }

  template <typename Scalar>
  Scalar calculate_max_abs_difference(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
      throw std::invalid_argument("Matrices must have same dimensions");
    }

    const int rows = static_cast<int>(A.rows());
    const int cols = static_cast<int>(A.cols());
    Scalar max_diff = static_cast<Scalar>(0);

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        Scalar diff = std::abs(A(i, j) - B(i, j));
        if (diff > max_diff) {
          max_diff = diff;
        }
      }
    }
    return max_diff;
  }

  template <typename Scalar>
  Scalar calculate_rmse(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
      throw std::invalid_argument("Matrices must have same dimensions");
    }

    const int rows = static_cast<int>(A.rows());
    const int cols = static_cast<int>(A.cols());
    Scalar sum_squared_error = static_cast<Scalar>(0);
    const int total_elements = rows * cols;

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        Scalar diff = A(i, j) - B(i, j);
        sum_squared_error += diff * diff;
      }
    }
    return std::sqrt(sum_squared_error / total_elements);
  }

  // Assert that `actual` matches `expected` in shape and is numerically close,
  // then print the standard difference metrics for the case.
  template <typename Scalar>
  void expect_matrix_near(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& actual,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& expected,
      double tolerance, const std::string& description = "") {
    EXPECT_EQ(actual.rows(), expected.rows())
        << "Row count mismatch: " << description;
    EXPECT_EQ(actual.cols(), expected.cols())
        << "Column count mismatch: " << description;

    const Scalar l2_diff = calculate_l2_difference(actual, expected);
    EXPECT_LE(l2_diff, tolerance)
        << "L2 difference exceeds tolerance: " << description;

    std::cout << description << "\nMax absolute difference: "
              << calculate_max_abs_difference(actual, expected)
              << "\nRMSE: " << calculate_rmse(actual, expected)
              << "\nL2 difference: " << l2_diff << std::endl;
  }
};

#endif  // TESTS_GTEST_BASE_H_
