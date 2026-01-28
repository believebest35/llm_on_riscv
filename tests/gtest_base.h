#ifndef TESTS_GTEST_BASE_H_
#define TESTS_GTEST_BASE_H_

#include <gtest/gtest.h>

#include <cmath>
#include <string>

class GTestBase : public ::testing::Test {
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

    Scalar max_diff = static_cast<Scalar>(0);

    for (int i = 0; i < A.rows(); ++i) {
      for (int j = 0; j < A.cols(); ++j) {
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

    Scalar sum_squared_error = static_cast<Scalar>(0);
    int total_elements = A.rows() * A.cols();

    for (int i = 0; i < A.rows(); ++i) {
      for (int j = 0; j < A.cols(); ++j) {
        Scalar diff = A(i, j) - B(i, j);
        sum_squared_error += diff * diff;
      }
    }

    return std::sqrt(sum_squared_error / total_elements);
  }
};

#endif  // TESTS_GTEST_BASE_H_
