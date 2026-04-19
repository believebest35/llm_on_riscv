#ifndef SUB_H
#define SUB_H

#include <Eigen/Dense>
#include <stdexcept>

/**
 * @brief Perform element-wise subtraction using Eigen library
 *
 * @tparam Scalar Data type (float, double, etc.)
 * @param A Left matrix (m x n)
 * @param B Right matrix (m x n)
 * @return C Result matrix (m x n) containing A - B.
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix_subtract_elementwise(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    throw std::invalid_argument(
        "Matrix dimensions incompatible for element-wise subtraction: "
        "A.rows() = " +
        std::to_string(A.rows()) + ", A.cols() = " + std::to_string(A.cols()) +
        ", B.rows() = " + std::to_string(B.rows()) +
        ", B.cols() = " + std::to_string(B.cols()));
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> C = A - B;
  return C;
}

#endif  // SUB_H
