#ifndef MATMUL_H
#define MATMUL_H

#include <Eigen/Dense>
#include <stdexcept>

/**
 * @brief Perform matrix multiplication using Eigen library.
 *
 * @tparam Scalar Data type (float, double, etc.)
 * @param A Left matrix (m x n)
 * @param B Right matrix (n x p)
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Result matrix
 * (m x p)
 */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matmul(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
  if (A.cols() != B.rows()) {
    throw std::invalid_argument(
        "matmul: A.cols() = " + std::to_string(A.cols()) +
        ", B.rows() = " + std::to_string(B.rows()));
  }

  return A * B;
}

#endif  // MATMUL_H