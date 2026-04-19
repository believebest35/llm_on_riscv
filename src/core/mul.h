#ifndef MUL_H
#define MUL_H

#include <Eigen/Dense>
#include <stdexcept>

/**
 * @brief Perform element-wise multiplication using Eigen library
 *
 * @tparam Scalar Data type (float, double, etc.)
 * @param A Left matrix (m x n)
 * @param B Right matrix (m x n)
 * @return C Result matrix (m x n)
 */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix_multiply_elementwise(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
  // Validate input dimensions
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    throw std::invalid_argument(
        "Matrix dimensions incompatible for element-wise multiplication: "
        "A.rows() = " +
        std::to_string(A.rows()) + ", A.cols() = " + std::to_string(A.cols()) +
        ", B.rows() = " + std::to_string(B.rows()) +
        ", B.cols() = " + std::to_string(B.cols()));
  }

  // Perform element-wise multiplication using Eigen's optimized implementation
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> C = A.cwiseProduct(B);
  return C;
}

#endif  // MUL_H
