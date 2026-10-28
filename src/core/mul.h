#ifndef MUL_H
#define MUL_H

#include <Eigen/Dense>
#include <stdexcept>
#include <string>

/**
 * @brief Perform element-wise multiplication using Eigen library (ONNX `Mul`).
 *
 * @tparam Scalar Data type (float, double, etc.)
 * @param A Left matrix (m x n)
 * @param B Right matrix (m x n)
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Result matrix
 * (m x n)
 */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> mul(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    throw std::invalid_argument(
        "mul: A = " + std::to_string(A.rows()) + "x" +
        std::to_string(A.cols()) + ", B = " + std::to_string(B.rows()) + "x" +
        std::to_string(B.cols()) + ", expected same shape");
  }
  return A.cwiseProduct(B);
}

#endif  // MUL_H
