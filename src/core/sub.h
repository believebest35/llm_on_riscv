#ifndef SUB_H
#define SUB_H

#include <Eigen/Dense>
#include <algorithm>
#include <stdexcept>

/**
 * @brief Perform element-wise subtraction with Numpy-style broadcasting.
 *
 * Both operands are 2-D matrices. A dimension of size 1 is broadcast to the
 * other operand's size, matching ONNX `Sub` for 2-D tensors.
 *
 * @tparam Scalar Data type (float, double, etc.)
 * @param A Left matrix (m x n)
 * @param B Right matrix (m x n), broadcastable to A
 * @return C Result matrix containing A - B.
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix_subtract_elementwise(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
  const Eigen::Index rows = std::max(A.rows(), B.rows());
  const Eigen::Index cols = std::max(A.cols(), B.cols());
  const bool a_broadcastable =
      (A.rows() == rows || A.rows() == 1) &&
      (A.cols() == cols || A.cols() == 1);
  const bool b_broadcastable =
      (B.rows() == rows || B.rows() == 1) &&
      (B.cols() == cols || B.cols() == 1);
  if (!a_broadcastable || !b_broadcastable) {
    throw std::invalid_argument(
        "Matrix dimensions incompatible for element-wise subtraction: "
        "A.rows() = " +
        std::to_string(A.rows()) + ", A.cols() = " + std::to_string(A.cols()) +
        ", B.rows() = " + std::to_string(B.rows()) +
        ", B.cols() = " + std::to_string(B.cols()));
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> C(rows, cols);
  for (Eigen::Index i = 0; i < rows; ++i) {
    for (Eigen::Index j = 0; j < cols; ++j) {
      const Eigen::Index ai = (A.rows() == 1) ? 0 : i;
      const Eigen::Index aj = (A.cols() == 1) ? 0 : j;
      const Eigen::Index bi = (B.rows() == 1) ? 0 : i;
      const Eigen::Index bj = (B.cols() == 1) ? 0 : j;
      C(i, j) = A(ai, aj) - B(bi, bj);
    }
  }
  return C;
}

#endif  // SUB_H
