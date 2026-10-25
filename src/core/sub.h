#ifndef SUB_H
#define SUB_H

#include <Eigen/Dense>
#include <algorithm>
#include <stdexcept>
#include <string>

/**
 * @brief Perform element-wise subtraction with Numpy-style broadcasting.
 *
 * Both operands are 2-D matrices. A dimension of size 1 is broadcast to the
 * other operand's size, matching ONNX `Sub` for 2-D tensors.
 *
 * @tparam Scalar Data type (float, double, etc.)
 * @param A Left matrix (m x n)
 * @param B Right matrix (m x n), broadcastable to A
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Result matrix
 *         containing A - B.
 */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> sub(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
  const int a_rows = static_cast<int>(A.rows());
  const int a_cols = static_cast<int>(A.cols());
  const int b_rows = static_cast<int>(B.rows());
  const int b_cols = static_cast<int>(B.cols());
  const int rows = std::max(a_rows, b_rows);
  const int cols = std::max(a_cols, b_cols);
  const bool a_broadcastable =
      (a_rows == rows || a_rows == 1) && (a_cols == cols || a_cols == 1);
  const bool b_broadcastable =
      (b_rows == rows || b_rows == 1) && (b_cols == cols || b_cols == 1);
  if (!a_broadcastable || !b_broadcastable) {
    throw std::invalid_argument(
        "sub: A = " + std::to_string(A.rows()) + "x" +
        std::to_string(A.cols()) + ", B = " + std::to_string(B.rows()) + "x" +
        std::to_string(B.cols()) + ", expected broadcastable shapes");
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const int ai = (a_rows == 1) ? 0 : i;
      const int aj = (a_cols == 1) ? 0 : j;
      const int bi = (b_rows == 1) ? 0 : i;
      const int bj = (b_cols == 1) ? 0 : j;
      output(i, j) = A(ai, aj) - B(bi, bj);
    }
  }
  return output;
}

#endif  // SUB_H
