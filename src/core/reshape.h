#ifndef RESHAPE_H
#define RESHAPE_H

#include <Eigen/Dense>
#include <stdexcept>

/**
 * @brief Change the shape of a matrix without altering data order (ONNX
 * `Reshape`).
 *
 * @tparam Scalar Element type of the matrix.
 * @param A Input matrix of size (rows x cols).
 * @param new_rows Number of output rows (>= 0).
 * @param new_cols Number of output columns (>= 0).
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Reshaped
 *         matrix (new_rows x new_cols), read/written in row-major order.
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> reshape(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    int new_rows, int new_cols) {
  if (new_rows < 0 || new_cols < 0) {
    throw std::invalid_argument("reshape: dimensions must be non-negative");
  }
  const std::int64_t total = A.size();
  if (static_cast<std::int64_t>(new_rows) * new_cols != total) {
    throw std::invalid_argument("reshape: total size mismatch");
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(new_rows, new_cols);
  // ONNX Reshape reads and writes in row-major order.
  for (std::int64_t i = 0; i < total; ++i) {
    out(i / new_cols, i % new_cols) = A(i / A.cols(), i % A.cols());
  }
  return out;
}

#endif  // RESHAPE_H
