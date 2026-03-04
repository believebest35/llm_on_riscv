#ifndef RESHAPE_H
#define RESHAPE_H

#include <Eigen/Dense>
#include <stdexcept>

/**
 * @brief Change the shape of a matrix without altering data order.
 *
 * This utility mirrors ONNX `Reshape` for 2-D tensors only.  The total
 * number of elements in the output shape must equal that of the input; the
 * matrix is interpreted in row-major (Eigen default) order when copying
 * data.
 *
 * @tparam Scalar Element type of the matrix.
 * @param A Input matrix of size (rows x cols).
 * @param new_rows Number of rows for the output matrix.  Must be >=0.
 * @param new_cols Number of columns for the output matrix.  Must be >=0.
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Reshaped
 *         matrix of size (new_rows x new_cols).
 * @throws std::invalid_argument if dimensions are negative or element counts
 *         mismatch.
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
  // Eigen stores in column-major by default, but data() gives column-major
  // linear storage.  We want row-major semantics comporting with typical
  // ONNX reshape.  Copy element-by-element in row-major order.
  for (std::int64_t i = 0; i < total; ++i) {
    // compute source row/col
    std::int64_t r = i / A.cols();
    std::int64_t c = i % A.cols();
    std::int64_t r2 = i / new_cols;
    std::int64_t c2 = i % new_cols;
    out(r2, c2) = A(r, c);
  }
  return out;
}

#endif  // RESHAPE_H
