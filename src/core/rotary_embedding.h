#ifndef ROTARY_EMBEDDING_H
#define ROTARY_EMBEDDING_H

#include <Eigen/Dense>
#include <stdexcept>

/**
 * @brief Apply rotary positional embeddings to a matrix.
 *
 * This helper implements the elementary rotation used in many transformer
 * models (sometimes called RoPE).  The input matrix is interpreted as a
 * sequence of 2-element feature pairs, and a corresponding pair of cosine and
 * sine factors is used to rotate each pair:
 *
 *   [x_0, x_1] -> [x_0 * cos - x_1 * sin,
 *                 x_0 * sin + x_1 * cos]
 *
 * The operation is performed per-row.  The cosine and sine factors may be
 * provided in the same shape as the input (typically each pair of columns has
 * identical cos/sin values), or they can be broadcast if one of the dimensions
 * is 1.
 *
 * This implementation is intentionally simple and works with 2-D matrices
 * only.  It requires that the number of columns be even and that the provided
 * cos/sin matrices be compatible for element-wise arithmetic.
 *
 * @tparam Scalar Element type (float/double/etc.).
 * @param data Input matrix of shape (rows x cols) with cols even.
 * @param cosines Matrix of same shape (or broadcastable) containing cosine
 *        coefficients.
 * @param sines Matrix of same shape (or broadcastable) containing sine
 *        coefficients.
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Rotated
 *         matrix of the same shape as `data`.
 * @throws std::invalid_argument if `cols` is odd or shapes incompatible.
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> rotary_embedding(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& data,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cosines,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& sines) {
  const int rows = data.rows();
  const int cols = data.cols();
  if (cols % 2 != 0) {
    throw std::invalid_argument(
        "rotary_embedding: number of columns must be even");
  }
  if ((cosines.rows() != rows && cosines.rows() != 1) ||
      (cosines.cols() != cols && cosines.cols() != 1)) {
    throw std::invalid_argument("rotary_embedding: cosines shape incompatible");
  }
  if ((sines.rows() != rows && sines.rows() != 1) ||
      (sines.cols() != cols && sines.cols() != 1)) {
    throw std::invalid_argument("rotary_embedding: sines shape incompatible");
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; j += 2) {
      Scalar x0 = data(i, j);
      Scalar x1 = data(i, j + 1);
      // fetch cos/sin with broadcasting rules
      Scalar c = cosines(i, j);
      Scalar s = sines(i, j);
      out(i, j) = x0 * c - x1 * s;
      out(i, j + 1) = x0 * s + x1 * c;
    }
  }
  return out;
}

#endif  // ROTARY_EMBEDDING_H
