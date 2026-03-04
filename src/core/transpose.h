#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include <Eigen/Dense>

/**
 * @brief Return the transpose of a matrix.
 *
 * A thin wrapper around Eigen's `transpose()` method; kept as a separate
 * helper to mirror ONNX `Transpose` semantics and to provide a consistent
 * interface for the unit tests.  Works for any scalar type.
 *
 * @tparam Scalar Type of the matrix elements.
 * @param A Input matrix of size (m x n).
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Transposed
 *         matrix of size (n x m).
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> transpose(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A) {
  // Eigen's transpose returns an expression; eval() yields an owned matrix.
  return A.transpose();
}

#endif  // TRANSPOSE_H
