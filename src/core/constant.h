#ifndef CONSTANT_H
#define CONSTANT_H

#include <Eigen/Dense>
#include <stdexcept>

/**
 * @brief Produce a matrix filled with a constant value.
 *
 * This helper mirrors the behavior of the ONNX `Constant` operator for the
 * common case where the constant is a scalar that should be broadcast to a
 * full tensor of a given shape.  It is intentionally simple and relies on
 * Eigen's `setConstant` method for efficiency.
 *
 * @tparam Scalar Type of the returned matrix and the constant value.
 * @param rows Number of rows in the output matrix (must be >= 0).
 * @param cols Number of columns in the output matrix (must be >= 0).
 * @param value Scalar value to fill the matrix with.
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix of
 *         size `(rows x cols)` filled with `value`.
 *
 * @throws std::invalid_argument if `rows` or `cols` is negative.
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> constant_matrix(
    int rows, int cols, Scalar value) {
  if (rows < 0 || cols < 0) {
    throw std::invalid_argument(
        "constant_matrix: dimensions must be non-negative");
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(rows, cols);
  result.setConstant(value);
  return result;
}

#endif  // CONSTANT_H
