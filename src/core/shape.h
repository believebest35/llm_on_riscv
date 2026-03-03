#ifndef SHAPE_H
#define SHAPE_H

#include <Eigen/Dense>
#include <stdexcept>

/**
 * @brief Compute the shape of a matrix and return it as a small vector.
 *
 * The result is a column vector of length two where the first element is the
 * number of rows and the second element is the number of columns of the input
 * matrix.  This mirrors the behavior of the ONNX `Shape` operator for 2‑D
 * tensors.
 *
 * @tparam Scalar Data type used for the output values.  This can be any numeric
 *        scalar (float, double, int32, int64, etc.) since the dimensions are
 *        cast to the requested type.
 * @param A Input matrix whose shape is queried.
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Column vector containing
 *         [rows, cols].
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> shape(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> result(2);
  result(0) = static_cast<Scalar>(A.rows());
  result(1) = static_cast<Scalar>(A.cols());
  return result;
}

#endif  // SHAPE_H
