#ifndef UNSQUEEZE_H
#define UNSQUEEZE_H

#include <Eigen/Dense>
#include <stdexcept>

/**
 * @brief "Unsqueeze" a matrix by inserting a new dimension of size 1.
 *
 * For the purposes of this lightweight helper we only support 2‑D input
 * matrices.  The new dimension is added either before or after the flattened
 * data depending on the requested axis.  This is not a full generalization of
 * ONNX Unsqueeze, but it is sufficient for simple scalar-to-vector or
 * vector-to-matrix tests and mirrors the limited uses seen in current
 * profiling output.
 *
 * The output is still a 2‑D matrix.  When `axis == 0` the input is flattened
 * row-major and returned as a 1 x N row vector.  When `axis == 1` the input is
 * flattened and returned as an N x 1 column vector.
 *
 * @tparam Scalar Element type stored in the matrix.
 * @param A Input matrix of arbitrary dimensions.
 * @param axis Axis at which to insert the new dimension: 0 or 1.
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Resulting
 *         2-D matrix with a singleton first or second dimension.
 * @throws std::invalid_argument if axis is not 0 or 1.
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> unsqueeze(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A, int axis) {
  if (axis != 0 && axis != 1) {
    throw std::invalid_argument("unsqueeze: axis must be 0 or 1");
  }

  const int total = static_cast<int>(A.size());
  if (axis == 0) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(1, total);
    for (int i = 0; i < total; ++i) {
      out(0, i) = A.data()[i];
    }
    return out;
  } else {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(total, 1);
    for (int i = 0; i < total; ++i) {
      out(i, 0) = A.data()[i];
    }
    return out;
  }
}

#endif  // UNSQUEEZE_H
