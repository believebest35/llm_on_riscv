#ifndef CONCAT_H
#define CONCAT_H

#include <Eigen/Dense>
#include <stdexcept>

/**
 * @brief Concatenate two matrices along a specified axis.
 *
 * A simple helper matching ONNX `Concat` semantics for two inputs and
 * two-dimensional tensors.  Only axes 0 (row-wise) and 1 (column-wise) are
 * supported.  The matrices must agree in size along the non-concatenation
 * axis.
 *
 * @tparam Scalar Element type stored in the matrices.
 * @param A First input matrix.
 * @param B Second input matrix.
 * @param axis Axis to concatenate along: 0 for vertical (stack rows), 1 for
 *             horizontal (append columns).
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Result of
 *         concatenation.
 * @throws std::invalid_argument if axis is invalid or dimensions mismatch.
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> concat(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B,
    int axis = 0) {
  if (axis != 0 && axis != 1) {
    throw std::invalid_argument("concat: axis must be 0 or 1");
  }

  if (axis == 0) {
    // stack rows: columns must match
    if (A.cols() != B.cols()) {
      throw std::invalid_argument("concat: column count mismatch");
    }
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(
        A.rows() + B.rows(), A.cols());
    out << A, B;
    return out;
  } else {
    // axis == 1
    if (A.rows() != B.rows()) {
      throw std::invalid_argument("concat: row count mismatch");
    }
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(
        A.rows(), A.cols() + B.cols());
    out << A, B;
    return out;
  }
}

#endif  // CONCAT_H
