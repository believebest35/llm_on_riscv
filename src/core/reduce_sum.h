#ifndef REDUCE_SUM_H
#define REDUCE_SUM_H

#include <Eigen/Dense>
#include <stdexcept>

/**
 * @brief Compute the sum over one axis of a 2-D matrix.
 *
 * This is a simplified version of the ONNX `ReduceSum` operator limited to
 * two-dimensional tensors.  Only axes 0 or 1 are supported.  The resulting
 * matrix has one dimension equal to 1 and the other equal to the corresponding
 * dimension of the input.
 *
 * When `axis == 0` the output shape is (1 x cols) and each element is the sum
 * of the elements in the corresponding column of the input.  When `axis == 1`
 * the output shape is (rows x 1) and each element is the sum of the elements
 * in the corresponding row.
 *
 * @tparam Scalar Type of elements stored in the input matrix.
 * @param A Input matrix of size (rows x cols).
 * @param axis Reduction axis (0 or 1).  Defaults to 1.
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Resulting
 *         summed matrix.
 * @throws std::invalid_argument if an invalid axis is provided.
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> reduce_sum(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    int axis = 1) {
  const int rows = A.rows();
  const int cols = A.cols();

  if (axis == 0) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(1, cols);
    result.setZero();
    for (int i = 0; i < rows; ++i) {
      result += A.row(i);
    }
    return result;
  } else if (axis == 1) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(rows, 1);
    result.setZero();
    for (int i = 0; i < rows; ++i) {
      // row(i) returns 1xcols; sum() collapses to scalar
      result(i, 0) = A.row(i).sum();
    }
    return result;
  } else {
    throw std::invalid_argument("reduce_sum: axis must be 0 or 1");
  }
}

#endif  // REDUCE_SUM_H
