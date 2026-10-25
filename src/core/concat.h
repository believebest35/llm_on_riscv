#ifndef CONCAT_H
#define CONCAT_H

#include <Eigen/Dense>
#include <stdexcept>
#include <string>

/**
 * @brief Concatenate two matrices along a specified axis (ONNX `Concat`).
 *
 * @tparam Scalar Element type stored in the matrices.
 * @param A First input matrix.
 * @param B Second input matrix.
 * @param axis Axis to concatenate along: 0 stacks rows, 1 appends columns;
 *        negative values count from the back.
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Result of
 *         concatenation.
 */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> concat(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B,
    int axis = 0) {
  if (axis < -2 || axis > 1) {
    throw std::invalid_argument("concat: axis = " + std::to_string(axis) +
                                ", expected in [-2, 1]");
  }
  if (axis < 0) {
    axis += 2;  // 2-D data: -1 -> 1, -2 -> 0
  }

  const int a_rows = static_cast<int>(A.rows());
  const int a_cols = static_cast<int>(A.cols());
  const int b_rows = static_cast<int>(B.rows());
  const int b_cols = static_cast<int>(B.cols());

  if (axis == 0) {
    // Axis 0 stacks rows, so the column counts must match.
    if (a_cols != b_cols) {
      throw std::invalid_argument(
          "concat: A.cols() = " + std::to_string(a_cols) +
          ", B.cols() = " + std::to_string(b_cols) + ", expected equal");
    }
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(
        a_rows + b_rows, a_cols);
    output << A, B;
    return output;
  } else {
    // Axis 1 appends columns, so the row counts must match.
    if (a_rows != b_rows) {
      throw std::invalid_argument(
          "concat: A.rows() = " + std::to_string(a_rows) +
          ", B.rows() = " + std::to_string(b_rows) + ", expected equal");
    }
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(
        a_rows, a_cols + b_cols);
    output << A, B;
    return output;
  }
}

#endif  // CONCAT_H
