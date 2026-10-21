#ifndef SHAPE_H
#define SHAPE_H

#include <Eigen/Dense>
#include <cstdint>

/**
 * @brief Compute the shape of a matrix and return it as an int64 vector.
 *
 * The result is a 2x1 column vector where the first element is the number of
 * rows and the second element is the number of columns of the input matrix.
 * ONNX `Shape` always outputs int64, regardless of the input element type.
 *
 * @tparam Scalar Element type of the input matrix.
 * @param A Input matrix whose shape is queried.
 * @return Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic>
 *         Column vector containing [rows, cols].
 */

template <typename Scalar>
Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> shape(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A) {
  Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> result(2, 1);
  result(0, 0) = static_cast<std::int64_t>(A.rows());
  result(1, 0) = static_cast<std::int64_t>(A.cols());
  return result;
}

#endif  // SHAPE_H
