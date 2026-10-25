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
 * Every input is valid; this kernel does not throw.
 *
 * @tparam Scalar Element type of the input matrix.
 * @param input Input matrix whose shape is queried.
 * @return Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic>
 *         Column vector containing [rows, cols].
 */
template <typename Scalar>
Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> shape(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input) {
  Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> output(2, 1);
  output(0, 0) = static_cast<std::int64_t>(input.rows());
  output(1, 0) = static_cast<std::int64_t>(input.cols());
  return output;
}

#endif  // SHAPE_H
