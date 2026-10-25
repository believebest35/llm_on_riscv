#ifndef UNSQUEEZE_H
#define UNSQUEEZE_H

#include <Eigen/Dense>
#include <stdexcept>
#include <string>

/**
 * @brief Unsqueeze a scalar or 1-D tensor stored as an Eigen matrix/vector
 * (ONNX `Unsqueeze`).
 *
 * Rank-0/1 tensors are represented as 2-D matrices here: a scalar is 1x1, a
 * vector is Nx1 or 1xN. The result is 1xN for axis=0 and Nx1 for axis=1, which
 * keeps the flattened representation consistent with the rest of the kernels.
 *
 * @tparam Scalar Element type of `input`.
 * @param input Input scalar (1x1) or 1-D vector (Nx1 or 1xN).
 * @param axis Axis at which to insert the new dimension: 0 gives a 1xN row
 *        vector, 1 gives an Nx1 column vector; negative values count from the
 *        back.
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Resulting
 *         2-D matrix (1xN or Nx1).
 */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> unsqueeze(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
    int axis) {
  if (axis < -2 || axis > 1) {
    throw std::invalid_argument("unsqueeze: axis = " + std::to_string(axis) +
                                ", expected in [-2, 1]");
  }
  if (axis < 0) {
    axis += 2;  // 2-D data: -1 -> 1, -2 -> 0
  }
  if (input.rows() != 1 && input.cols() != 1) {
    throw std::invalid_argument(
        "unsqueeze: input = " + std::to_string(input.rows()) + "x" +
        std::to_string(input.cols()) + ", expected a scalar or 1-D vector");
  }

  const int rows = static_cast<int>(input.rows());
  const int cols = static_cast<int>(input.cols());
  const int N = rows * cols;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output;
  if (axis == 0) {
    output.resize(1, N);
  } else {
    output.resize(N, 1);
  }
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const int k = i * cols + j;
      if (axis == 0) {
        output(0, k) = input(i, j);
      } else {
        output(k, 0) = input(i, j);
      }
    }
  }
  return output;
}

#endif  // UNSQUEEZE_H
