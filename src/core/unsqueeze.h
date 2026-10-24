#ifndef UNSQUEEZE_H
#define UNSQUEEZE_H

#include <Eigen/Dense>
#include <stdexcept>

/**
 * @brief Unsqueeze a scalar or 1-D tensor stored as an Eigen matrix/vector
 * (ONNX `Unsqueeze`).
 *
 * Rank-0/1 tensors are represented as 2-D matrices here: a scalar is 1x1, a
 * vector is Nx1 or 1xN. The result is 1xN for axes=0 and Nx1 for axes=1, which
 * keeps the flattened representation consistent with the rest of the kernels.
 *
 * @tparam Scalar Element type of `data`.
 * @param data Input scalar (1x1) or 1-D vector (Nx1 or 1xN).
 * @param axes Axis at which to insert the new dimension: 0 gives a 1xN row
 *        vector, 1 gives an Nx1 column vector.
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Resulting
 *         2-D matrix (1xN or Nx1).
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> unsqueeze(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& data,
    int axes) {
  if (axes != 0 && axes != 1) {
    throw std::invalid_argument("unsqueeze: axes must be 0 or 1");
  }
  if (data.rows() != 1 && data.cols() != 1) {
    throw std::invalid_argument(
        "unsqueeze: input must be a scalar or 1-D vector");
  }

  const int rows = static_cast<int>(data.rows());
  const int cols = static_cast<int>(data.cols());
  const int N = rows * cols;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> expanded;
  if (axes == 0) {
    expanded.resize(1, N);
  } else {
    expanded.resize(N, 1);
  }
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const int k = i * cols + j;
      if (axes == 0) {
        expanded(0, k) = data(i, j);
      } else {
        expanded(k, 0) = data(i, j);
      }
    }
  }
  return expanded;
}

#endif  // UNSQUEEZE_H
