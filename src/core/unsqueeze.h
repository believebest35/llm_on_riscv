#ifndef UNSQUEEZE_H
#define UNSQUEEZE_H

#include <Eigen/Dense>
#include <stdexcept>

/**
 * @brief "Unsqueeze" a 1-D vector by inserting a new dimension of size 1
 * (ONNX `Unsqueeze`).
 *
 * @tparam Scalar Element type.
 * @param data Input 1-D vector of size N.
 * @param axes Axis at which to insert the new dimension: 0 gives a 1 x N row
 *        vector, 1 gives an N x 1 column vector.
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Resulting
 *         2-D matrix (1xN or Nx1).
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> unsqueeze(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& data, int axes) {
  if (axes != 0 && axes != 1) {
    throw std::invalid_argument("unsqueeze: axes must be 0 or 1");
  }

  const int N = data.size();
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> expanded;
  if (axes == 0) {
    expanded.resize(1, N);
    expanded.row(0) = data.transpose();
  } else {
    expanded.resize(N, 1);
    expanded.col(0) = data;
  }
  return expanded;
}

#endif  // UNSQUEEZE_H
