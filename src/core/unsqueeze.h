#ifndef UNSQUEEZE_H
#define UNSQUEEZE_H

#include <Eigen/Dense>
#include <stdexcept>

/**
 * @brief "Unsqueeze" a 1-D vector by inserting a new dimension of size 1.
 *
 * Input data is a 1-D vector of size N.
 * unsqueeze inserts a new dimension of size 1 at the specified axes,
 * producing a 2-D output without changing data order.
 *
 * axes == 0: N-element vector → 1 x N row vector
 * axes == 1: N-element vector → N x 1 column vector
 *
 * @tparam Scalar Element type.
 * @param data Input 1-D vector of size N.
 * @param axes Axis at which to insert the new dimension: 0 or 1.
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Resulting
 *         2-D matrix (1xN or Nx1).
 * @throws std::invalid_argument if axes is not 0 or 1.
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> unsqueeze(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& data, int axes) {
  if (axes != 0 && axes != 1) {
    throw std::invalid_argument("unsqueeze: axes must be 0 or 1");
  }

  const int N = data.size();
  if (N == 0) {
    throw std::invalid_argument("unsqueeze: input vector cannot be empty");
  }

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
