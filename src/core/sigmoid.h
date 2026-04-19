#ifndef SIGMOID_H
#define SIGMOID_H

#include <Eigen/Dense>

/**
 * @brief Apply element-wise sigmoid function to a matrix.
 *
 * Computes 1 / (1 + exp(-x)) for each entry of the input matrix.
 * Works with any scalar type that supports std::exp.
 *
 * @tparam Scalar Element type of the matrix.
 * @param X Input matrix.
 * @return Y Output matrix of the same shape with sigmoid applied element-wise.
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> sigmoid(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y = X;
  for (std::int64_t i = 0; i < Y.size(); ++i) {
    Y.data()[i] = static_cast<Scalar>(1) /
                    (static_cast<Scalar>(1) + std::exp(-Y.data()[i]));
  }
  return Y;
}

#endif  // SIGMOID_H
