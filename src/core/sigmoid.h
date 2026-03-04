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
 * @param A Input matrix.
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix of the
 *         same shape with sigmoid applied element-wise.
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> sigmoid(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out = A;
  for (std::int64_t i = 0; i < out.size(); ++i) {
    out.data()[i] = static_cast<Scalar>(1) /
                    (static_cast<Scalar>(1) + std::exp(-out.data()[i]));
  }
  return out;
}

#endif  // SIGMOID_H
