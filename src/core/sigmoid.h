#ifndef SIGMOID_H
#define SIGMOID_H

#include <Eigen/Dense>

/**
 * @brief Apply element-wise sigmoid function to a matrix.
 *
 * Computes 1 / (1 + exp(-x)) for each entry of the input matrix, mirroring
 * the ONNX `Sigmoid` operator.
 *
 * @tparam Scalar Element type of the matrix.
 * @param X Input matrix.
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Output matrix
 * of the same shape with sigmoid applied element-wise.
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> sigmoid(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y(X.rows(), X.cols());
  Y.array() = static_cast<Scalar>(1) /
              (static_cast<Scalar>(1) + (-X).array().exp());
  return Y;
}

#endif  // SIGMOID_H
