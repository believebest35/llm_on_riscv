#ifndef SIGMOID_H
#define SIGMOID_H

#include <Eigen/Dense>

/**
 * @brief Apply element-wise sigmoid function to a matrix (ONNX `Sigmoid`).
 *
 * Computes 1 / (1 + exp(-input)) for each entry of the input matrix, mirroring
 * the ONNX `Sigmoid` operator. Every input is valid; this kernel does not
 * throw.
 *
 * @tparam Scalar Element type of the matrix.
 * @param input Input matrix.
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Output matrix
 * of the same shape with sigmoid applied element-wise.
 */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> sigmoid(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input) {
  const int rows = static_cast<int>(input.rows());
  const int cols = static_cast<int>(input.cols());
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(rows, cols);
  output.array() =
      static_cast<Scalar>(1) /
      (static_cast<Scalar>(1) + (-input).array().exp());
  return output;
}

#endif  // SIGMOID_H
