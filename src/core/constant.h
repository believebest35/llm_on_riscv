#ifndef CONSTANT_H
#define CONSTANT_H

#include <Eigen/Dense>

/**
 * @brief Return a constant matrix tensor.
 *
 * This helper mirrors the behavior of the ONNX `Constant` operator where the
 * constant is provided directly as a matrix tensor.
 *
 * @tparam Scalar Type of the input and output matrices.
 * @param value Input constant tensor, shape:
 *        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>.
 * @return Output tensor, shape:
 *         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>.
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> constant_matrix(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& value) {
  return value;
}

#endif  // CONSTANT_H
