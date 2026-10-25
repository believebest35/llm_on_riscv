#ifndef CAST_H
#define CAST_H

#include <Eigen/Dense>

/**
 * @brief Cast matrix elements from one scalar type to another.
 *
 * This helper mirrors the ONNX `Cast` operator with the simplest semantics:
 * perform a C++ static_cast on each element. Only numeric scalar types that
 * Eigen supports are usable. Every input is valid; this kernel does not throw.
 *
 * @tparam OutScalar Destination element type.
 * @tparam InScalar Source element type.
 * @param input Input matrix of size (m x n).
 * @return Eigen::Matrix<OutScalar, Eigen::Dynamic, Eigen::Dynamic> Matrix of
 *         identical shape with elements converted to OutScalar.
 */
template <typename OutScalar, typename InScalar>
Eigen::Matrix<OutScalar, Eigen::Dynamic, Eigen::Dynamic> cast(
    const Eigen::Matrix<InScalar, Eigen::Dynamic, Eigen::Dynamic>& input) {
  return input.template cast<OutScalar>();
}

#endif  // CAST_H
