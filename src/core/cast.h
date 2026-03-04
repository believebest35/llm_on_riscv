#ifndef CAST_H
#define CAST_H

#include <Eigen/Dense>
#include <stdexcept>

/**
 * @brief Cast matrix elements from one scalar type to another.
 *
 * This helper mirrors the ONNX `Cast` operator with the simplest semantics:
 * perform a C++ static_cast on each element.  Only numeric scalar types that
 * Eigen supports are usable.
 *
 * @tparam OutScalar Destination element type.
 * @tparam InScalar Source element type.
 * @param A Input matrix of size (m x n).
 * @return Eigen::Matrix<OutScalar, Eigen::Dynamic, Eigen::Dynamic> Matrix of
 *         identical shape with elements converted to OutScalar.
 */

template <typename OutScalar, typename InScalar>
Eigen::Matrix<OutScalar, Eigen::Dynamic, Eigen::Dynamic> cast_matrix(
    const Eigen::Matrix<InScalar, Eigen::Dynamic, Eigen::Dynamic>& A) {
  // Eigen provides a convenient cast() member function for this purpose.
  return A.template cast<OutScalar>();
}

#endif  // CAST_H
