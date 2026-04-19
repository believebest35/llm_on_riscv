#ifndef SIMPLIFIED_LAYERNORM_H
#define SIMPLIFIED_LAYERNORM_H

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>

/**
 * @brief SimplifiedLayerNormalization (RMSNorm-style) over the last dimension.
 *
 * Inputs: `X`, `scale`. Output: `Y` (return value). Optional `epsilon` for
 * numerical stability (default 1e-5).
 *
 * For `X` (m x n) and `scale` (n),
 *   rms_i = sqrt(mean_j(X(i,j)^2) + epsilon)
 *   Y(i,j) = (X(i,j) / rms_i) * scale(j)
 *
 * @tparam Scalar Element type of `X`, `scale`, and `Y`.
 * @param X Input matrix (m x n).
 * @param scale Column vector of length n (same as `X.cols()`).
 * @param epsilon Small positive constant.
 * @return Y Same shape as `X`.
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
simplified_layer_normalization(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& scale,
    Scalar epsilon = static_cast<Scalar>(1e-5)) {
  if (X.cols() != scale.size()) {
    throw std::invalid_argument(
        "SimplifiedLayerNormalization dimension mismatch: "
        "X.cols() = " +
        std::to_string(X.cols()) +
        ", scale.size() = " + std::to_string(scale.size()));
  }
  if (epsilon <= static_cast<Scalar>(0)) {
    throw std::invalid_argument(
        "SimplifiedLayerNormalization: epsilon must be > 0");
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y(X.rows(), X.cols());

  for (int i = 0; i < X.rows(); ++i) {
    const Scalar mean_sq =
        X.row(i).array().square().sum() / static_cast<Scalar>(X.cols());
    const Scalar rms = std::sqrt(mean_sq + epsilon);
    Y.row(i) = (X.row(i) / rms).cwiseProduct(scale.transpose());
  }

  return Y;
}

#endif  // SIMPLIFIED_LAYERNORM_H
