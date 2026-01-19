#ifndef SIMPLIFIED_LAYERNORM_H
#define SIMPLIFIED_LAYERNORM_H

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

/**
 * @brief SimplifiedLayerNormalization (RMSNorm-style) over the last dimension.
 *
 * For input X (m x n) and scale weight (n),
 *   rms_i = sqrt(mean_j(X(i,j)^2) + epsilon)
 *   Y(i,j) = (X(i,j) / rms_i) * weight(j)
 *
 * This is commonly used as a simplified LayerNorm variant without
 * mean-centering and typically without bias.
 *
 * @tparam Scalar Data type (float, double, etc.)
 * @param X Input matrix (m x n)
 * @param weight Scale vector (n)
 * @param epsilon Small constant for numerical stability
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Output (m x n)
 */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
simplified_layer_normalization(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& weight,
    Scalar epsilon = static_cast<Scalar>(1e-5)) {
  if (X.cols() != weight.size()) {
    throw std::invalid_argument(
        "SimplifiedLayerNormalization dimension mismatch: "
        "X.cols() = " +
        std::to_string(X.cols()) +
        ", weight.size() = " + std::to_string(weight.size()));
  }
  if (epsilon <= static_cast<Scalar>(0)) {
    throw std::invalid_argument("epsilon must be > 0");
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y(X.rows(), X.cols());

  // Normalize each row (last dimension) by RMS, then apply per-channel scale.
  for (int i = 0; i < X.rows(); ++i) {
    // mean(x^2)
    const Scalar mean_sq =
        X.row(i).array().square().sum() / static_cast<Scalar>(X.cols());
    const Scalar rms = std::sqrt(mean_sq + epsilon);
    Y.row(i) = (X.row(i) / rms).cwiseProduct(weight.transpose());
  }

  return Y;
}

#endif  // SIMPLIFIED_LAYERNORM_H
