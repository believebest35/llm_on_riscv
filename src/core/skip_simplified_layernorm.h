#ifndef SKIP_SIMPLIFIED_LAYERNORM_H
#define SKIP_SIMPLIFIED_LAYERNORM_H

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

/**
 * @brief SkipSimplifiedLayerNormalization: adds skip (residual) to input X and
 * performs simplified layer normalization (RMS-style) over the last dimension.
 *
 * Y = ( (X + skip) / rms ) * gamma
 *
 * @tparam Scalar
 * @param X Input matrix (m x n)
 * @param skip Skip/residual matrix (m x n)
 * @param gamma Scale vector (n)
 * @param epsilon Small constant for numerical stability
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Output (m x n)
 */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
skip_simplified_layer_normalization(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& skip,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& gamma,
    Scalar epsilon = static_cast<Scalar>(1e-5)) {
  if (X.rows() != skip.rows() || X.cols() != skip.cols()) {
    throw std::invalid_argument(
        "SkipSimplifiedLayerNormalization dimension mismatch: X and skip must "
        "have same shape");
  }
  if (X.cols() != gamma.size()) {
    throw std::invalid_argument(
        "SkipSimplifiedLayerNormalization dimension mismatch: X.cols() = " +
        std::to_string(X.cols()) +
        ", gamma.size() = " + std::to_string(gamma.size()));
  }
  if (epsilon <= static_cast<Scalar>(0)) {
    throw std::invalid_argument("epsilon must be > 0");
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y(X.rows(), X.cols());

  for (int i = 0; i < X.rows(); ++i) {
    const auto row = X.row(i) + skip.row(i);
    const Scalar mean_sq =
        row.array().square().sum() / static_cast<Scalar>(X.cols());
    const Scalar rms = std::sqrt(mean_sq + epsilon);
    Y.row(i) = (row / rms).cwiseProduct(gamma.transpose());
  }

  return Y;
}

#endif  // SKIP_SIMPLIFIED_LAYERNORM_H
