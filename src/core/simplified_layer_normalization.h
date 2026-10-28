#ifndef SIMPLIFIED_LAYER_NORMALIZATION_H
#define SIMPLIFIED_LAYER_NORMALIZATION_H

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>

/**
 * @brief SimplifiedLayerNormalization (RMSNorm-style) over the last dimension
 *        (ONNX `SimplifiedLayerNormalization`).
 *
 * Inputs: `input`, `scale`. Output: `output` (return value). Optional
 * `epsilon` for numerical stability (default 1e-5).
 *
 * For `input` (m x n) and `scale` (n),
 *   rms_i = sqrt(mean_j(input(i,j)^2) + epsilon)
 *   output(i,j) = (input(i,j) / rms_i) * scale(j)
 *
 * `scale` corresponds to `gamma` in SkipSimplifiedLayerNormalization; both are
 * the per-column scale factor.
 *
 * @tparam Scalar Element type of `input`, `scale`, and `output`.
 * @param input Input matrix (m x n).
 * @param scale Column vector of length n (same as `input.cols()`).
 * @param epsilon Small positive constant.
 * @return output Same shape as `input`.
 */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
simplified_layer_normalization(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& scale,
    Scalar epsilon = static_cast<Scalar>(1e-5)) {
  const int rows = static_cast<int>(input.rows());
  const int cols = static_cast<int>(input.cols());
  const int scale_size = static_cast<int>(scale.size());
  if (cols != scale_size) {
    throw std::invalid_argument(
        "simplified_layer_normalization: input.cols() = " +
        std::to_string(cols) + ", scale.size() = " +
        std::to_string(scale_size) + ", expected equal");
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(rows, cols);

  for (int i = 0; i < rows; ++i) {
    const Scalar mean_sq =
        input.row(i).array().square().sum() / static_cast<Scalar>(cols);
    const Scalar rms = std::sqrt(mean_sq + epsilon);
    output.row(i) = (input.row(i) / rms).cwiseProduct(scale.transpose());
  }
  return output;
}

#endif  // SIMPLIFIED_LAYER_NORMALIZATION_H
