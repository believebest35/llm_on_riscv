#ifndef SKIP_SIMPLIFIED_LAYER_NORMALIZATION_H
#define SKIP_SIMPLIFIED_LAYER_NORMALIZATION_H

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>

/**
 * @brief SkipSimplifiedLayerNormalization: adds skip (residual) to the input
 * and performs simplified layer normalization (RMS-style) over the last
 * dimension.
 *
 * output = ( (input + skip) / rms ) * gamma
 *
 * Returns both outputs used by the model:
 * - output: normalized output (m x n)
 * - input_skip_bias_sum: input + skip (m x n)
 *
 * `gamma` corresponds to `scale` in SimplifiedLayerNormalization.
 *
 * @tparam Scalar Element type of `input`, `skip`, `gamma`, and the outputs.
 * @param input Input matrix (m x n).
 * @param skip Skip/residual matrix (m x n).
 * @param gamma Scale vector (n).
 * @param epsilon Small constant for numerical stability.
 * @return std::tuple<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
 *         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> output and
 *         input_skip_bias_sum, both (m x n).
 */
template <typename Scalar>
std::tuple<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
           Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
skip_simplified_layer_normalization(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& skip,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& gamma,
    Scalar epsilon = static_cast<Scalar>(1e-5)) {
  const int rows = static_cast<int>(input.rows());
  const int cols = static_cast<int>(input.cols());
  const int skip_rows = static_cast<int>(skip.rows());
  const int skip_cols = static_cast<int>(skip.cols());
  const int gamma_size = static_cast<int>(gamma.size());
  if (rows != skip_rows || cols != skip_cols) {
    throw std::invalid_argument(
        "skip_simplified_layer_normalization: input = " +
        std::to_string(rows) + "x" + std::to_string(cols) + ", skip = " +
        std::to_string(skip_rows) + "x" + std::to_string(skip_cols) +
        ", expected same shape");
  }
  if (cols != gamma_size) {
    throw std::invalid_argument(
        "skip_simplified_layer_normalization: input.cols() = " +
        std::to_string(cols) + ", gamma.size() = " +
        std::to_string(gamma_size) + ", expected equal");
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(rows, cols);
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> input_skip_bias_sum(
      rows, cols);

  for (int i = 0; i < rows; ++i) {
    input_skip_bias_sum.row(i) = input.row(i) + skip.row(i);
    const auto row = input_skip_bias_sum.row(i);
    const Scalar mean_sq =
        row.array().square().sum() / static_cast<Scalar>(cols);
    const Scalar rms = std::sqrt(mean_sq + epsilon);
    output.row(i) = (row / rms).cwiseProduct(gamma.transpose());
  }
  return {output, input_skip_bias_sum};
}

#endif  // SKIP_SIMPLIFIED_LAYER_NORMALIZATION_H
