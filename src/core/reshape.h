#ifndef RESHAPE_H
#define RESHAPE_H

#include <Eigen/Dense>
#include <stdexcept>
#include <string>

/**
 * @brief Change the shape of a matrix without altering data order (ONNX
 * `Reshape`).
 *
 * Follows ONNX 2-D Reshape semantics: a dimension of 0 copies the
 * corresponding input dimension, and -1 infers the dimension from the total
 * number of elements.
 *
 * @tparam Scalar Element type of the matrix.
 * @param input Input matrix of size (rows x cols).
 * @param new_rows Number of output rows (>= 0, or -1 to infer).
 * @param new_cols Number of output columns (>= 0, or -1 to infer).
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Reshaped
 *         matrix (new_rows x new_cols), read/written in row-major order.
 */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> reshape(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
    int new_rows, int new_cols) {
  if (new_rows < -1 || new_cols < -1) {
    throw std::invalid_argument(
        "reshape: new shape = " + std::to_string(new_rows) + "x" +
        std::to_string(new_cols) + ", expected dimensions >= -1");
  }

  const int rows = static_cast<int>(input.rows());
  const int cols = static_cast<int>(input.cols());
  const int total = rows * cols;
  if (new_rows == 0) {
    new_rows = rows;
  }
  if (new_cols == 0) {
    new_cols = cols;
  }

  if (new_rows == -1 && new_cols == -1) {
    throw std::invalid_argument(
        "reshape: new shape = -1x-1, expected at most one -1");
  }
  if (new_rows == -1) {
    if (new_cols == 0) {
      throw std::invalid_argument(
          "reshape: new shape = -1x0, expected a non-zero dimension to infer");
    }
    if (total % new_cols != 0) {
      throw std::invalid_argument(
          "reshape: new shape = -1x" + std::to_string(new_cols) +
          ", input size = " + std::to_string(total) + ", expected divisible");
    }
    new_rows = total / new_cols;
  } else if (new_cols == -1) {
    if (new_rows == 0) {
      throw std::invalid_argument(
          "reshape: new shape = 0x-1, expected a non-zero dimension to infer");
    }
    if (total % new_rows != 0) {
      throw std::invalid_argument(
          "reshape: new shape = " + std::to_string(new_rows) + "x-1" +
          ", input size = " + std::to_string(total) + ", expected divisible");
    }
    new_cols = total / new_rows;
  }

  if (new_rows * new_cols != total) {
    throw std::invalid_argument(
        "reshape: new shape = " + std::to_string(new_rows) + "x" +
        std::to_string(new_cols) + " = " + std::to_string(new_rows * new_cols) +
        ", expected input size = " + std::to_string(total));
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(new_rows,
                                                              new_cols);
  // ONNX Reshape reads and writes in row-major order.
  for (int i = 0; i < total; ++i) {
    output(i / new_cols, i % new_cols) = input(i / cols, i % cols);
  }
  return output;
}

#endif  // RESHAPE_H
