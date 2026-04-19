#ifndef REDUCE_SUM_H
#define REDUCE_SUM_H

#include <Eigen/Dense>
#include <set>
#include <stdexcept>
#include <vector>

/**
 * @brief Sum-reduction over selected axes of a 2-D matrix.
 *
 * Simplified ONNX `ReduceSum` for two-dimensional `data`. Each axis must be
 * 0 or 1. Duplicate entries in `axes` are ignored. The reduced tensor keeps
 * degenerate dimensions of size 1 (same shape convention as the previous
 * single-axis implementation).
 *
 * @tparam Scalar Element type of `data` and `reduced`.
 * @param data Input matrix (rows x cols).
 * @param axes Non-empty list of axes to reduce (each 0 or 1 for 2-D data).
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> `reduced`
 *         tensor: axis 0 only -> (1 x cols); axis 1 only -> (rows x 1);
 *         both axes -> (1 x 1) scalar block with the total sum.
 * @throws std::invalid_argument if `axes` is empty or any axis is not 0 or 1.
 */

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> reduce_sum(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& data,
    const std::vector<int>& axes) {
  if (axes.empty()) {
    throw std::invalid_argument("reduce_sum: axes must be non-empty");
  }

  std::set<int> unique_axes;
  for (int a : axes) {
    if (a < 0 || a > 1) {
      throw std::invalid_argument("reduce_sum: axis must be 0 or 1 for 2-D");
    }
    unique_axes.insert(a);
  }

  const int rows = data.rows();
  const int cols = data.cols();

  if (unique_axes.size() == 2) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> reduced(1, 1);
    reduced(0, 0) = data.sum();
    return reduced;
  }

  if (*unique_axes.begin() == 0) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> reduced(1, cols);
    reduced.setZero();
    for (int i = 0; i < rows; ++i) {
      reduced += data.row(i);
    }
    return reduced;
  }

  // axis 1 only
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> reduced(rows, 1);
  reduced.setZero();
  for (int i = 0; i < rows; ++i) {
    reduced(i, 0) = data.row(i).sum();
  }
  return reduced;
}

#endif  // REDUCE_SUM_H
