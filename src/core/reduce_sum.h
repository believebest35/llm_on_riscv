#ifndef REDUCE_SUM_H
#define REDUCE_SUM_H

#include <Eigen/Dense>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Sum-reduction over selected axes of a 2-D matrix (ONNX `ReduceSum`).
 *
 * @tparam Scalar Element type of `input` and `output`.
 * @param input Input matrix (rows x cols).
 * @param axes Non-empty list of axes to reduce (each in [-2, 1] for 2-D data).
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> `output`
 *         tensor: axis 0 only -> (1 x cols); axis 1 only -> (rows x 1);
 *         both axes -> (1 x 1) scalar block with the total sum.
 */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> reduce_sum(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
    const std::vector<int>& axes) {
  if (axes.empty()) {
    throw std::invalid_argument(
        "reduce_sum: axes is empty, expected at least one axis");
  }

  std::set<int> unique_axes;
  for (int a : axes) {
    if (a < -2 || a > 1) {
      throw std::invalid_argument("reduce_sum: axis = " + std::to_string(a) +
                                  ", expected in [-2, 1]");
    }
    if (a < 0) {
      a += 2;  // 2-D data: -1 -> 1, -2 -> 0
    }
    unique_axes.insert(a);
  }

  if (unique_axes.size() == 2) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(1, 1);
    output(0, 0) = input.sum();
    return output;
  }
  if (*unique_axes.begin() == 0) {
    return input.colwise().sum();
  }
  return input.rowwise().sum();
}

#endif  // REDUCE_SUM_H
