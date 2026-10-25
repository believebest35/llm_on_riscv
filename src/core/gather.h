#ifndef GATHER_H
#define GATHER_H

#include <Eigen/Dense>
#include <stdexcept>
#include <string>

/**
 * @brief Select rows or columns of a 2-D matrix by index (ONNX `Gather`).
 *
 * @tparam Scalar Data type stored in the input matrix.
 * @tparam IndexType Integer type used for indices (e.g. int, int64_t).
 * @param input Input matrix of size (rows x cols).
 * @param indices Matrix of integer indices; any shape, traversed row-major.
 * @param axis Axis along which to gather: 0 selects rows, 1 selects columns;
 *        negative values count from the back.
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Gathered
 *         values: (num_indices x cols) for axis 0, (rows x num_indices) for
 *         axis 1.
 */
template <typename Scalar, typename IndexType>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> gather(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
    const Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic>& indices,
    int axis = 0) {
  if (axis < -2 || axis > 1) {
    throw std::invalid_argument("gather: axis = " + std::to_string(axis) +
                                ", expected in [-2, 1]");
  }
  if (axis < 0) {
    axis += 2;  // 2-D data: -1 -> 1, -2 -> 0
  }

  const int rows = static_cast<int>(input.rows());
  const int cols = static_cast<int>(input.cols());
  const int index_rows = static_cast<int>(indices.rows());
  const int index_cols = static_cast<int>(indices.cols());
  const int num_indices = static_cast<int>(indices.size());

  if (axis == 0) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(num_indices,
                                                                 cols);
    int out_i = 0;
    for (int i = 0; i < index_rows; ++i) {
      for (int j = 0; j < index_cols; ++j) {
        IndexType idx = indices(i, j);
        if (idx < 0) {
          idx += rows;  // ONNX allows negative indices counting from the end
        }
        if (idx < 0 || idx >= rows) {
          throw std::invalid_argument(
              "gather: index = " + std::to_string(idx) + ", expected in [0, " +
              std::to_string(rows) + ")");
        }
        output.row(out_i++) = input.row(static_cast<int>(idx));
      }
    }
    return output;
  } else {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(rows,
                                                                 num_indices);
    int out_j = 0;
    for (int i = 0; i < index_rows; ++i) {
      for (int j = 0; j < index_cols; ++j) {
        IndexType idx = indices(i, j);
        if (idx < 0) {
          idx += cols;  // ONNX allows negative indices counting from the end
        }
        if (idx < 0 || idx >= cols) {
          throw std::invalid_argument(
              "gather: index = " + std::to_string(idx) + ", expected in [0, " +
              std::to_string(cols) + ")");
        }
        output.col(out_j++) = input.col(static_cast<int>(idx));
      }
    }
    return output;
  }
}

#endif  // GATHER_H
