#ifndef GATHER_H
#define GATHER_H

#include <Eigen/Dense>
#include <stdexcept>

/**
 * @brief Select rows or columns of a 2-D matrix by index (ONNX `Gather`).
 *
 * @tparam Scalar Data type stored in the input matrix.
 * @tparam IndexType Integer type used for indices (e.g. int, int64_t).
 * @param data Input matrix of size (rows x cols).
 * @param indices Matrix of integer indices; any shape, traversed row-major.
 * @param axis Axis along which to gather: 0 selects rows, 1 selects columns;
 *        negative values count from the back.
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Gathered
 *         values: (num_indices x cols) for axis 0, (rows x num_indices) for
 *         axis 1.
 */
template <typename Scalar, typename IndexType>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> gather(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& data,
    const Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic>& indices,
    int axis = 0) {
  if (axis < 0) {
    axis += 2;  // 2-D data: -1 -> 1, -2 -> 0
  }
  if (axis != 0 && axis != 1) {
    throw std::invalid_argument("gather: axis must be 0 or 1");
  }

  const int rows = data.rows();
  const int cols = data.cols();
  const int num_indices = static_cast<int>(indices.size());

  if (axis == 0) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(num_indices,
                                                                 cols);
    int out_i = 0;
    for (int i = 0; i < indices.rows(); ++i) {
      for (int j = 0; j < indices.cols(); ++j) {
        IndexType idx = indices(i, j);
        if (idx < 0) {
          idx += rows;  // ONNX allows negative indices counting from the end
        }
        if (idx < 0 || idx >= rows) {
          throw std::invalid_argument("gather: index out of bounds");
        }
        result.row(out_i++) = data.row(static_cast<int>(idx));
      }
    }
    return result;
  } else {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(rows,
                                                                 num_indices);
    int out_j = 0;
    for (int i = 0; i < indices.rows(); ++i) {
      for (int j = 0; j < indices.cols(); ++j) {
        IndexType idx = indices(i, j);
        if (idx < 0) {
          idx += cols;  // ONNX allows negative indices counting from the end
        }
        if (idx < 0 || idx >= cols) {
          throw std::invalid_argument("gather: index out of bounds");
        }
        result.col(out_j++) = data.col(static_cast<int>(idx));
      }
    }
    return result;
  }
}

#endif  // GATHER_H
