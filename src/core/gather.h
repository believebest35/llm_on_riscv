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
 *         values: (num_indices x cols) for axis 0, (num_indices x rows) for
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
  const int axis_size = (axis == 0) ? rows : cols;
  const int kept_size = (axis == 0) ? cols : rows;

  // ONNX keeps indices.shape + the non-gathered data dim, so the gathered axis
  // is flattened first: (num_indices x cols) for axis 0, (num_indices x rows)
  // for axis 1.
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(num_indices,
                                                               kept_size);
  int out_i = 0;
  for (int i = 0; i < indices.rows(); ++i) {
    for (int j = 0; j < indices.cols(); ++j) {
      IndexType idx = indices(i, j);
      if (idx < 0) {
        idx += axis_size;  // ONNX allows negative indices counting from the end
      }
      if (idx < 0 || idx >= axis_size) {
        throw std::invalid_argument("gather: index out of bounds");
      }
      const int k = static_cast<int>(idx);
      if (axis == 0) {
        result.row(out_i) = data.row(k);
      } else {
        result.row(out_i) = data.col(k).transpose();
      }
      ++out_i;
    }
  }
  return result;
}

#endif  // GATHER_H
