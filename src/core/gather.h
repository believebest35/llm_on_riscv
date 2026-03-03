#ifndef GATHER_H
#define GATHER_H

#include <Eigen/Dense>
#include <stdexcept>

/**
 * @brief Select rows or columns from a 2D matrix using index tensor.
 *
 * This is a very simple helper matching the semantics of the ONNX Gather
 * operator specialized to two-dimensional matrices.  Only integer indices are
 * supported, and the output is always a matrix with one dimension equal to
 * the number of indices and the other dimension equal to the remaining
 * dimension of the input matrix.
 *
 * The `axis` parameter controls whether the gathering is performed along
 * rows (axis=0, the default) or columns (axis=1).  The index tensor may be
 * of arbitrary shape; it is flattened before indexing, and the resulting
 * row/column order follows row-major traversal of the indices matrix.
 *
 * The output shape is effectively
 *   indices.shape + data.shape[axis+1..]
 * flattened to 2D.  For axis=0 the result has dimensions
 * (num_indices x data.cols()).  For axis=1 the result has dimensions
 * (data.rows() x num_indices).
 *
 * @tparam Scalar Data type stored in the input matrix.
 * @tparam IndexType Integer type used for indices (e.g. int, int64_t).
 * @param data Input matrix of size (rows x cols).
 * @param indices Matrix of integer indices; can be any shape.
 * @param axis Axis along which to gather: 0 selects rows, 1 selects columns.
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Output
 *         matrix of gathered values.
 * @throws std::invalid_argument for invalid axis or out-of-bounds indices.
 */

template <typename Scalar, typename IndexType>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> gather(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& data,
    const Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic>& indices,
    int axis = 0) {
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
