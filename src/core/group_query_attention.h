#ifndef GROUP_QUERY_ATTENTION_H
#define GROUP_QUERY_ATTENTION_H

#include <Eigen/Dense>
#include <stdexcept>
#include <tuple>

/**
 * @brief Simplified GroupQueryAttention-like operator for testing.
 *
 * This toy implementation does **not** implement the full semantics of the
 * ONNX `GroupQueryAttention` operator.  It is intentionally minimal: the
 * output tensor Y is simply a copy of the input queries, and the present
 * key/value caches are produced by concatenating the provided past state
 * with the current keys/values along the first (row) dimension.
 *
 * The function accepts nine input matrices corresponding to the typical
 * inputs of the ONNX node.  Inputs which are not used by the simple
 * reference implementation (bias, mask_index, and two optional "unused"
 * tensors) are validated only for their expected shapes.
 *
 * This header is paired with a comprehensive unit test that exercises the
 * various shape checks and compares results against a golden reference.
 *
 * @tparam Scalar Numeric type (float, double, etc.)
 * @param Q Queries tensor flattened to 2-D: (batch*sequence) x Qdim
 * @param K Keys tensor flattened to 2-D: (batch*sequence) x Kdim
 * @param V Values tensor flattened to 2-D: (batch*sequence) x Kdim
 * @param past_k Previous key cache: arbitrary number of rows x Kdim
 * @param past_v Previous value cache: arbitrary number of rows x Kdim
 * @param bias Bias tensor, expected to have a single column
 * @param mask_index Attention mask index, must be a scalar matrix (1x1)
 * @param unused1 Placeholder for optional input (ignored)
 * @param unused2 Placeholder for optional input (ignored)
 * @return tuple containing {Y, present_k, present_v}
 * @throws std::invalid_argument when the inputs have incompatible shapes.
 */

template <typename Scalar>
std::tuple<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
           Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
           Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
group_query_attention(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Q,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& K,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& V,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& past_k,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& past_v,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& bias,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& mask_index,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& unused1,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& unused2) {
  // basic dimension checks
  if (Q.rows() != K.rows() || Q.rows() != V.rows()) {
    throw std::invalid_argument("Q, K and V must have the same number of rows");
  }
  if (Q.cols() != 2 * K.cols()) {
    throw std::invalid_argument(
        "Query dimension must be twice the key/value dimension");
  }
  if (K.cols() != V.cols()) {
    throw std::invalid_argument("K and V must have the same number of cols");
  }
  if (past_k.cols() != K.cols() || past_v.cols() != V.cols()) {
    throw std::invalid_argument(
        "Past key/value caches must have the same number of columns as "
        "current K/V");
  }
  if (bias.cols() != 1) {
    throw std::invalid_argument("Bias tensor must have one column");
  }
  if (mask_index.size() != 1) {
    throw std::invalid_argument("Mask index must be a scalar");
  }

  // Y is just a copy of Q in this toy implementation
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y = Q;

  // present_k/v are simple row-wise concatenations of past and current.
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> present_k;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> present_v;

  present_k.resize(past_k.rows() + K.rows(), K.cols());
  present_v.resize(past_v.rows() + V.rows(), V.cols());

  if (past_k.rows() > 0) {
    present_k.topRows(past_k.rows()) = past_k;
  }
  if (past_v.rows() > 0) {
    present_v.topRows(past_v.rows()) = past_v;
  }

  present_k.bottomRows(K.rows()) = K;
  present_v.bottomRows(V.rows()) = V;

  return {Y, present_k, present_v};
}

#endif  // GROUP_QUERY_ATTENTION_H
