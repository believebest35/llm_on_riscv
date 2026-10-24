#ifndef ROTARY_EMBEDDING_H
#define ROTARY_EMBEDDING_H

#include <Eigen/Dense>
#include <stdexcept>

/**
 * @brief Llama-style rotate_half on the last dimension: [x1, x2] -> [-x2, x1].
 *
 * @tparam Scalar Element type of the row.
 * @param x Input row of even length.
 * @param out Output row of the same length.
 */
template <typename Scalar>
void rotary_rotate_half_row(
    const Eigen::Ref<const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>>& x,
    Eigen::Ref<Eigen::Matrix<Scalar, 1, Eigen::Dynamic>> out) {
  const int d = static_cast<int>(x.size());
  const int h = d / 2;
  out.head(h) = -x.tail(h);
  out.tail(h) = x.head(h);
}

/**
 * @brief Apply rotary positional embeddings with a precomputed cos/sin cache.
 *
 * Batch size is fixed to 1, so the input is (sequence_length, hidden_dim).
 * position_id is the absolute position of the first row: 0 for prefill, the
 * current token position for decode. Row r uses position_id + r.
 *
 * Same formula as HF Llama: cos/sin for a head each have length head_dim/2,
 * broadcast by concatenating [cos, cos] along the head axis, then
 *   output = input * cos_full + rotate_half(input) * sin_full.
 *
 * @tparam Scalar Element type of the input, caches, and output.
 * @param input Input matrix (sequence_length x hidden_dim).
 * @param position_id Absolute position of the first input row.
 * @param cos_cache Cosine cache (max_positions x head_dim / 2).
 * @param sin_cache Sine cache (max_positions x head_dim / 2).
 * @param num_heads Number of heads; hidden_dim must be divisible by it.
 * @return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Rotated
 *         matrix (sequence_length x hidden_dim).
 */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> rotary_embedding(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
    int position_id,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cos_cache,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& sin_cache,
    int num_heads) {
  const int rows = static_cast<int>(input.rows());
  const int hidden_dim = static_cast<int>(input.cols());
  if (num_heads <= 0 || hidden_dim % num_heads != 0) {
    throw std::invalid_argument(
        "rotary_embedding: hidden_dim must be divisible by num_heads");
  }
  const int head_dim = hidden_dim / num_heads;
  if (head_dim % 2 != 0) {
    throw std::invalid_argument("rotary_embedding: head_dim must be even");
  }
  const int rope_dim = head_dim / 2;

  if (cos_cache.cols() != rope_dim || sin_cache.cols() != rope_dim) {
    throw std::invalid_argument(
        "rotary_embedding: cache last dim must equal head_dim/2");
  }
  if (cos_cache.rows() != sin_cache.rows()) {
    throw std::invalid_argument(
        "rotary_embedding: cos_cache and sin_cache row count must match");
  }
  if (position_id < 0 || position_id + rows > cos_cache.rows()) {
    throw std::out_of_range("rotary_embedding: position out of cache bounds");
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(rows,
                                                               hidden_dim);
  Eigen::Matrix<Scalar, 1, Eigen::Dynamic> rh(1, head_dim);
  Eigen::Matrix<Scalar, 1, Eigen::Dynamic> cos_full(1, head_dim);
  Eigen::Matrix<Scalar, 1, Eigen::Dynamic> sin_full(1, head_dim);

  for (int r = 0; r < rows; ++r) {
    const int pos = position_id + r;

    cos_full.leftCols(rope_dim) = cos_cache.row(pos);
    cos_full.rightCols(rope_dim) = cos_cache.row(pos);
    sin_full.leftCols(rope_dim) = sin_cache.row(pos);
    sin_full.rightCols(rope_dim) = sin_cache.row(pos);

    for (int h = 0; h < num_heads; ++h) {
      const int off = h * head_dim;
      auto x = input.row(r).segment(off, head_dim);
      rotary_rotate_half_row<Scalar>(x, rh);
      output.row(r).segment(off, head_dim).array() =
          x.array() * cos_full.array() + rh.array() * sin_full.array();
    }
  }
  return output;
}

#endif  // ROTARY_EMBEDDING_H
