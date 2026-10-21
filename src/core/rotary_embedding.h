#ifndef ROTARY_EMBEDDING_H
#define ROTARY_EMBEDDING_H

#include <Eigen/Dense>
#include <optional>
#include <stdexcept>
#include <string>

/**
 * Llama-style rotate_half on the last dimension: x = [x1, x2] -> [-x2, x1].
 * head_dim must be even.
 */
template <typename Scalar>
void rotary_rotate_half_row(
    const Eigen::Ref<const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>>& x,
    Eigen::Ref<Eigen::Matrix<Scalar, 1, Eigen::Dynamic>> out) {
  const Eigen::Index d = x.size();
  const Eigen::Index h = d / 2;
  out.head(h) = -x.tail(h);
  out.tail(h) = x.head(h);
}

/**
 * Apply rotary positional embeddings (RoPE) with precomputed cos/sin cache.
 *
 * Logical I/O (matching graph / ONNX-style RotaryEmbedding):
 * - input:  (batch * sequence_length, hidden_dim)
 * - position_ids: optional int64 vector. A length-1 vector is the starting
 *   position for each sequence (row r gets position_ids[0] + r %
 *   sequence_length), matching ORT's single-element position_ids input. A
 *   length-(batch*sequence_length) vector gives one absolute position per row.
 *   If omitted or empty, positions are implicit for prefill: row r has position
 *   (r % sequence_length).
 * - cos_cache, sin_cache: (max_positions, head_dim/2), use rows indexed by
 *   position (max row index must stay < cache.rows()).
 * - num_heads: hidden_dim must be divisible; head_dim = hidden_dim / num_heads.
 *
 * Same formula as common HF Llama: cos/sin for head are each length head_dim/2,
 * broadcast by concatenating [cos, cos] along the head axis, then
 *   output = input * cos_full + rotate_half(input) * sin_full.
 *
 * @param sequence_length Used only when position_ids is not provided (prefill).
 */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> rotary_embedding(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
    const std::optional<Eigen::Matrix<int, Eigen::Dynamic, 1>>& position_ids,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cos_cache,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& sin_cache,
    int sequence_length, int num_heads) {
  const Eigen::Index rows = input.rows();
  const Eigen::Index hidden_dim = input.cols();
  if (hidden_dim % num_heads != 0) {
    throw std::invalid_argument(
        "rotary_embedding: hidden_dim must be divisible by num_heads");
  }
  const Eigen::Index head_dim = hidden_dim / num_heads;
  if (head_dim % 2 != 0) {
    throw std::invalid_argument("rotary_embedding: head_dim must be even");
  }
  const Eigen::Index rope_dim = head_dim / 2;

  if (cos_cache.cols() != rope_dim || sin_cache.cols() != rope_dim) {
    throw std::invalid_argument(
        "rotary_embedding: cache last dim must equal head_dim/2");
  }
  if (cos_cache.rows() != sin_cache.rows()) {
    throw std::invalid_argument(
        "rotary_embedding: cos_cache and sin_cache row count must match");
  }

  const Eigen::Index position_count =
      position_ids.has_value() ? position_ids->size() : 0;
  if (position_count != 0 && position_count != 1 && position_count != rows) {
    throw std::invalid_argument(
        "rotary_embedding: position_ids must contain 1 or input_rows values");
  }
  if (position_count != rows) {
    if (sequence_length <= 0) {
      throw std::invalid_argument(
          "rotary_embedding: sequence_length > 0 required when position_ids "
          "has fewer than input_rows values");
    }
    if (rows % sequence_length != 0) {
      throw std::invalid_argument(
          "rotary_embedding: input rows must be multiple of sequence_length");
    }
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(rows, hidden_dim);
  Eigen::Matrix<Scalar, 1, Eigen::Dynamic> rh(1, head_dim);
  Eigen::Matrix<Scalar, 1, Eigen::Dynamic> cos_full(1, head_dim);
  Eigen::Matrix<Scalar, 1, Eigen::Dynamic> sin_full(1, head_dim);

  for (Eigen::Index r = 0; r < rows; ++r) {
    int pos = 0;
    if (position_count == 1) {
      pos = (*position_ids)(0) + static_cast<int>(r % sequence_length);
    } else if (position_count == rows) {
      pos = (*position_ids)(r);
    } else {
      pos = static_cast<int>(r % sequence_length);
    }
    if (pos < 0 || pos >= cos_cache.rows()) {
      throw std::out_of_range(
          "rotary_embedding: position index " + std::to_string(pos) +
          " out of cache bounds [0, " + std::to_string(cos_cache.rows()) + ")");
    }

    cos_full.leftCols(rope_dim) = cos_cache.row(pos);
    cos_full.rightCols(rope_dim) = cos_cache.row(pos);
    sin_full.leftCols(rope_dim) = sin_cache.row(pos);
    sin_full.rightCols(rope_dim) = sin_cache.row(pos);

    for (int h = 0; h < num_heads; ++h) {
      const Eigen::Index off = h * head_dim;
      auto x = input.row(r).segment(off, head_dim);
      rotary_rotate_half_row<Scalar>(x, rh);
      output.row(r).segment(off, head_dim).array() =
          x.array() * cos_full.array() + rh.array() * sin_full.array();
    }
  }
  return output;
}

#endif  // ROTARY_EMBEDDING_H
