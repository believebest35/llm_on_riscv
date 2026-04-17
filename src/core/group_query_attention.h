#ifndef GROUP_QUERY_ATTENTION_H
#define GROUP_QUERY_ATTENTION_H

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

/**
 * @brief Group Query Attention kernel for inference.
 *
 * Note: this implementation is specialized for batch_size == 1.
 *
 * Shapes:
 * - query:      (sequence, 2048) = (S, num_query_heads * head_dim)
 * - key/value:  (sequence, 1024) = (S, num_kv_heads * head_dim)
 * - past_key/v: (8 * past_sequence, 128)
 * - seqlens_k:  scalar, valid KV length after concat
 * - total_sequence_length: scalar, must equal past_sequence + sequence
 *
 * Returns:
 * - output:      (sequence, 2048)
 * - present_k/v: (8 * total_sequence_length, 128)
 */
template <typename Scalar>
std::tuple<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
           Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
           Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
group_query_attention(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& query,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& key,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& value,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& past_key,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& past_value,
    int seqlens_k,
    int total_sequence_length) {
  constexpr int kNumQueryHeads = 16;
  constexpr int kNumKeyValueHeads = 8;
  constexpr int kHeadDim = 128;

  if (query.rows() <= 0) {
    throw std::invalid_argument("query sequence length must be positive");
  }
  if (query.rows() != key.rows() || query.rows() != value.rows()) {
    throw std::invalid_argument("query/key/value must share sequence length");
  }
  if (query.cols() != kNumQueryHeads * kHeadDim) {
    throw std::invalid_argument("query hidden size must be 2048");
  }
  if (key.cols() != kNumKeyValueHeads * kHeadDim ||
      value.cols() != kNumKeyValueHeads * kHeadDim) {
    throw std::invalid_argument("key/value hidden size must be 1024");
  }
  if (past_key.cols() != kHeadDim || past_value.cols() != kHeadDim) {
    throw std::invalid_argument("past key/value second dimension must be 128");
  }
  if (past_key.rows() != past_value.rows()) {
    throw std::invalid_argument("past key/value rows must match");
  }
  if (past_key.rows() % kNumKeyValueHeads != 0) {
    throw std::invalid_argument("past key/value rows must be divisible by 8");
  }

  const int sequence = query.rows();
  const int past_sequence = past_key.rows() / kNumKeyValueHeads;

  if (total_sequence_length != past_sequence + sequence) {
    throw std::invalid_argument(
        "total_sequence_length must equal past_sequence_length + sequence_length");
  }
  if (total_sequence_length <= 0) {
    throw std::invalid_argument("total_sequence_length must be positive");
  }
  if (seqlens_k <= 0 || seqlens_k > total_sequence_length) {
    throw std::invalid_argument("seqlens_k value out of valid range");
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> present_key(
      kNumKeyValueHeads * total_sequence_length, kHeadDim);
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> present_value(
      kNumKeyValueHeads * total_sequence_length, kHeadDim);

  for (int h = 0; h < kNumKeyValueHeads; ++h) {
    if (past_sequence > 0) {
      present_key.block(h * total_sequence_length, 0, past_sequence, kHeadDim) =
          past_key.block(h * past_sequence, 0, past_sequence, kHeadDim);
      present_value.block(h * total_sequence_length, 0, past_sequence, kHeadDim) =
          past_value.block(h * past_sequence, 0, past_sequence, kHeadDim);
    }
    present_key.block(h * total_sequence_length + past_sequence, 0, sequence, kHeadDim) =
        key.block(0, h * kHeadDim, sequence, kHeadDim);
    present_value.block(h * total_sequence_length + past_sequence, 0, sequence, kHeadDim) =
        value.block(0, h * kHeadDim, sequence, kHeadDim);
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(sequence, kNumQueryHeads * kHeadDim);
  const Scalar scale = static_cast<Scalar>(1.0 / std::sqrt(static_cast<double>(kHeadDim)));
  const Scalar neg_inf = -std::numeric_limits<Scalar>::infinity();

  // Build mask once: causal mask (by query position) + seqlens mask (by key position).
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask(sequence, total_sequence_length);
  for (int qs = 0; qs < sequence; ++qs) {
    const int causal_limit = past_sequence + qs;
    for (int ks = 0; ks < total_sequence_length; ++ks) {
      valid_mask(qs, ks) = (ks <= causal_limit) && (ks < seqlens_k);
    }
  }
  const Eigen::VectorXi valid_counts =
      valid_mask.cast<int>().matrix().rowwise().sum();
  if ((valid_counts.array() == 0).any()) {
    throw std::invalid_argument("no valid key positions after applying masks");
  }

  for (int qh = 0; qh < kNumQueryHeads; ++qh) {
    const int kvh = qh / 2;
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> q_block =
        query.block(0, qh * kHeadDim, sequence, kHeadDim);
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> k_block =
        present_key.block(kvh * total_sequence_length, 0, total_sequence_length, kHeadDim);
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> v_block =
        present_value.block(kvh * total_sequence_length, 0, total_sequence_length, kHeadDim);

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> scores =
        (q_block * k_block.transpose()) * scale;
    scores.array() = valid_mask.select(scores.array(), neg_inf);

    // Keep a row-wise loop for numerically stable softmax.
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> probs(sequence, total_sequence_length);
    for (int qs = 0; qs < sequence; ++qs) {
      const Scalar max_score = scores.row(qs).maxCoeff();
      Eigen::Array<Scalar, 1, Eigen::Dynamic> row_exp =
          (scores.row(qs).array() - max_score).exp();
      const Scalar exp_sum = row_exp.sum();
      if (exp_sum == static_cast<Scalar>(0)) {
        throw std::invalid_argument("softmax normalization sum became zero");
      }
      probs.row(qs) = (row_exp / exp_sum).matrix();
    }

    output.block(0, qh * kHeadDim, sequence, kHeadDim) = probs * v_block;
  }

  return {output, present_key, present_value};
}

#endif  // GROUP_QUERY_ATTENTION_H
