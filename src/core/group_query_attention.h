#ifndef GROUP_QUERY_ATTENTION_H
#define GROUP_QUERY_ATTENTION_H

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

/**
 * @brief Group query attention for batch size one.
 *
 * Batch size is fixed to 1, so all inputs are flattened to 2-D.
 *
 * @tparam Scalar Element type of the inputs, caches, and outputs.
 * @param query Query matrix (sequence x 2048).
 * @param key Key matrix (sequence x 1024).
 * @param value Value matrix (sequence x 1024).
 * @param past_key Past key cache (8 * past_sequence x 128).
 * @param past_value Past value cache (8 * past_sequence x 128).
 * @param seqlens_k Last valid KV index (total_sequence_length - 1).
 * @param total_sequence_length Must equal past_sequence + sequence.
 * @return std::tuple<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
 *         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
 *         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> Output
 *         (sequence x 2048), present_key and present_value
 *         (8 * total_sequence_length x 128).
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

  if (query.rows() != key.rows() || query.rows() != value.rows()) {
    throw std::invalid_argument(
        "group_query_attention: query.rows() = " +
        std::to_string(query.rows()) + ", key.rows() = " +
        std::to_string(key.rows()) + ", value.rows() = " +
        std::to_string(value.rows()) + ", expected equal");
  }
  if (query.cols() != kNumQueryHeads * kHeadDim) {
    throw std::invalid_argument(
        "group_query_attention: query.cols() = " +
        std::to_string(query.cols()) + ", expected " +
        std::to_string(kNumQueryHeads * kHeadDim));
  }
  if (key.cols() != kNumKeyValueHeads * kHeadDim ||
      value.cols() != kNumKeyValueHeads * kHeadDim) {
    throw std::invalid_argument(
        "group_query_attention: key.cols() = " +
        std::to_string(key.cols()) + ", value.cols() = " +
        std::to_string(value.cols()) + ", expected " +
        std::to_string(kNumKeyValueHeads * kHeadDim));
  }
  if (past_key.cols() != kHeadDim || past_value.cols() != kHeadDim) {
    throw std::invalid_argument(
        "group_query_attention: past_key.cols() = " +
        std::to_string(past_key.cols()) + ", past_value.cols() = " +
        std::to_string(past_value.cols()) + ", expected " +
        std::to_string(kHeadDim));
  }
  if (past_key.rows() != past_value.rows()) {
    throw std::invalid_argument(
        "group_query_attention: past_key.rows() = " +
        std::to_string(past_key.rows()) + ", past_value.rows() = " +
        std::to_string(past_value.rows()) + ", expected equal");
  }
  if (past_key.rows() % kNumKeyValueHeads != 0) {
    throw std::invalid_argument(
        "group_query_attention: past_key.rows() = " +
        std::to_string(past_key.rows()) + ", expected divisible by " +
        std::to_string(kNumKeyValueHeads));
  }

  const int sequence = static_cast<int>(query.rows());
  const int past_sequence =
      static_cast<int>(past_key.rows() / kNumKeyValueHeads);

  if (total_sequence_length != past_sequence + sequence) {
    throw std::invalid_argument(
        "group_query_attention: total_sequence_length = " +
        std::to_string(total_sequence_length) +
        ", expected past_sequence_length + sequence_length = " +
        std::to_string(past_sequence + sequence));
  }
  if (seqlens_k < 0 || seqlens_k >= total_sequence_length) {
    throw std::invalid_argument(
        "group_query_attention: seqlens_k = " +
        std::to_string(seqlens_k) + ", expected in [0, " +
        std::to_string(total_sequence_length) + ")");
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> present_key(
      kNumKeyValueHeads * total_sequence_length, kHeadDim);
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> present_value(
      kNumKeyValueHeads * total_sequence_length, kHeadDim);

  for (int h = 0; h < kNumKeyValueHeads; ++h) {
    if (past_sequence > 0) {
      present_key.block(h * total_sequence_length, 0, past_sequence,
                        kHeadDim) =
          past_key.block(h * past_sequence, 0, past_sequence, kHeadDim);
      present_value.block(h * total_sequence_length, 0, past_sequence,
                          kHeadDim) =
          past_value.block(h * past_sequence, 0, past_sequence, kHeadDim);
    }
    present_key.block(h * total_sequence_length + past_sequence, 0, sequence,
                      kHeadDim) =
        key.block(0, h * kHeadDim, sequence, kHeadDim);
    present_value.block(h * total_sequence_length + past_sequence, 0, sequence,
                        kHeadDim) =
        value.block(0, h * kHeadDim, sequence, kHeadDim);
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(
      sequence, kNumQueryHeads * kHeadDim);
  const Scalar scale = static_cast<Scalar>(
      1.0 / std::sqrt(static_cast<double>(kHeadDim)));
  const Scalar neg_inf = -std::numeric_limits<Scalar>::infinity();

  // Build the mask once: causal mask by query position and seqlens mask by key
  // position.
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask(
      sequence, total_sequence_length);
  for (int qs = 0; qs < sequence; ++qs) {
    const int causal_limit = past_sequence + qs;
    for (int ks = 0; ks < total_sequence_length; ++ks) {
      valid_mask(qs, ks) = (ks <= causal_limit) && (ks <= seqlens_k);
    }
  }

  for (int qh = 0; qh < kNumQueryHeads; ++qh) {
    const int kvh = qh / 2;
    const int kv_offset = kvh * total_sequence_length;
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> q_block =
        query.block(0, qh * kHeadDim, sequence, kHeadDim);
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> k_block =
        present_key.block(kv_offset, 0, total_sequence_length, kHeadDim);
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> v_block =
        present_value.block(kv_offset, 0, total_sequence_length, kHeadDim);

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> scores =
        (q_block * k_block.transpose()) * scale;
    scores.array() = valid_mask.select(scores.array(), neg_inf);

    // Keep a row-wise loop for numerically stable softmax.
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> probs(
        sequence, total_sequence_length);
    for (int qs = 0; qs < sequence; ++qs) {
      const Scalar max_score = scores.row(qs).maxCoeff();
      Eigen::Array<Scalar, 1, Eigen::Dynamic> row_exp =
          (scores.row(qs).array() - max_score).exp();
      const Scalar exp_sum = row_exp.sum();
      probs.row(qs) = (row_exp / exp_sum).matrix();
    }

    output.block(0, qh * kHeadDim, sequence, kHeadDim) = probs * v_block;
  }
  return {output, present_key, present_value};
}

#endif  // GROUP_QUERY_ATTENTION_H
