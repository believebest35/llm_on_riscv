#ifndef GROUP_QUERY_ATTENTION_H
#define GROUP_QUERY_ATTENTION_H

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

/**
 * @brief Group query attention for batch size one (ONNX
 * `GroupQueryAttention`).
 *
 * Batch size is fixed to 1, so all inputs are flattened to 2-D. `num_heads` and
 * `kv_num_heads` mirror the ONNX attributes of the same names; `head_dim` is
 * inferred as query.cols() / num_heads. Only the graph configuration is
 * supported: softcap = 0 and do_rotary = 0 (RoPE is applied by the separate
 * rotary_embedding kernel); rotary_interleaved is irrelevant in that case.
 * The scale attribute is assumed to be 1 / sqrt(head_dim), which matches the
 * Qwen3 graph.
 *
 * @tparam Scalar Element type of the inputs, caches, and outputs.
 * @param query Query matrix (sequence x num_heads * head_dim).
 * @param key Key matrix (sequence x kv_num_heads * head_dim).
 * @param value Value matrix (sequence x kv_num_heads * head_dim).
 * @param past_key Past key cache (kv_num_heads * past_sequence x head_dim).
 * @param past_value Past value cache (kv_num_heads * past_sequence x
 *        head_dim).
 * @param num_heads Number of query heads (ONNX attribute `num_heads`).
 * @param kv_num_heads Number of key/value heads (ONNX attribute
 *        `kv_num_heads`).
 * @param seqlens_k Last valid KV index (total_sequence_length - 1).
 * @param total_sequence_length Must equal past_sequence + sequence.
 * @return std::tuple<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
 *         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
 *         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> Output
 *         (sequence x num_heads * head_dim), present_key and present_value
 *         (kv_num_heads * total_sequence_length x head_dim).
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
    int num_heads, int kv_num_heads, int seqlens_k,
    int total_sequence_length) {
  if (num_heads <= 0) {
    throw std::invalid_argument(
        "group_query_attention: num_heads = " + std::to_string(num_heads) +
        ", expected > 0");
  }
  if (kv_num_heads <= 0) {
    throw std::invalid_argument(
        "group_query_attention: kv_num_heads = " +
        std::to_string(kv_num_heads) + ", expected > 0");
  }
  if (query.cols() % num_heads != 0) {
    throw std::invalid_argument(
        "group_query_attention: query.cols() = " +
        std::to_string(query.cols()) +
        ", expected divisible by num_heads = " + std::to_string(num_heads));
  }
  const int head_dim = static_cast<int>(query.cols()) / num_heads;
  if (head_dim <= 0) {
    throw std::invalid_argument(
        "group_query_attention: query.cols() = " +
        std::to_string(query.cols()) + ", expected a positive head_dim");
  }
  if (num_heads % kv_num_heads != 0) {
    throw std::invalid_argument(
        "group_query_attention: num_heads = " + std::to_string(num_heads) +
        ", kv_num_heads = " + std::to_string(kv_num_heads) +
        ", expected num_heads divisible by kv_num_heads");
  }
  if (query.rows() != key.rows() || query.rows() != value.rows()) {
    throw std::invalid_argument(
        "group_query_attention: query.rows() = " +
        std::to_string(query.rows()) + ", key.rows() = " +
        std::to_string(key.rows()) + ", value.rows() = " +
        std::to_string(value.rows()) + ", expected equal");
  }
  if (key.cols() != kv_num_heads * head_dim ||
      value.cols() != kv_num_heads * head_dim) {
    throw std::invalid_argument(
        "group_query_attention: key.cols() = " +
        std::to_string(key.cols()) + ", value.cols() = " +
        std::to_string(value.cols()) + ", expected kv_num_heads * head_dim = " +
        std::to_string(kv_num_heads * head_dim));
  }
  if (past_key.cols() != head_dim || past_value.cols() != head_dim) {
    throw std::invalid_argument(
        "group_query_attention: past_key.cols() = " +
        std::to_string(past_key.cols()) + ", past_value.cols() = " +
        std::to_string(past_value.cols()) + ", expected " +
        std::to_string(head_dim));
  }
  if (past_key.rows() != past_value.rows()) {
    throw std::invalid_argument(
        "group_query_attention: past_key.rows() = " +
        std::to_string(past_key.rows()) + ", past_value.rows() = " +
        std::to_string(past_value.rows()) + ", expected equal");
  }
  if (past_key.rows() % kv_num_heads != 0) {
    throw std::invalid_argument(
        "group_query_attention: past_key.rows() = " +
        std::to_string(past_key.rows()) + ", expected divisible by " +
        std::to_string(kv_num_heads));
  }

  const int sequence = static_cast<int>(query.rows());
  const int past_sequence =
      static_cast<int>(past_key.rows() / kv_num_heads);

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
      kv_num_heads * total_sequence_length, head_dim);
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> present_value(
      kv_num_heads * total_sequence_length, head_dim);

  for (int h = 0; h < kv_num_heads; ++h) {
    if (past_sequence > 0) {
      present_key.block(h * total_sequence_length, 0, past_sequence,
                        head_dim) =
          past_key.block(h * past_sequence, 0, past_sequence, head_dim);
      present_value.block(h * total_sequence_length, 0, past_sequence,
                          head_dim) =
          past_value.block(h * past_sequence, 0, past_sequence, head_dim);
    }
    present_key.block(h * total_sequence_length + past_sequence, 0, sequence,
                      head_dim) =
        key.block(0, h * head_dim, sequence, head_dim);
    present_value.block(h * total_sequence_length + past_sequence, 0, sequence,
                        head_dim) =
        value.block(0, h * head_dim, sequence, head_dim);
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(
      sequence, num_heads * head_dim);
  const Scalar scale = static_cast<Scalar>(
      1.0 / std::sqrt(static_cast<double>(head_dim)));
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

  const int group_size = num_heads / kv_num_heads;
  for (int qh = 0; qh < num_heads; ++qh) {
    const int kvh = qh / group_size;
    const int kv_offset = kvh * total_sequence_length;
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> q_block =
        query.block(0, qh * head_dim, sequence, head_dim);
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> k_block =
        present_key.block(kv_offset, 0, total_sequence_length, head_dim);
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> v_block =
        present_value.block(kv_offset, 0, total_sequence_length, head_dim);

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

    output.block(0, qh * head_dim, sequence, head_dim) = probs * v_block;
  }
  return {output, present_key, present_value};
}

#endif  // GROUP_QUERY_ATTENTION_H
