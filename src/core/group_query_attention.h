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
    for (int s = 0; s < past_sequence; ++s) {
      for (int d = 0; d < kHeadDim; ++d) {
        present_key(h * total_sequence_length + s, d) = past_key(h * past_sequence + s, d);
        present_value(h * total_sequence_length + s, d) = past_value(h * past_sequence + s, d);
      }
    }
    for (int s = 0; s < sequence; ++s) {
      const int target_pos = past_sequence + s;
      const int hidden_base = h * kHeadDim;
      for (int d = 0; d < kHeadDim; ++d) {
        present_key(h * total_sequence_length + target_pos, d) = key(s, hidden_base + d);
        present_value(h * total_sequence_length + target_pos, d) = value(s, hidden_base + d);
      }
    }
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(sequence, kNumQueryHeads * kHeadDim);
  const Scalar scale = static_cast<Scalar>(1.0 / std::sqrt(static_cast<double>(kHeadDim)));
  std::vector<Scalar> scores(total_sequence_length, static_cast<Scalar>(0));
  std::vector<Scalar> probs(total_sequence_length, static_cast<Scalar>(0));

  for (int qh = 0; qh < kNumQueryHeads; ++qh) {
    const int kvh = qh / 2;
    const int q_hidden_base = qh * kHeadDim;
    for (int qs = 0; qs < sequence; ++qs) {
      const int causal_limit = past_sequence + qs;
      int valid_count = 0;
      Scalar max_score = -std::numeric_limits<Scalar>::infinity();

      for (int ks = 0; ks < total_sequence_length; ++ks) {
        if (ks > causal_limit || ks >= seqlens_k) {
          scores[ks] = -std::numeric_limits<Scalar>::infinity();
          continue;
        }

        Scalar dot = static_cast<Scalar>(0);
        for (int d = 0; d < kHeadDim; ++d) {
          dot += query(qs, q_hidden_base + d) * present_key(kvh * total_sequence_length + ks, d);
        }
        scores[ks] = dot * scale;
        if (scores[ks] > max_score) {
          max_score = scores[ks];
        }
        ++valid_count;
      }

      if (valid_count == 0) {
        throw std::invalid_argument("no valid key positions after applying masks");
      }

      Scalar exp_sum = static_cast<Scalar>(0);
      for (int ks = 0; ks < total_sequence_length; ++ks) {
        if (!std::isfinite(scores[ks])) {
          probs[ks] = static_cast<Scalar>(0);
          continue;
        }
        probs[ks] = static_cast<Scalar>(std::exp(scores[ks] - max_score));
        exp_sum += probs[ks];
      }
      if (exp_sum == static_cast<Scalar>(0)) {
        throw std::invalid_argument("softmax normalization sum became zero");
      }
      for (int ks = 0; ks < total_sequence_length; ++ks) {
        probs[ks] /= exp_sum;
      }

      for (int d = 0; d < kHeadDim; ++d) {
        Scalar acc = static_cast<Scalar>(0);
        for (int ks = 0; ks < total_sequence_length; ++ks) {
          if (probs[ks] == static_cast<Scalar>(0)) {
            continue;
          }
          acc += probs[ks] * present_value(kvh * total_sequence_length + ks, d);
        }
        output(qs, q_hidden_base + d) = acc;
      }
    }
  }

  return {output, present_key, present_value};
}

#endif  // GROUP_QUERY_ATTENTION_H
