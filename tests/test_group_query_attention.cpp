#include <limits>

#include "core/group_query_attention.h"
#include "gtest_base.h"

class GroupQueryAttentionTest : public GTestBase {
 protected:
  static constexpr int kNumQueryHeads = 16;
  static constexpr int kNumKeyValueHeads = 8;
  static constexpr int kHeadDim = 128;
  static constexpr int kQueryHidden = kNumQueryHeads * kHeadDim;
  static constexpr int kKvHidden = kNumKeyValueHeads * kHeadDim;

  template <typename Scalar>
  std::tuple<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
             Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
             Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
  golden_reference_group_query_attention(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& query,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& key,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& value,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& past_key,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& past_value,
      int seqlens_k, int total_sequence_length) {
    const int sequence = static_cast<int>(query.rows());
    const int past_sequence =
        static_cast<int>(past_key.rows() / kNumKeyValueHeads);

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> present_key(
        kNumKeyValueHeads * total_sequence_length, kHeadDim);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> present_value(
        kNumKeyValueHeads * total_sequence_length, kHeadDim);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(
        sequence, kQueryHidden);

    for (int h = 0; h < kNumKeyValueHeads; ++h) {
      for (int s = 0; s < past_sequence; ++s) {
        for (int d = 0; d < kHeadDim; ++d) {
          present_key(h * total_sequence_length + s, d) =
              past_key(h * past_sequence + s, d);
          present_value(h * total_sequence_length + s, d) =
              past_value(h * past_sequence + s, d);
        }
      }
      for (int s = 0; s < sequence; ++s) {
        for (int d = 0; d < kHeadDim; ++d) {
          present_key(h * total_sequence_length + past_sequence + s, d) =
              key(s, h * kHeadDim + d);
          present_value(h * total_sequence_length + past_sequence + s, d) =
              value(s, h * kHeadDim + d);
        }
      }
    }

    const Scalar scale =
        static_cast<Scalar>(1.0 / std::sqrt(static_cast<double>(kHeadDim)));
    std::vector<Scalar> scores(total_sequence_length,
                               static_cast<Scalar>(0));
    std::vector<Scalar> probs(total_sequence_length,
                              static_cast<Scalar>(0));

    for (int qh = 0; qh < kNumQueryHeads; ++qh) {
      const int kvh = qh / 2;
      for (int qs = 0; qs < sequence; ++qs) {
        const int causal_limit = past_sequence + qs;
        Scalar max_score = -std::numeric_limits<Scalar>::infinity();

        for (int ks = 0; ks < total_sequence_length; ++ks) {
          if (ks > causal_limit || ks > seqlens_k) {
            scores[ks] = -std::numeric_limits<Scalar>::infinity();
            continue;
          }
          Scalar dot = static_cast<Scalar>(0);
          for (int d = 0; d < kHeadDim; ++d) {
            dot += query(qs, qh * kHeadDim + d) *
                   present_key(kvh * total_sequence_length + ks, d);
          }
          scores[ks] = dot * scale;
          if (scores[ks] > max_score) {
            max_score = scores[ks];
          }
        }

        Scalar sum = static_cast<Scalar>(0);
        for (int ks = 0; ks < total_sequence_length; ++ks) {
          if (!std::isfinite(scores[ks])) {
            probs[ks] = static_cast<Scalar>(0);
            continue;
          }
          probs[ks] =
              static_cast<Scalar>(std::exp(scores[ks] - max_score));
          sum += probs[ks];
        }
        for (int ks = 0; ks < total_sequence_length; ++ks) {
          probs[ks] /= sum;
        }

        for (int d = 0; d < kHeadDim; ++d) {
          Scalar acc = static_cast<Scalar>(0);
          for (int ks = 0; ks < total_sequence_length; ++ks) {
            acc += probs[ks] *
                   present_value(kvh * total_sequence_length + ks, d);
          }
          output(qs, qh * kHeadDim + d) = acc;
        }
      }
    }
    return {output, present_key, present_value};
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& query,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& key,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& value,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& past_key,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& past_value,
      int seqlens_k, int total_sequence_length, Scalar tolerance,
      const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_output,
        impl_pk, impl_pv;
    ASSERT_NO_THROW(
        std::tie(impl_output, impl_pk, impl_pv) =
            group_query_attention<Scalar>(query, key, value, past_key,
                                          past_value, seqlens_k,
                                          total_sequence_length))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_output,
        golden_pk, golden_pv;
    ASSERT_NO_THROW(
        std::tie(golden_output, golden_pk, golden_pv) =
            golden_reference_group_query_attention(
                query, key, value, past_key, past_value, seqlens_k,
                total_sequence_length))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_output, golden_output, tolerance,
                       test_description + " [output]");
    expect_matrix_near(impl_pk, golden_pk, tolerance,
                       test_description + " [present_key]");
    expect_matrix_near(impl_pv, golden_pv, tolerance,
                       test_description + " [present_value]");
  }
};

TEST_F(GroupQueryAttentionTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {3, 0},
      {1, 4},
      {1, 8},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {4, 3},
  };

  const float tolerance = 1e-6f;

  for (const auto& [seq, past_seq] : test_cases) {
    const int total_seq = past_seq + seq;
    const int seqlens_k = total_seq - 1;

    auto query =
        generate_random_matrix<float>(seq, kQueryHidden, -0.5f, 0.5f);
    auto key =
        generate_random_matrix<float>(seq, kKvHidden, -0.5f, 0.5f);
    auto value =
        generate_random_matrix<float>(seq, kKvHidden, -0.5f, 0.5f);
    auto past_key = generate_random_matrix<float>(
        kNumKeyValueHeads * past_seq, kHeadDim, -0.5f, 0.5f);
    auto past_value = generate_random_matrix<float>(
        kNumKeyValueHeads * past_seq, kHeadDim, -0.5f, 0.5f);

    std::string test_name = "GQA seq=" + std::to_string(seq) +
                            " past_seq=" + std::to_string(past_seq);

    test_with_golden(query, key, value, past_key, past_value, seqlens_k,
                     total_seq, tolerance, test_name);
  }
}

TEST_F(GroupQueryAttentionTest, InvalidInputs) {
  constexpr int kSeq = 2;

  auto query = generate_random_matrix<float>(kSeq, kQueryHidden, -0.5f, 0.5f);
  auto key = generate_random_matrix<float>(kSeq, kKvHidden, -0.5f, 0.5f);
  auto value = generate_random_matrix<float>(kSeq, kKvHidden, -0.5f, 0.5f);
  Eigen::MatrixXf empty_past(0, kHeadDim);

  // Wrong query hidden size.
  auto bad_query = generate_random_matrix<float>(kSeq, kKvHidden);
  EXPECT_THROW(group_query_attention<float>(bad_query, key, value, empty_past,
                                            empty_past, kSeq, kSeq),
               std::invalid_argument);

  // seqlens_k is the last valid KV index, so it must be in
  // [0, total_sequence_length).
  EXPECT_THROW(group_query_attention<float>(query, key, value, empty_past,
                                            empty_past, -1, kSeq),
               std::invalid_argument);
  EXPECT_THROW(group_query_attention<float>(query, key, value, empty_past,
                                            empty_past, kSeq, kSeq),
               std::invalid_argument);
}
