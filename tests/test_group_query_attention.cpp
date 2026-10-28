#include "core/group_query_attention.h"
#include "gtest_base.h"

class GroupQueryAttentionTest : public GTestBase {
 protected:
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
      int num_heads, int kv_num_heads, int seqlens_k,
      int total_sequence_length) {
    const int sequence = static_cast<int>(query.rows());
    const int head_dim = static_cast<int>(query.cols()) / num_heads;
    const int past_sequence =
        static_cast<int>(past_key.rows() / kv_num_heads);
    const int group_size = num_heads / kv_num_heads;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> present_key(
        kv_num_heads * total_sequence_length, head_dim);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> present_value(
        kv_num_heads * total_sequence_length, head_dim);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(
        sequence, num_heads * head_dim);

    for (int h = 0; h < kv_num_heads; ++h) {
      for (int s = 0; s < past_sequence; ++s) {
        for (int d = 0; d < head_dim; ++d) {
          present_key(h * total_sequence_length + s, d) =
              past_key(h * past_sequence + s, d);
          present_value(h * total_sequence_length + s, d) =
              past_value(h * past_sequence + s, d);
        }
      }
      for (int s = 0; s < sequence; ++s) {
        for (int d = 0; d < head_dim; ++d) {
          present_key(h * total_sequence_length + past_sequence + s, d) =
              key(s, h * head_dim + d);
          present_value(h * total_sequence_length + past_sequence + s, d) =
              value(s, h * head_dim + d);
        }
      }
    }

    const Scalar scale =
        static_cast<Scalar>(1.0 / std::sqrt(static_cast<double>(head_dim)));
    std::vector<Scalar> scores(total_sequence_length,
                               static_cast<Scalar>(0));
    std::vector<Scalar> probs(total_sequence_length,
                              static_cast<Scalar>(0));

    for (int qh = 0; qh < num_heads; ++qh) {
      const int kvh = qh / group_size;
      for (int qs = 0; qs < sequence; ++qs) {
        const int causal_limit = past_sequence + qs;
        Scalar max_score = -std::numeric_limits<Scalar>::infinity();

        for (int ks = 0; ks < total_sequence_length; ++ks) {
          if (ks > causal_limit || ks > seqlens_k) {
            scores[ks] = -std::numeric_limits<Scalar>::infinity();
            continue;
          }
          Scalar dot = static_cast<Scalar>(0);
          for (int d = 0; d < head_dim; ++d) {
            dot += query(qs, qh * head_dim + d) *
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

        for (int d = 0; d < head_dim; ++d) {
          Scalar acc = static_cast<Scalar>(0);
          for (int ks = 0; ks < total_sequence_length; ++ks) {
            acc += probs[ks] *
                   present_value(kvh * total_sequence_length + ks, d);
          }
          output(qs, qh * head_dim + d) = acc;
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
      int num_heads, int kv_num_heads, int seqlens_k,
      int total_sequence_length, Scalar tolerance,
      const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_output,
        impl_pk, impl_pv;
    ASSERT_NO_THROW(
        std::tie(impl_output, impl_pk, impl_pv) =
            group_query_attention<Scalar>(query, key, value, past_key,
                                          past_value, num_heads, kv_num_heads,
                                          seqlens_k, total_sequence_length))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_output,
        golden_pk, golden_pv;
    ASSERT_NO_THROW(
        std::tie(golden_output, golden_pk, golden_pv) =
            golden_reference_group_query_attention<Scalar>(
                query, key, value, past_key, past_value, num_heads,
                kv_num_heads, seqlens_k, total_sequence_length))
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
  const std::vector<std::tuple<int, int, int, int, int>> test_cases = {
      {3, 0, 16, 8, 128},
      {1, 4, 16, 8, 128},
      {1, 8, 16, 8, 128},
      {2, 1, 4, 2, 8},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {4, 3, 16, 8, 128},
  };

  const float tolerance = 1e-6f;

  for (const auto& [sequence, past_sequence, num_heads, kv_num_heads,
                    head_dim] : test_cases) {
    const int total_sequence_length = past_sequence + sequence;
    const int seqlens_k = total_sequence_length - 1;

    auto query = generate_random_matrix<float>(sequence, num_heads * head_dim,
                                               -0.5f, 0.5f);
    auto key = generate_random_matrix<float>(sequence, kv_num_heads * head_dim,
                                             -0.5f, 0.5f);
    auto value = generate_random_matrix<float>(
        sequence, kv_num_heads * head_dim, -0.5f, 0.5f);
    auto past_key = generate_random_matrix<float>(
        kv_num_heads * past_sequence, head_dim, -0.5f, 0.5f);
    auto past_value = generate_random_matrix<float>(
        kv_num_heads * past_sequence, head_dim, -0.5f, 0.5f);

    std::string test_description =
        "GroupQueryAttention " + std::to_string(sequence) + "x" +
        std::to_string(num_heads * head_dim) + " past_sequence=" +
        std::to_string(past_sequence) + " num_heads=" +
        std::to_string(num_heads) + " kv_num_heads=" +
        std::to_string(kv_num_heads) + " head_dim=" +
        std::to_string(head_dim);

    test_with_golden(query, key, value, past_key, past_value, num_heads,
                     kv_num_heads, seqlens_k, total_sequence_length,
                     tolerance, test_description);
  }
}

TEST_F(GroupQueryAttentionTest, InvalidInputs) {
  const int sequence = 2;

  auto query = generate_random_matrix<float>(sequence, 2048, -0.5f, 0.5f);
  auto key = generate_random_matrix<float>(sequence, 1024, -0.5f, 0.5f);
  auto value = generate_random_matrix<float>(sequence, 1024, -0.5f, 0.5f);
  Eigen::MatrixXf empty_past(0, 128);

  // num_heads and kv_num_heads must be positive.
  EXPECT_THROW(group_query_attention<float>(
                   query, key, value, empty_past, empty_past, 0, 8,
                   sequence - 1, sequence),
               std::invalid_argument);
  EXPECT_THROW(group_query_attention<float>(
                   query, key, value, empty_past, empty_past, 16, 0,
                   sequence - 1, sequence),
               std::invalid_argument);

  // query hidden size must be divisible by num_heads.
  auto indivisible_query = generate_random_matrix<float>(sequence, 2047);
  EXPECT_THROW(group_query_attention<float>(
                   indivisible_query, key, value, empty_past, empty_past,
                   16, 8, sequence - 1, sequence),
               std::invalid_argument);

  // A zero-width query yields head_dim == 0.
  Eigen::MatrixXf empty_query(sequence, 0);
  Eigen::MatrixXf empty_key(sequence, 0);
  Eigen::MatrixXf empty_value(sequence, 0);
  Eigen::MatrixXf empty_zero_past(0, 0);
  EXPECT_THROW(group_query_attention<float>(
                   empty_query, empty_key, empty_value, empty_zero_past,
                   empty_zero_past, 16, 8, sequence - 1, sequence),
               std::invalid_argument);

  // num_heads must be divisible by kv_num_heads.
  auto group_query = generate_random_matrix<float>(sequence, 1536);
  auto group_key = generate_random_matrix<float>(sequence, 1024);
  auto group_value = generate_random_matrix<float>(sequence, 1024);
  EXPECT_THROW(group_query_attention<float>(
                   group_query, group_key, group_value, empty_past,
                   empty_past, 12, 8, sequence - 1, sequence),
               std::invalid_argument);

  // query/key/value must share the sequence length.
  auto short_key = generate_random_matrix<float>(sequence - 1, 1024);
  EXPECT_THROW(group_query_attention<float>(
                   query, short_key, value, empty_past, empty_past,
                   16, 8, sequence - 1, sequence),
               std::invalid_argument);

  // key/value hidden sizes must equal kv_num_heads * head_dim.
  auto bad_value = generate_random_matrix<float>(sequence, 896);
  EXPECT_THROW(group_query_attention<float>(
                   query, key, bad_value, empty_past, empty_past,
                   16, 8, sequence - 1, sequence),
               std::invalid_argument);

  // Past caches must use head_dim columns.
  Eigen::MatrixXf bad_past(0, 64);
  EXPECT_THROW(group_query_attention<float>(
                   query, key, value, bad_past, empty_past, 16, 8,
                   sequence - 1, sequence),
               std::invalid_argument);

  // Past key/value caches must share the row count.
  Eigen::MatrixXf mismatched_past_key(16, 128);
  Eigen::MatrixXf mismatched_past_value(8, 128);
  EXPECT_THROW(group_query_attention<float>(
                   query, key, value, mismatched_past_key,
                   mismatched_past_value, 16, 8, sequence - 1, sequence),
               std::invalid_argument);

  // Past rows must be divisible by kv_num_heads.
  Eigen::MatrixXf ragged_past_key(1, 128);
  Eigen::MatrixXf ragged_past_value(1, 128);
  EXPECT_THROW(group_query_attention<float>(
                   query, key, value, ragged_past_key, ragged_past_value,
                   16, 8, sequence - 1, sequence),
               std::invalid_argument);

  // total_sequence_length must equal past_sequence + sequence.
  EXPECT_THROW(group_query_attention<float>(
                   query, key, value, empty_past, empty_past, 16, 8,
                   sequence - 1, sequence + 1),
               std::invalid_argument);

  // seqlens_k is the last valid KV index, so it must be in
  // [0, total_sequence_length).
  EXPECT_THROW(group_query_attention<float>(
                   query, key, value, empty_past, empty_past, 16, 8,
                   -1, sequence),
               std::invalid_argument);
  EXPECT_THROW(group_query_attention<float>(
                   query, key, value, empty_past, empty_past, 16, 8,
                   sequence, sequence),
               std::invalid_argument);
}
