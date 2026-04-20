#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "core/group_query_attention.h"
#include "gtest_base.h"

class GroupQueryAttentionTest : public GTestBase {
 protected:
  static constexpr int kNumQueryHeads = 16;
  static constexpr int kNumKvHeads = 8;
  static constexpr int kHeadDim = 128;
  static constexpr int kQueryHidden = kNumQueryHeads * kHeadDim;
  static constexpr int kKvHidden = kNumKvHeads * kHeadDim;

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
    const int sequence = query.rows();
    const int past_sequence = past_key.rows() / kNumKvHeads;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> present_key(
        kNumKvHeads * total_sequence_length, kHeadDim);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> present_value(
        kNumKvHeads * total_sequence_length, kHeadDim);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(
        sequence, kQueryHidden);

    for (int h = 0; h < kNumKvHeads; ++h) {
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
          if (ks > causal_limit || ks >= seqlens_k) {
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

    EXPECT_EQ(impl_output.rows(), golden_output.rows())
        << "Output row mismatch: " << test_description;
    EXPECT_EQ(impl_output.cols(), golden_output.cols())
        << "Output col mismatch: " << test_description;
    EXPECT_EQ(impl_pk.rows(), golden_pk.rows())
        << "PresentKey row mismatch: " << test_description;
    EXPECT_EQ(impl_pk.cols(), golden_pk.cols())
        << "PresentKey col mismatch: " << test_description;
    EXPECT_EQ(impl_pv.rows(), golden_pv.rows())
        << "PresentValue row mismatch: " << test_description;
    EXPECT_EQ(impl_pv.cols(), golden_pv.cols())
        << "PresentValue col mismatch: " << test_description;

    Scalar l2_diff_output =
        calculate_l2_difference(impl_output, golden_output);
    EXPECT_LE(l2_diff_output, tolerance)
        << "Output L2 diff too large: " << test_description;

    Scalar l2_diff_pk = calculate_l2_difference(impl_pk, golden_pk);
    EXPECT_LE(l2_diff_pk, tolerance)
        << "PresentKey L2 diff too large: " << test_description;

    Scalar l2_diff_pv = calculate_l2_difference(impl_pv, golden_pv);
    EXPECT_LE(l2_diff_pv, tolerance)
        << "PresentValue L2 diff too large: " << test_description;

    std::cout << test_description
              << "\n  Output  - Max abs diff: "
              << calculate_max_abs_difference(impl_output, golden_output)
              << " RMSE: " << calculate_rmse(impl_output, golden_output)
              << " L2: " << l2_diff_output
              << "\n  PresKey - Max abs diff: "
              << calculate_max_abs_difference(impl_pk, golden_pk)
              << " RMSE: " << calculate_rmse(impl_pk, golden_pk)
              << " L2: " << l2_diff_pk
              << "\n  PresVal - Max abs diff: "
              << calculate_max_abs_difference(impl_pv, golden_pv)
              << " RMSE: " << calculate_rmse(impl_pv, golden_pv)
              << " L2: " << l2_diff_pv << std::endl;
  }
};

TEST_F(GroupQueryAttentionTest, RandomMatrices) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {3, 0},
      {1, 4},
      {1, 8},
  };

  const float tolerance = 1e-3f;

  for (const auto& [seq, past_seq] : test_cases) {
    const int total_seq = past_seq + seq;
    const int seqlens_k = total_seq;

    auto query =
        generate_random_matrix<float>(seq, kQueryHidden, -0.5f, 0.5f);
    auto key =
        generate_random_matrix<float>(seq, kKvHidden, -0.5f, 0.5f);
    auto value =
        generate_random_matrix<float>(seq, kKvHidden, -0.5f, 0.5f);
    auto past_key = generate_random_matrix<float>(
        kNumKvHeads * past_seq, kHeadDim, -0.5f, 0.5f);
    auto past_value = generate_random_matrix<float>(
        kNumKvHeads * past_seq, kHeadDim, -0.5f, 0.5f);

    std::string test_name = "GQA seq=" + std::to_string(seq) +
                            " past_seq=" + std::to_string(past_seq);

    test_with_golden(query, key, value, past_key, past_value, seqlens_k,
                     total_seq, tolerance, test_name);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
