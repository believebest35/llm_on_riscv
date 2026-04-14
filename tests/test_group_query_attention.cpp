#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

#include "core/group_query_attention.h"

namespace {

constexpr int kNumQueryHeads = 16;
constexpr int kNumKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kQueryHidden = kNumQueryHeads * kHeadDim;
constexpr int kKvHidden = kNumKvHeads * kHeadDim;

template <typename Scalar>
Scalar almost_equal_eps() { return static_cast<Scalar>(1e-5); }

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> make_matrix(
    int rows, int cols, Scalar base) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> m(rows, cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      m(r, c) = base + static_cast<Scalar>((r * 7 + c % 17) * 0.01);
    }
  }
  return m;
}

template <typename Scalar>
void expect_matrix_near(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& lhs,
                        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& rhs) {
  ASSERT_EQ(lhs.rows(), rhs.rows());
  ASSERT_EQ(lhs.cols(), rhs.cols());
  for (int r = 0; r < lhs.rows(); ++r) {
    for (int c = 0; c < lhs.cols(); ++c) {
      EXPECT_NEAR(lhs(r, c), rhs(r, c), almost_equal_eps<Scalar>());
    }
  }
}

template <typename Scalar>
std::tuple<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
           Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
           Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
reference_gqa(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& query,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& key,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& value,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& past_key,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& past_value,
    int seqlens_k,
    int total_sequence_length) {
  const int sequence = query.rows();
  const int past_sequence = past_key.rows() / kNumKvHeads;

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> present_key(
      kNumKvHeads * total_sequence_length, kHeadDim);
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> present_value(
      kNumKvHeads * total_sequence_length, kHeadDim);
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(sequence, kQueryHidden);

  for (int h = 0; h < kNumKvHeads; ++h) {
    for (int s = 0; s < past_sequence; ++s) {
      for (int d = 0; d < kHeadDim; ++d) {
        present_key(h * total_sequence_length + s, d) = past_key(h * past_sequence + s, d);
        present_value(h * total_sequence_length + s, d) = past_value(h * past_sequence + s, d);
      }
    }
    for (int s = 0; s < sequence; ++s) {
      for (int d = 0; d < kHeadDim; ++d) {
        present_key(h * total_sequence_length + past_sequence + s, d) = key(s, h * kHeadDim + d);
        present_value(h * total_sequence_length + past_sequence + s, d) =
            value(s, h * kHeadDim + d);
      }
    }
  }

  const Scalar scale = static_cast<Scalar>(1.0 / std::sqrt(static_cast<double>(kHeadDim)));
  std::vector<Scalar> scores(total_sequence_length, static_cast<Scalar>(0));
  std::vector<Scalar> probs(total_sequence_length, static_cast<Scalar>(0));

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
          dot += query(qs, qh * kHeadDim + d) * present_key(kvh * total_sequence_length + ks, d);
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
        probs[ks] = static_cast<Scalar>(std::exp(scores[ks] - max_score));
        sum += probs[ks];
      }
      for (int ks = 0; ks < total_sequence_length; ++ks) {
        probs[ks] /= sum;
      }

      for (int d = 0; d < kHeadDim; ++d) {
        Scalar acc = static_cast<Scalar>(0);
        for (int ks = 0; ks < total_sequence_length; ++ks) {
          acc += probs[ks] * present_value(kvh * total_sequence_length + ks, d);
        }
        output(qs, qh * kHeadDim + d) = acc;
      }
    }
  }

  return {output, present_key, present_value};
}

}  // namespace

TEST(GroupQueryAttentionTest, PrefillMatchesReference) {
  auto query = make_matrix<float>(3, kQueryHidden, 0.1f);
  auto key = make_matrix<float>(3, kKvHidden, 0.2f);
  auto value = make_matrix<float>(3, kKvHidden, 0.3f);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> past_key(0, kHeadDim);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> past_value(0, kHeadDim);
  const int seqlens_k = 3;

  auto impl =
      group_query_attention<float>(query, key, value, past_key, past_value, seqlens_k, 3);
  auto ref = reference_gqa<float>(query, key, value, past_key, past_value, seqlens_k, 3);

  expect_matrix_near(std::get<0>(impl), std::get<0>(ref));
  expect_matrix_near(std::get<1>(impl), std::get<1>(ref));
  expect_matrix_near(std::get<2>(impl), std::get<2>(ref));
}

TEST(GroupQueryAttentionTest, DecodeMatchesReferenceAndConcatCache) {
  auto query = make_matrix<float>(1, kQueryHidden, 0.4f);
  auto key = make_matrix<float>(1, kKvHidden, 0.5f);
  auto value = make_matrix<float>(1, kKvHidden, 0.6f);
  auto past_key = make_matrix<float>(kNumKvHeads * 4, kHeadDim, -0.2f);
  auto past_value = make_matrix<float>(kNumKvHeads * 4, kHeadDim, -0.1f);
  const int seqlens_k = 5;

  auto impl =
      group_query_attention<float>(query, key, value, past_key, past_value, seqlens_k, 5);
  auto ref = reference_gqa<float>(query, key, value, past_key, past_value, seqlens_k, 5);

  expect_matrix_near(std::get<0>(impl), std::get<0>(ref));
  expect_matrix_near(std::get<1>(impl), std::get<1>(ref));
  expect_matrix_near(std::get<2>(impl), std::get<2>(ref));
}

TEST(GroupQueryAttentionTest, GroupMappingUsesHeadPairs) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> query(1, kQueryHidden);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> key(1, kKvHidden);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> value(1, kKvHidden);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> past_key(0, kHeadDim);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> past_value(0, kHeadDim);
  const int seqlens_k = 1;

  for (int h = 0; h < kNumQueryHeads; ++h) {
    for (int d = 0; d < kHeadDim; ++d) {
      query(0, h * kHeadDim + d) = static_cast<float>(h + 1);
    }
  }
  for (int h = 0; h < kNumKvHeads; ++h) {
    for (int d = 0; d < kHeadDim; ++d) {
      key(0, h * kHeadDim + d) = 1.0f;
      value(0, h * kHeadDim + d) = static_cast<float>(100 + h);
    }
  }

  auto impl =
      group_query_attention<float>(query, key, value, past_key, past_value, seqlens_k, 1);

  const auto& out = std::get<0>(impl);
  for (int qh = 0; qh < kNumQueryHeads; ++qh) {
    const int kvh = qh / 2;
    for (int d = 0; d < kHeadDim; ++d) {
      EXPECT_NEAR(out(0, qh * kHeadDim + d), static_cast<float>(100 + kvh), 1e-6);
    }
  }
}

TEST(GroupQueryAttentionTest, ThrowsForInvalidShapesAndLengths) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> query(2, kQueryHidden);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> key(2, kKvHidden);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> value(2, kKvHidden);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> past_key(kNumKvHeads, kHeadDim);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> past_value(kNumKvHeads, kHeadDim);
  const int seqlens_k = 3;

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> bad_query_hidden(2, kQueryHidden - 1);
  EXPECT_THROW(group_query_attention<float>(bad_query_hidden, key, value, past_key, past_value,
                                            seqlens_k, 3),
               std::invalid_argument);

  EXPECT_THROW(
      group_query_attention<float>(query, key, value, past_key, past_value, seqlens_k, 100),
      std::invalid_argument);

  EXPECT_THROW(group_query_attention<float>(query, key, value, past_key, past_value,
                                            0, 3),
               std::invalid_argument);
  EXPECT_THROW(group_query_attention<float>(query, key, value, past_key, past_value,
                                            4, 3),
               std::invalid_argument);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
