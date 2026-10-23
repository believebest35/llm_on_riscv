#include <gtest/gtest.h>

#include <string>
#include <tuple>
#include <vector>

#include "core/rotary_embedding.h"
#include "gtest_base.h"

class RotaryEmbeddingTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_rotary_embedding(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
      int position_id,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cos_cache,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& sin_cache,
      int num_heads) {
    const Eigen::Index rows = input.rows();
    const Eigen::Index hidden_dim = input.cols();
    const Eigen::Index head_dim = hidden_dim / num_heads;
    const Eigen::Index rope_dim = head_dim / 2;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(rows,
                                                                  hidden_dim);
    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> rh(1, head_dim);
    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> cos_full(1, head_dim);
    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> sin_full(1, head_dim);

    for (Eigen::Index r = 0; r < rows; ++r) {
      const Eigen::Index pos = static_cast<Eigen::Index>(position_id) + r;

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

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
      int position_id,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cos_cache,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& sin_cache,
      int num_heads, Scalar tolerance,
      const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result = rotary_embedding<Scalar>(
                        input, position_id, cos_cache, sin_cache, num_heads))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_rotary_embedding(
                        input, position_id, cos_cache, sin_cache, num_heads))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_result, golden_result, tolerance, test_description);
  }
};

namespace {

// Build a genuine RoPE table for positions [0, rows) so that
// cos^2 + sin^2 == 1 and the transform is a pure rotation.
void fill_rope_table(Eigen::MatrixXf& cos_cache, Eigen::MatrixXf& sin_cache,
                     int cache_rows, int rows, int rope_dim, int head_dim) {
  cos_cache = Eigen::MatrixXf::Zero(cache_rows, rope_dim);
  sin_cache = Eigen::MatrixXf::Zero(cache_rows, rope_dim);
  for (int p = 0; p < rows; ++p) {
    for (int d = 0; d < rope_dim; ++d) {
      const float freq = std::pow(10000.0f, -2.0f * d / head_dim);
      cos_cache(p, d) = std::cos(static_cast<float>(p) * freq);
      sin_cache(p, d) = std::sin(static_cast<float>(p) * freq);
    }
  }
}

}  // namespace


TEST_F(RotaryEmbeddingTest, ValidInputs) {
  const std::vector<std::tuple<int, int, int>> test_cases = {
      {1, 1, 4}, {2, 2, 4}, {3, 4, 8}, {1, 16, 128}, {3, 16, 128},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {4, 16, 128}, {4, 8, 128},
  };

  const float tolerance = 1e-4f;

  for (const auto& [seq, num_heads, head_dim] : test_cases) {
    const int hidden_dim = num_heads * head_dim;
    const int rows = seq;
    const int max_pos = 128;

    auto input = generate_random_matrix<float>(rows, hidden_dim, -0.5f, 0.5f);
    auto cos_cache =
        generate_random_matrix<float>(max_pos, head_dim / 2, -1.0f, 1.0f);
    auto sin_cache =
        generate_random_matrix<float>(max_pos, head_dim / 2, -1.0f, 1.0f);

    std::string test_name = "RotaryEmbedding seq=" + std::to_string(seq) +
                            " heads=" + std::to_string(num_heads) +
                            " head_dim=" + std::to_string(head_dim);

    test_with_golden(input, 0, cos_cache, sin_cache, num_heads, tolerance,
                     test_name);
  }

  // Hand-picked cases: genuine cos/sin table over four positions.
  {
    constexpr int kHeads = 4;
    constexpr int kHeadDim = 8;
    constexpr int kRows = 4;

    Eigen::MatrixXf cos_cache;
    Eigen::MatrixXf sin_cache;
    fill_rope_table(cos_cache, sin_cache, 16, kRows, kHeadDim / 2, kHeadDim);

    auto input = generate_random_matrix<float>(kRows, kHeads * kHeadDim,
                                               -0.5f, 0.5f);

    test_with_golden(input, 0, cos_cache, sin_cache, kHeads, 1e-5f,
                     "RoPE genuine table");
  }

  // Single start position: rows use position_id + r.
  {
    constexpr int kHeads = 4;
    constexpr int kHeadDim = 8;
    constexpr int kRows = 6;

    Eigen::MatrixXf cos_cache;
    Eigen::MatrixXf sin_cache;
    fill_rope_table(cos_cache, sin_cache, 16, kRows, kHeadDim / 2, kHeadDim);

    auto input =
        generate_random_matrix<float>(kRows, kHeads * kHeadDim, -0.5f, 0.5f);

    test_with_golden(input, 3, cos_cache, sin_cache, kHeads, 1e-5f,
                     "RoPE single start position");
  }

  // Full-size cache from docs/01_dev.md: (40960, 64) for head_dim=128.
  {
    constexpr int kSeq = 4;
    constexpr int kHeads = 16;
    constexpr int kHeadDim = 128;
    constexpr int kMaxPos = 40960;

    auto input =
        generate_random_matrix<float>(kSeq, kHeads * kHeadDim, -0.5f, 0.5f);
    auto cos_cache = generate_random_matrix<float>(
        kMaxPos, kHeadDim / 2, -1.0f, 1.0f);
    auto sin_cache = generate_random_matrix<float>(
        kMaxPos, kHeadDim / 2, -1.0f, 1.0f);

    test_with_golden(input, 0, cos_cache, sin_cache, kHeads, 1e-4f,
                     "RoPE cache 40960x64");
  }
}




TEST_F(RotaryEmbeddingTest, InvalidInputs) {
  constexpr int kHeads = 4;
  constexpr int kHeadDim = 8;

  Eigen::MatrixXf cos_cache;
  Eigen::MatrixXf sin_cache;
  fill_rope_table(cos_cache, sin_cache, 16, 4, kHeadDim / 2, kHeadDim);

  // hidden_dim (6) is not divisible by num_heads (4).
  auto odd_input = generate_random_matrix<float>(2, 6);
  EXPECT_THROW(rotary_embedding<float>(odd_input, 0, cos_cache, sin_cache,
                                       kHeads),
               std::invalid_argument);

  // Cache last dim must equal head_dim / 2.
  auto bad_cache = generate_random_matrix<float>(16, kHeadDim);
  auto input = generate_random_matrix<float>(2, kHeads * kHeadDim);
  EXPECT_THROW(rotary_embedding<float>(input, 0, bad_cache, sin_cache, kHeads),
               std::invalid_argument);

  // position_id + input rows must stay within the cache.
  EXPECT_THROW(rotary_embedding<float>(input, 15, cos_cache, sin_cache, kHeads),
               std::out_of_range);
}
