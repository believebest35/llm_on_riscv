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
    const int rows = static_cast<int>(input.rows());
    const int hidden_dim = static_cast<int>(input.cols());
    const int head_dim = hidden_dim / num_heads;
    const int rope_dim = head_dim / 2;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(rows,
                                                                  hidden_dim);

    for (int r = 0; r < rows; ++r) {
      const int pos = position_id + r;
      for (int h = 0; h < num_heads; ++h) {
        const int off = h * head_dim;
        for (int d = 0; d < head_dim; ++d) {
          const Scalar x = input(r, off + d);
          const int cache_d = (d < rope_dim) ? d : d - rope_dim;
          const Scalar c = cos_cache(pos, cache_d);
          const Scalar s = sin_cache(pos, cache_d);
          // rotate_half: [x1, x2] -> [-x2, x1]
          const Scalar rh = (d < rope_dim)
                                ? -input(r, off + d + rope_dim)
                                : input(r, off + d - rope_dim);
          output(r, off + d) = x * c + rh * s;
        }
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
  const std::vector<std::tuple<int, int, int, int, int>> test_cases = {
      {1, 1, 4, 128, 0},    {2, 2, 4, 128, 0},   {3, 4, 8, 128, 0},
      {6, 4, 8, 16, 3},     {1, 16, 128, 128, 0}, {3, 16, 128, 128, 0},
      // docs/01_dev.md (Qwen3-0.6B) shapes, with (40960, 64) caches.
      {4, 16, 128, 40960, 0}, {4, 8, 128, 40960, 0},
  };

  const float tolerance = 0.0f;

  for (const auto& [seq, num_heads, head_dim, cache_rows, position_id] :
       test_cases) {
    const int hidden_dim = num_heads * head_dim;

    auto input = generate_random_matrix<float>(seq, hidden_dim, -0.5f, 0.5f);
    Eigen::MatrixXf cos_cache;
    Eigen::MatrixXf sin_cache;
    fill_rope_table(cos_cache, sin_cache, cache_rows, cache_rows,
                    head_dim / 2, head_dim);

    std::string test_name = "RotaryEmbedding seq=" + std::to_string(seq) +
                            " heads=" + std::to_string(num_heads) +
                            " head_dim=" + std::to_string(head_dim) +
                            " position_id=" + std::to_string(position_id);

    test_with_golden(input, position_id, cos_cache, sin_cache, num_heads,
                     tolerance, test_name);
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
