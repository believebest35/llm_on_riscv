#include <gtest/gtest.h>

#include <string>

#include "core/rotary_embedding.h"
#include "gtest_base.h"

class RotaryEmbeddingTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_llama_rope_row(
      const Eigen::Ref<const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>>& x,
      const Eigen::Ref<const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>>& cos_half,
      const Eigen::Ref<const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>>& sin_half) {
    const int head_dim = static_cast<int>(x.size());
    const int rope_dim = head_dim / 2;
    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> rh(1, head_dim);
    rotary_rotate_half_row<Scalar>(x, rh);
    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> cos_full(1, head_dim);
    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> sin_full(1, head_dim);
    cos_full.leftCols(rope_dim) = cos_half;
    cos_full.rightCols(rope_dim) = cos_half;
    sin_full.leftCols(rope_dim) = sin_half;
    sin_full.rightCols(rope_dim) = sin_half;
    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> y(1, head_dim);
    y.array() = x.array() * cos_full.array() + rh.array() * sin_full.array();
    return y;
  }
};

TEST_F(RotaryEmbeddingTest, RotateHalfDim4) {
  Eigen::Matrix<float, 1, Eigen::Dynamic> x(1, 4);
  x << 1.f, 2.f, 3.f, 4.f;
  Eigen::Matrix<float, 1, Eigen::Dynamic> rh(1, 4);
  rotary_rotate_half_row<float>(x, rh);
  EXPECT_FLOAT_EQ(rh(0), -3.f);
  EXPECT_FLOAT_EQ(rh(1), -4.f);
  EXPECT_FLOAT_EQ(rh(2), 1.f);
  EXPECT_FLOAT_EQ(rh(3), 2.f);
}

TEST_F(RotaryEmbeddingTest, SingleHeadExplicitPosition) {
  const int head_dim = 4;
  const int max_pos = 8;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> cache_cos(max_pos,
                                                                 head_dim / 2);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> cache_sin(max_pos,
                                                                 head_dim / 2);
  cache_cos.setZero();
  cache_sin.setZero();
  cache_cos.row(3) << 0.5f, 0.25f;
  cache_sin.row(3) << 0.25f, 0.5f;

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> input(1, head_dim);
  input << 1.f, 2.f, 3.f, 4.f;

  Eigen::Matrix<int, Eigen::Dynamic, 1> pos(1);
  pos(0) = 3;

  std::optional<Eigen::Matrix<int, Eigen::Dynamic, 1>> opt_pos = pos;
  auto out = rotary_embedding<float>(input, opt_pos, cache_cos, cache_sin, 1,
                                     /*num_heads=*/1);

  auto want = golden_llama_rope_row<float>(input.row(0), cache_cos.row(3),
                                           cache_sin.row(3));
  for (int i = 0; i < head_dim; ++i) {
    EXPECT_NEAR(out(0, i), want(i), 1e-5f) << "i=" << i;
  }
}

TEST_F(RotaryEmbeddingTest, PrefillImplicitPositionsTwoBatch) {
  const int seq = 2;
  const int batch = 2;
  const int head_dim = 4;
  const int num_heads = 1;
  const int max_pos = 16;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> cache_cos(max_pos,
                                                                 head_dim / 2);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> cache_sin(max_pos,
                                                                 head_dim / 2);
  for (int p = 0; p < max_pos; ++p) {
    for (int j = 0; j < head_dim / 2; ++j) {
      cache_cos(p, j) = static_cast<float>(0.1f * (p + 1) + 0.01f * j);
      cache_sin(p, j) = static_cast<float>(0.05f * (p + 1) + 0.02f * j);
    }
  }

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> input(batch * seq,
                                                               head_dim);
  input.row(0) << 1.f, 0.f, 0.f, 0.f;
  input.row(1) << 0.f, 1.f, 0.f, 0.f;
  input.row(2) << 0.f, 0.f, 1.f, 0.f;
  input.row(3) << 0.f, 0.f, 0.f, 1.f;

  std::optional<Eigen::Matrix<int, Eigen::Dynamic, 1>> no_pos;
  auto out = rotary_embedding<float>(input, no_pos, cache_cos, cache_sin, seq,
                                     num_heads);

  for (int r = 0; r < batch * seq; ++r) {
    const int p = r % seq;
    auto want = golden_llama_rope_row<float>(input.row(r), cache_cos.row(p),
                                             cache_sin.row(p));
    for (int i = 0; i < head_dim; ++i) {
      EXPECT_NEAR(out(r, i), want(i), 1e-5f)
          << "r=" << r << " i=" << i;
    }
  }
}

TEST_F(RotaryEmbeddingTest, MultiHead2048Layout) {
  const int num_heads = 16;
  const int head_dim = 128;
  const int hidden_dim = num_heads * head_dim;
  const int seq = 3;
  const int rows = seq;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> cache_cos(
      kRotaryEmbeddingMaxSeqLen, head_dim / 2);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> cache_sin(
      kRotaryEmbeddingMaxSeqLen, head_dim / 2);
  cache_cos.setRandom();
  cache_sin.setRandom();

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> input =
      generate_random_matrix<float>(rows, hidden_dim, -0.5f, 0.5f);

  std::optional<Eigen::Matrix<int, Eigen::Dynamic, 1>> no_pos;
  auto out = rotary_embedding<float>(input, no_pos, cache_cos, cache_sin, seq,
                                     num_heads);

  for (int r = 0; r < rows; ++r) {
    const int p = r % seq;
    for (int h = 0; h < num_heads; ++h) {
      auto x = input.row(r).segment(h * head_dim, head_dim);
      auto want = golden_llama_rope_row<float>(x, cache_cos.row(p),
                                               cache_sin.row(p));
      for (int i = 0; i < head_dim; ++i) {
        EXPECT_NEAR(out(r, h * head_dim + i), want(i), 1e-4f)
            << "r=" << r << " h=" << h << " i=" << i;
      }
    }
  }
}

TEST_F(RotaryEmbeddingTest, ErrorCases) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> input(2, 4);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> cos_t(10, 2);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> sin_t(10, 2);
  cos_t.setZero();
  sin_t.setZero();

  std::optional<Eigen::Matrix<int, Eigen::Dynamic, 1>> no_pos;
  EXPECT_THROW(
      (void)rotary_embedding<float>(input, no_pos, cos_t, sin_t, 0, 1),
      std::invalid_argument);

  Eigen::Matrix<int, Eigen::Dynamic, 1> bad_pos(1);
  bad_pos(0) = 0;
  std::optional<Eigen::Matrix<int, Eigen::Dynamic, 1>> opt_bad = bad_pos;
  EXPECT_THROW((void)rotary_embedding<float>(input, opt_bad, cos_t, sin_t, 2, 1),
               std::invalid_argument);

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> bad_cos(10, 3);
  bad_cos.setZero();
  EXPECT_THROW(
      (void)rotary_embedding<float>(input, no_pos, bad_cos, sin_t, 2, 1),
      std::invalid_argument);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
