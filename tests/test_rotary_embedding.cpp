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
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cos_cache,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& sin_cache,
      int sequence_length, int num_heads) {
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
      int pos = static_cast<int>(r % sequence_length);

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
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cos_cache,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& sin_cache,
      int sequence_length, int num_heads, Scalar tolerance,
      const std::string& test_description = "") {
    std::optional<Eigen::Matrix<int, Eigen::Dynamic, 1>> no_pos;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result = rotary_embedding<Scalar>(
                        input, no_pos, cos_cache, sin_cache, sequence_length,
                        num_heads))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_rotary_embedding(
                        input, cos_cache, sin_cache, sequence_length, num_heads))
        << "Golden reference failed: " << test_description;

    EXPECT_EQ(impl_result.rows(), golden_result.rows())
        << "Row count mismatch: " << test_description;
    EXPECT_EQ(impl_result.cols(), golden_result.cols())
        << "Column count mismatch: " << test_description;

    Scalar l2_diff = calculate_l2_difference(impl_result, golden_result);
    EXPECT_LE(l2_diff, tolerance)
        << "L2 difference exceeds tolerance: " << test_description;

    std::cout << test_description << "\nMax absolute difference: "
              << calculate_max_abs_difference(impl_result, golden_result)
              << "\nRMSE: " << calculate_rmse(impl_result, golden_result)
              << "\nL2 difference: " << l2_diff << std::endl;
  }
};

TEST_F(RotaryEmbeddingTest, RandomMatrices) {
  const std::vector<std::tuple<int, int, int>> test_cases = {
      {1, 1, 4}, {2, 2, 4}, {3, 4, 8}, {1, 16, 128}, {3, 16, 128},
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

    test_with_golden(input, cos_cache, sin_cache, seq, num_heads, tolerance,
                     test_name);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
