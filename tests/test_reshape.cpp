#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/reshape.h"
#include "gtest_base.h"

class ReshapeTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_reshape(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      int new_rows, int new_cols) {
    const std::int64_t total = A.size();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(new_rows,
                                                              new_cols);
    for (std::int64_t i = 0; i < total; ++i) {
      std::int64_t r = i / A.cols();
      std::int64_t c = i % A.cols();
      std::int64_t r2 = i / new_cols;
      std::int64_t c2 = i % new_cols;
      out(r2, c2) = A(r, c);
    }
    return out;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      int new_rows, int new_cols, Scalar tolerance,
      const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result = reshape<Scalar>(A, new_rows, new_cols))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(
        golden_result = golden_reference_reshape<Scalar>(A, new_rows, new_cols))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_result, golden_result, tolerance, test_description);
  }
};

TEST_F(ReshapeTest, ValidInputs) {
  const std::vector<std::tuple<int, int, int, int>> test_cases = {
      {2, 3, 3, 2}, {2, 3, 1, 6}, {2, 3, 6, 1},
      {4, 4, 2, 8}, {64, 128, 128, 64}, {256, 256, 1, 65536},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {4, 2048, 64, 128}, {4, 1024, 32, 128},
      {64, 128, 4, 2048}, {32, 128, 4, 1024},
  };

  const float tolerance = 1e-5;

  for (const auto& [rows, cols, new_rows, new_cols] : test_cases) {
    auto A = generate_random_matrix<float>(rows, cols, -5.0f, 5.0f);

    std::string test_name = "Reshape " + std::to_string(rows) + "x" +
                            std::to_string(cols) + " -> " +
                            std::to_string(new_rows) + "x" +
                            std::to_string(new_cols);

    test_with_golden(A, new_rows, new_cols, tolerance, test_name);
  }
}


TEST_F(ReshapeTest, InvalidInputs) {
  auto A = generate_random_matrix<float>(2, 3);
  EXPECT_THROW(reshape<float>(A, 4, 4), std::invalid_argument);
  EXPECT_THROW(reshape<float>(A, -1, 6), std::invalid_argument);
}
