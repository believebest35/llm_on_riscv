#include "core/reshape.h"
#include "gtest_base.h"

class ReshapeTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_reshape(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
      int new_rows, int new_cols) {
    // ONNX Reshape: 0 copies the input dimension, -1 is inferred.
    if (new_rows == 0) {
      new_rows = static_cast<int>(input.rows());
    }
    if (new_cols == 0) {
      new_cols = static_cast<int>(input.cols());
    }
    if (new_rows == -1) {
      new_rows = static_cast<int>(input.size()) / new_cols;
    } else if (new_cols == -1) {
      new_cols = static_cast<int>(input.size()) / new_rows;
    }

    const std::int64_t total = input.size();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(new_rows,
                                                              new_cols);
    for (std::int64_t i = 0; i < total; ++i) {
      std::int64_t r = i / input.cols();
      std::int64_t c = i % input.cols();
      std::int64_t r2 = i / new_cols;
      std::int64_t c2 = i % new_cols;
      output(r2, c2) = input(r, c);
    }
    return output;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
      int new_rows, int new_cols, Scalar tolerance,
      const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_output;
    ASSERT_NO_THROW(impl_output = reshape<Scalar>(input, new_rows, new_cols))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_output;
    ASSERT_NO_THROW(golden_output = golden_reference_reshape<Scalar>(
                        input, new_rows, new_cols))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_output, golden_output, tolerance, test_description);
  }
};

TEST_F(ReshapeTest, ValidInputs) {
  const std::vector<std::tuple<int, int, int, int>> test_cases = {
      {2, 3, 3, 2},
      {2, 3, 1, 6},
      {2, 3, 6, 1},
      {2, 3, -1, 3},
      {2, 3, -1, 6},
      {2, 3, 0, 3},
      {2, 3, 2, 0},
      {4, 4, 2, 8},
      {64, 128, 128, 64},
      {256, 256, 1, 65536},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {1, 1, 1, 1},
      {4, 2048, 64, 128},
      {4, 1024, 32, 128},
      {64, 128, 4, 2048},
      {32, 128, 4, 1024},
  };

  const float tolerance = 0.0f;

  for (const auto& [rows, cols, new_rows, new_cols] : test_cases) {
    auto input = generate_random_matrix<float>(rows, cols, -5.0f, 5.0f);

    std::string test_description = "Reshape " + std::to_string(rows) + "x" +
                            std::to_string(cols) + " -> " +
                            std::to_string(new_rows) + "x" +
                            std::to_string(new_cols);

    test_with_golden(input, new_rows, new_cols, tolerance, test_description);
  }
}

TEST_F(ReshapeTest, InvalidInputs) {
  auto input = generate_random_matrix<float>(2, 3);
  EXPECT_THROW(reshape<float>(input, 4, 4), std::invalid_argument);
  EXPECT_THROW(reshape<float>(input, -2, 6), std::invalid_argument);
  EXPECT_THROW(reshape<float>(input, -1, -1), std::invalid_argument);
  EXPECT_THROW(reshape<float>(input, -1, 4), std::invalid_argument);
  EXPECT_THROW(reshape<float>(input, 4, -1), std::invalid_argument);

  // A zero-copy dimension that resolves to zero cannot be used to infer -1.
  Eigen::MatrixXf empty_cols(2, 0);
  EXPECT_THROW(reshape<float>(empty_cols, -1, 0), std::invalid_argument);
  Eigen::MatrixXf empty_rows(0, 3);
  EXPECT_THROW(reshape<float>(empty_rows, 0, -1), std::invalid_argument);
}
