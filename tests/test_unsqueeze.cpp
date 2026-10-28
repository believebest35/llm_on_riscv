#include "core/unsqueeze.h"
#include "gtest_base.h"

class UnsqueezeTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_unsqueeze(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
      int axis) {
    if (axis < 0) {
      axis += 2;  // 2-D input: -1 -> 1, -2 -> 0
    }
    const int rows = static_cast<int>(input.rows());
    const int cols = static_cast<int>(input.cols());
    const int num_elements = rows * cols;
    if (axis == 0) {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(
          1, num_elements);
      int k = 0;
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          output(0, k++) = input(i, j);
        }
      }
      return output;
    } else {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output(
          num_elements, 1);
      int k = 0;
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          output(k++, 0) = input(i, j);
        }
      }
      return output;
    }
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
      int axis, Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_output;
    ASSERT_NO_THROW(impl_output = unsqueeze<Scalar>(input, axis))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_output;
    ASSERT_NO_THROW(
        golden_output = golden_reference_unsqueeze<Scalar>(input, axis))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_output, golden_output, tolerance, test_description);
  }
};

TEST_F(UnsqueezeTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1},
      {5, 0},
      {5, 1},
      {10, 0},
      {100, 0},
      {100, 1},
      {5, -2},
      {10, -1},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {1, 0},
  };

  const float tolerance = 0.0f;

  for (const auto& [size, axis] : test_cases) {
    auto input = generate_random_vector<float>(size, -2.0f, 2.0f);

    std::string test_description = "Unsqueeze " + std::to_string(size) +
                            "x1 axis=" + std::to_string(axis);

    test_with_golden<float>(input, axis, tolerance, test_description);
  }
}

TEST_F(UnsqueezeTest, InvalidInputs) {
  auto input = generate_random_vector<float>(4);
  EXPECT_THROW(unsqueeze<float>(input, 2), std::invalid_argument);
  EXPECT_THROW(unsqueeze<float>(input, -3), std::invalid_argument);

  auto matrix = generate_random_matrix<float>(2, 3);
  EXPECT_THROW(unsqueeze(matrix, 0), std::invalid_argument);
}
