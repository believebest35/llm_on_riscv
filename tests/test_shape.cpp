#include <cstdint>

#include "core/shape.h"
#include "gtest_base.h"

class ShapeTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_shape(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input) {
    Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> output(2, 1);
    const std::int64_t dims[2] = {static_cast<std::int64_t>(input.rows()),
                                  static_cast<std::int64_t>(input.cols())};
    for (int i = 0; i < 2; ++i) {
      output(i, 0) = dims[i];
    }
    return output;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input,
      Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> impl_output;
    ASSERT_NO_THROW(impl_output = shape(input))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> golden_output;
    ASSERT_NO_THROW(golden_output = golden_reference_shape(input))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_output, golden_output, tolerance, test_description);
  }
};

TEST_F(ShapeTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1}, {1, 1024}, {1024, 1}, {2, 3}, {64, 64}, {256, 256}, {1024, 2048},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {1, 4}, {1, 7},
  };

  const float tolerance = 0.0f;

  for (const auto& [rows, cols] : test_cases) {
    auto input = generate_random_matrix<float>(rows, cols, -1.0f, 1.0f);

    std::string test_name = "Shape " + std::to_string(rows) + "x" +
                            std::to_string(cols);

    test_with_golden(input, tolerance, test_name);
  }
}
