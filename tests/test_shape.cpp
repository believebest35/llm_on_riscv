#include <cstdint>

#include "core/shape.h"
#include "gtest_base.h"

class ShapeTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_shape(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A) {
    Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> result(2, 1);
    result(0, 0) = A.rows();
    result(1, 0) = A.cols();
    return result;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      double tolerance, const std::string& test_description = "") {
    Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result = shape(A))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_shape(A))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_result, golden_result, tolerance, test_description);
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
    auto A = generate_random_matrix<float>(rows, cols, -1.0f, 1.0f);

    std::string test_name = "Shape " + std::to_string(rows) + "x" +
                            std::to_string(cols);

    test_with_golden(A, tolerance, test_name);
  }
}
