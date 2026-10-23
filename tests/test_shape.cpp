#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "core/shape.h"
#include "gtest_base.h"

class ShapeTest : public GTestBase {};

TEST_F(ShapeTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1}, {1, 1024}, {1024, 1}, {2, 3}, {64, 64}, {256, 256}, {1024, 2048},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {1, 4}, {1, 7},
  };

  for (const auto& [rows, cols] : test_cases) {
    auto A = generate_random_matrix<float>(rows, cols, -1.0f, 1.0f);

    Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> result;
    ASSERT_NO_THROW(result = shape(A))
        << "Shape " << rows << "x" << cols;

    EXPECT_EQ(result.rows(), 2) << "Shape " << rows << "x" << cols;
    EXPECT_EQ(result.cols(), 1) << "Shape " << rows << "x" << cols;
    EXPECT_EQ(result(0, 0), rows) << "Shape " << rows << "x" << cols;
    EXPECT_EQ(result(1, 0), cols) << "Shape " << rows << "x" << cols;
  }
}

