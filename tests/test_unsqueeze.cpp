#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/unsqueeze.h"
#include "gtest_base.h"

class UnsqueezeTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_unsqueeze(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& data, int axes) {
    const int N = data.size();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> expanded;
    if (axes == 0) {
      expanded.resize(1, N);
      expanded.row(0) = data.transpose();
    } else {
      expanded.resize(N, 1);
      expanded.col(0) = data;
    }
    return expanded;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& data, int axes,
      Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result = unsqueeze(data, axes))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_unsqueeze<Scalar>(data, axes))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_result, golden_result, tolerance, test_description);
  }
};

TEST_F(UnsqueezeTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 0}, {1, 1}, {5, 0}, {5, 1}, {10, 0}, {100, 0}, {100, 1},
  };

  const float tolerance = 1e-5;

  for (const auto& [size, axes] : test_cases) {
    auto data = generate_random_vector<float>(size, -2.0f, 2.0f);

    std::string test_name = "Unsqueeze size=" + std::to_string(size) +
                            " axes=" + std::to_string(axes);

    test_with_golden(data, axes, tolerance, test_name);
  }

  // Scalar (1x1) and 1-D tensors passed as 2-D matrices, as used by the
  // Qwen3 shape-construction path.
  {
    Eigen::MatrixXf scalar(1, 1);
    scalar << 7.0f;

    Eigen::MatrixXf scalar_row;
    ASSERT_NO_THROW(scalar_row = unsqueeze(scalar, 0));
    EXPECT_EQ(scalar_row.rows(), 1);
    EXPECT_EQ(scalar_row.cols(), 1);
    EXPECT_FLOAT_EQ(scalar_row(0, 0), 7.0f);

    Eigen::MatrixXf vector = generate_random_vector<float>(5, -1.0f, 1.0f);
    Eigen::MatrixXf row;
    ASSERT_NO_THROW(row = unsqueeze(vector, 0));
    EXPECT_EQ(row.rows(), 1);
    EXPECT_EQ(row.cols(), 5);
    for (int i = 0; i < 5; ++i) {
      EXPECT_FLOAT_EQ(row(0, i), vector(i, 0));
    }

    Eigen::MatrixXf col;
    ASSERT_NO_THROW(col = unsqueeze(row, 1));
    EXPECT_EQ(col.rows(), 5);
    EXPECT_EQ(col.cols(), 1);
    for (int i = 0; i < 5; ++i) {
      EXPECT_FLOAT_EQ(col(i, 0), row(0, i));
    }
  }
}


TEST_F(UnsqueezeTest, InvalidInputs) {
  auto data = generate_random_vector<float>(4);
  EXPECT_THROW(unsqueeze(data, 2), std::invalid_argument);

  auto matrix = generate_random_matrix<float>(2, 3);
  EXPECT_THROW(unsqueeze(matrix, 0), std::invalid_argument);
}
