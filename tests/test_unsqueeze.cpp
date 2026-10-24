#include "core/unsqueeze.h"
#include "gtest_base.h"

class UnsqueezeTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_unsqueeze(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& data, int axes) {
    const int N = static_cast<int>(data.size());
    if (axes == 0) {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> expanded(1, N);
      int k = 0;
      for (int i = 0; i < data.rows(); ++i) {
        for (int j = 0; j < data.cols(); ++j) {
          expanded(0, k++) = data(i, j);
        }
      }
      return expanded;
    } else {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> expanded(N, 1);
      int k = 0;
      for (int i = 0; i < data.rows(); ++i) {
        for (int j = 0; j < data.cols(); ++j) {
          expanded(k++, 0) = data(i, j);
        }
      }
      return expanded;
    }
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& data, int axes,
      Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result = unsqueeze<Scalar>(data, axes))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_unsqueeze<Scalar>(data, axes))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_result, golden_result, tolerance, test_description);
  }
};

TEST_F(UnsqueezeTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1}, {5, 0}, {5, 1}, {10, 0}, {100, 0}, {100, 1},
      // docs/01_dev.md (Qwen3-0.6B) shapes: scalar () -> (1,).
      {1, 0},
  };

  const float tolerance = 0.0f;

  for (const auto& [size, axes] : test_cases) {
    auto data = generate_random_vector<float>(size, -2.0f, 2.0f);

    std::string test_name = "Unsqueeze size=" + std::to_string(size) +
                            " axes=" + std::to_string(axes);

    test_with_golden(data, axes, tolerance, test_name);
  }
}

TEST_F(UnsqueezeTest, InvalidInputs) {
  auto data = generate_random_vector<float>(4);
  EXPECT_THROW(unsqueeze<float>(data, 2), std::invalid_argument);

  auto matrix = generate_random_matrix<float>(2, 3);
  EXPECT_THROW(unsqueeze(matrix, 0), std::invalid_argument);
}
