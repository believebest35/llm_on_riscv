#include "core/transpose.h"
#include "gtest_base.h"

class TransposeTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_transpose(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(A.cols(),
                                                                 A.rows());
    for (int i = 0; i < A.rows(); ++i) {
      for (int j = 0; j < A.cols(); ++j) {
        result(j, i) = A(i, j);
      }
    }
    return result;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      Scalar tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW(impl_result = transpose<Scalar>(A))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_transpose(A))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_result, golden_result, tolerance, test_description);
  }
};

TEST_F(TransposeTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1}, {1, 1024}, {1024, 1}, {2, 3}, {64, 128}, {256, 512},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {151936, 1024},
  };

  const float tolerance = 0.0f;

  for (const auto& [rows, cols] : test_cases) {
    auto A = generate_random_matrix<float>(rows, cols, -1.0f, 1.0f);

    std::string test_name = "Transpose " + std::to_string(rows) + "x" +
                            std::to_string(cols);

    test_with_golden(A, tolerance, test_name);
  }
}

