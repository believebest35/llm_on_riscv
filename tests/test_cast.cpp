#include "core/cast.h"
#include "gtest_base.h"

class CastTest : public GTestBase {
 protected:
  template <typename Out, typename In>
  Eigen::Matrix<Out, Eigen::Dynamic, Eigen::Dynamic> golden_reference_cast(
      const Eigen::Matrix<In, Eigen::Dynamic, Eigen::Dynamic>& A) {
    Eigen::Matrix<Out, Eigen::Dynamic, Eigen::Dynamic> B(A.rows(), A.cols());
    for (int i = 0; i < A.rows(); ++i) {
      for (int j = 0; j < A.cols(); ++j) {
        B(i, j) = static_cast<Out>(A(i, j));
      }
    }
    return B;
  }

  template <typename Out, typename In>
  void test_with_golden(
      const Eigen::Matrix<In, Eigen::Dynamic, Eigen::Dynamic>& A,
      double tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Out, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW((impl_result = cast<Out, In>(A)))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Out, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW((golden_result = golden_reference_cast<Out, In>(A)))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_result, golden_result, tolerance, test_description);
  }
};

TEST_F(CastTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1024}, {1024, 1}, {2, 3}, {64, 64}, {256, 256}, {1024, 1024},
      // docs/01_dev.md (Qwen3-0.6B) shapes: scalar () and (batch, 1).
      {1, 1},
  };

  const double tolerance = 0.0;

  for (const auto& [rows, cols] : test_cases) {
    auto A = generate_random_matrix<float>(rows, cols, -1.0f, 1.0f);

    const std::string shape =
        std::to_string(rows) + "x" + std::to_string(cols);

    test_with_golden<float, float>(A, tolerance, "Cast float->float " + shape);
    test_with_golden<double, float>(A, tolerance, "Cast float->double " + shape);
    test_with_golden<int, float>(A, tolerance, "Cast float->int " + shape);
  }
}
