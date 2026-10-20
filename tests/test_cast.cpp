#include <gtest/gtest.h>

#include <string>

#include "core/cast.h"
#include "gtest_base.h"

class CastTest : public GTestBase {
 protected:
  template <typename Out, typename In>
  Eigen::Matrix<Out, Eigen::Dynamic, Eigen::Dynamic> golden_reference_cast(
      const Eigen::Matrix<In, Eigen::Dynamic, Eigen::Dynamic>& A) {
    Eigen::Matrix<Out, Eigen::Dynamic, Eigen::Dynamic> B(A.rows(), A.cols());
    for (int i = 0; i < A.size(); ++i) {
      B.data()[i] = static_cast<Out>(A.data()[i]);
    }
    return B;
  }

  template <typename Out, typename In>
  void test_with_golden(
      const Eigen::Matrix<In, Eigen::Dynamic, Eigen::Dynamic>& A,
      double tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Out, Eigen::Dynamic, Eigen::Dynamic> impl_result;
    ASSERT_NO_THROW((impl_result = cast_matrix<Out, In>(A)))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Out, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW((golden_result = golden_reference_cast<Out, In>(A)))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_result, golden_result, tolerance, test_description);
  }
};

TEST_F(CastTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1}, {1, 1024}, {1024, 1}, {2, 3}, {64, 64}, {256, 256}, {1024, 1024},
  };

  const double tolerance = 1e-5;

  for (const auto& [rows, cols] : test_cases) {
    auto A = generate_random_matrix<float>(rows, cols, -1.0f, 1.0f);

    std::string test_name = "Cast float->float " + std::to_string(rows) + "x" +
                            std::to_string(cols);

    test_with_golden<float, float>(A, tolerance, test_name);
  }

  // Hand-picked cases: non-random values / alternative
  // argument forms, checked against the same reference.
  {
    const auto A = generate_random_matrix<float>(4, 8, -2.0f, 2.0f);

    const auto as_int = cast_matrix<int, float>(A);
    EXPECT_EQ(as_int.rows(), A.rows());
    EXPECT_EQ(as_int.cols(), A.cols());
    for (int i = 0; i < A.size(); ++i) {
      EXPECT_EQ(as_int.data()[i], static_cast<int>(A.data()[i]));
    }

    test_with_golden<double, float>(A, 1e-6, "Cast float->double");
  }
}


