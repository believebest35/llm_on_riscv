#include <gtest/gtest.h>

#include <cmath>
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

    EXPECT_EQ(impl_result.rows(), golden_result.rows())
        << "Row count mismatch: " << test_description;
    EXPECT_EQ(impl_result.cols(), golden_result.cols())
        << "Column count mismatch: " << test_description;

    double max_diff = 0.0;
    double sum_sq = 0.0;
    int total = impl_result.size();
    for (int i = 0; i < total; ++i) {
      double diff = std::abs(static_cast<double>(impl_result.data()[i]) -
                             static_cast<double>(golden_result.data()[i]));
      max_diff = std::max(max_diff, diff);
      sum_sq += diff * diff;
    }
    double rmse = std::sqrt(sum_sq / total);
    double l2_diff = std::sqrt(sum_sq);

    EXPECT_LE(l2_diff, tolerance)
        << "L2 difference exceeds tolerance: " << test_description;

    std::cout << test_description << "\nMax absolute difference: " << max_diff
              << "\nRMSE: " << rmse << "\nL2 difference: " << l2_diff
              << std::endl;
  }
};

TEST_F(CastTest, RandomMatrices) {
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
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
