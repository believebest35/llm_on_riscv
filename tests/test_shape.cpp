#include <gtest/gtest.h>

#include <string>

#include "core/shape.h"
#include "gtest_base.h"

class ShapeTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> golden_reference_shape(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> out(2);
    out(0) = static_cast<Scalar>(A.rows());
    out(1) = static_cast<Scalar>(A.cols());
    return out;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const std::string& test_description = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> impl_result;
    ASSERT_NO_THROW(impl_result = shape(A))
        << "Implementation threw for: " << test_description;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_shape(A))
        << "Golden reference threw for: " << test_description;

    EXPECT_EQ(impl_result.rows(), golden_result.rows())
        << "Row count mismatch: " << test_description;
    EXPECT_EQ(impl_result.cols(), golden_result.cols())
        << "Column count mismatch: " << test_description;

    // compare each element exactly since these are integers in practice
    for (int i = 0; i < impl_result.size(); ++i) {
      EXPECT_EQ(impl_result(i), golden_result(i))
          << "Value mismatch at index " << i << " (" << test_description << ")";
    }
  }
};

TEST_F(ShapeTest, VariousSizes) {
  const std::vector<std::pair<int, int>> test_cases = {
      {1, 1}, {1, 1024}, {1024, 1}, {2, 3}, {64, 64}, {256, 256}, {1024, 2048},
  };

  for (const auto& [m, n] : test_cases) {
    auto A = generate_random_matrix<float>(m, n, -1.0f, 1.0f);
    std::string desc =
        "Shape of " + std::to_string(m) + "x" + std::to_string(n);
    test_with_golden(A, desc);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
