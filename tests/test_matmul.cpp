#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <string>

#include "core/matmul.h"

class MatMulTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Use fixed seed for reproducible tests
    std::srand(42);
  }

  // Generate random matrix with specified dimensions
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> generate_random_matrix(
      int rows, int cols, Scalar min_val = -1.0, Scalar max_val = 1.0) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> mat(rows, cols);
    mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Random(rows,
                                                                        cols);
    mat = (mat.array() + 1.0) / 2.0 * (max_val - min_val) + min_val;
    return mat;
  }

  // Golden reference implementation - triple nested loop
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_reference_matmul(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
    const int m = A.rows();
    const int n = A.cols();
    const int p = B.cols();

    if (n != B.rows()) {
      throw std::invalid_argument("Matrix dimensions incompatible");
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> C(m, p);
    C.setZero();

    // Simple triple-nested loop implementation
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < p; ++j) {
        Scalar sum = static_cast<Scalar>(0);
        for (int k = 0; k < n; ++k) {
          sum += A(i, k) * B(k, j);
        }
        C(i, j) = sum;
      }
    }

    return C;
  }

  // Calculate maximum absolute difference between two matrices
  template <typename Scalar>
  Scalar calculate_max_abs_difference(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
      throw std::invalid_argument("Matrices must have same dimensions");
    }

    Scalar max_diff = static_cast<Scalar>(0);

    for (int i = 0; i < A.rows(); ++i) {
      for (int j = 0; j < A.cols(); ++j) {
        Scalar diff = std::abs(A(i, j) - B(i, j));
        if (diff > max_diff) {
          max_diff = diff;
        }
      }
    }

    return max_diff;
  }

  // Calculate root mean square error between two matrices
  template <typename Scalar>
  Scalar calculate_rmse(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
      throw std::invalid_argument("Matrices must have same dimensions");
    }

    Scalar sum_squared_error = static_cast<Scalar>(0);
    int total_elements = A.rows() * A.cols();

    for (int i = 0; i < A.rows(); ++i) {
      for (int j = 0; j < A.cols(); ++j) {
        Scalar diff = A(i, j) - B(i, j);
        sum_squared_error += diff * diff;
      }
    }

    return std::sqrt(sum_squared_error / total_elements);
  }

  // Compare matrices with tolerance
  template <typename Scalar>
  ::testing::AssertionResult compare_matrices_with_tolerance(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B,
      Scalar tolerance, const std::string& matrix_name = "Result") {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
      return ::testing::AssertionFailure()
             << matrix_name << " dimensions mismatch: "
             << "(" << A.rows() << "x" << A.cols() << ") vs "
             << "(" << B.rows() << "x" << B.cols() << ")";
    }

    for (int i = 0; i < A.rows(); ++i) {
      for (int j = 0; j < A.cols(); ++j) {
        Scalar diff = std::abs(A(i, j) - B(i, j));
        if (diff > tolerance) {
          return ::testing::AssertionFailure()
                 << matrix_name << " differs at position (" << i << ", " << j
                 << "): " << A(i, j) << " vs " << B(i, j) << " (diff = " << diff
                 << ", tolerance = " << tolerance << ")";
        }
      }
    }

    return ::testing::AssertionSuccess();
  }

  // Test helper function with automatic golden reference comparison
  template <typename Scalar>
  void test_matmul_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B,
      Scalar tolerance, const std::string& test_description = "") {
    // Perform multiplication using Eigen implementation
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_result;
    ASSERT_NO_THROW(eigen_result = matrix_multiply(A, B))
        << "Eigen implementation failed: " << test_description;

    // Compute golden reference
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden_result;
    ASSERT_NO_THROW(golden_result = golden_reference_matmul(A, B))
        << "Golden reference failed: " << test_description;

    // Check dimensions
    EXPECT_EQ(eigen_result.rows(), A.rows())
        << "Result rows incorrect: " << test_description;
    EXPECT_EQ(eigen_result.cols(), B.cols())
        << "Result cols incorrect: " << test_description;
    EXPECT_EQ(eigen_result.rows(), golden_result.rows())
        << "Row count mismatch with golden: " << test_description;
    EXPECT_EQ(eigen_result.cols(), golden_result.cols())
        << "Column count mismatch with golden: " << test_description;

    // Compare with golden reference using tolerance
    auto comparison_result = compare_matrices_with_tolerance(
        eigen_result, golden_result, tolerance, "Matrix multiplication result");

    EXPECT_TRUE(comparison_result)
        << test_description << "\n"
        << "Max absolute difference: "
        << calculate_max_abs_difference(eigen_result, golden_result)
        << "\nRMSE: " << calculate_rmse(eigen_result, golden_result);
  }
};

TEST_F(MatMulTest, RandomMatrices) {
  // Test multiple random matrix sizes
  const std::vector<std::tuple<int, int, int>> test_cases = {
      {1, 1, 1},        // Scalar
      {1, 1024, 1},     // Vector
      {1024, 1, 2048},  // Vector
      {2, 3, 4},        // Small matrix
      {64, 64, 64},     // Medium matrix
      {256, 256, 256},  // Larger matrix
                        // {1024, 1024, 2048}, // real shape
                        // {2048, 1024, 1024}, // real shape
                        // {1024, 2048, 1024}, // real shape
                        // {1024, 1024, 3072}, // real shape
                        // {1024, 3072, 1024}  // real shape
  };

  const float tolerance = 1e-4;

  for (const auto& [m, n, p] : test_cases) {
    auto A = generate_random_matrix<float>(m, n, -1.0, 1.0);
    auto B = generate_random_matrix<float>(n, p, -1.0, 1.0);

    std::string test_name = "Random matrix " + std::to_string(m) + "x" +
                            std::to_string(n) + " * " + std::to_string(n) +
                            "x" + std::to_string(p);

    test_matmul_with_golden(A, B, tolerance, test_name);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Optional: Configure Google Test
  ::testing::GTEST_FLAG(print_time) = false;

  return RUN_ALL_TESTS();
}