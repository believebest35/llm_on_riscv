#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/reshape.h"
#include "gtest_base.h"

class ReshapeTest : public GTestBase {
 protected:
  template <typename Scalar>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  golden_reference_reshape(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      int new_rows, int new_cols) {
    const std::int64_t total = A.size();
    if (new_rows < 0 || new_cols < 0) {
      throw std::invalid_argument("negative dims");
    }
    if (static_cast<std::int64_t>(new_rows) * new_cols != total) {
      throw std::invalid_argument("size mismatch");
    }
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(new_rows,
                                                              new_cols);
    for (std::int64_t i = 0; i < total; ++i) {
      std::int64_t r = i / A.cols();
      std::int64_t c = i % A.cols();
      std::int64_t r2 = i / new_cols;
      std::int64_t c2 = i % new_cols;
      out(r2, c2) = A(r, c);
    }
    return out;
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
      int new_rows, int new_cols, const std::string& desc = "") {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl;
    ASSERT_NO_THROW(impl = reshape<Scalar>(A, new_rows, new_cols))
        << "implementation threw: " << desc;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> golden;
    ASSERT_NO_THROW(golden =
                        golden_reference_reshape<Scalar>(A, new_rows, new_cols))
        << "golden threw: " << desc;

    EXPECT_EQ(impl.rows(), golden.rows()) << "row mismatch: " << desc;
    EXPECT_EQ(impl.cols(), golden.cols()) << "col mismatch: " << desc;

    const auto numel = impl.rows() * impl.cols();
    for (int i = 0; i < numel; ++i) {
      EXPECT_EQ(impl.data()[i], golden.data()[i])
          << "value mismatch at " << i << " (" << desc << ")";
    }
  }
};

TEST_F(ReshapeTest, Simple) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(2, 3);
  A << 1, 2, 3, 4, 5, 6;
  test_with_golden<float>(A, 3, 2, "2x3->3x2");
  test_with_golden<float>(A, 1, 6, "2x3->1x6");
  test_with_golden<float>(A, 6, 1, "2x3->6x1");
}

TEST_F(ReshapeTest, Random) {
  std::vector<std::tuple<int, int, int, int>> dims = {{2, 4, 4, 2},
                                                      {5, 2, 10, 1}};
  for (auto [r, c, nr, nc] : dims) {
    auto M = generate_random_matrix<float>(r, c, -5, 5);
    test_with_golden<float>(M, nr, nc, "random reshape");
  }
}

TEST_F(ReshapeTest, ErrorCases) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(2, 2);
  A.setRandom();
  EXPECT_THROW(reshape<float>(A, 3, 2), std::invalid_argument);
  EXPECT_THROW(reshape<float>(A, -1, 4), std::invalid_argument);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
