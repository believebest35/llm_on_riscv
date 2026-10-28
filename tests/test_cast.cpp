#include "core/cast.h"
#include "gtest_base.h"

class CastTest : public GTestBase {
 protected:
  template <typename Out, typename In>
  Eigen::Matrix<Out, Eigen::Dynamic, Eigen::Dynamic> golden_reference_cast(
      const Eigen::Matrix<In, Eigen::Dynamic, Eigen::Dynamic>& input) {
    const int rows = static_cast<int>(input.rows());
    const int cols = static_cast<int>(input.cols());
    Eigen::Matrix<Out, Eigen::Dynamic, Eigen::Dynamic> output(rows, cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        output(i, j) = static_cast<Out>(input(i, j));
      }
    }
    return output;
  }

  template <typename Out, typename In>
  void test_with_golden(
      const Eigen::Matrix<In, Eigen::Dynamic, Eigen::Dynamic>& input,
      In tolerance, const std::string& test_description = "") {
    Eigen::Matrix<Out, Eigen::Dynamic, Eigen::Dynamic> impl_output;
    ASSERT_NO_THROW((impl_output = cast<Out, In>(input)))
        << "Implementation failed: " << test_description;

    Eigen::Matrix<Out, Eigen::Dynamic, Eigen::Dynamic> golden_output;
    ASSERT_NO_THROW((golden_output = golden_reference_cast<Out, In>(input)))
        << "Golden reference failed: " << test_description;

    expect_matrix_near(impl_output, golden_output, tolerance, test_description);
  }
};

TEST_F(CastTest, ValidInputs) {
  const std::vector<std::tuple<int, int>> test_cases = {
      {1, 1024},
      {1024, 1},
      {2, 3},
      {64, 64},
      {256, 256},
      {1024, 1024},
      // docs/01_dev.md (Qwen3-0.6B) shapes.
      {1, 1},
  };

  const float tolerance = 0.0f;

  for (const auto& [rows, cols] : test_cases) {
    auto input = generate_random_matrix<float>(rows, cols, -1.0f, 1.0f);

    const std::string shape =
        std::to_string(rows) + "x" + std::to_string(cols);

    test_with_golden<float, float>(input, tolerance,
                                   "Cast " + shape + " to=float");
    test_with_golden<double, float>(input, tolerance,
                                    "Cast " + shape + " to=double");
    test_with_golden<int, float>(input, tolerance,
                                 "Cast " + shape + " to=int");
  }
}
