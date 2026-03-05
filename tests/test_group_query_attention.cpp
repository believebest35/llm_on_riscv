#include <gtest/gtest.h>

#include <tuple>

#include "core/group_query_attention.h"
#include "gtest_base.h"

class GroupQueryAttentionTest : public GTestBase {
 protected:
  template <typename Scalar>
  std::tuple<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
             Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
             Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
  golden_reference(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Q,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& K,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& V,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& past_k,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& past_v,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& bias,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& mask_index,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& unused1,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& unused2) {
    // replicate the simple logic from the implementation
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y = Q;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> present_k;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> present_v;

    present_k.resize(past_k.rows() + K.rows(), K.cols());
    present_v.resize(past_v.rows() + V.rows(), V.cols());

    if (past_k.rows() > 0) {
      present_k.topRows(past_k.rows()) = past_k;
    }
    if (past_v.rows() > 0) {
      present_v.topRows(past_v.rows()) = past_v;
    }
    present_k.bottomRows(K.rows()) = K;
    present_v.bottomRows(V.rows()) = V;

    return {Y, present_k, present_v};
  }

  template <typename Scalar>
  void test_with_golden(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Q,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& K,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& V,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& past_k,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& past_v,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& bias,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& mask_index,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& unused1,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& unused2,
      const std::string& desc = "") {
    std::tuple<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
               Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
               Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        impl;
    ASSERT_NO_THROW(
        impl = group_query_attention<Scalar>(Q, K, V, past_k, past_v, bias,
                                             mask_index, unused1, unused2))
        << "implementation threw: " << desc;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_Y;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_pk;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> impl_pv;
    std::tie(impl_Y, impl_pk, impl_pv) = impl;

    std::tuple<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
               Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
               Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        golden;
    ASSERT_NO_THROW(golden =
                        golden_reference<Scalar>(Q, K, V, past_k, past_v, bias,
                                                 mask_index, unused1, unused2))
        << "golden threw: " << desc;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> gold_Y;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> gold_pk;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> gold_pv;
    std::tie(gold_Y, gold_pk, gold_pv) = golden;

    EXPECT_EQ(impl_Y.rows(), gold_Y.rows()) << "Y row mismatch: " << desc;
    EXPECT_EQ(impl_Y.cols(), gold_Y.cols()) << "Y col mismatch: " << desc;
    EXPECT_EQ(impl_pk.rows(), gold_pk.rows()) << "pk row mismatch: " << desc;
    EXPECT_EQ(impl_pk.cols(), gold_pk.cols()) << "pk col mismatch: " << desc;
    EXPECT_EQ(impl_pv.rows(), gold_pv.rows()) << "pv row mismatch: " << desc;
    EXPECT_EQ(impl_pv.cols(), gold_pv.cols()) << "pv col mismatch: " << desc;

    // elementwise compare
    for (int i = 0; i < impl_Y.size(); ++i) {
      EXPECT_NEAR(impl_Y.data()[i], gold_Y.data()[i], 1e-6)
          << "Y value mismatch at " << i << " (" << desc << ")";
    }
    for (int i = 0; i < impl_pk.size(); ++i) {
      EXPECT_NEAR(impl_pk.data()[i], gold_pk.data()[i], 1e-6)
          << "pk value mismatch at " << i << " (" << desc << ")";
    }
    for (int i = 0; i < impl_pv.size(); ++i) {
      EXPECT_NEAR(impl_pv.data()[i], gold_pv.data()[i], 1e-6)
          << "pv value mismatch at " << i << " (" << desc << ")";
    }
  }
};

TEST_F(GroupQueryAttentionTest, Simple) {
  // choose small dims: rows 2, Qdim=4, Kdim=2
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Q(2, 4);
  Q << 1, 2, 3, 4, 5, 6, 7, 8;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> K(2, 2);
  K << 1, 1, 2, 2;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> V(2, 2);
  V << 3, 3, 4, 4;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> past_k(1, 2);
  past_k << 7, 7;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> past_v(1, 2);
  past_v << 8, 8;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> bias(1, 1);
  bias << 0;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mask_index(1, 1);
  mask_index << 1;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> unused1(0, 0);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> unused2(0, 0);
  test_with_golden<float>(Q, K, V, past_k, past_v, bias, mask_index, unused1,
                          unused2, "simple");
}

TEST_F(GroupQueryAttentionTest, Random) {
  auto Q = generate_random_matrix<float>(3, 6, -1, 1);
  // ensure Q.cols == 2 * K.cols
  auto K = generate_random_matrix<float>(3, 3, -1, 1);
  auto V = generate_random_matrix<float>(3, 3, -1, 1);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> past_k(2, 3);
  past_k.setRandom();
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> past_v(2, 3);
  past_v.setRandom();
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> bias(3, 1);
  bias.setZero();
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mask_index(1, 1);
  mask_index << 0;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> unused1(1, 1);
  unused1 << 0;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> unused2(2, 2);
  unused2.setZero();
  test_with_golden<float>(Q, K, V, past_k, past_v, bias, mask_index, unused1,
                          unused2, "random");
}

TEST_F(GroupQueryAttentionTest, ErrorCases) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Q(1, 4);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> K(1, 3);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> V(1, 3);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> past_k(1, 3);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> past_v(1, 3);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> bias(1, 2);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mask_index(2, 2);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> unused1(0, 0);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> unused2(0, 0);

  // mismatched row counts
  EXPECT_THROW(group_query_attention<float>(
                   Q, K, V, past_k, past_v, bias,
                   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>(1, 1),
                   unused1, unused2),
               std::invalid_argument);

  // query dim wrong relative to K
  EXPECT_THROW(group_query_attention<float>(
                   Q, K, V, past_k, past_v,
                   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>(1, 1),
                   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>(1, 1),
                   unused1, unused2),
               std::invalid_argument);

  // bias wrong shape
  EXPECT_THROW(group_query_attention<float>(
                   Q, K, V, past_k, past_v, bias,
                   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>(1, 1),
                   unused1, unused2),
               std::invalid_argument);

  // mask_index not scalar
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> ok_bias(1, 1);
  ok_bias << 0;
  EXPECT_THROW(group_query_attention<float>(Q, K, V, past_k, past_v, ok_bias,
                                            mask_index, unused1, unused2),
               std::invalid_argument);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(print_time) = false;
  return RUN_ALL_TESTS();
}
