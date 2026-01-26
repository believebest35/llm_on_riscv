# 架构重构实施示例

本文档展示如何将现有代码重构为推荐的架构设计。

## 示例1：Tensor抽象层实现

### 当前问题
- 直接使用Eigen Matrix，耦合度高
- 无法灵活切换后端
- 缺少统一的类型系统

### 重构方案

#### Step 1: 创建Tensor基础类

```cpp
// src/core/tensor/tensor.h
#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include "core/types.h"
#include "core/memory/allocator.h"

namespace llm_runtime {
namespace core {

class Tensor {
public:
  Tensor(DataType dtype, const Shape& shape);
  Tensor(DataType dtype, const Shape& shape, 
         void* data, bool owns_data = false);
  ~Tensor();
  
  // 禁止拷贝，允许移动
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&&) noexcept;
  Tensor& operator=(Tensor&&) noexcept;
  
  // 访问器
  DataType dtype() const { return dtype_; }
  const Shape& shape() const { return shape_; }
  int64_t numel() const { return shape_.total_elements(); }
  size_t bytes() const { return numel() * dtype_size(dtype_); }
  
  // 数据访问
  void* data() { return data_; }
  const void* data() const { return data_; }
  
  template<typename T>
  T* data_as() {
    return static_cast<T*>(data_);
  }
  
  template<typename T>
  const T* data_as() const {
    return static_cast<const T*>(data_);
  }
  
  // 内存管理
  void allocate(std::shared_ptr<Allocator> allocator = nullptr);
  void deallocate();
  bool is_allocated() const { return data_ != nullptr; }
  
  // 工具方法
  Tensor clone() const;
  void fill_zero();
  
private:
  DataType dtype_;
  Shape shape_;
  void* data_;
  bool owns_data_;
  std::shared_ptr<Allocator> allocator_;
  
  static size_t dtype_size(DataType dtype);
};

} // namespace core
} // namespace llm_runtime
```

#### Step 2: 实现Shape类

```cpp
// src/core/tensor/shape.h
#pragma once

#include <vector>
#include <cstdint>
#include <string>

namespace llm_runtime {
namespace core {

class Shape {
public:
  Shape() = default;
  explicit Shape(std::vector<int64_t> dims);
  
  // 访问器
  int64_t rank() const { return static_cast<int64_t>(dims_.size()); }
  int64_t dim(int64_t idx) const;
  const std::vector<int64_t>& dims() const { return dims_; }
  
  // 计算
  int64_t total_elements() const;
  bool is_dynamic() const;  // 检查是否有-1（动态维度）
  
  // 操作
  Shape squeeze(int64_t axis) const;
  Shape unsqueeze(int64_t axis) const;
  Shape reshape(const std::vector<int64_t>& new_dims) const;
  
  // 比较
  bool operator==(const Shape& other) const;
  bool operator!=(const Shape& other) const;
  
  // 字符串表示
  std::string to_string() const;
  
private:
  std::vector<int64_t> dims_;
};

} // namespace core
} // namespace llm_runtime
```

#### Step 3: 实现类型系统

```cpp
// src/core/types.h
#pragma once

namespace llm_runtime {
namespace core {

enum class DataType {
  FLOAT32 = 0,
  FLOAT16,
  INT32,
  INT64,
  UINT8,
  BOOL,
  // 添加更多类型...
};

inline const char* dtype_to_string(DataType dtype) {
  switch (dtype) {
    case DataType::FLOAT32: return "float32";
    case DataType::FLOAT16: return "float16";
    case DataType::INT32: return "int32";
    case DataType::INT64: return "int64";
    default: return "unknown";
  }
}

} // namespace core
} // namespace llm_runtime
```

## 示例2：重构MatMul算子

### 当前代码
```cpp
// src/core/matmul.h (旧)
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix_multiply(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B);
```

### 重构后代码

#### Step 1: 创建算子基类

```cpp
// src/operators/base/operator.h
#pragma once

#include "core/tensor/tensor.h"
#include <vector>
#include <map>
#include <string>
#include <memory>

namespace llm_runtime {
namespace operators {

class Attribute {
public:
  enum Type { FLOAT, INT, STRING, TENSOR, FLOATS, INTS };
  
  template<typename T>
  T get() const;
  
private:
  Type type_;
  std::variant<float, int64_t, std::string, core::Tensor,
                std::vector<float>, std::vector<int64_t>> value_;
};

class Operator {
public:
  virtual ~Operator() = default;
  
  // 执行算子
  virtual std::vector<core::Tensor> forward(
      const std::vector<core::Tensor>& inputs) = 0;
  
  // 类型推断
  virtual std::vector<core::DataType> infer_output_types(
      const std::vector<core::DataType>& input_types) const = 0;
  
  // Shape推断
  virtual std::vector<core::Shape> infer_output_shapes(
      const std::vector<core::Shape>& input_shapes) const = 0;
  
  // 算子名称
  virtual const char* name() const = 0;
  
  // 属性设置
  void set_attributes(const std::map<std::string, Attribute>& attrs) {
    attributes_ = attrs;
  }
  
protected:
  std::map<std::string, Attribute> attributes_;
  
  // 验证输入
  void validate_inputs(const std::vector<core::Tensor>& inputs,
                       size_t expected_count) const;
};

} // namespace operators
} // namespace llm_runtime
```

#### Step 2: 实现MatMul算子

```cpp
// src/operators/math/matmul.h
#pragma once

#include "operators/base/operator.h"
#include "backend/cpu/blas.h"
#include <memory>

namespace llm_runtime {
namespace operators {

class MatMulOp : public Operator {
public:
  MatMulOp();
  explicit MatMulOp(std::shared_ptr<backend::BLAS> blas);
  
  const char* name() const override { return "MatMul"; }
  
  std::vector<core::Tensor> forward(
      const std::vector<core::Tensor>& inputs) override;
  
  std::vector<core::DataType> infer_output_types(
      const std::vector<core::DataType>& input_types) const override;
  
  std::vector<core::Shape> infer_output_shapes(
      const std::vector<core::Shape>& input_shapes) const override;
  
private:
  std::shared_ptr<backend::BLAS> blas_;
  
  void validate_inputs(const std::vector<core::Tensor>& inputs) const;
};

} // namespace operators
} // namespace llm_runtime
```

```cpp
// src/operators/math/matmul.cpp
#include "operators/math/matmul.h"
#include "core/tensor/tensor.h"
#include <stdexcept>

namespace llm_runtime {
namespace operators {

MatMulOp::MatMulOp() 
    : blas_(backend::create_blas("auto")) {
}

MatMulOp::MatMulOp(std::shared_ptr<backend::BLAS> blas)
    : blas_(blas) {
}

std::vector<core::Tensor> MatMulOp::forward(
    const std::vector<core::Tensor>& inputs) {
  
  validate_inputs(inputs);
  
  const auto& A = inputs[0];
  const auto& B = inputs[1];
  
  // 推断输出shape
  auto output_shapes = infer_output_shapes({A.shape(), B.shape()});
  const auto& output_shape = output_shapes[0];
  
  // 创建输出tensor
  core::Tensor output(A.dtype(), output_shape);
  output.allocate();
  
  // 执行矩阵乘法
  const int m = A.shape().dim(0);
  const int n = B.shape().dim(1);
  const int k = A.shape().dim(1);
  
  const float* A_data = A.data_as<float>();
  const float* B_data = B.data_as<float>();
  float* C_data = output.data_as<float>();
  
  // 使用BLAS后端
  blas_->gemm(
      false, false,  // 不转置
      m, n, k,
      1.0f, A_data, k,  // alpha=1.0, lda=k
      B_data, n,        // ldb=n
      0.0f, C_data, n   // beta=0.0, ldc=n
  );
  
  return {std::move(output)};
}

std::vector<core::DataType> MatMulOp::infer_output_types(
    const std::vector<core::DataType>& input_types) const {
  
  if (input_types.size() != 2) {
    throw std::invalid_argument("MatMul requires 2 inputs");
  }
  
  // 输出类型与第一个输入相同
  return {input_types[0]};
}

std::vector<core::Shape> MatMulOp::infer_output_shapes(
    const std::vector<core::Shape>& input_shapes) const {
  
  if (input_shapes.size() != 2) {
    throw std::invalid_argument("MatMul requires 2 inputs");
  }
  
  const auto& A_shape = input_shapes[0];
  const auto& B_shape = input_shapes[1];
  
  if (A_shape.rank() != 2 || B_shape.rank() != 2) {
    throw std::invalid_argument("MatMul inputs must be 2D");
  }
  
  if (A_shape.dim(1) != B_shape.dim(0)) {
    throw std::invalid_argument(
        "MatMul dimension mismatch: A.cols() != B.rows()");
  }
  
  // 输出shape: [A.rows(), B.cols()]
  return {core::Shape({A_shape.dim(0), B_shape.dim(1)})};
}

void MatMulOp::validate_inputs(
    const std::vector<core::Tensor>& inputs) const {
  
  if (inputs.size() != 2) {
    throw std::invalid_argument("MatMul requires 2 inputs");
  }
  
  const auto& A = inputs[0];
  const auto& B = inputs[1];
  
  if (!A.is_allocated() || !B.is_allocated()) {
    throw std::runtime_error("MatMul inputs must be allocated");
  }
  
  if (A.dtype() != core::DataType::FLOAT32 ||
      B.dtype() != core::DataType::FLOAT32) {
    throw std::invalid_argument("MatMul currently only supports float32");
  }
  
  // 验证shape
  infer_output_shapes({A.shape(), B.shape()});
}

} // namespace operators
} // namespace llm_runtime
```

## 示例3：BLAS抽象层

### 创建BLAS接口

```cpp
// src/backend/cpu/blas.h
#pragma once

#include <memory>
#include <string>

namespace llm_runtime {
namespace backend {

class BLAS {
public:
  virtual ~BLAS() = default;
  
  // GEMM: C = alpha * op(A) * op(B) + beta * C
  virtual void gemm(
      bool trans_a, bool trans_b,
      int m, int n, int k,
      float alpha, const float* A, int lda,
      const float* B, int ldb,
      float beta, float* C, int ldc) = 0;
  
  // 其他BLAS操作...
};

// 工厂函数
std::unique_ptr<BLAS> create_blas(const std::string& backend = "auto");

} // namespace backend
} // namespace llm_runtime
```

### 实现Eigen后端

```cpp
// src/backend/cpu/eigen_blas.h
#pragma once

#include "backend/cpu/blas.h"
#include <Eigen/Dense>

namespace llm_runtime {
namespace backend {

class EigenBLAS : public BLAS {
public:
  void gemm(bool trans_a, bool trans_b,
            int m, int n, int k,
            float alpha, const float* A, int lda,
            const float* B, int ldb,
            float beta, float* C, int ldc) override {
    
    using MatrixMap = Eigen::Map<const Eigen::MatrixXf>;
    using MatrixMapMut = Eigen::Map<Eigen::MatrixXf>;
    
    MatrixMapMut C_map(C, m, n);
    
    if (!trans_a && !trans_b) {
      MatrixMap A_map(A, m, k);
      MatrixMap B_map(B, k, n);
      C_map = alpha * A_map * B_map + beta * C_map;
    } else {
      // 处理转置情况...
    }
  }
};

} // namespace backend
} // namespace llm_runtime
```

### 实现RVV后端（占位符）

```cpp
// src/backend/cpu/rvv/rvv_blas.h
#pragma once

#include "backend/cpu/blas.h"

namespace llm_runtime {
namespace backend {

class RVVBLAS : public BLAS {
public:
  void gemm(bool trans_a, bool trans_b,
            int m, int n, int k,
            float alpha, const float* A, int lda,
            const float* B, int ldb,
            float beta, float* C, int ldc) override {
    
    // TODO: 实现RVV优化的GEMM
    // 当前回退到简单实现
    simple_gemm(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  
private:
  void simple_gemm(int m, int n, int k,
                   float alpha, const float* A, int lda,
                   const float* B, int ldb,
                   float beta, float* C, int ldc);
};

} // namespace backend
} // namespace llm_runtime
```

## 示例4：测试代码迁移

### 旧测试代码
```cpp
// tests/test_matmul.cpp (旧)
TEST_F(MatMulTest, RandomMatrices) {
  auto A = generate_random_matrix<float>(1024, 1024);
  auto B = generate_random_matrix<float>(1024, 2048);
  auto result = matrix_multiply(A, B);
  // ...
}
```

### 新测试代码

```cpp
// tests/test_matmul.cpp (新)
#include <gtest/gtest.h>
#include "operators/math/matmul.h"
#include "core/tensor/tensor.h"

class MatMulTest : public ::testing::Test {
protected:
  void SetUp() override {
    op_ = std::make_unique<operators::MatMulOp>();
  }
  
  core::Tensor create_random_tensor(const core::Shape& shape) {
    core::Tensor tensor(core::DataType::FLOAT32, shape);
    tensor.allocate();
    
    // 填充随机数据
    float* data = tensor.data_as<float>();
    int64_t numel = tensor.numel();
    for (int64_t i = 0; i < numel; ++i) {
      data[i] = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
    }
    
    return tensor;
  }
  
  std::unique_ptr<operators::MatMulOp> op_;
};

TEST_F(MatMulTest, BasicMatMul) {
  auto A = create_random_tensor(core::Shape({1024, 1024}));
  auto B = create_random_tensor(core::Shape({1024, 2048}));
  
  auto outputs = op_->forward({A, B});
  ASSERT_EQ(outputs.size(), 1);
  
  const auto& C = outputs[0];
  EXPECT_EQ(C.shape(), core::Shape({1024, 2048}));
  EXPECT_EQ(C.dtype(), core::DataType::FLOAT32);
}
```

## 迁移检查清单

### Phase 1: 基础架构
- [ ] 创建新的目录结构
- [ ] 实现Tensor类
- [ ] 实现Shape类
- [ ] 实现Allocator接口
- [ ] 实现基础类型系统

### Phase 2: 算子重构
- [ ] 创建Operator基类
- [ ] 重构MatMul算子
- [ ] 重构Mul算子
- [ ] 重构LayerNorm算子
- [ ] 更新所有测试

### Phase 3: 后端抽象
- [ ] 创建BLAS接口
- [ ] 实现Eigen后端
- [ ] 实现简单CPU后端（回退）
- [ ] 添加RVV后端框架

### Phase 4: 集成测试
- [ ] 端到端测试
- [ ] 性能对比测试
- [ ] 内存泄漏检查

## 注意事项

1. **向后兼容**：可以保留旧的接口作为wrapper，逐步迁移
2. **测试优先**：确保每个重构步骤都有测试覆盖
3. **增量迁移**：不要一次性重构所有代码
4. **性能监控**：确保重构不引入性能回归
