# 架构快速参考

## 目录结构

```
llm_on_riscv/
├── include/llm_runtime/          # 公共API（用户可见）
│   ├── llm_runtime.h            # 主头文件
│   ├── tensor.h                 # Tensor API
│   ├── model.h                  # Model API
│   └── inference.h              # Inference API
│
├── src/
│   ├── core/                    # 核心抽象层（无外部依赖）
│   │   ├── tensor/             # Tensor实现
│   │   ├── memory/              # 内存管理
│   │   └── types.h              # 类型定义
│   │
│   ├── operators/               # 算子层（依赖core）
│   │   ├── base/                # 算子基类
│   │   ├── math/                # 数学算子
│   │   ├── tensor/              # 张量操作
│   │   ├── normalization/       # 归一化
│   │   ├── attention/           # 注意力
│   │   └── activation/          # 激活函数
│   │
│   ├── graph/                   # 图执行引擎（依赖operators）
│   │   ├── onnx/                # ONNX解析
│   │   ├── executor.h           # 执行引擎
│   │   └── node.h               # 节点抽象
│   │
│   └── backend/                 # 后端实现（可被operators使用）
│       ├── cpu/                 # CPU后端
│       │   ├── blas.h           # BLAS抽象
│       │   └── rvv/             # RISC-V向量扩展
│       └── allocator/           # 内存分配器
│
├── third_party/                 # 第三方库
│   ├── eigen/                  # Eigen（可选）
│   ├── gtest/                  # GoogleTest
│   └── onnx/                   # ONNX（可选）
│
└── tests/                       # 测试
    ├── unit/                    # 单元测试
    └── integration/             # 集成测试
```

## 依赖关系图

```
Application Layer
    ↓
Graph Execution Engine
    ↓
Operator Layer ──→ Backend Layer
    ↓
Tensor & Memory Layer
```

## 核心类关系

```
Tensor ──uses──→ Allocator
  ↑
  │
Operator ──creates──→ Tensor
  ↑
  │
GraphExecutor ──executes──→ Operator
```

## 关键接口

### Tensor接口
```cpp
class Tensor {
  DataType dtype();
  const Shape& shape();
  void* data();
  template<typename T> T* data_as();
};
```

### Operator接口
```cpp
class Operator {
  virtual std::vector<Tensor> forward(
      const std::vector<Tensor>& inputs) = 0;
  virtual std::vector<Shape> infer_output_shapes(
      const std::vector<Shape>& input_shapes) const = 0;
};
```

### BLAS接口
```cpp
class BLAS {
  virtual void gemm(...) = 0;
  // 其他BLAS操作
};
```

## 使用示例

### 1. 创建Tensor
```cpp
#include "llm_runtime/tensor.h"

Shape shape({2, 3, 1024});
Tensor tensor(DataType::FLOAT32, shape);
tensor.allocate();
```

### 2. 使用算子
```cpp
#include "operators/math/matmul.h"

MatMulOp op;
auto inputs = {tensor_a, tensor_b};
auto outputs = op.forward(inputs);
```

### 3. 执行模型
```cpp
#include "llm_runtime/inference.h"

Model model;
model.load("qwen3_0.6b.onnx");

auto inputs = prepare_inputs(...);
auto outputs = model.run(inputs);
```

## 迁移指南

### 从当前代码迁移

1. **替换Eigen Matrix**
   ```cpp
   // 旧代码
   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> m;
   
   // 新代码
   Tensor tensor(DataType::FLOAT32, Shape({rows, cols}));
   float* data = tensor.data_as<float>();
   ```

2. **算子接口统一**
   ```cpp
   // 旧代码
   matrix_multiply(A, B);
   
   // 新代码
   MatMulOp op;
   op.forward({tensor_a, tensor_b});
   ```

3. **内存管理**
   ```cpp
   // 旧代码：Eigen自动管理
   // 新代码：显式管理
   Tensor tensor(...);
   tensor.allocate();
   // ... 使用
   tensor.deallocate();
   ```

## 配置选项

### CMake选项
```cmake
option(USE_EIGEN "Use Eigen backend" ON)
option(USE_RVV "Enable RISC-V Vector extension" OFF)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
```

### 运行时配置
```cpp
Config::instance().set("backend", "rvv");
Config::instance().set("num_threads", "4");
Config::instance().set("memory_pool_size", "1GB");
```
