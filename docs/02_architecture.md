# 架构分析与优化建议

## 一、项目现状分析

### 1.1 当前架构

**优点：**
- ✅ 清晰的目录结构（src/core/, tests/, third_party/）
- ✅ 使用现代C++17标准
- ✅ 完善的测试框架（GoogleTest）
- ✅ 代码格式化工具（clang-format, pre-commit）

**存在的问题：**
- ❌ 缺少统一的Tensor抽象层
- ❌ 算子实现分散，没有统一的接口
- ❌ 缺少ONNX模型加载和解析模块
- ❌ 缺少图执行引擎
- ❌ 第三方库管理不够灵活（Eigen可能不适合RISC-V优化）
- ❌ 缺少内存管理抽象
- ❌ 头文件组织不够清晰，缺少分层

### 1.2 技术挑战

1. **RISC-V架构特点**
   - 向量扩展（RVV）支持有限
   - 内存带宽可能受限
   - 需要针对性的优化策略

2. **ONNX模型执行**
   - 需要图解析和拓扑排序
   - 动态shape支持
   - 内存池管理

3. **性能优化**
   - 算子融合机会
   - 内存布局优化
   - SIMD/向量化优化

## 二、推荐架构设计

### 2.1 整体架构分层

```
┌─────────────────────────────────────────┐
│         Application Layer               │
│  (Model Loading, Inference API)        │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Graph Execution Engine          │
│  (ONNX Graph, Node Execution)           │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Operator Layer                  │
│  (MatMul, Attention, LayerNorm, etc.)  │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Tensor & Memory Layer           │
│  (Tensor Abstraction, Memory Pool)      │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Backend Layer                   │
│  (CPU/RVV, BLAS, Memory Allocator)     │
└─────────────────────────────────────────┘
```

### 2.2 目录结构建议

```
llm_on_riscv/
├── include/                    # 公共头文件（对外API）
│   └── llm_runtime/
│       ├── tensor.h
│       ├── model.h
│       └── inference.h
├── src/
│   ├── core/                   # 核心抽象层
│   │   ├── tensor/            # Tensor实现
│   │   │   ├── tensor.h
│   │   │   ├── tensor.cpp
│   │   │   └── shape.h
│   │   ├── memory/            # 内存管理
│   │   │   ├── allocator.h
│   │   │   ├── memory_pool.h
│   │   │   └── memory_pool.cpp
│   │   └── types.h            # 类型定义
│   ├── graph/                  # 图执行引擎
│   │   ├── onnx/              # ONNX解析
│   │   │   ├── model_loader.h
│   │   │   ├── model_loader.cpp
│   │   │   └── onnx_types.h
│   │   ├── executor.h         # 执行引擎
│   │   ├── executor.cpp
│   │   └── node.h             # 节点抽象
│   ├── operators/              # 算子实现
│   │   ├── base/              # 算子基类
│   │   │   ├── operator.h
│   │   │   └── operator_registry.h
│   │   ├── math/              # 数学算子
│   │   │   ├── matmul.h
│   │   │   ├── matmul.cpp
│   │   │   ├── mul.h
│   │   │   └── add.h
│   │   ├── tensor/            # 张量操作
│   │   │   ├── reshape.h
│   │   │   ├── transpose.h
│   │   │   └── concat.h
│   │   ├── normalization/     # 归一化算子
│   │   │   ├── layernorm.h
│   │   │   └── skip_layernorm.h
│   │   ├── attention/         # 注意力算子
│   │   │   ├── rotary_embedding.h
│   │   │   └── group_query_attention.h
│   │   └── activation/        # 激活函数
│   │       └── sigmoid.h
│   └── backend/                # 后端实现
│       ├── cpu/               # CPU后端
│       │   ├── blas.h         # BLAS接口
│       │   ├── blas.cpp
│       │   └── rvv/           # RISC-V向量扩展
│       │       ├── rvv_matmul.h
│       │       └── rvv_utils.h
│       └── allocator/         # 内存分配器
│           ├── aligned_allocator.h
│           └── pool_allocator.h
├── third_party/                # 第三方库
│   ├── onnx/                  # ONNX protobuf（可选）
│   ├── eigen/                 # Eigen（当前使用）
│   └── gtest/                 # GoogleTest
├── tests/                      # 测试
│   ├── unit/                  # 单元测试
│   │   ├── tensor/
│   │   ├── operators/
│   │   └── graph/
│   └── integration/           # 集成测试
│       └── onnx_models/
├── examples/                   # 示例代码
│   └── inference_example.cpp
└── tools/                      # 工具脚本
    └── benchmark/
```

## 三、核心组件设计

### 3.1 Tensor抽象层

**设计目标：**
- 统一的张量表示
- 支持多种数据类型（float32, int32, int64等）
- 内存布局控制（行主序/列主序）
- Shape动态管理

**建议实现：**

```cpp
// src/core/tensor/tensor.h
namespace llm_runtime {

enum class DataType {
  FLOAT32,
  INT32,
  INT64,
  // ...
};

class Shape {
public:
  Shape() = default;
  Shape(std::vector<int64_t> dims);
  
  int64_t rank() const;
  int64_t dim(int idx) const;
  int64_t total_elements() const;
  bool is_dynamic() const;
  
private:
  std::vector<int64_t> dims_;
};

class Tensor {
public:
  Tensor(DataType dtype, Shape shape);
  Tensor(DataType dtype, Shape shape, void* data, bool owns_data = false);
  
  // 访问器
  DataType dtype() const { return dtype_; }
  const Shape& shape() const { return shape_; }
  void* data() { return data_; }
  const void* data() const { return data_; }
  
  // 类型安全的访问
  template<typename T>
  T* data_as() { return static_cast<T*>(data_); }
  
  template<typename T>
  const T* data_as() const { return static_cast<const T*>(data_); }
  
  // 内存管理
  void allocate();
  void deallocate();
  
private:
  DataType dtype_;
  Shape shape_;
  void* data_;
  bool owns_data_;
  std::shared_ptr<Allocator> allocator_;
};

} // namespace llm_runtime
```

### 3.2 算子基类设计

**设计目标：**
- 统一的算子接口
- 算子注册机制
- 支持算子融合
- 类型推断

**建议实现：**

```cpp
// src/operators/base/operator.h
namespace llm_runtime {

class Operator {
public:
  virtual ~Operator() = default;
  
  // 算子执行
  virtual std::vector<Tensor> forward(
      const std::vector<Tensor>& inputs) = 0;
  
  // 类型推断
  virtual std::vector<DataType> infer_output_types(
      const std::vector<DataType>& input_types) const = 0;
  
  // Shape推断
  virtual std::vector<Shape> infer_output_shapes(
      const std::vector<Shape>& input_shapes) const = 0;
  
  // 算子名称
  virtual const char* name() const = 0;
  
protected:
  // 验证输入
  void validate_inputs(const std::vector<Tensor>& inputs) const;
};

// 算子注册表
class OperatorRegistry {
public:
  static OperatorRegistry& instance();
  
  void register_operator(const std::string& name,
                        std::unique_ptr<Operator> op);
  
  std::unique_ptr<Operator> create_operator(
      const std::string& name,
      const std::map<std::string, Attribute>& attrs);
  
private:
  std::map<std::string, std::function<std::unique_ptr<Operator>(
      const std::map<std::string, Attribute>&)>> factories_;
};

} // namespace llm_runtime
```

### 3.3 内存管理

**设计目标：**
- 内存池管理
- 对齐分配（SIMD优化需要）
- 内存复用
- 统计和调试

**建议实现：**

```cpp
// src/core/memory/memory_pool.h
namespace llm_runtime {

class Allocator {
public:
  virtual ~Allocator() = default;
  virtual void* allocate(size_t size, size_t alignment = 16) = 0;
  virtual void deallocate(void* ptr) = 0;
};

class PoolAllocator : public Allocator {
public:
  PoolAllocator(size_t pool_size);
  ~PoolAllocator();
  
  void* allocate(size_t size, size_t alignment = 16) override;
  void deallocate(void* ptr) override;
  
  // 统计信息
  size_t allocated_bytes() const;
  size_t peak_allocated_bytes() const;
  
private:
  struct Block {
    void* ptr;
    size_t size;
    bool in_use;
  };
  
  void* pool_;
  size_t pool_size_;
  std::vector<Block> blocks_;
};

} // namespace llm_runtime
```

### 3.4 图执行引擎

**设计目标：**
- ONNX图解析
- 拓扑排序
- 执行计划优化
- 内存规划

**建议实现：**

```cpp
// src/graph/executor.h
namespace llm_runtime {

class GraphNode {
public:
  std::string name;
  std::string op_type;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::map<std::string, Attribute> attributes;
};

class GraphExecutor {
public:
  // 加载ONNX模型
  void load_model(const std::string& model_path);
  
  // 执行推理
  std::map<std::string, Tensor> run(
      const std::map<std::string, Tensor>& inputs);
  
  // 设置执行选项
  void set_execution_options(const ExecutionOptions& opts);
  
private:
  // 构建执行图
  void build_execution_graph();
  
  // 拓扑排序
  std::vector<GraphNode*> topological_sort();
  
  // 内存规划
  void plan_memory();
  
  std::vector<GraphNode> nodes_;
  std::map<std::string, Tensor> tensors_;
  std::shared_ptr<MemoryPool> memory_pool_;
};

} // namespace llm_runtime
```

## 四、头文件组织建议

### 4.1 公共API头文件（include/）

**原则：**
- 只暴露必要的公共接口
- 用户只需要包含少量头文件
- 隐藏实现细节

```
include/
└── llm_runtime/
    ├── llm_runtime.h          # 主头文件（包含所有公共API）
    ├── tensor.h               # Tensor API
    ├── model.h                # Model加载API
    └── inference.h            # 推理API
```

### 4.2 内部头文件（src/）

**原则：**
- 按模块组织
- 使用前向声明减少依赖
- 清晰的依赖关系

**依赖规则：**
- `core/` 不依赖 `operators/` 或 `graph/`
- `operators/` 依赖 `core/`
- `graph/` 依赖 `operators/` 和 `core/`
- `backend/` 可被 `operators/` 使用

### 4.3 头文件包含顺序

```cpp
// 1. 对应的头文件
#include "my_class.h"

// 2. C/C++标准库
#include <vector>
#include <memory>

// 3. 第三方库
#include <Eigen/Dense>

// 4. 项目内部头文件
#include "core/tensor/tensor.h"
#include "operators/base/operator.h"
```

## 五、第三方库管理

### 5.1 当前问题

- Eigen可能不是RISC-V优化的最佳选择
- 缺少灵活的BLAS后端切换
- 第三方库版本管理不明确

### 5.2 建议方案

#### 方案A：使用CMake FetchContent（推荐）

```cmake
# cmake/dependencies.cmake
include(FetchContent)

# Eigen（可选，用于开发/测试）
option(USE_EIGEN "Use Eigen library" ON)
if(USE_EIGEN)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
  )
  FetchContent_MakeAvailable(eigen)
endif()

# ONNX（如果需要运行时解析）
option(USE_ONNX "Use ONNX runtime" OFF)
if(USE_ONNX)
  # 只包含protobuf定义，不包含完整runtime
endif()
```

#### 方案B：使用Git Submodules（当前方案改进）

```
third_party/
├── eigen/          # Git submodule
├── gtest/          # Git submodule
└── onnx/           # Git submodule（可选）
```

**改进：**
- 添加版本锁定文件 `third_party/versions.txt`
- 添加 `scripts/setup_deps.sh` 脚本

#### 方案C：抽象BLAS接口（关键）

```cpp
// src/backend/cpu/blas.h
namespace llm_runtime {
namespace backend {

class BLAS {
public:
  virtual ~BLAS() = default;
  
  // GEMM: C = alpha * A * B + beta * C
  virtual void gemm(
      bool trans_a, bool trans_b,
      int m, int n, int k,
      float alpha, const float* A, int lda,
      const float* B, int ldb,
      float beta, float* C, int ldc) = 0;
  
  // 其他BLAS操作...
};

// 实现：Eigen后端
class EigenBLAS : public BLAS {
  // ...
};

// 实现：RVV优化后端
class RVVBLAS : public BLAS {
  // ...
};

// 工厂函数
std::unique_ptr<BLAS> create_blas(const std::string& backend = "auto");

} // namespace backend
} // namespace llm_runtime
```

### 5.3 RISC-V特定优化

**建议：**
1. **条件编译支持RVV**
   ```cpp
   #ifdef __riscv_vector
   #include "backend/cpu/rvv/rvv_matmul.h"
   #endif
   ```

2. **运行时检测**
   ```cpp
   bool has_rvv_support();
   bool has_rvv_fp16_support();
   ```

3. **分层实现**
   - Level 1: 纯C++实现（可移植）
   - Level 2: 内联汇编优化（RISC-V特定）
   - Level 3: RVV向量化（如果支持）

## 六、基础类设计建议

### 6.1 核心基础类

1. **Tensor** - 张量抽象（已讨论）

2. **Shape** - Shape管理（已讨论）

3. **Attribute** - 算子属性
   ```cpp
   class Attribute {
   public:
     enum Type { FLOAT, INT, STRING, TENSOR, FLOATS, INTS };
     
     template<typename T>
     T get() const;
     
   private:
     Type type_;
     std::variant<float, int64_t, std::string, Tensor, 
                  std::vector<float>, std::vector<int64_t>> value_;
   };
   ```

4. **Error/Status** - 错误处理
   ```cpp
   class Status {
   public:
     static Status OK();
     static Status Error(const std::string& msg);
     
     bool is_ok() const;
     const std::string& message() const;
   };
   ```

### 6.2 工具类

1. **Logger** - 日志系统
   ```cpp
   class Logger {
   public:
     enum Level { DEBUG, INFO, WARN, ERROR };
     static void log(Level level, const std::string& msg);
   };
   ```

2. **Profiler** - 性能分析
   ```cpp
   class Profiler {
   public:
     void start(const std::string& name);
     void stop(const std::string& name);
     void report() const;
   };
   ```

3. **Config** - 配置管理
   ```cpp
   class Config {
   public:
     void set(const std::string& key, const std::string& value);
     std::string get(const std::string& key, 
                     const std::string& default_val = "") const;
   };
   ```

## 七、实施路线图

### Phase 1: 基础架构重构（1-2周）

1. ✅ 创建新的目录结构
2. ✅ 实现Tensor和Shape类
3. ✅ 实现内存分配器
4. ✅ 重构现有算子使用新接口

### Phase 2: 图执行引擎（2-3周）

1. ✅ 实现ONNX模型加载（简化版）
2. ✅ 实现图执行引擎
3. ✅ 实现节点注册机制
4. ✅ 端到端测试

### Phase 3: 算子完善（持续）

1. ✅ 实现所有必需算子
2. ✅ 添加单元测试
3. ✅ 性能优化

### Phase 4: RISC-V优化（2-3周）

1. ✅ 添加RVV支持检测
2. ✅ 实现RVV优化的关键算子
3. ✅ 交叉编译支持
4. ✅ 在模拟器上测试

## 八、最佳实践建议

### 8.1 代码组织

- **单一职责原则**：每个类/文件只做一件事
- **依赖倒置**：高层模块不依赖低层模块，都依赖抽象
- **接口隔离**：使用小的、专门的接口

### 8.2 性能考虑

- **内存对齐**：SIMD操作需要16/32字节对齐
- **缓存友好**：注意内存访问模式
- **零拷贝**：尽可能复用内存
- **延迟分配**：只在需要时分配内存

### 8.3 可移植性

- **平台抽象**：使用抽象层隔离平台差异
- **条件编译**：使用宏控制平台特定代码
- **运行时检测**：检测硬件特性而非编译时假设

### 8.4 测试策略

- **单元测试**：每个算子独立测试
- **集成测试**：端到端模型推理测试
- **性能测试**：基准测试套件
- **正确性测试**：与ONNX Runtime对比

## 九、总结

### 关键改进点

1. **统一Tensor抽象** - 所有算子使用统一的Tensor接口
2. **算子注册机制** - 便于扩展和维护
3. **内存池管理** - 提高性能和内存效率
4. **BLAS抽象层** - 支持多种后端，便于RISC-V优化
5. **清晰的模块划分** - 降低耦合，提高可维护性

### 下一步行动

1. 创建新的目录结构
2. 实现核心基础类（Tensor, Shape, Allocator）
3. 重构现有算子
4. 逐步添加新功能

这个架构设计为项目提供了清晰的路线图，既保证了当前功能的正确性，又为未来的RISC-V优化和扩展留下了空间。
