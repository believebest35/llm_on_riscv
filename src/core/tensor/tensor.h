// Basic Tensor class used by operators and the execution engine.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "core/tensor/shape.h"
#include "core/types.h"

namespace llm_on_riscv {
namespace core {

class Allocator;  // forward declaration

class Tensor {
 public:
  Tensor(DataType dtype, const Shape& shape);
  Tensor(DataType dtype, const Shape& shape, void* data, bool owns_data);
  ~Tensor();

  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;

  DataType Dtype() const { return dtype_; }
  const Shape& GetShape() const { return shape_; }

  std::int64_t NumElements() const;
  std::size_t Bytes() const;

  void* Data() { return data_; }
  const void* Data() const { return data_; }

  template <typename T>
  T* DataAs() {
    return static_cast<T*>(data_);
  }

  template <typename T>
  const T* DataAs() const {
    return static_cast<const T*>(data_);
  }

  bool IsAllocated() const { return data_ != nullptr; }

  void Allocate(std::shared_ptr<Allocator> allocator = nullptr);
  void Deallocate();

  Tensor Clone() const;
  void FillZero();

 private:
  DataType dtype_;
  Shape shape_;
  void* data_;
  bool owns_data_;
  std::shared_ptr<Allocator> allocator_;
};

}  // namespace core
}  // namespace llm_on_riscv

