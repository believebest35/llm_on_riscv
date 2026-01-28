#include "core/tensor/tensor.h"

#include <cstring>
#include <stdexcept>

#include "core/memory/allocator.h"
#include "core/types.h"

namespace llm_on_riscv {
namespace core {

Tensor::Tensor(DataType dtype, const Shape& shape)
    : dtype_(dtype),
      shape_(shape),
      data_(nullptr),
      owns_data_(false),
      allocator_(GetDefaultAllocator()) {}

Tensor::Tensor(DataType dtype, const Shape& shape, void* data, bool owns_data)
    : dtype_(dtype),
      shape_(shape),
      data_(data),
      owns_data_(owns_data),
      allocator_(GetDefaultAllocator()) {}

Tensor::~Tensor() { Deallocate(); }

Tensor::Tensor(Tensor&& other) noexcept
    : dtype_(other.dtype_),
      shape_(other.shape_),
      data_(other.data_),
      owns_data_(other.owns_data_),
      allocator_(std::move(other.allocator_)) {
  other.data_ = nullptr;
  other.owns_data_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    Deallocate();
    dtype_ = other.dtype_;
    shape_ = other.shape_;
    data_ = other.data_;
    owns_data_ = other.owns_data_;
    allocator_ = std::move(other.allocator_);

    other.data_ = nullptr;
    other.owns_data_ = false;
  }
  return *this;
}

std::int64_t Tensor::NumElements() const { return shape_.NumElements(); }

std::size_t Tensor::Bytes() const {
  auto numel = NumElements();
  if (numel < 0) {
    throw std::runtime_error("Tensor::Bytes: shape has unknown size (negative dimension)");
  }
  return static_cast<std::size_t>(numel) * SizeOf(dtype_);
}

void Tensor::Allocate(std::shared_ptr<Allocator> allocator) {
  if (data_ != nullptr && owns_data_) {
    // Already allocated.
    return;
  }
  if (!allocator) {
    allocator = GetDefaultAllocator();
  }
  allocator_ = allocator;

  const auto num_bytes = Bytes();
  if (num_bytes == 0) {
    data_ = nullptr;
    owns_data_ = false;
    return;
  }

  void* ptr = allocator_->Allocate(num_bytes);
  if (!ptr) {
    throw std::bad_alloc();
  }
  data_ = ptr;
  owns_data_ = true;
}

void Tensor::Deallocate() {
  if (data_ != nullptr && owns_data_ && allocator_) {
    allocator_->Deallocate(data_);
  }
  data_ = nullptr;
  owns_data_ = false;
}

Tensor Tensor::Clone() const {
  Tensor copy(dtype_, shape_);
  copy.Allocate();

  if (data_ != nullptr && copy.data_ != nullptr) {
    std::memcpy(copy.data_, data_, Bytes());
  }
  return copy;
}

void Tensor::FillZero() {
  if (data_ == nullptr) {
    Allocate();
  }
  if (data_ != nullptr) {
    std::memset(data_, 0, Bytes());
  }
}

}  // namespace core
}  // namespace llm_on_riscv

