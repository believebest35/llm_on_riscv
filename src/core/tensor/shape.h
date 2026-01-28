// Simple shape representation for tensors.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llm_on_riscv {
namespace core {

class Shape {
 public:
  Shape() = default;
  explicit Shape(std::vector<std::int64_t> dims) : dims_(std::move(dims)) {}

  // Rank of the tensor (number of dimensions).
  std::int64_t Rank() const {
    return static_cast<std::int64_t>(dims_.size());
  }

  // Access a specific dimension (no bounds check).
  std::int64_t Dim(std::int64_t idx) const { return dims_.at(idx); }

  const std::vector<std::int64_t>& Dims() const { return dims_; }

  // Total number of elements. If any dimension is negative (e.g. dynamic),
  // return -1 to indicate "unknown".
  std::int64_t NumElements() const {
    if (dims_.empty()) {
      return 0;
    }
    std::int64_t total = 1;
    for (auto d : dims_) {
      if (d < 0) {
        return -1;
      }
      total *= d;
    }
    return total;
  }

  bool operator==(const Shape& other) const { return dims_.size() == other.dims_.size() && dims_ == other.dims_; }
  bool operator!=(const Shape& other) const { return !(*this == other); }

  std::string ToString() const {
    std::string result = "[";
    for (std::size_t i = 0; i < dims_.size(); ++i) {
      result += std::to_string(dims_[i]);
      if (i + 1 < dims_.size()) {
        result += ", ";
      }
    }
    result += "]";
    return result;
  }

 private:
  std::vector<std::int64_t> dims_;
};

}  // namespace core
}  // namespace llm_on_riscv

