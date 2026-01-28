// A simple linear memory pool implementing the Allocator interface.
// This is intentionally minimal and correctness-oriented; you can
// optimize or replace it later for RISC-V targets.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/memory/allocator.h"

namespace llm_on_riscv {
namespace core {

class MemoryPool final : public Allocator {
 public:
  explicit MemoryPool(std::size_t pool_size_bytes);
  ~MemoryPool() override;

  void* Allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) override;
  void Deallocate(void* ptr) override;

  std::size_t PoolSize() const { return pool_size_bytes_; }
  std::size_t Used() const { return offset_; }

 private:
  void* base_;               // base pointer of the pool
  std::size_t pool_size_bytes_;
  std::size_t offset_;       // current allocation offset

  // For very small project we do not attempt to recycle freed blocks;
  // Deallocate is a no-op for pool allocations. Non-pooled allocations
  // (fallback) are tracked separately.
  std::vector<void*> fallback_allocations_;
};

}  // namespace core
}  // namespace llm_on_riscv

