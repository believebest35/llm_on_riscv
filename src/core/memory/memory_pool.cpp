#include "core/memory/memory_pool.h"

#include <algorithm>
#include <cstdlib>

namespace llm_on_riscv {
namespace core {

namespace {
// Align `offset` up to `alignment` (power of two).
std::size_t AlignUp(std::size_t offset, std::size_t alignment) {
  const std::size_t mask = alignment - 1;
  return (offset + mask) & ~mask;
}
}  // namespace

MemoryPool::MemoryPool(std::size_t pool_size_bytes)
    : base_(nullptr), pool_size_bytes_(pool_size_bytes), offset_(0) {
  if (pool_size_bytes_ > 0) {
    base_ = std::malloc(pool_size_bytes_);
  }
}

MemoryPool::~MemoryPool() {
  // Free pooled memory.
  if (base_ != nullptr) {
    std::free(base_);
    base_ = nullptr;
  }

  // Free any fallback allocations.
  for (void* ptr : fallback_allocations_) {
    std::free(ptr);
  }
  fallback_allocations_.clear();
}

void* MemoryPool::Allocate(std::size_t size, std::size_t alignment) {
  if (size == 0) {
    return nullptr;
  }

  if (base_ != nullptr && pool_size_bytes_ > 0) {
    std::size_t aligned_offset = AlignUp(offset_, alignment);
    if (aligned_offset + size <= pool_size_bytes_) {
      auto* ptr = static_cast<std::uint8_t*>(base_) + aligned_offset;
      offset_ = aligned_offset + size;
      return ptr;
    }
  }

  // Fallback: allocate from the heap using operator new (ignoring explicit
  // alignment request for simplicity; it will still be suitably aligned for
  // most types).
  void* heap_ptr = ::operator new(size, std::nothrow);
  if (heap_ptr != nullptr) {
    fallback_allocations_.push_back(heap_ptr);
  }
  return heap_ptr;
}

void MemoryPool::Deallocate(void* /*ptr*/) {
  // For simplicity we do not recycle individual blocks from the pool.
  // Fallback allocations are freed in the destructor.
}

}  // namespace core
}  // namespace llm_on_riscv

