// Simple allocator abstraction used by tensors and memory pools.

#pragma once

#include <cstddef>
#include <memory>
#include <new>

namespace llm_on_riscv {
namespace core {

class Allocator {
 public:
  virtual ~Allocator() = default;

  // Allocate `size` bytes with at least `alignment` alignment.
  virtual void* Allocate(std::size_t size,
                         std::size_t alignment = alignof(std::max_align_t)) = 0;

  // Free memory previously allocated by Allocate.
  virtual void Deallocate(void* ptr) = 0;
};

// A very small default allocator which simply forwards to ::operator new/delete.
class DefaultAllocator final : public Allocator {
 public:
  void* Allocate(std::size_t size,
                 std::size_t alignment) override {
    (void)alignment;  // alignment is ignored; ::operator new is suitably aligned.
    return ::operator new(size, std::nothrow);
  }

  void Deallocate(void* ptr) override { ::operator delete(ptr); }
};

// Helper to get a shared default allocator instance.
inline std::shared_ptr<Allocator> GetDefaultAllocator() {
  static std::shared_ptr<Allocator> allocator =
      std::make_shared<DefaultAllocator>();
  return allocator;
}

}  // namespace core
}  // namespace llm_on_riscv


