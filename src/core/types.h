// Basic type definitions for the core runtime.

#pragma once

#include <cstddef>
#include <cstdint>

namespace llm_on_riscv {
namespace core {

// Data types supported by the runtime. Extend as needed.
enum class DataType {
  FLOAT32 = 0,
  FLOAT16,
  INT32,
  INT64,
  BOOL,
  UINT8,
};

// Return the size in bytes of a single element of the given DataType.
inline std::size_t SizeOf(DataType dtype) {
  switch (dtype) {
    case DataType::FLOAT32:
      return 4;
    case DataType::FLOAT16:
      return 2;
    case DataType::INT32:
      return 4;
    case DataType::INT64:
      return 8;
    case DataType::BOOL:
      return 1;
    case DataType::UINT8:
      return 1;
    default:
      return 0;
  }
}

// Convert DataType to a human‑readable string (useful for logging / errors).
inline const char* ToString(DataType dtype) {
  switch (dtype) {
    case DataType::FLOAT32:
      return "float32";
    case DataType::FLOAT16:
      return "float16";
    case DataType::INT32:
      return "int32";
    case DataType::INT64:
      return "int64";
    case DataType::BOOL:
      return "bool";
    case DataType::UINT8:
      return "uint8";
    default:
      return "unknown";
  }
}

}  // namespace core
}  // namespace llm_on_riscv

