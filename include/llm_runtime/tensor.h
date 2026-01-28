// Public Tensor/Shape/DataType facade header.
// External users should include this file instead of the internal
// core headers, so that we can evolve internals without breaking API.

#pragma once

#include "core/tensor/shape.h"
#include "core/tensor/tensor.h"
#include "core/types.h"

namespace llm_on_riscv {

// Re-export core types into the top-level namespace for convenience.
using DataType = core::DataType;
using Shape = core::Shape;
using Tensor = core::Tensor;

}  // namespace llm_on_riscv

