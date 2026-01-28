// Simple operator registry for mapping op names to factory functions.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "core/operators/base/operator.h"

namespace llm_on_riscv {
namespace core {
namespace operators {

class OperatorRegistry {
 public:
  using OperatorFactory = std::function<std::unique_ptr<Operator>()>;

  // Singleton access.
  static OperatorRegistry& Instance() {
    static OperatorRegistry instance;
    return instance;
  }

  // Register a factory for the given operator type name.
  void Register(const std::string& op_type, OperatorFactory factory) {
    factories_[op_type] = std::move(factory);
  }

  // Create an operator instance for the given type name.
  // Returns nullptr if the type is unknown.
  [[nodiscard]] std::unique_ptr<Operator> Create(
      const std::string& op_type) const {
    auto it = factories_.find(op_type);
    if (it == factories_.end()) {
      return nullptr;
    }
    return (it->second)();
  }

 private:
  OperatorRegistry() = default;

  std::unordered_map<std::string, OperatorFactory> factories_;
};

// Helper macro for registering an operator type. Usage:
//   REGISTER_OPERATOR("Add", AddOp);
// This creates a single inline variable whose initializer performs the
// registration at static-init time.
#define REGISTER_OPERATOR(OP_NAME, OP_CLASS)                                    \
  inline const bool g_registered_op_##OP_CLASS = []() {                         \
    ::llm_on_riscv::core::operators::OperatorRegistry::Instance().Register(     \
        (OP_NAME),                                                              \
        []() -> std::unique_ptr<::llm_on_riscv::core::operators::Operator> {    \
          return std::make_unique<OP_CLASS>();                                  \
        });                                                                      \
    return true;                                                                \
  }()

}  // namespace operators
}  // namespace core
}  // namespace llm_on_riscv

